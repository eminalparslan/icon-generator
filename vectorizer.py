import os
import time
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pydiffvg
import svgpathtools

from typing import Tuple, Optional

"""
Some of the utility function are adapted from https://github.com/etaisella/svgLearner
    canonizePaths, tensor2SVG, renderSVG, parts of IconDataset
"""

MAX_CURVES = 32

def canonize_image(image: torch.Tensor) -> torch.Tensor:
    # convert to grayscale
    image = image.mean(dim=-1)
    image = image / image.max()
    image = 1 - image
    return image.unsqueeze(0)

def decanonize_image(image: torch.Tensor) -> torch.Tensor:
    # expand to rgb and add alpha channel
    image_rgb = image.repeat(3, 1, 1)
    alpha = torch.ones_like(image)
    image_rgba = torch.cat([image_rgb, alpha], dim=0)
    return image_rgba

class IconDataset(Dataset):
    def __init__(self, data_dir: str):
        render = pydiffvg.RenderFunction.apply
        self.paths = []
        self.images = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if not file.endswith('.svg'):
                    continue

                file_path = os.path.join(root, file)

                # converts all SVG elements to paths
                paths, _ = svgpathtools.svg2paths(file_path) # type: ignore

                path_tensors = []
                skip = False
                for path in paths:
                    path_points = []
                    for segment in path:
                        if isinstance(segment, svgpathtools.CubicBezier):
                            path_points.append([segment.start.real, segment.start.imag])
                            path_points.append([segment.control1.real, segment.control1.imag])
                            path_points.append([segment.control2.real, segment.control2.imag])
                            path_points.append([segment.end.real, segment.end.imag])
                        elif isinstance(segment, svgpathtools.Line):
                            # convert line to cubic bezier by adding control points
                            # along the line at 1/3 and 2/3 positions
                            t1 = segment.start + (segment.end - segment.start) * 0.333
                            t2 = segment.start + (segment.end - segment.start) * 0.667
                            path_points.append([segment.start.real, segment.start.imag])
                            path_points.append([t1.real, t1.imag])
                            path_points.append([t2.real, t2.imag])
                            path_points.append([segment.end.real, segment.end.imag])
                        elif isinstance(segment, svgpathtools.Arc):
                            skip = True
                            break
                        else:
                            raise ValueError(f"Unsupported path segment type: {type(segment)}")
                    path_tensors.append(torch.tensor(path_points, dtype=torch.float32))

                # skip if too many points or svg includes arc
                if skip or sum(len(p) for p in path_tensors) > MAX_CURVES * 4:
                    continue

                # TODO: possibly skip icons that have stroke fill

                canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(file_path)

                path_tensors = torch.concat(path_tensors)
                path_tensors = path_tensors.reshape(-1, 2)
                path_tensors = path_tensors / torch.tensor([canvas_width, canvas_height])
                assert path_tensors.min() >= 0 and path_tensors.max() <= 1
                path_tensors = path_tensors.reshape(-1, 4, 2)
                self.paths.append(path_tensors)

                # scale up rasterized image
                scale_factor = 16
                canvas_width *= scale_factor
                canvas_height *= scale_factor
                for shape in shapes:
                    shape.points[:, 0] *= scale_factor
                    shape.points[:, 1] *= scale_factor

                scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
                img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

                self.images.append(img)

        assert len(self.paths) == len(self.images)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = canonize_image(self.images[idx])
        paths = self.paths[idx]
        return image, paths


class BezierVectorizer(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, MAX_CURVES * 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate Bezier curves from image.

        Args:
            x: Input image tensor of shape (B, C, H, W)

        Returns:
            curve_points: Tensor of Bezier curve points (B, num_curves, 4, 2)
        """
        assert x.dim() == 4, f"Expected 4D tensor (B,C,H,W), got {x.dim()}D"
        assert x.size(1) == 1, f"Expected 1 channel, got {x.size(1)}"

        curve_points = self.layers(x).view(-1, MAX_CURVES, 4, 2)
        return torch.sigmoid(curve_points)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model

class DiffusionVectorizer(nn.Module):
    def __init__(self,
                 embedding_dim: int = 256,
                 time_dim: int = 256,
                 num_steps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()

        # Diffusion parameters
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Calculate diffusion schedule
        self.beta = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2),  # 192x192
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # 96x96
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # 48x48
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # 8x8
        )

        # Middle blocks with time conditioning
        self.middle = nn.ModuleList([
            TimestepEmbedSequential(128, time_dim) for _ in range(3)
        ])

        # Decoder that outputs curve parameters directly
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, MAX_CURVES * 8),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Embed time
        t = self.time_mlp(timesteps)

        # Encode
        h = self.encoder(x)

        # Apply middle blocks with time conditioning
        for block in self.middle:
            h = block(h, t)

        # Decode to curve parameters
        out = self.decoder(h)
        out = out.view(-1, MAX_CURVES, 4, 2)
        return torch.sigmoid(out)

    def sample(self, x, num_steps=None):
        num_steps = num_steps or self.num_steps
        device = x.device

        # Start from random noise
        shape = (x.shape[0], MAX_CURVES, 4, 2)
        curves = torch.randn(shape, device=device)

        # Gradually denoise
        for t in reversed(range(num_steps)):
            t_batch = torch.ones(x.shape[0], device=device) * t

            # Get model prediction
            pred = self(x, t_batch)

            # Update sample
            alpha_t = self.alpha[t]
            alpha_t_hat = self.alpha_hat[t]
            beta_t = self.beta[t]

            if t > 0:
                noise = torch.randn_like(curves)
            else:
                noise = 0

            curves = (1 / torch.sqrt(alpha_t)) * (
                curves - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_hat)) * noise
            ) + torch.sqrt(beta_t) * pred
            # curves = (1 / torch.sqrt(alpha_t)) * (
            #     curves - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_hat)) * pred
            # ) + torch.sqrt(beta_t) * noise

        out = torch.sigmoid(curves)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimestepEmbedSequential(nn.Module):
    def __init__(self, channels, time_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, t):
        h = self.net(x)
        time_emb = self.time_mlp(t)
        h = h + time_emb[..., None, None]
        return h


def decanonize_paths(paths_tensor: torch.Tensor, im_height: int=384, im_width: int=384) -> torch.Tensor:
    assert paths_tensor.size(-1) == 2, "Last dimension should be 2 (x,y coordinates)"
    device = paths_tensor.device
    # Scale back to image dimensions
    paths_tensor = paths_tensor * torch.tensor([im_width, im_height], device=device)
    return paths_tensor

class RenderBatch(Dataset):
    def __init__(self, batch_paths: torch.Tensor):
        self.batch_paths = batch_paths

    def __len__(self):
        return self.batch_paths.shape[0]

    def __getitem__(self, idx):
        path = self.batch_paths[idx]
        curves = decanonize_paths(path)
        rendered = renderSVG(*tensor2SVG(curves))
        return canonize_image(rendered)

def train_bezier_vectorizer(train_loader, val_loader, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BezierVectorizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    alpha = 0.8

    try:
        for epoch in tqdm(range(epochs)):
            model.train()
            losses = []

            for i, (batch_images, batch_paths) in enumerate(train_loader):
                batch_images = batch_images.to(device)
                batch_paths = batch_paths.to(device)

                # TODO: paths -> SVG -> raster (boxfiltered?)

                optimizer.zero_grad()
                pred_curve_points = model(batch_images)

                assert pred_curve_points.min() >= 0 and pred_curve_points.max() <= 1
                assert batch_paths.min() >= 0 and batch_paths.max() <= 1
                direct_loss = F.binary_cross_entropy(pred_curve_points, batch_paths)


                rendered_images = RenderBatch(pred_curve_points)
                rendered_loader = DataLoader(rendered_images, batch_size=batch_images.shape[0], shuffle=False)
                rendered_images = next(iter(rendered_loader))

                # if i % 5 == 0:
                #     pydiffvg.imwrite(decanonize_image(rendered_images[0].cpu()), os.path.join("./testing_outputs/", f"training_{epoch}_{i}.png"))

                perceptual_loss = F.mse_loss(rendered_images, batch_images)

                loss = (1 - alpha) * direct_loss + alpha * perceptual_loss
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: Train loss: {np.mean(losses)}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_images, batch_paths in val_loader:
                batch_images = batch_images.to(device)
                batch_paths = batch_paths.to(device)

                pred_curve_points = model(batch_images)
                direct_loss = F.binary_cross_entropy(pred_curve_points, batch_paths)

                rendered_images = RenderBatch(pred_curve_points)
                rendered_loader = DataLoader(rendered_images, batch_size=batch_images.shape[0], shuffle=False)
                rendered_images = next(iter(rendered_loader))

                perceptual_loss = F.mse_loss(rendered_images, batch_images)

                loss = (1 - alpha) * direct_loss + alpha * perceptual_loss
                val_losses.append(loss.item())

        print(f"Validation loss: {np.mean(val_losses)}")

    except KeyboardInterrupt:
        print("Training interrupted")

    return model

def train_diffusion_vectorizer(train_loader, val_loader, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DiffusionVectorizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    try:
        for epoch in tqdm(range(epochs)):
            model.train()
            losses = []

            for batch_images, batch_paths in train_loader:
                batch_images = batch_images.to(device)
                batch_paths = batch_paths.to(device)

                optimizer.zero_grad()

                # Sample random timesteps
                t = torch.randint(0, model.num_steps, (batch_images.shape[0],), device=device)

                # Calculate loss
                pred = model.forward(batch_images, t)

                noise = torch.randn_like(batch_paths)
                alpha_t_hat = model.alpha_hat[t].view(-1, 1, 1, 1)
                noisy_paths = torch.sqrt(alpha_t_hat) * batch_paths + torch.sqrt(1 - alpha_t_hat) * noise

                direct_loss = F.mse_loss(pred, batch_paths)

                # rendered_images = RenderBatch(pred)
                # rendered_loader = DataLoader(rendered_images, batch_size=batch_images.shape[0], shuffle=False)
                # rendered_images = next(iter(rendered_loader))

                # perceptual_loss = F.mse_loss(rendered_images, batch_images)

                # alpha = 0.8
                # loss = (1 - alpha) * direct_loss + alpha * perceptual_loss
                loss = direct_loss
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: Train loss: {np.mean(losses)}")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_images, batch_paths in val_loader:
                batch_images = batch_images.to(device)
                batch_paths = batch_paths.to(device)

                # Sample at different timesteps
                t = torch.randint(0, model.num_steps, (batch_images.shape[0],), device=device)

                pred = model.forward(batch_images, t)

                direct_loss = F.mse_loss(pred, batch_paths)

                rendered_dataset = RenderBatch(pred)
                rendered_loader = DataLoader(rendered_dataset, batch_size=batch_images.shape[0], shuffle=False)
                rendered_images = next(iter(rendered_loader))

                perceptual_loss = F.mse_loss(rendered_images, batch_images)

                alpha = 0.8
                loss = (1 - alpha) * direct_loss + alpha * perceptual_loss

                val_losses.append(loss.item())

        print(f"Validation loss: {np.mean(val_losses)}")

    except KeyboardInterrupt:
        print("Training interrupted")

    return model

# from https://github.com/etaisella/svgLearner
def tensor2SVG(paths_tensor: torch.Tensor, stroke_width: float=3.0, stroke_color: list=[0.0, 0.0, 0.0, 1.0]):
    device = paths_tensor.device
    paths_tensor = paths_tensor.reshape(-1, 4, 2)
    num_paths = paths_tensor.shape[0]

    shapes=[]
    shape_groups=[]
    for i in range(num_paths):
        path_points = paths_tensor[i]
        shape = pydiffvg.Path(num_control_points=torch.zeros(1, dtype=torch.int32, device=device) + 2,
                                    points=path_points,
                                    stroke_width=torch.tensor(stroke_width, device=device),
                                    is_closed=False)
        shapes.append(shape)
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1], device=device),
                                            fill_color=None,
                                            stroke_color=torch.tensor(stroke_color, device=device))
        shape_groups.append(path_group)

    return shapes, shape_groups

def renderSVG(shapes, shape_groups):
    render = pydiffvg.RenderFunction.apply
    width, height = 384, 384
    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    img: torch.Tensor = render(width, height, 2, 2, 0, None, *scene_args) # pyright: ignore
    return img

if __name__ == "__main__":
    dataset = IconDataset("./outlined/")
    print(f"{len(dataset)} icons loaded")

    # num paths histogram
    num_paths_counter = Counter([len(p) for p in dataset.paths])
    print(f"{num_paths_counter=}")

    train_len = int(len(dataset)*0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    # needed to ensure each element in batch is the same size
    def collate_fn(data):
        images, paths = zip(*data)
        # pad paths to 32 curves with 0
        batch_size = len(paths)
        padded_targets = torch.zeros((batch_size, 32, 4, 2))
        for i, path_set in enumerate(paths):
            num_curves = len(path_set)
            padded_targets[i, :num_curves] = path_set
        return torch.stack(images), padded_targets

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    train_batch_images, train_batch_paths = next(iter(train_loader))
    print(train_batch_images.shape)
    print(train_batch_paths.shape)

    # model = train_diffusion_vectorizer(train_loader, val_loader, epochs=100)
    # model.save(f"./models/diffusion_model_{time.time()}.pth")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = DiffusionVectorizer.load(os.path.join("./models/", "diffusion_model_1733427256.0879426.pth")).to(device)

    model = train_bezier_vectorizer(train_loader, val_loader, epochs=100)
    model.save(f"./models/model_{time.time()}.pth")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = BezierVectorizer.load(os.path.join("./models/", "model_actually_works.pth"), embedding_dim=256).to(device)

    model.eval()
    print(f"{model=}")

    n = 5
    out_path = "./testing_outputs"

    it = iter(val_dataset)
    for i in range(n):
        example_image, example_points = next(it)

        pydiffvg.imwrite(decanonize_image(example_image.cpu()).permute(1, 2, 0), os.path.join(out_path, f"example_image_{i}.png"))

        # curves = model(example_image.unsqueeze(0))
        # curves = decanonize_paths(pred_curve_points)

        with torch.no_grad():
            # curves = model.sample(example_image.unsqueeze(0))
            curves = model(example_image.unsqueeze(0))
            curves = decanonize_paths(curves[0])  # Take first batch and denormalize

        shapes, shape_groups = tensor2SVG(curves, stroke_width=3.0, stroke_color=[0.0, 0.0, 0.0, 1.0])
        image = renderSVG(shapes, shape_groups).cpu()
        rgb_image = torch.ones((image.shape[0], image.shape[1], 3)) * 255
        mask = image[:, :, 3] > 0
        rgb_image[mask] = image[mask][:, :3]
        pydiffvg.imwrite(rgb_image, os.path.join(out_path, f"example_vectorized_{i}.png"))

    import code; code.interact(local=locals())
