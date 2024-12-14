import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pydiffvg
from pytorch_msssim import MS_SSIM

from svg_utils import tensor2SVG, renderSVG, svg2path_tensors, decanonize_paths

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
        self.paths = []
        self.images = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if not file.endswith('.svg'):
                    continue

                file_path = os.path.join(root, file)

                path_tensors = svg2path_tensors(file_path, max_curves=MAX_CURVES)
                if path_tensors is None:
                    continue

                curves = decanonize_paths(path_tensors)
                image = renderSVG(*tensor2SVG(curves))

                self.paths.append(path_tensors)
                self.images.append(image)

        assert len(self.paths) == len(self.images)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = canonize_image(self.images[idx])
        paths = self.paths[idx]
        return image, paths


class BezierVectorizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.LazyLinear(1024),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),

            nn.Linear(1024, 2048),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),

            nn.Linear(2048, MAX_CURVES * 8)
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

def boxBlurfilter(in_channels: int=1, out_channels: int=1, kernel_size: int=7):
    box_filter = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
    box_filter.weight.data = torch.ones(in_channels, out_channels, kernel_size, kernel_size).float() / (kernel_size ** 2)
    box_filter.weight.requires_grad = False
    return box_filter

def curve_continuity_loss(curves):
    # Compute distance between end of one curve and start of next
    start_points = curves[:, 1:, 0, :]  # Start points of all curves except first
    end_points = curves[:, :-1, -1, :]  # End points of all curves except last
    return F.mse_loss(start_points, end_points)

def train_bezier_vectorizer(train_loader, val_loader, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BezierVectorizer().to(device)
    # TUNE: lr and weight_decay
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # ssim_loss = MS_SSIM(win_size=11, size_average=True, data_range=1.0, channel=1).to(device)

    alpha = 0.2

    try:
        for epoch in tqdm(range(epochs)):
            model.train()
            losses = []

            for i, (batch_images, batch_paths) in enumerate(train_loader):
                batch_images = batch_images.to(device)
                batch_paths = batch_paths.to(device)

                optimizer.zero_grad()
                pred_curve_points = model(batch_images)
                direct_loss = F.binary_cross_entropy(pred_curve_points, batch_paths)

                rendered_loader = DataLoader(
                    RenderBatch(pred_curve_points),
                    batch_size=batch_images.shape[0],
                    shuffle=False
                )
                rendered_images = next(iter(rendered_loader))

                perceptual_loss = F.mse_loss(rendered_images, batch_images)
                # perceptual_loss = 1 - ssim_loss(batch_images, rendered_images)

                # cont_loss = curve_continuity_loss(pred_curve_points)

                loss = (1 - alpha) * direct_loss + alpha * perceptual_loss #+ 0.2 * cont_loss
                # loss = direct_loss
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

                rendered_loader = DataLoader(
                    RenderBatch(pred_curve_points),
                    batch_size=batch_images.shape[0],
                    shuffle=False
                )
                rendered_images = next(iter(rendered_loader))

                perceptual_loss = F.mse_loss(rendered_images, batch_images)
                # perceptual_loss = 1 - ssim_loss(batch_images, rendered_images)

                # cont_loss = curve_continuity_loss(pred_curve_points)

                loss = (1 - alpha) * direct_loss + alpha * perceptual_loss #+ 0.2 * cont_loss
                # loss = direct_loss
                val_losses.append(loss.item())

        print(f"Validation loss: {np.mean(val_losses)}")

    except KeyboardInterrupt:
        print("Training interrupted")

    return model

if __name__ == "__main__":
    dataset = IconDataset("./material-design-icons/svg/outlined/")
    print(f"{len(dataset)} icons loaded")

    # num paths histogram
    num_paths_counter = Counter([len(p) for p in dataset.paths])
    print(f"{num_paths_counter=}")

    train_len = int(len(dataset)*0.9)
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

    model = train_bezier_vectorizer(train_loader, val_loader, epochs=100)
    # model.save(f"./models/model_{time.time()}.pth")

    # model = BezierVectorizer.load(os.path.join("./models/", "model_actually_works.pth")).to(device)

    print(f"{model=}")
    model.eval()

    n = 5
    out_path = "./testing_outputs"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    boxfilter = boxBlurfilter(kernel_size=12).to(device)

    it = iter(train_loader)
    for i in range(n):
        example_image, _ = next(it)
        example_image = example_image[0]
        # example_image = boxfilter(example_image)[0]

        display_image = decanonize_image(example_image.cpu()).permute(1, 2, 0)
        pydiffvg.imwrite(display_image, os.path.join(out_path, f"example_image_{i}.png"))

        with torch.no_grad():
            curves = model(example_image.unsqueeze(0))
            curves = decanonize_paths(curves[0])

        shapes, shape_groups = tensor2SVG(curves, stroke_width=3.0, stroke_color=[0.0, 0.0, 0.0, 1.0])
        image = renderSVG(shapes, shape_groups).cpu()

        rgb_image = torch.ones((image.shape[0], image.shape[1], 3)) * 255
        mask = image[:, :, 3] > 0
        rgb_image[mask] = image[mask][:, :3]

        pydiffvg.imwrite(rgb_image, os.path.join(out_path, f"example_vectorized_{i}.png"))

    import code; code.interact(local=locals())
