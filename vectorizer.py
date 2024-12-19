import os
import time
from collections import Counter
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import GaussianBlur
import numpy as np
from tqdm import tqdm
import pydiffvg
from pytorch_msssim import MS_SSIM

from svg_utils import tensor2SVG, renderSVG, svg2path_tensors, decanonize_paths
from typing import OrderedDict

"""
Some of the utility function are adapted from https://github.com/etaisella/svgLearner
    canonizePaths, tensor2SVG, renderSVG, parts of IconDataset
"""

MAX_CURVES = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, data_dir: str, add_noise: bool = False):
        self.add_noise = add_noise
        self.skipped = 0
        self.paths = []

        # for root, dirs, files in os.walk(data_dir):
        #     for file in files:
        #         if not file.endswith('.svg'):
        #             continue

        #         file_path = os.path.join(root, file)

        #         path_tensors = svg2path_tensors(file_path, max_curves=MAX_CURVES)
        #         if path_tensors is None:
        #             continue

        #         self.paths.append(path_tensors)

        with open(data_dir, "r") as f:
            for line in f:
                svg = line.split(" [code] ")
                if len(svg) != 2:
                    self.skipped += 1
                    continue
                path_tensors = svg2path_tensors(svg[1], max_curves=MAX_CURVES, verbose=False)
                if path_tensors is None:
                    self.skipped += 1
                    continue

                self.paths.append(path_tensors)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        paths = self.paths[idx]
        curves = decanonize_paths(paths)

        if self.add_noise:
            noise_std = 5
            noise = torch.randn_like(curves) * noise_std
            curves = curves + noise

        image = canonize_image(renderSVG(*tensor2SVG(curves)))
        return image, paths

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int=384, patch_size: int=32, width: int=768, layers: int=4, heads: int=12, output_dim: int=MAX_CURVES*8):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, padding=2)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv0(x)  # shape = [*, 3, 224, 224]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x.reshape(-1, MAX_CURVES, 4, 2).sigmoid()

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

            nn.LazyLinear(768),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),

            nn.Linear(768, 2048),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),

            nn.Linear(2048, 4096),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),

            nn.Linear(4096, MAX_CURVES * 8)
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

def curve_continuity_loss(curves):
    """Compute distance between end of one curve and start of next"""
    # start points of all curves except first
    start_points = curves[:, 1:, 0, :]
    # end points of all curves except last
    end_points = curves[:, :-1, -1, :]
    return F.mse_loss(start_points, end_points)

def train_bezier_vectorizer(train_loader, val_loader, epochs=100):
    model = BezierVectorizer().to(device)
    # model = VisionTransformer().to(device)
    # TODO: lr and weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # gfilter = GaussianBlur(kernel_size=13, sigma=3.0).to(device)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=5,
    #     min_lr=1e-6
    # )

    # ssim_loss = MS_SSIM(win_size=11, size_average=True, data_range=1.0, channel=1).to(device)

    alpha = 0.1
    # best_val_loss = float('inf')

    def run_batch(batch_images: torch.Tensor, batch_paths: torch.Tensor) -> torch.Tensor:
        pred_curve_points = model(batch_images)
        direct_loss = F.binary_cross_entropy(pred_curve_points, batch_paths)

        # rendered_loader = DataLoader(
        #     RenderBatch(pred_curve_points),
        #     batch_size=batch_images.shape[0],
        #     shuffle=False
        # )
        # rendered_images = next(iter(rendered_loader))

        # # perceptual_loss = F.mse_loss(gfilter(rendered_images), gfilter(batch_images))
        # perceptual_loss = 1 - ssim_loss(gfilter(batch_images), gfilter(rendered_images))

        cont_loss = curve_continuity_loss(pred_curve_points)

        loss = (1 - alpha) * direct_loss + 0.2 * cont_loss# + alpha * perceptual_loss
        # loss = direct_loss
        return loss

    try:
        for epoch in tqdm(range(epochs)):
            model.train()
            losses = []
            for batch_images, batch_paths in train_loader:
                batch_images = batch_images.to(device)
                batch_paths = batch_paths.to(device)
                optimizer.zero_grad()
                loss = run_batch(batch_images, batch_paths)
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
                    loss = run_batch(batch_images, batch_paths)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f"Validation loss: {avg_val_loss}")

            # scheduler.step(avg_val_loss)
            # if avg_val_loss < best_val_loss:
            #     best_val_loss = avg_val_loss
            #     torch.save(model.state_dict(), 'best_model.pth')

    except KeyboardInterrupt:
        print("Training interrupted")

    return model

if __name__ == "__main__":
    batch_size = 32

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

    # dataset = IconDataset("./material-design-icons/svg/outlined/", add_noise=True)
    dataset = IconDataset("./gensvg/output/data.txt", add_noise=True)
    print(f"{len(dataset)} icons loaded, {dataset.skipped} skipped")

    # num paths histogram
    num_paths_counter = Counter([len(p) for p in dataset.paths])
    print(f"{num_paths_counter=}")

    train_len = int(len(dataset)*0.9)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    train_batch_images, train_batch_paths = next(iter(train_loader))
    print(train_batch_images.shape)
    print(train_batch_paths.shape)

    # # save images
    # out_path = "./testing_outputs"
    # for i, image in enumerate(train_batch_images):
    #     image = image.cpu()
    #     image = decanonize_image(image).permute(1, 2, 0)
    #     print(image.shape)
    #     pydiffvg.imwrite(image, os.path.join(out_path, f"g_train_image_{i}.png"))

    model = train_bezier_vectorizer(train_loader, val_loader, epochs=100)
    model.save(f"./models/model_{time.time()}.pth")

    # model = BezierVectorizer.load(os.path.join("./models/", "model_1734231214.6171353.pth")).to(device)

    print(f"{model=}")
    model.eval()

    n = 5
    out_path = "./testing_outputs"

    # example_images, _ = next(iter(val_loader))
    example_images, _ = next(iter(train_loader))
    for i in range(8):
        example_image = example_images[i]
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
