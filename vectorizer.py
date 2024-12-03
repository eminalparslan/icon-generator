import os
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pydiffvg
import svgpathtools

from typing import Tuple, List, Optional

"""
Some of the utility function are adapted from https://github.com/etaisella/svgLearner
    - canonizePaths
    - tensor2SVG
    - renderSVG
    - parts of IconDataset
"""

def canonize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0, 1)
    # use alpha to composite on white background
    rgb = image[:, :, :3]
    alpha = image[:, :, 3:]
    white_bg = torch.ones_like(rgb)
    image = rgb * alpha + white_bg * (1 - alpha)
    image = image.permute(2, 0, 1)
    return image

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
                if skip or sum(len(p) for p in path_tensors) > 32 * 4:
                    continue

                path_tensors = torch.concat(path_tensors)
                path_tensors = path_tensors.reshape(-1, 4, 2)
                self.paths.append(path_tensors)

                canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(file_path)

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
    def __init__(self,
                 image_size: int = 384,
                 max_curves: int = 32,
                 embedding_dim: int = 256):
        """
        Neural network to convert images to cubic Bezier curves.

        Args:
            image_size: Size of input image (square)
            max_curves: Maximum number of Bezier curves to generate
            embedding_dim: Dimensionality of internal feature representation
        """
        super().__init__()

        self.max_curves = max_curves

        # Backbone feature extractor
        self.feature_extractor = nn.Sequential(
            # Initial convolutional layers with increasing receptive field
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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
            nn.MaxPool2d(2)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Curve count prediction
        self.curve_count_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.max_curves),
            nn.Sigmoid()  # Probability for each potential curve
        )

        # Curve point generator
        self.curve_point_generator = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, self.max_curves * 8)  # 8 coordinates per curve (4 points * x,y)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate Bezier curves from image.

        Args:
            x: Input image tensor of shape (B, C, H, W)

        Returns:
            curve_points: Tensor of Bezier curve points (B, num_curves, 4, 2)
            curve_probabilities: Tensor of curve existence probabilities (B, num_curves)
        """
        # Extract global image features
        features = self.feature_extractor(x)
        global_features = self.global_pool(features).squeeze(-1).squeeze(-1)

        # Predict number of curves
        curve_probabilities = self.curve_count_predictor(global_features)

        # Generate curve points
        raw_curve_points = self.curve_point_generator(global_features)

        # Reshape and normalize curve points
        curve_points = raw_curve_points.view(-1, self.max_curves, 4, 2)
        curve_points = torch.sigmoid(curve_points)  # Normalize to [0,1]

        return curve_points, curve_probabilities

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model


class BezierVectorizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                pred_curve_points: torch.Tensor,
                pred_curve_prob: torch.Tensor,
                target_paths: torch.Tensor) -> torch.Tensor:
        batch_size = pred_curve_points.shape[0]

        target_probs = torch.zeros_like(pred_curve_prob)
        for i in range(batch_size):
            # FIXME: find a better way to do this
            # -8.0 because 4 point curves with 2 x,y values each
            # not 0 because paths are canonized
            num_actual_curves = (target_paths[i].sum(dim=(1,2)) != -8.0).sum()
            target_probs[i, :num_actual_curves] = 1.0

        # binary cross entropy loss for curve existence
        prob_loss = nn.BCELoss()(pred_curve_prob, target_probs)

        # flatten points and expand probabilities
        pred_points = pred_curve_points.view(batch_size, -1, 2)
        padded_targets = target_paths.view(batch_size, -1, 2)
        # flattens curves * points into a single dimension
        point_probs = pred_curve_prob.unsqueeze(-1).expand(-1, -1, 4).reshape(batch_size, -1, 1)
        # FIXME: expands the probabilities to each point in the curve
        # point_probs = pred_curve_prob.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 2)

        point_diffs = (pred_points - padded_targets) ** 2
        weighted_diffs = point_diffs * point_probs
        point_loss = weighted_diffs.mean()

        # add probability regularization term
        # prob_reg = -torch.mean(torch.log(pred_curve_prob + 1e-10) + torch.log(1 - pred_curve_prob + 1e-10))

        total_loss = point_loss + prob_loss

        return total_loss


class PerceptualVectorizationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Combined loss function with both point-based and perceptual components

        Args:
            alpha: Weight for balancing point-based and perceptual losses
        """
        super().__init__()
        self.point_loss = BezierVectorizationLoss()
        self.alpha = alpha

    def forward(self,
                pred_curve_points: torch.Tensor,
                pred_curve_prob: torch.Tensor,
                target_paths: List[torch.Tensor],
                target_image: torch.Tensor,
                timestep: Optional[int] = None) -> torch.Tensor:
        point_loss = self.point_loss(pred_curve_points, pred_curve_prob, target_paths)

        batch_size = pred_curve_points.shape[0]
        rendered_images = []

        for i in range(batch_size):
            curves = pred_curve_points[i][pred_curve_prob[i] > 0.5]
            if len(curves) == 0:
                rendered = torch.ones_like(target_image[i])
                rendered_images.append(rendered)
                continue

            curves = decanonize_paths(curves)
            rendered = renderSVG(*tensor2SVG(curves))
            rendered = canonize_image(rendered)
            rendered_images.append(rendered)

        rendered_images = torch.stack(rendered_images)

        perceptual_loss = torch.nn.functional.mse_loss(rendered_images, target_image)

        total_loss = (1 - self.alpha) * point_loss + self.alpha * perceptual_loss

        return total_loss

def canonize_paths(paths_tensor: torch.Tensor, im_height: int=384, im_width: int=384) -> torch.Tensor:
    device = paths_tensor.device
    # Normalize to [0,1] range by dividing by image dimensions
    paths_tensor = paths_tensor / torch.tensor([im_width, im_height], device=device)
    return paths_tensor

def decanonize_paths(paths_tensor: torch.Tensor, im_height: int=384, im_width: int=384) -> torch.Tensor:
    device = paths_tensor.device
    # Scale back to image dimensions
    paths_tensor = paths_tensor * torch.tensor([im_width, im_height], device=device)
    return paths_tensor

def train_bezier_vectorizer(train_loader, val_loader, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BezierVectorizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = PerceptualVectorizationLoss(alpha=0.80).to(device)

    try:
        for epoch in tqdm(range(epochs)):
            model.train()

            losses = []
            for i, (batch_images, batch_paths) in enumerate(train_loader):
                batch_images = batch_images.to(device)
                batch_paths = canonize_paths(batch_paths).to(device)

                # TODO: make sure images are in correct format everywhere
                # TODO: paths -> SVG -> raster (boxfiltered?)
                # TODO: bigger model?
                # TODO: maybe just adapts Etai's code initially?

                optimizer.zero_grad()
                pred_curve_points, pred_curve_prob = model(batch_images)

                loss = criterion(pred_curve_points, pred_curve_prob, batch_paths, batch_images, i)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: Train loss: {np.mean(losses)}")

        model.eval()
        val_loss = 0
        for batch_images, batch_paths in val_loader:
            batch_images = batch_images.to(device)
            batch_paths = canonize_paths(batch_paths).to(device)

            pred_curve_points, pred_curve_prob = model(batch_images)
            val_loss += criterion(pred_curve_points, pred_curve_prob, batch_paths, batch_images).item()

        print(f"Validation loss: {val_loss / len(val_loader)}")

    except KeyboardInterrupt:
        print("Training interrupted")

    return model


# from https://github.com/etaisella/svgLearner
def tensor2SVG(paths_tensor: torch.Tensor, stroke_width: float=3.0, stroke_color: list=[0.0, 0.0, 0.0, 1.0]):
    device = paths_tensor.device
    paths_tensor = paths_tensor.reshape(-1, 4, 2)
    paths_tensor = torch.clamp(paths_tensor, 0, 384)
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

    for batch_images, batch_paths in train_loader:
        print(batch_images.shape)
        print(batch_paths.shape)
        break

    model = train_bezier_vectorizer(train_loader, val_loader, epochs=100)
    model.save(f"./models/model_{time.time()}.pth")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = BezierVectorizer.load("./models/model_1733114716.6607993.pth", image_size=384, max_curves=32, embedding_dim=256).to(device)

    model.eval()
    print(f"{model=}")

    n = 5
    out_path = "./testing_outputs"

    it = iter(val_dataset)
    for i in range(n):
        example_image, example_points = next(it)

        pydiffvg.imwrite(example_image.permute(1, 2, 0).cpu(), os.path.join(out_path, f"example_image_{i}.png"))

        pred_curve_points, pred_curve_prob = model(example_image.unsqueeze(0))
        print(f"{pred_curve_points.shape=}")
        pred_curve_points = pred_curve_points[pred_curve_prob > 0.5]
        pred_curve_points = decanonize_paths(pred_curve_points)
        print(f"{pred_curve_points.shape=}")

        shapes, shape_groups = tensor2SVG(pred_curve_points, stroke_width=3.0, stroke_color=[0.0, 0.0, 0.0, 1.0])
        image = renderSVG(shapes, shape_groups)
        pydiffvg.imwrite(image.cpu(), os.path.join(out_path, f"example_vectorized_{i}.png"))

    import code; code.interact(local=locals())
