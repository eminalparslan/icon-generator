import os
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pydiffvg

from typing import Tuple, List


class IconDataset(Dataset):
    def __init__(self, data_dir: str):
        render = pydiffvg.RenderFunction.apply
        self.paths = []
        self.images = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.svg'):
                    path = os.path.join(root, file)
                    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(path)

                    scale_factor = 16
                    canvas_width *= scale_factor
                    canvas_height *= scale_factor
                    skip = False
                    for shape in shapes:
                        # make sure SVG is only made up of Paths
                        if not isinstance(shape, pydiffvg.Path):
                            skip = True
                            break
                        shape.points[:, 0] *= scale_factor
                        shape.points[:, 1] *= scale_factor

                    if skip:
                        continue

                    self.paths.append([p.points.detach().cpu().numpy() for p in shapes])

                    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
                    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

                    self.images.append(img)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.images[idx].clamp(0, 1)
        # use alpha to composite on white background
        rgb = image[:, :, :3]
        alpha = image[:, :, 3:]
        white_bg = torch.ones_like(rgb)
        image = rgb * alpha + white_bg * (1 - alpha)
        image = image.permute(2, 0, 1)
        # image = image[:, :, :3].permute(2, 0, 1)
        # paths = torch.tensor(self.paths[idx], dtype=torch.float32)
        paths = [torch.from_numpy(p).float() / 384 for p in self.paths[idx]]
        return image, paths


class BezierVectorizer(nn.Module):
    def __init__(self,
                 image_size: int = 384,
                 max_curves: int = 8,
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
        """
        Loss function that:
        1. Pads predictions or targets to match shorter length
        2. Uses probabilities to weight the importance of each curve
        3. Calculates weighted distance between corresponding points
        """
        device = pred_curve_points.device
        batch_size = pred_curve_points.shape[0]

        # flatten predicted points and expand probabilities
        pred_points = pred_curve_points.view(batch_size, -1, 2)  # Shape: (batch_size, num_points, 2)
        point_probs = pred_curve_prob.unsqueeze(-1).expand(-1, -1, 4).reshape(batch_size, -1, 1)

        max_target_len = max(len(target) for target in target_paths)

        # truncate predictions to match target length
        pred_points = pred_points[:, :max_target_len, :]
        point_probs = point_probs[:, :max_target_len, :]

        # pad targets up to max_target_len
        padded_targets = torch.zeros(batch_size, max_target_len, 2, device=device)
        target_masks = torch.zeros(batch_size, max_target_len, 1, device=device)

        for i in range(batch_size):
            target = target_paths[i]
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, device=device)
            length = target.shape[0]
            padded_targets[i, :length] = target
            target_masks[i, :length] = 1

        # calculate weighted MSE loss between predicted and target points
        point_diffs = (pred_points - padded_targets) ** 2
        weighted_diffs = point_diffs * point_probs * target_masks
        point_loss = weighted_diffs.mean()

        # Add probability regularization term
        prob_reg = -torch.mean(torch.log(pred_curve_prob + 1e-10) + torch.log(1 - pred_curve_prob + 1e-10))

        total_loss = point_loss + 0.1 * prob_reg

        return total_loss

# Example usage and training setup
def train_bezier_vectorizer(train_loader, val_loader, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BezierVectorizer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = BezierVectorizationLoss()

    for epoch in tqdm(range(epochs)):
        model.train()

        losses = []
        for batch_images, batch_paths, mask in train_loader:
            batch_images = batch_images.to(device)
            batch_paths = batch_paths.to(device)
            mask = mask.to(device)

            # TODO: canonize paths
            # TODO: paths -> SVG -> raster (boxfiltered?)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            pred_curve_points, pred_curve_prob = model(batch_images)

            # Compute loss
            batch_paths = batch_paths[mask]
            loss = criterion(pred_curve_points, pred_curve_prob, batch_paths)
            losses.append(loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Train loss: {np.mean(losses)}")

    model.eval()
    val_loss = 0
    for batch_images, batch_paths, mask in val_loader:
        batch_images = batch_images.to(device)
        batch_paths = batch_paths.to(device)
        mask = mask.to(device)

        pred_curve_points, pred_curve_prob = model(batch_images)
        batch_paths = batch_paths[mask]
        val_loss += criterion(pred_curve_points, pred_curve_prob, batch_paths).item()

    print(f"Validation loss: {val_loss / len(val_loader)}")

    return model


# from https://github.com/etaisella/svgLearner
def tensor2SVG(paths_tensor: torch.Tensor, stroke_width: float=3.0, stroke_color: list=[0.0, 0.0, 0.0, 1.0]):
    paths_tensor = paths_tensor.reshape(-1, 4, 2)
    num_paths = paths_tensor.shape[0]

    shapes=[]
    shape_groups=[]
    for i in range(num_paths):
        path_points = paths_tensor[i]
        shape = pydiffvg.Path(num_control_points=torch.zeros(1, dtype=torch.int32) + 2,
                                    points=path_points,
                                    stroke_width=torch.tensor(stroke_width),
                                    is_closed=False)
        shapes.append(shape)
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                            fill_color=None,
                                            stroke_color=torch.tensor(stroke_color))
        shape_groups.append(path_group)

    return shapes, shape_groups

def renderSVG(shapes, shape_groups):
    render = pydiffvg.RenderFunction.apply
    width, height = 384, 384
    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    img = render(width, height, 2, 2, 0, None, *scene_args)
    return img

if __name__ == "__main__":
    dataset = IconDataset("./outlined/")
    print(f"{len(dataset)} icons loaded")

    # num paths histogram
    print(Counter([len(p) for p in dataset.paths]))

    # for image, path in dataset:
    #     print(image.shape)
    #     pydiffvg.imwrite(image.cpu(), "./test.png")
    #     break

    train_len = int(len(dataset)*0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])

    # needed to ensure each element in batch is the same size
    def collate_fn(data):
        images, paths = zip(*data)

        images = torch.stack(images)

        # Find max dimensions
        max_paths = max(len(paths) for paths in paths)
        max_points = max(max(path.size(0) for path in paths) for paths in paths)

        # Create padded tensor and mask
        batch_size = len(paths)
        padded_targets = torch.zeros((batch_size, max_paths, max_points, 2))
        mask = torch.zeros((batch_size, max_paths, max_points), dtype=torch.bool)

        # Fill in the actual path data and mask
        for i, paths in enumerate(paths):
            for j, path in enumerate(paths):
                padded_targets[i, j, :path.size(0), :] = path
                mask[i, j, :path.size(0)] = True

        return images, padded_targets, mask

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # model = train_bezier_vectorizer(train_loader, val_loader, epochs=100)
    # model.save(f"./models/model_{time.time()}.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BezierVectorizer.load("./models/model_1733114716.6607993.pth", image_size=384, max_curves=8, embedding_dim=256).to(device)

    model.eval()
    print(f"{model=}")

    n = 5  # Number of examples to process
    out_path = "./testing_outputs"

    it = iter(val_dataset)
    for i in range(n):
        example_image, example_points = next(it)
        # print(f"{example_points=}")

        pydiffvg.imwrite(example_image.permute(1, 2, 0).cpu(), os.path.join(out_path, f"example_image_{i}.png"))

        pred_curve_points, pred_curve_prob = model(example_image.unsqueeze(0))
        print(f"{pred_curve_points.shape=}")
        pred_curve_points = pred_curve_points[pred_curve_prob > 0.5]
        pred_curve_points *= 384
        print(f"{pred_curve_points.shape=}")

        shapes, shape_groups = tensor2SVG(pred_curve_points, stroke_width=3.0, stroke_color=[0.0, 0.0, 0.0, 1.0])
        image = renderSVG(shapes, shape_groups)
        pydiffvg.imwrite(image.cpu(), os.path.join(out_path, f"example_vectorized_{i}.png"))

    import code; code.interact(local=locals())
