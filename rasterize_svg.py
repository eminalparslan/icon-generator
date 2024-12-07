import os
import torch
import pydiffvg

from svg_utils import tensor2SVG, renderSVG, svg2path_tensors, decanonize_paths

path = "./material-design-icons/svg/outlined/"
out_path = "./outlined_rasterized/train/"

for file in os.listdir(path):
    svg = os.path.join(path, file)

    path_tensors = svg2path_tensors(svg)
    if path_tensors is None:
        continue

    curves = decanonize_paths(path_tensors)
    image = renderSVG(*tensor2SVG(curves)).cpu()

    rgb_image = torch.ones((image.shape[0], image.shape[1], 3)) * 255
    mask = image[:, :, 3] > 0
    rgb_image[mask] = image[mask][:, :3]

    pydiffvg.imwrite(rgb_image, os.path.join(out_path, file[:-4] + ".png"), gamma=2.2)
