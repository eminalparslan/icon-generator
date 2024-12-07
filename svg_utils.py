import torch
import pydiffvg
import svgpathtools

from typing import Optional

def tensor2SVG(paths_tensor: torch.Tensor, stroke_width: float=3.0, stroke_color: list=[0.0, 0.0, 0.0, 1.0]):
    device = paths_tensor.device
    paths_tensor = paths_tensor.reshape(-1, 4, 2)
    num_paths = paths_tensor.shape[0]

    shapes=[]
    shape_groups=[]
    for i in range(num_paths):
        path_points = paths_tensor[i]
        shape = pydiffvg.Path(
            num_control_points=torch.zeros(1, dtype=torch.int32, device=device) + 2,
            points=path_points,
            stroke_width=torch.tensor(stroke_width, device=device),
            is_closed=False
        )
        shapes.append(shape)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1], device=device),
            fill_color=None,
            stroke_color=torch.tensor(stroke_color, device=device)
        )
        shape_groups.append(path_group)

    return shapes, shape_groups

def renderSVG(shapes, shape_groups):
    render = pydiffvg.RenderFunction.apply
    width, height = 384, 384
    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    img: torch.Tensor = render(width, height, 2, 2, 0, None, *scene_args) # pyright: ignore
    return img

def decanonize_paths(paths_tensor: torch.Tensor, im_height: int=384, im_width: int=384) -> torch.Tensor:
    assert paths_tensor.size(-1) == 2, "Last dimension should be 2 (x,y coordinates)"
    device = paths_tensor.device
    # Scale back to image dimensions
    paths_tensor = paths_tensor * torch.tensor([im_width, im_height], device=device)
    return paths_tensor

def svg2path_tensors(file_path: str, max_curves: Optional[int]=None) -> Optional[torch.Tensor]:
    # converts all SVG elements to paths
    paths, _, attrs = svgpathtools.svg2paths2(file_path) # type: ignore

    path_tensors = []
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
                return
            else:
                raise ValueError(f"Unsupported path segment type: {type(segment)}")
        path_tensors.append(torch.tensor(path_points, dtype=torch.float32))

    if max_curves is not None and sum(len(p) for p in path_tensors) > max_curves * 4:
        return

    canvas_width, canvas_height = float(attrs["width"]), float(attrs["height"])

    path_tensors = torch.concat(path_tensors)
    path_tensors = path_tensors.reshape(-1, 2)
    path_tensors = path_tensors / torch.tensor([canvas_width, canvas_height])
    if path_tensors.min() < 0 or path_tensors.max() > 1:
        print("WARNING: Curves out of bounds. Clipping to [0,1]")
        path_tensors = path_tensors.clamp(0, 1)
    path_tensors = path_tensors.reshape(-1, 4, 2)
    return path_tensors
