import torch
import pydiffvg
import svgpathtools

from svg_to_paths import svg2paths2

from typing import Optional

def tensor2SVG(paths_tensor: torch.Tensor, stroke_width: float=3.0, stroke_color: list=[0.0, 0.0, 0.0, 1.0]):
    device = paths_tensor.device
    paths_tensor = paths_tensor.reshape(-1, 4, 2)
    num_paths = paths_tensor.shape[0]

    shapes=[]
    shape_groups=[]
    for i in range(num_paths):
        path_points = paths_tensor[i]
        if torch.all(torch.abs(path_points) < 1e-4):
            continue
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

def svg2path_tensors(file_path: str, max_curves: Optional[int]=None, verbose: bool=False) -> Optional[torch.Tensor]:
    # converts all SVG elements to paths
    try:
        paths, _, attrs = svg2paths2(file_path) # type: ignore
    except Exception as e:
        if verbose:
            print(f"Error parsing SVG: {e}")
        return

    path_tensors = []
    for path in paths:
        path_points = []
        for segment in path:
            if isinstance(segment, svgpathtools.CubicBezier):
                path_points.append([segment.start.real, segment.start.imag])
                path_points.append([segment.control1.real, segment.control1.imag])
                path_points.append([segment.control2.real, segment.control2.imag])
                path_points.append([segment.end.real, segment.end.imag])
            elif isinstance(segment, svgpathtools.QuadraticBezier):
                # https://stackoverflow.com/a/63059651
                cp1 = segment.start + (segment.control - segment.start) * 0.667
                cp2 = segment.end + (segment.control - segment.end) * 0.667
                path_points.append([segment.start.real, segment.start.imag])
                path_points.append([cp1.real, cp1.imag])
                path_points.append([cp2.real, cp2.imag])
                path_points.append([segment.end.real, segment.end.imag])
            elif isinstance(segment, svgpathtools.Line):
                # convert line to cubic bezier by adding control points
                # along the line at 1/3 and 2/3 positions
                cp1 = segment.start + (segment.end - segment.start) * 0.333
                cp2 = segment.start + (segment.end - segment.start) * 0.667
                path_points.append([segment.start.real, segment.start.imag])
                path_points.append([cp1.real, cp1.imag])
                path_points.append([cp2.real, cp2.imag])
                path_points.append([segment.end.real, segment.end.imag])
            else:
                if verbose:
                    print(f"Unsupported path segment type: {type(segment)}")
                return
        path_tensors.append(torch.tensor(path_points, dtype=torch.float32))

    if not path_tensors:
        if verbose:
            print("No paths found in SVG")
        return

    # check if this SVG is representable with `max_curves` Bezier curves
    num_curves = sum(len(p) for p in path_tensors)
    if max_curves is not None and num_curves > max_curves * 4:
        if verbose:
            print(f"SVG is too complex ({num_curves} curves)")
        return

    # Parse viewBox or fall back to width/height
    if "viewBox" in attrs:
        # viewBox format is "min-x min-y width height"
        try:
            viewbox = attrs["viewBox"].split()
            if len(viewbox) != 4:
                if verbose:
                    print("Invalid viewBox format")
                return
            min_x, min_y, width, height = map(float, viewbox)
        except ValueError:
            if verbose:
                print("Could not parse viewBox values")
            return
    else:
        # Fall back to width/height
        if "width" not in attrs or "height" not in attrs:
            if verbose:
                print("No viewBox or width/height found in SVG")
            return

        width, height = attrs["width"], attrs["height"]
        if width.endswith("px"):
            width = width[:-2]
        if height.endswith("px"):
            height = height[:-2]
        if width.endswith("%") or height.endswith("%"):
            if verbose:
                print("Percentage width or height found in SVG")
            return
        try:
            width, height = float(width), float(height)
            min_x, min_y = 0, 0
        except ValueError:
            if verbose:
                print("Could not parse width/height values")
            return

    path_tensors = torch.concat(path_tensors)
    if path_tensors.shape[0] == 0:
        if verbose:
            print("No paths found in SVG")
        return
    assert path_tensors.dim() == 2, f"Expected 2D tensor, got {path_tensors.dim()}D ({path_tensors.shape})"

    # normalize using viewBox coordinates
    path_tensors = path_tensors - torch.tensor([min_x, min_y])
    path_tensors = path_tensors / torch.tensor([width, height])
    path_tensors = path_tensors.reshape(-1, 4, 2)

    # get only start and end points (indices 0 and 3 of each curve)
    starts_and_ends = path_tensors[:, [0,3], :]
    if starts_and_ends.min() < 0 or starts_and_ends.max() > 1:
        if verbose:
            print(f"Curves out of bounds ({starts_and_ends.min()} to {starts_and_ends.max()})")
        return

    # FIXME: probably breaks some SVGs by clamping control points
    path_tensors = path_tensors.clamp(0, 1)
    return path_tensors
