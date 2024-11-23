import pydiffvg
import torch

pydiffvg.set_use_gpu(torch.cuda.is_available())


canvas_width = 510
canvas_height = 510

shapes = pydiffvg.from_svg_path('M510,255c0-20.4-17.85-38.25-38.25-38.25H331.5L204,12.75h-51l63.75,204H76.5l-38.25-51H0L25.5,255L0,344.25h38.25l38.25-51h140.25l-63.75,204h51l127.5-204h140.25C492.15,293.25,510,275.4,510,255z')
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]), fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))
shape_groups = [path_group]

# shapes = pydiffvg.from_svg_path("m336-280 144-144 144 144 56-56-144-144 144-144-56-56-144 144-144-144-56 56 144 144-144 144 56 56ZM480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z")
# scale_factor = 0.5
# for shape in shapes:
#     shape.points[:, 0] = (shape.points[:, 0] + 880) * scale_factor
#     shape.points[:, 1] = (shape.points[:, 1] + 880) * scale_factor
# print(shapes[0].points)
# print(shapes[1].points)
# print(shapes[2].points)
# path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0]))
# path_group2 = pydiffvg.ShapeGroup(shape_ids=torch.tensor([1]), fill_color=None, stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
# path_group3 = pydiffvg.ShapeGroup(shape_ids=torch.tensor([2]), fill_color=None, stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
# shape_groups = [path_group, path_group2, path_group3]

scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
render = pydiffvg.RenderFunction.apply
img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
pydiffvg.imwrite(img.cpu(), "output.png", gamma=2.2)
