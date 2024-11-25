import os
import pydiffvg

path = "./outlined/"
# out_path = "./outlined_rasterized/"
out_path = "./"

for file in os.listdir(path):
    svg = os.path.join(path, file)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg)

    scale_factor = 16
    canvas_width *= scale_factor
    canvas_height *= scale_factor
    for shape in shapes:
        shape.points[:, 0] *= scale_factor
        shape.points[:, 1] *= scale_factor

    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    pydiffvg.imwrite(img.cpu(), os.path.join(out_path, file[:-4] + ".png"), gamma=2.2)
    break
