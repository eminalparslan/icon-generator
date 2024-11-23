import os
import pydiffvg

path = "./material_icons/"

# svg = os.path.join(path, "cancel_24dp_E8EAED_FILL0_wght400_GRAD0_opsz24.svg")
svg = os.path.join(path, "home_24dp_E8EAED_FILL0_wght400_GRAD0_opsz24.svg")
canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg)

# svg canvas needs to be shifted and scaled to show up in output
for shape in shapes:
    shape.points[:, 0] += canvas_width
    shape.points[:, 0] *= 0.5
    shape.points[:, 1] += canvas_height
    shape.points[:, 1] *= 0.5

render = pydiffvg.RenderFunction.apply
scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
pydiffvg.imwrite(img.cpu(), "output.png", gamma=2.2)
