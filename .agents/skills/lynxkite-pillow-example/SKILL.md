---
name: lynxkite-pillow-example
description: Collection of operations - Open image, Save image, Crop, Flip horizontally, Flip vertically, Blur, Detail, Edge enhance, To grayscale, View image
---

**Open image:**

parameters:
  - filename: <class 'str'> = None - .

usage:
output_variable = lynxkite_pillow_example.open_image(filename=<filename_value>)

**Save image:**

parameters:
  - filename: <class 'str'> = None - .
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.save_image(filename=<filename_value>, image=<image_variable>)

**Crop:**

parameters:
  - top: <class 'int'> = None - .
  - left: <class 'int'> = None - .
  - bottom: <class 'int'> = None - .
  - right: <class 'int'> = None - .
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.crop(top=<top_value>, left=<left_value>, bottom=<bottom_value>, right=<right_value>, image=<image_variable>)

**Flip horizontally:**

parameters:
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.flip_horizontally(image=<image_variable>)

**Flip vertically:**

parameters:
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.flip_vertically(image=<image_variable>)

**Blur:**

parameters:
  - radius: <class 'float'> = 5 - .
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.blur(radius=<radius_value>, image=<image_variable>)

**Detail:**

parameters:
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.detail(image=<image_variable>)

**Edge enhance:**

parameters:
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.edge_enhance(image=<image_variable>)

**To grayscale:**

parameters:
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.to_grayscale(image=<image_variable>)

**View image:**

parameters:
  - image: <class 'PIL.Image.Image'> = None - .

usage:
output_variable = lynxkite_pillow_example.view_image(image=<image_variable>)
