**Crop:**

```python
@op("Crop", icon="crop")
def crop(image: Image.Image, *, top: int, left: int, bottom: int, right: int):
    return image.crop((left, top, right, bottom))

```
