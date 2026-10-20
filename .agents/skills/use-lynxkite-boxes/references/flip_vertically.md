**Flip vertically:**

```python
@op("Flip vertically", icon="flip-vertical")
def flip_vertically(image: Image.Image):
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

```
