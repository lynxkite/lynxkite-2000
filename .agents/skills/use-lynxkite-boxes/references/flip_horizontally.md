**Flip horizontally:**

```python
@op("Flip horizontally", icon="flip-horizontal")
def flip_horizontally(image: Image.Image):
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

```
