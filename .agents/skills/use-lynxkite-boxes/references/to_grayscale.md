**To grayscale:**

```python
@op("To grayscale", icon="filters-filled")
def to_grayscale(image: Image.Image):
    return image.convert("L")

```
