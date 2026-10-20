**Blur:**

```python
@op("Blur", icon="filters-filled")
def blur(image: Image.Image, *, radius: float = 5):
    return image.filter(ImageFilter.GaussianBlur(radius))

```
