**Detail:**

```python
@op("Detail", icon="filters-filled")
def detail(image: Image.Image):
    return image.filter(ImageFilter.DETAIL)

```
