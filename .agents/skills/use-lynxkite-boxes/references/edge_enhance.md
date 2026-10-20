**Edge enhance:**

```python
@op("Edge enhance", icon="filters-filled")
def edge_enhance(image: Image.Image):
    return image.filter(ImageFilter.EDGE_ENHANCE)

```
