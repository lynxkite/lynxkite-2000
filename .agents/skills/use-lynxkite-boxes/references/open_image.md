**Open image:**

```python
@op("Open image", color="blue", icon="photo-filled")
def open_image(*, filename: str):
    with fsspec.open(filename, "rb") as f:
        data = io.BytesIO(f.read())
    return Image.open(data)

```
