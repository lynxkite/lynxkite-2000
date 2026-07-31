**Save image:**

```python
@op("Save image", color="green", icon="device-floppy")
def save_image(image: Image.Image, *, filename: str):
    with fsspec.open(filename, "wb") as f:
        image.save(f)

```
