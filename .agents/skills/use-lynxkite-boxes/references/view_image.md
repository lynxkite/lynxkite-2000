**View image:**

```python
@op("View image", view="image", color="green", icon="photo-filled")
def view_image(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="webp")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = "data:image/jpeg;base64," + b64
    return data_url

```
