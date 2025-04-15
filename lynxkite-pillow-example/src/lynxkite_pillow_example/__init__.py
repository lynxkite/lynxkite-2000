"""Demo for how easily we can provide a UI for popular open-source tools."""

from lynxkite.core import ops
from lynxkite.core.executors import one_by_one
from PIL import Image, ImageFilter
import base64
import fsspec
import io

ENV = "Pillow"
op = ops.op_registration(ENV)
one_by_one.register(ENV, cache=False)


@op("Open image")
def open_image(*, filename: str):
    with fsspec.open(filename, "rb") as f:
        data = io.BytesIO(f.read())
    return Image.open(data)


@op("Save image")
def save_image(image: Image, *, filename: str):
    with fsspec.open(filename, "wb") as f:
        image.save(f)


@op("Crop")
def crop(image: Image, *, top: int, left: int, bottom: int, right: int):
    return image.crop((left, top, right, bottom))


@op("Flip horizontally")
def flip_horizontally(image: Image):
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


@op("Flip vertically")
def flip_vertically(image: Image):
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


@op("Blur")
def blur(image: Image, *, radius: float = 5):
    return image.filter(ImageFilter.GaussianBlur(radius))


@op("Detail")
def detail(image: Image):
    return image.filter(ImageFilter.DETAIL)


@op("Edge enhance")
def edge_enhance(image: Image):
    return image.filter(ImageFilter.EDGE_ENHANCE)


@op("To grayscale")
def to_grayscale(image: Image):
    return image.convert("L")


@op("View image", view="image")
def view_image(image: Image):
    buffered = io.BytesIO()
    image.save(buffered, format="webp")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = "data:image/jpeg;base64," + b64
    return data_url
