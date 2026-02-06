"""Demo for how easily we can provide a UI for popular open-source tools."""

from lynxkite_core import ops
from lynxkite_core.executors import simple
from PIL import Image, ImageFilter
import base64
import fsspec
import io

ENV = "Pillow"
op = ops.op_registration(ENV)
simple.register(ENV)


@op("Open image", color="blue", icon="photo-filled")
def open_image(*, filename: str):
    with fsspec.open(filename, "rb") as f:
        data = io.BytesIO(f.read())
    return Image.open(data)


@op("Save image", color="green", icon="device-floppy")
def save_image(image: Image.Image, *, filename: str):
    with fsspec.open(filename, "wb") as f:
        image.save(f)


@op("Crop", icon="crop")
def crop(image: Image.Image, *, top: int, left: int, bottom: int, right: int):
    return image.crop((left, top, right, bottom))


@op("Flip horizontally", icon="flip-horizontal")
def flip_horizontally(image: Image.Image):
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


@op("Flip vertically", icon="flip-vertical")
def flip_vertically(image: Image.Image):
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


@op("Blur", icon="filters-filled")
def blur(image: Image.Image, *, radius: float = 5):
    return image.filter(ImageFilter.GaussianBlur(radius))


@op("Detail", icon="filters-filled")
def detail(image: Image.Image):
    return image.filter(ImageFilter.DETAIL)


@op("Edge enhance", icon="filters-filled")
def edge_enhance(image: Image.Image):
    return image.filter(ImageFilter.EDGE_ENHANCE)


@op("To grayscale", icon="filters-filled")
def to_grayscale(image: Image.Image):
    return image.convert("L")


@op("View image", view="image", color="green", icon="photo-filled")
def view_image(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="webp")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = "data:image/jpeg;base64," + b64
    return data_url
