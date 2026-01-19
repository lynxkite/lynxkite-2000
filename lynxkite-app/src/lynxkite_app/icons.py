"""Retrieves icons for the frontend, with persistent caching."""

import json
import fastapi
import joblib
import os.path

mem = joblib.Memory(".icon-cache")
router = fastapi.APIRouter()


def load_icons():
    path = os.path.join(os.path.dirname(__file__), "web_assets", "tabler-icons.json")
    with open(path) as f:
        return json.load(f)


icons = load_icons()


@mem.cache
def _get_icon(icon_name: str):
    body = icons["icons"][icon_name]["body"]
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'


@router.get("/api/icons/{icon_name}")
def get_icon(icon_name: str):
    svg = _get_icon(icon_name)
    return fastapi.Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=2592000"},
    )
