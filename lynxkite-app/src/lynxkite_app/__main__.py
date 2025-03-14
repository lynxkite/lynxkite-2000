import uvicorn
from .main import app  # noqa: F401
import os


def main():
    port = int(os.environ.get("PORT", "8000"))
    reload = bool(os.environ.get("LYNXKITE_RELOAD", ""))
    uvicorn.run(
        "lynxkite_app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        loop="asyncio",
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
