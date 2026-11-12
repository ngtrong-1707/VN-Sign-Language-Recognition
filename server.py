"""
Root server entrypoint for VSL Vision platform.
Imports and serves the modernized modular FastAPI application in src/vsl,
or automatically switches to the lightweight serverless app on Vercel.
"""

import os
import sys
from pathlib import Path

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if os.environ.get("VERCEL"):
    from vsl.serverless.app import app  # noqa: F401
else:
    import uvicorn
    from vsl.config import settings
    from vsl.api.server import app  # noqa: F401

    if __name__ == "__main__":
        print(f"\n[SERVER] Đang khởi chạy máy chủ VSL Vision tại http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
        uvicorn.run("vsl.api.server:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT, reload=settings.DEBUG)
