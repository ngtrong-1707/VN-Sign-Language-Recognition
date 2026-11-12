"""
Vercel FastAPI entrypoint — defines a top-level `app = FastAPI(...)` instance
for Vercel's native FastAPI builder and attaches all serverless routes.
"""

import sys
from pathlib import Path
from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vsl.serverless.app import configure_serverless_app  # noqa: E402

app = FastAPI(
    title="VSL Vision (Serverless)",
    version="2.2.0",
    description="Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt — bản triển khai serverless (MediaPipe WASM + ONNX Runtime).",
)
configure_serverless_app(app)
