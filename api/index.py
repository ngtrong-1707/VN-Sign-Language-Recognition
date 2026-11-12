"""
Vercel entrypoint — exposes the serverless FastAPI app as `app`.
"""

import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from vsl.serverless.app import app  # noqa: E402,F401
except Exception as exc:  # pragma: no cover - diagnostic fallback on cloud cold start
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    _err_msg = str(exc)
    _err_tb = traceback.format_exc()
    app = FastAPI(title="VSL Vision Startup Diagnostic")

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def _startup_error(path: str = ""):
        return JSONResponse(
            status_code=500,
            content={
                "status": "startup_error",
                "error": _err_msg,
                "traceback": _err_tb,
                "cwd": str(Path.cwd()),
                "root_dir": str(ROOT_DIR),
            },
        )
