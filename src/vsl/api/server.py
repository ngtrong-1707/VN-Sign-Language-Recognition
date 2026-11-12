"""
FastAPI Main Application Server for VSL Recognition Platform.
Integrates WebSockets, REST APIs, and Jinja2 frontend views.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vsl.config import settings, ACTIONS, ACTIONS_METADATA
from vsl.db.base import init_db
from vsl.models.engine import VSLInferenceEngine
from vsl.api.routes_dict import router as dict_router
from vsl.api.routes_feedback import router as feedback_router
from vsl.api.routes_video import router as video_router
from vsl.api.routes_camera import router as camera_router
from vsl.api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure database initialized
    await init_db()
    # Preload inference engine
    app.state.engine = VSLInferenceEngine(use_onnx=settings.USE_ONNX)
    yield
    # Shutdown logic if any


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Hệ thống Nhận diện Cử chỉ Ngôn ngữ Ký hiệu Tiếng Việt Thời gian thực (Production Ready)",
    lifespan=lifespan
)

# Mount static, assets, and templates
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
assets_dir = settings.BASE_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# Include API Routers
app.include_router(dict_router)
app.include_router(feedback_router)
app.include_router(video_router)
app.include_router(camera_router)
app.include_router(ws_router)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the main Hope:RE web studio dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "actions_metadata": ACTIONS_METADATA,
            "device": settings.DEVICE.upper(),
            "total_params": "428.5K",
            "actions_count": len(ACTIONS),
            "engine_type": "ONNX Runtime (CPU <1ms)" if settings.USE_ONNX else "PyTorch Native",
            "deploy_mode": "server"
        }
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "mode": "server", "version": settings.VERSION}


@app.get("/api/system_stats")
async def get_system_stats():
    """Returns telemetry data: device, runtime, architecture, features."""
    return JSONResponse(content={
        "status": "online",
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "device": settings.DEVICE.upper(),
        "engine": "ONNX Runtime" if settings.USE_ONNX else "PyTorch",
        "input_features": settings.INPUT_FEATURE_DIM,
        "sequence_length": settings.SEQUENCE_LENGTH,
        "architecture": "Bi-directional GRU with Temporal Attention (VSL_BiGRU v2.4)",
        "vocabulary_size": len(ACTIONS)
    })


def main():
    """CLI Entrypoint for vsl-server."""
    import uvicorn
    uvicorn.run(
        "vsl.api.server:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=False,
        log_level="info"
    )
