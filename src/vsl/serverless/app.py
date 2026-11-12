"""
VSL Vision — Serverless FastAPI application (Vercel).

Differences from the full server (`vsl.api.server`):
- No PyTorch / OpenCV / MediaPipe on the server: landmarks are extracted in the
  browser with MediaPipe Tasks Vision (WASM) and sent as 258-d vectors.
- No WebSocket (Vercel Functions are request/response): live inference uses
  `POST /api/predict` on a client-side 30-frame sliding window.
- No database: the dictionary is served from `label_map.json`, feedback is
  acknowledged but not persisted.
"""

from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from vsl import __version__
from vsl.serverless import core

app = FastAPI(
    title="VSL Vision (Serverless)",
    version=__version__,
    description="Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt — bản triển khai serverless (MediaPipe WASM + ONNX Runtime).",
)

app.mount("/static", StaticFiles(directory=str(core.ROOT_DIR / "static")), name="static")
if (core.ROOT_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(core.ROOT_DIR / "assets")), name="assets")
templates = Jinja2Templates(directory=str(core.ROOT_DIR / "templates"))

_engine = None


def get_engine() -> core.LiteInferenceEngine:
    global _engine
    if _engine is None:
        _engine = core.LiteInferenceEngine()
    return _engine


class PredictRequest(BaseModel):
    frames: List[List[float]] = Field(..., description="Cửa sổ 30 frames × 258 đặc trưng landmark")


class TimelineRequest(BaseModel):
    frames: List[List[float]] = Field(..., description="Chuỗi frames × 258 đặc trưng landmark theo thời gian")
    fps: float = Field(20.0, gt=0, le=120, description="Tốc độ lấy mẫu frame")


class FeedbackRequest(BaseModel):
    predicted_action: str
    actual_action: str
    confidence: float = 0.0
    session_id: str | None = None
    notes: str | None = None


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    meta = core.load_actions_metadata()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "actions_metadata": meta,
            "device": "CPU",
            "total_params": "428.5K",
            "actions_count": len(meta),
            "engine_type": "ONNX Runtime · Serverless (MediaPipe WASM)",
            "deploy_mode": "serverless",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "mode": "serverless", "version": __version__}


@app.get("/api/system_stats")
async def get_system_stats():
    return {
        "status": "online",
        "app_name": "VSL Vision",
        "version": __version__,
        "device": "CPU",
        "engine": "ONNX Runtime (Serverless)",
        "mode": "serverless",
        "input_features": core.FEATURE_DIM,
        "sequence_length": core.SEQUENCE_LENGTH,
        "architecture": "Bi-directional GRU with Temporal Attention (VSL_BiGRU v2.4)",
        "vocabulary_size": len(core.load_actions_metadata()),
    }


@app.get("/api/dictionary")
async def get_dictionary():
    meta = core.load_actions_metadata()
    data = [
        {
            "id": item["id"] + 1,
            "name": item["name"],
            "display": item["display"],
            "category": item.get("category", "Chung"),
            "desc": item.get("desc", ""),
            "icon": item.get("icon", "👋"),
        }
        for item in meta
    ]
    return {"status": "success", "total": len(data), "data": data}


@app.get("/api/dictionary/{action_key}")
async def get_dictionary_item(action_key: str):
    for item in core.load_actions_metadata():
        if item["name"] == action_key:
            return {"status": "success", "data": {**item, "id": item["id"] + 1}}
    raise HTTPException(status_code=404, detail=f"Cử chỉ '{action_key}' không tồn tại trong từ điển.")


@app.post("/api/predict")
async def predict_window(payload: PredictRequest):
    try:
        window = core.to_window(payload.frames)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not core.hands_present(window[-1]):
        return {"type": "idle", "buffer_size": 0, "required": core.SEQUENCE_LENGTH}
    return core.prediction_payload(get_engine().predict(window))


@app.post("/api/predict_timeline")
async def predict_timeline(payload: TimelineRequest):
    if len(payload.frames) > core.MAX_TIMELINE_FRAMES:
        raise HTTPException(status_code=413, detail=f"Tối đa {core.MAX_TIMELINE_FRAMES} frames mỗi lần phân tích.")
    if payload.frames and len(payload.frames[0]) != core.FEATURE_DIM:
        raise HTTPException(status_code=422, detail=f"Mỗi frame phải có đúng {core.FEATURE_DIM} giá trị.")
    return core.build_timeline(payload.frames, payload.fps, get_engine())


@app.get("/api/predict_video/demo_sequence/{action_name}")
async def demo_sequence(action_name: str):
    data = core.demo_sequence_payload(action_name, get_engine())
    if data is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu mẫu cho cử chỉ '{action_name}'.")
    return JSONResponse(content=data)


@app.post("/api/predict_video/demo_timeline")
async def demo_timeline():
    return core.run_demo_timeline(get_engine())


@app.post("/api/predict_video")
async def predict_video_upload():
    raise HTTPException(
        status_code=501,
        detail="Bản serverless phân tích video ngay trên trình duyệt và gửi landmarks tới /api/predict_timeline.",
    )


@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackRequest):
    print(f"[FEEDBACK] {payload.predicted_action} -> {payload.actual_action} ({payload.confidence:.2f}) {payload.notes or ''}")
    return {
        "status": "success",
        "message": "Cảm ơn bạn đã đóng góp phản hồi để cải thiện mô hình!",
        "feedback_id": None,
        "persisted": False,
    }

