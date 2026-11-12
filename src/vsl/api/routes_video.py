"""
REST API routes for offline video file analysis, chronological timeline extraction,
landmark-window prediction, and interactive gesture simulation from the VSL dataset.
"""

import os
import time
from typing import List

import cv2
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vsl.config import settings
from vsl.core.detector import MediaPipeHolisticDetector
from vsl.core.extractor import extract_landmarks
from vsl.models.engine import VSLInferenceEngine
from vsl.serverless import core

router = APIRouter(tags=["Video & Prediction"])


class PredictRequest(BaseModel):
    frames: List[List[float]] = Field(..., description="Cửa sổ 30 frames × 258 đặc trưng landmark")


class TimelineRequest(BaseModel):
    frames: List[List[float]] = Field(..., description="Chuỗi frames × 258 đặc trưng landmark theo thời gian")
    fps: float = Field(20.0, gt=0, le=120, description="Tốc độ lấy mẫu frame")


def _engine(request: Request):
    return getattr(request.app.state, "engine", None) or VSLInferenceEngine(use_onnx=settings.USE_ONNX)


@router.post("/api/predict_video")
async def predict_video_file(request: Request, file: UploadFile = File(...)):
    """
    Processes an uploaded video file frame-by-frame and produces a chronological
    timeline of recognized Vietnamese Sign Language gestures with timestamps.
    """
    safe_name = os.path.basename(file.filename or "upload.mp4")
    temp_path = str(settings.LOGS_DIR / f"temp_{int(time.time())}_{safe_name}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = MediaPipeHolisticDetector()
    frames = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Training data was recorded on mirrored webcam frames
            frame = cv2.flip(frame, 1)
            frames.append(extract_landmarks(detector.process(frame)))
    finally:
        cap.release()
        detector.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return JSONResponse(content=core.build_timeline(frames, fps, _engine(request)))


@router.post("/api/predict")
async def predict_window(request: Request, payload: PredictRequest):
    """Classifies a single 30-frame landmark window (used by HTTP / serverless clients)."""
    try:
        window = core.to_window(payload.frames)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not core.hands_present(window[-1]):
        return {"type": "idle", "buffer_size": 0, "required": core.SEQUENCE_LENGTH}
    return core.prediction_payload(_engine(request).predict(window), settings.PREDICTION_THRESHOLD)


@router.post("/api/predict_timeline")
async def predict_timeline(request: Request, payload: TimelineRequest):
    """Builds a sign timeline from landmarks extracted client-side (browser MediaPipe)."""
    if len(payload.frames) > core.MAX_TIMELINE_FRAMES:
        raise HTTPException(status_code=413, detail=f"Tối đa {core.MAX_TIMELINE_FRAMES} frames mỗi lần phân tích.")
    if payload.frames and len(payload.frames[0]) != core.FEATURE_DIM:
        raise HTTPException(status_code=422, detail=f"Mỗi frame phải có đúng {core.FEATURE_DIM} giá trị.")
    return core.build_timeline(payload.frames, payload.fps, _engine(request), settings.PREDICTION_THRESHOLD)


@router.get("/api/predict_video/demo_sequence/{action_name}")
async def get_demo_sequence(action_name: str, request: Request, sample_idx: int = 0):
    """
    Loads a 30-frame landmark sequence of the requested sign so the browser can
    simulate real-time 2D skeleton playback + ONNX inference.
    """
    data = core.demo_sequence_payload(action_name, _engine(request), sample_idx)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu mẫu cho cử chỉ '{action_name}'.")
    return JSONResponse(content=data)


@router.post("/api/predict_video/demo_timeline")
async def predict_demo_timeline(request: Request):
    """
    Runs a multi-sign demonstration timeline ("Xin chào" -> "Tôi" -> "Yêu" -> "Việt Nam" -> "Cảm ơn")
    through the real inference engine for instant testing of the Video Analysis tab.
    """
    return JSONResponse(content=core.run_demo_timeline(_engine(request)))
