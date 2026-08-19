"""
Module: Real-Time FastAPI & WebSocket Backend Server for Vietnamese Sign Language (VSL).
Supports Live Webcam Streaming, Realtime Inference, Video File Analysis, and VSL Dictionary.
"""

import os
import sys
import time
import json
import base64
import collections

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import torch

import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from models.vsl_model import VSL_BiGRU
from utils.landmark_utils import (
    MediaPipeHolisticDetector,
    extract_landmarks
)


app = FastAPI(
    title="Real-time Vietnamese Sign Language Recognition (VSL)",
    description="Deep Learning VSL Translation System with Hope:RE UI Aesthetics",
    version="2.0.0"
)

# Mount Static Files & Jinja2 Templates
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)


class ModelEngine:
    def __init__(self):
        self.device = config.DEVICE
        self.actions = config.ACTIONS
        self.display_map = config.ACTION_DISPLAY_MAP
        self.icon_map = config.ACTION_ICON_MAP
        self.seq_len = config.SEQUENCE_LENGTH

        # Load Neural Network
        self.model = VSL_BiGRU(
            input_dim=config.INPUT_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_classes=len(self.actions),
            dropout=config.DROPOUT
        ).to(self.device)

        if os.path.exists(config.BEST_MODEL_PATH):
            print(f"[ENGINE] Nạp checkpoint mô hình từ: {config.BEST_MODEL_PATH}")
            self.model.load_state_dict(
                torch.load(config.BEST_MODEL_PATH, map_location=self.device, weights_only=True)
            )
        else:
            print("[ENGINE] Cảnh báo: Checkpoint chưa tồn tại. Đang khởi tạo trọng số mặc định.")

        self.model.eval()
        self.total_params = sum(p.numel() for p in self.model.parameters())

    def predict(self, sequence_window):
        """
        Input: np.ndarray shape (30, 258)
        Output: (predicted_name, display_name, icon, confidence, top5_list, latency_ms)
        """
        start_t = time.perf_counter()
        tensor_in = torch.tensor(
            sequence_window, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        predicted_name = self.actions[best_idx]
        display_name = self.display_map.get(predicted_name, predicted_name)
        icon = self.icon_map.get(predicted_name, "👋")

        # Top 5 probabilities
        top5_indices = np.argsort(probs)[::-1][:5]
        top5 = []
        for idx in top5_indices:
            name = self.actions[idx]
            top5.append({
                "name": name,
                "display": self.display_map.get(name, name),
                "icon": self.icon_map.get(name, "✨"),
                "probability": float(probs[idx])
            })

        return predicted_name, display_name, icon, confidence, top5, latency_ms


engine = ModelEngine()


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the main Hope:RE styled web interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "actions_metadata": config.ACTIONS_METADATA,
        "device": config.DEVICE.upper(),
        "total_params": f"{engine.total_params / 1e3:.1f}K",
        "actions_count": len(config.ACTIONS)
    })


@app.get("/api/dictionary")
async def get_dictionary():
    """Returns full catalog of 20 VSL signs with descriptions and icons."""
    return JSONResponse(content={
        "status": "success",
        "total": len(config.ACTIONS_METADATA),
        "data": config.ACTIONS_METADATA
    })


@app.get("/api/system_stats")
async def get_system_stats():
    """Returns telemetry data: device, memory, parameter count, architecture."""
    return JSONResponse(content={
        "device": config.DEVICE.upper(),
        "cuda_available": torch.cuda.is_available(),
        "parameters": engine.total_params,
        "input_features": config.INPUT_FEATURE_DIM,
        "sequence_length": config.SEQUENCE_LENGTH,
        "architecture": "Bi-directional GRU with Self-Attention (VSL_BiGRU v2.4)",
        "vocabulary_size": len(config.ACTIONS)
    })


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    High-speed Bidirectional WebSocket for real-time video stream & landmark inference.
    """
    await websocket.accept()
    buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
    detector = MediaPipeHolisticDetector()

    try:
        while True:
            # Receive frame as base64 JPEG from browser
            data = await websocket.receive_text()
            msg = json.loads(data)

            if "image" in msg:
                img_data = base64.b64decode(msg["image"].split(",")[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # MediaPipe Landmark Processing
                results = detector.process(frame)
                keypoints = extract_landmarks(results)
                buffer.append(keypoints)

                # If buffer is full (30 frames), run inference
                if len(buffer) == config.SEQUENCE_LENGTH:
                    window_arr = np.array(buffer)
                    pred_name, display_name, icon, conf, top5, latency_ms = engine.predict(window_arr)

                    # Check if hands detected
                    hand_res = results.get("hand")
                    has_hands = bool(hand_res and len(hand_res.hand_landmarks) > 0)

                    await websocket.send_json({
                        "type": "prediction",
                        "name": pred_name,
                        "display": display_name,
                        "icon": icon,
                        "confidence": conf,
                        "is_confident": conf >= config.PREDICTION_THRESHOLD,
                        "top5": top5,
                        "latency_ms": round(latency_ms, 2),
                        "has_hands": has_hands,
                        "buffer_size": len(buffer)
                    })
                else:
                    await websocket.send_json({
                        "type": "buffering",
                        "buffer_size": len(buffer),
                        "required": config.SEQUENCE_LENGTH
                    })

    except WebSocketDisconnect:
        print("[WS] Client đã ngắt kết nối WebSocket.")
    except Exception as e:
        print(f"[WS LỖI] {e}")
    finally:
        detector.close()


@app.post("/api/predict_video")
async def predict_video_file(file: UploadFile = File(...)):
    """
    Processes an uploaded video file frame-by-frame and produces a chronological
    timeline of recognized Vietnamese Sign Language gestures with timestamps!
    """
    temp_path = os.path.join(config.LOGS_DIR, f"temp_{int(time.time())}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
    timeline = []
    detector = MediaPipeHolisticDetector()

    frame_idx = 0
    last_detected = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = detector.process(frame)
        keypoints = extract_landmarks(results)
        buffer.append(keypoints)

        if len(buffer) == config.SEQUENCE_LENGTH and frame_idx % 5 == 0:
            window_arr = np.array(buffer)
            pred_name, display_name, icon, conf, top5, _ = engine.predict(window_arr)

            if conf >= config.PREDICTION_THRESHOLD and pred_name != last_detected:
                timestamp_sec = round(frame_idx / fps, 2)
                timeline.append({
                    "timestamp": f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}",
                    "second": timestamp_sec,
                    "name": pred_name,
                    "display": display_name,
                    "icon": icon,
                    "confidence": round(conf * 100, 1)
                })
                last_detected = pred_name

    cap.release()
    detector.close()
    if os.path.exists(temp_path):
        os.remove(temp_path)


    full_sentence = " ".join([item["display"] for item in timeline])
    return JSONResponse(content={
        "status": "success",
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2),
        "sentence": full_sentence if full_sentence else "Không phát hiện cử chỉ VSL rõ ràng",
        "timeline": timeline
    })


if __name__ == "__main__":
    import uvicorn
    print(f"\n[SERVER] Đang khởi chạy máy chủ tại http://localhost:{config.SERVER_PORT}")
    uvicorn.run("server:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=True)
