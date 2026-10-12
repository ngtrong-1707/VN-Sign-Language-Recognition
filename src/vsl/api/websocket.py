"""
High-speed Bidirectional WebSocket Endpoint for Real-time Landmark Stream.
Supports both:
1. Direct Landmark Vectors (Client-side MediaPipe WASM - Ultra-lightweight, <1KB/frame)
2. Base64 JPEG Video Frames (Server-side MediaPipe fallback)
"""

import json
import base64
from collections import deque
import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vsl.config import settings
from vsl.models.engine import VSLInferenceEngine
from vsl.core.extractor import extract_landmarks
from vsl.core.detector import MediaPipeHolisticDetector

router = APIRouter(tags=["Stream"])


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = deque(maxlen=settings.SEQUENCE_LENGTH)
    engine = VSLInferenceEngine(use_onnx=settings.USE_ONNX)
    detector = None

    try:
        while True:
            text_data = await websocket.receive_text()
            msg = json.loads(text_data)

            keypoints = None

            # Mode 1: Client sent direct landmark vector (Fastest, zero server vision overhead)
            if "landmarks" in msg:
                raw_landmarks = msg["landmarks"]
                if len(raw_landmarks) == settings.INPUT_FEATURE_DIM:
                    keypoints = np.array(raw_landmarks, dtype=np.float32)

            # Mode 2: Client sent base64 image (Fallback server vision pipeline)
            elif "image" in msg:
                if detector is None:
                    detector = MediaPipeHolisticDetector()

                img_bytes = base64.b64decode(msg["image"].split(",")[1])
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    results = detector.process(frame)
                    keypoints = extract_landmarks(results)

            if keypoints is not None:
                buffer.append(keypoints)

                if len(buffer) == settings.SEQUENCE_LENGTH:
                    window_arr = np.array(buffer)
                    pred_name, display_name, icon, conf, top5, latency_ms = engine.predict(window_arr)

                    await websocket.send_json({
                        "type": "prediction",
                        "name": pred_name,
                        "display": display_name,
                        "icon": icon,
                        "confidence": conf,
                        "is_confident": conf >= settings.PREDICTION_THRESHOLD,
                        "top5": top5,
                        "latency_ms": round(latency_ms, 2),
                        "buffer_size": len(buffer)
                    })
                else:
                    await websocket.send_json({
                        "type": "buffering",
                        "buffer_size": len(buffer),
                        "required": settings.SEQUENCE_LENGTH
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        if detector:
            detector.close()
