"""
REST API route for offline video file analysis and chronological timeline extraction.
"""

import os
import time
from collections import deque
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from vsl.config import settings
from vsl.core.detector import MediaPipeHolisticDetector
from vsl.core.extractor import extract_landmarks
from vsl.models.engine import VSLInferenceEngine

router = APIRouter(prefix="/api/predict_video", tags=["Video"])


@router.post("")
async def predict_video_file(file: UploadFile = File(...)):
    """
    Processes an uploaded video file frame-by-frame and produces a chronological
    timeline of recognized Vietnamese Sign Language gestures with timestamps.
    """
    temp_path = str(settings.LOGS_DIR / f"temp_{int(time.time())}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buffer = deque(maxlen=settings.SEQUENCE_LENGTH)
    timeline = []
    detector = MediaPipeHolisticDetector()
    engine = VSLInferenceEngine(use_onnx=settings.USE_ONNX)

    frame_idx = 0
    last_detected = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            results = detector.process(frame)
            keypoints = extract_landmarks(results)
            buffer.append(keypoints)

            if len(buffer) == settings.SEQUENCE_LENGTH and frame_idx % 5 == 0:
                window_arr = np.array(buffer)
                pred_name, display_name, icon, conf, top5, _ = engine.predict(window_arr)

                if conf >= settings.PREDICTION_THRESHOLD and pred_name != last_detected:
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

    finally:
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
