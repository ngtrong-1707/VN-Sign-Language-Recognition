"""
CLI Entrypoint for Live OpenCV HUD Real-time Recognition.
"""

import sys
import time
from collections import deque
import cv2
import numpy as np

from vsl.config import settings, ACTIONS
from vsl.models.engine import VSLInferenceEngine
from vsl.core.detector import MediaPipeHolisticDetector
from vsl.core.extractor import extract_landmarks
from vsl.core.renderer import draw_styled_landmarks, draw_vietnamese_text
from vsl.core.smoothing import TemporalPredictionSmoother


def run_hud(camera_index: int = 0):
    engine = VSLInferenceEngine(use_onnx=settings.USE_ONNX)
    detector = MediaPipeHolisticDetector()
    smoother = TemporalPredictionSmoother(
        window_size=settings.HISTORY_SMOOTHING_LEN,
        threshold=settings.PREDICTION_THRESHOLD
    )

    buffer = deque(maxlen=settings.SEQUENCE_LENGTH)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    current_pred = None
    conf = 0.0
    probs = None

    print("\n" + "=" * 65)
    print(" HỆ THỐNG NHẬN DIỆN VSL THEO THỜI GIAN THỰC (CLI OPENCV HUD)")
    print("=" * 65)
    print(" - Nhấn 'c' để xóa câu dịch hiện tại.")
    print(" - Nhấn 'q' để thoát chương trình.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[CẢNH BÁO] Không thể đọc luồng camera.")
                break

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)

            # 1. MediaPipe Landmark Extraction
            results = detector.process(frame)
            draw_styled_landmarks(frame, results)

            # 2. Append features to sliding window
            keypoints = extract_landmarks(results)
            buffer.append(keypoints)

            # 3. Model Inference & Temporal Smoothing
            if len(buffer) == settings.SEQUENCE_LENGTH:
                window_arr = np.array(buffer)
                current_pred, disp_name, icon, conf, top5, latency_ms = engine.predict(window_arr)
                smoother.update(disp_name, conf)

            # 4. Draw HUD Dashboard
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 25), -1)
            cv2.line(frame, (0, 65), (w, 65), (0, 200, 255), 2)

            sentence_str = smoother.get_sentence_text() or "[Đang chờ ký hiệu...]"
            frame = draw_vietnamese_text(
                frame,
                f"VĂN BẢN DỊCH: {sentence_str}",
                (20, 16),
                font_size=24,
                font_color=(255, 255, 255)
            )

            # Bottom status
            cv2.rectangle(frame, (0, h - 35), (w, h), (15, 15, 20), -1)
            status_info = f"FPS: {fps:.1f}  |  Buffer: {len(buffer)}/{settings.SEQUENCE_LENGTH}  |  [C]: Xóa câu  |  [Q]: Thoát"
            frame = draw_vietnamese_text(frame, status_info, (15, h - 28), font_size=16, font_color=(180, 180, 180))

            # Center Badge if confident
            if current_pred and conf >= settings.PREDICTION_THRESHOLD:
                badge_text = f"{current_pred} ({conf*100:.0f}%)"
                frame = draw_vietnamese_text(
                    frame,
                    badge_text,
                    (30, 85),
                    font_size=28,
                    font_color=(0, 255, 100),
                    bg_color=(0, 50, 20),
                    padding=8
                )

            cv2.imshow("VSL Real-Time Recognition (Production Engine)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                smoother.clear()
                print("[INFO] Đã xóa câu dịch.")

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hud()
