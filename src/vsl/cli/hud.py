"""
CLI Entrypoint for Live OpenCV HUD Real-time Recognition.
Supports device enumeration, camera index selection, RTSP/HTTP streams, and custom resolutions.
"""

import sys
import time
import argparse
from collections import deque
from typing import Union
import cv2
import numpy as np

from vsl.config import settings, ACTIONS
from vsl.models.engine import VSLInferenceEngine
from vsl.core.detector import MediaPipeHolisticDetector
from vsl.core.extractor import extract_landmarks
from vsl.core.renderer import draw_styled_landmarks, draw_vietnamese_text
from vsl.core.smoothing import TemporalPredictionSmoother


def scan_available_cameras(max_tested: int = 4):
    """Probes system video capture indices to detect connected webcams."""
    cameras = []
    print("\n[CAMERA DISCOVERY] Đang quét các cổng camera kết nối...")
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            cap.release()
            if ret:
                cameras.append({"index": i, "width": w, "height": h, "fps": fps})
                print(f"  ✓ Camera [{i}]: Độ phân giải {w}x{h} @ {fps}fps")
    if not cameras:
        print("  ⚠️ Không tìm thấy thiết bị camera khả dụng nào.")
    print()
    return cameras


def run_hud(camera_source: Union[int, str] = 0, width: int = 1280, height: int = 720, use_onnx: bool = True):
    # Try converting numeric string to int
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    print("\n" + "=" * 65)
    print(" HỆ THỐNG NHẬN DIỆN VSL THEO THỜI GIAN THỰC (CLI OPENCV HUD)")
    print("=" * 65)
    print(f" - Nguồn Camera: {camera_source}")
    print(f" - Độ phân giải mục tiêu: {width}x{height}")
    print(f" - Động cơ suy luận: {'ONNX Runtime' if use_onnx else 'PyTorch Native'}")
    print(" - Nhấn 'c' để xóa câu dịch hiện tại.")
    print(" - Nhấn 'q' để thoát chương trình.\n")

    engine = VSLInferenceEngine(use_onnx=use_onnx)
    detector = MediaPipeHolisticDetector()
    smoother = TemporalPredictionSmoother(
        window_size=settings.HISTORY_SMOOTHING_LEN,
        threshold=settings.PREDICTION_THRESHOLD
    )

    buffer = deque(maxlen=settings.SEQUENCE_LENGTH)
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[LỖI] Không thể kết nối tới nguồn camera: {camera_source}")
        print("Mẹo: Dùng cờ '--list-cameras' để kiểm tra các cổng camera đang kết nối.")
        return

    prev_time = time.time()
    current_pred = None
    conf = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[CẢNH BÁO] Luồng camera bị ngắt quãng hoặc kết thúc.")
                break

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Mirror view for natural webcam interaction
            if isinstance(camera_source, int):
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
            status_info = f"Cam: {camera_source}  |  FPS: {fps:.1f}  |  Buffer: {len(buffer)}/{settings.SEQUENCE_LENGTH}  |  [C]: Xóa câu  |  [Q]: Thoát"
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


def main():
    parser = argparse.ArgumentParser(description="Chạy nhận diện VSL qua camera thời gian thực.")
    parser.add_argument("-c", "--camera", default=0, help="Chỉ số camera (0, 1, 2) hoặc đường dẫn RTSP/HTTP/File video.")
    parser.add_argument("-W", "--width", type=int, default=1280, help="Độ rộng khung hình camera (mặc định 1280).")
    parser.add_argument("-H", "--height", type=int, default=720, help="Độ cao khung hình camera (mặc định 720).")
    parser.add_argument("--list-cameras", action="store_true", help="Liệt kê toàn bộ camera khả dụng rồi thoát.")
    parser.add_argument("--pytorch", action="store_true", help="Sử dụng động cơ PyTorch thay vì ONNX Runtime.")

    args = parser.parse_args()

    if args.list_cameras:
        scan_available_cameras()
        return

    run_hud(
        camera_source=args.camera,
        width=args.width,
        height=args.height,
        use_onnx=not args.pytorch
    )


if __name__ == "__main__":
    main()
