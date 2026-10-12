"""
Data Collection CLI for Vietnamese Sign Language (VSL).
Records 30-frame sequential landmarks from webcam.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vsl.config import settings, ACTIONS
from vsl.core.detector import MediaPipeHolisticDetector
from vsl.core.extractor import extract_landmarks
from vsl.core.renderer import draw_styled_landmarks, draw_vietnamese_text


def create_directory_structure():
    """
    Creates dataset directories:
    data/sequences/<action_name>/<sequence_number>/<frame_number>.npy
    """
    for action in ACTIONS:
        action_dir_name = action.replace(" ", "_")
        for sequence in range(settings.NO_SEQUENCES):
            dir_path = settings.DATA_DIR / action_dir_name / str(sequence)
            dir_path.mkdir(parents=True, exist_ok=True)


def draw_hud(image, action, sequence_idx, frame_idx, is_capturing, countdown=0):
    height, width, _ = image.shape
    cv2.rectangle(image, (0, 0), (width, 70), (25, 25, 25), -1)
    cv2.line(image, (0, 70), (width, 70), (0, 255, 200), 2)

    status_text = f"Ky hieu: {action}  |  Mau: {sequence_idx + 1}/{settings.NO_SEQUENCES}"
    image = draw_vietnamese_text(
        image,
        status_text,
        (20, 15),
        font_size=24,
        font_color=(0, 255, 200)
    )

    if not is_capturing:
        overlay = image.copy()
        cv2.rectangle(overlay, (width // 4, height // 3), (3 * width // 4, 2 * height // 3), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        cv2.rectangle(image, (width // 4, height // 3), (3 * width // 4, 2 * height // 3), (0, 200, 255), 2)

        image = draw_vietnamese_text(
            image,
            f"CHUAN BI CHO MAU {sequence_idx + 1}",
            (width // 4 + 30, height // 3 + 30),
            font_size=22,
            font_color=(255, 255, 255)
        )
        image = draw_vietnamese_text(
            image,
            f"Bat dau sau: {countdown:.1f}s",
            (width // 4 + 70, height // 3 + 80),
            font_size=28,
            font_color=(0, 255, 255)
        )
    else:
        progress_width = int((frame_idx + 1) / settings.SEQUENCE_LENGTH * (width - 40))
        cv2.rectangle(image, (20, height - 40), (width - 20, height - 20), (50, 50, 50), -1)
        cv2.rectangle(image, (20, height - 40), (20 + progress_width, height - 20), (0, 0, 255), -1)

        cv2.putText(
            image,
            f"DANG THU DU LIEU: Frame {frame_idx + 1}/{settings.SEQUENCE_LENGTH}",
            (20, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return image


def run_collection():
    create_directory_structure()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 65)
    print(" BẮT ĐẦU THU THẬP DỮ LIỆU NGÔN NGỮ KÝ HIỆU TIẾNG VIỆT (VSL)")
    print("=" * 65)
    print(f"Tổng số từ/ký hiệu: {len(ACTIONS)}")
    print(f"Số mẫu mỗi từ: {settings.NO_SEQUENCES} samples")
    print(f"Số frames mỗi mẫu: {settings.SEQUENCE_LENGTH} frames")
    print("Nhấn 'q' bất kỳ lúc nào để thoát.\n")

    detector = MediaPipeHolisticDetector()
    try:
        for action in ACTIONS:
            action_dir_name = action.replace(" ", "_")
            print(f"\n>>> Chuẩn bị thu thập cử chỉ: '{action}'")

            for sequence in range(settings.NO_SEQUENCES):
                # 1. Countdown Phase
                start_time = time.time()
                while time.time() - start_time < settings.CAPTURE_DELAY_SEC:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    results = detector.process(frame)
                    draw_styled_landmarks(frame, results)

                    time_remaining = settings.CAPTURE_DELAY_SEC - (time.time() - start_time)
                    frame = draw_hud(
                        frame, action, sequence, 0, is_capturing=False, countdown=time_remaining
                    )

                    cv2.imshow("VSL Data Collection Engine", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

                # 2. Recording Phase
                for frame_num in range(settings.SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    results = detector.process(frame)
                    draw_styled_landmarks(frame, results)

                    keypoints = extract_landmarks(results)
                    npy_path = settings.DATA_DIR / action_dir_name / str(sequence) / f"{frame_num}.npy"
                    np.save(str(npy_path), keypoints)

                    frame = draw_hud(
                        frame, action, sequence, frame_num, is_capturing=True
                    )

                    cv2.imshow("VSL Data Collection Engine", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[HOÀN TẤT] Quá trình thu thập kết thúc.")


if __name__ == "__main__":
    run_collection()
