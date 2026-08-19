import os
import sys
import time
import cv2
import numpy as np

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import config
from utils.landmark_utils import (
    MediaPipeHolisticDetector,
    draw_styled_landmarks,
    extract_landmarks,
    draw_vietnamese_text,
)



def create_directory_structure():
    """
    Creates dataset directories:
    data/sequences/<action_name>/<sequence_number>/<frame_number>.npy
    """
    for action in config.ACTIONS:
        action_dir_name = action.replace(" ", "_")
        for sequence in range(config.NO_SEQUENCES):
            dir_path = os.path.join(config.DATA_DIR, action_dir_name, str(sequence))
            os.makedirs(dir_path, exist_ok=True)


def draw_hud(image, action, sequence_idx, frame_idx, is_capturing, countdown=0):
    """
    Draws modern, informative HUD overlay during data collection.
    """
    height, width, _ = image.shape
    
    # Top Status Bar
    cv2.rectangle(image, (0, 0), (width, 70), (25, 25, 25), -1)
    cv2.line(image, (0, 70), (width, 70), (0, 255, 200), 2)
    
    status_text = f"Ky hieu: {action}  |  Mau: {sequence_idx + 1}/{config.NO_SEQUENCES}"
    image = draw_vietnamese_text(
        image,
        status_text,
        (20, 15),
        font_size=24,
        font_color=(0, 255, 200)
    )

    if not is_capturing:
        # Preparation Countdown Modal
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
        # Recording progress indicator
        progress_width = int((frame_idx + 1) / config.SEQUENCE_LENGTH * (width - 40))
        cv2.rectangle(image, (20, height - 40), (width - 20, height - 20), (50, 50, 50), -1)
        cv2.rectangle(image, (20, height - 40), (20 + progress_width, height - 20), (0, 0, 255), -1)
        
        cv2.putText(
            image,
            f"DANG THU AM/HINH: Frame {frame_idx + 1}/{config.SEQUENCE_LENGTH}",
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
    # Set high camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "=" * 60)
    print(" BẮT ĐẦU THU THẬP DỮ LIỆU NGÔN NGỮ KÝ HIỆU TIẾNG VIỆT (VSL)")
    print("=" * 60)
    print(f"Tổng số từ/ký hiệu: {len(config.ACTIONS)}")
    print(f"Số mẫu mỗi từ: {config.NO_SEQUENCES} samples")
    print(f"Số frames mỗi mẫu: {config.SEQUENCE_LENGTH} frames")
    print("Nhấn 'q' bất kỳ lúc nào để thoát.\n")

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for action in config.ACTIONS:
            action_dir_name = action.replace(" ", "_")
            print(f"\n>>> Chuẩn bị thu thập cử chỉ: '{action}'")

            for sequence in range(config.NO_SEQUENCES):
                # 1. Countdown Phase (2 seconds for user to get in position)
                start_time = time.time()
                while time.time() - start_time < config.CAPTURE_DELAY_SEC:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)  # Mirror view
                    results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(frame, results)

                    time_remaining = config.CAPTURE_DELAY_SEC - (time.time() - start_time)
                    frame = draw_hud(
                        frame, action, sequence, 0, is_capturing=False, countdown=time_remaining
                    )

                    cv2.imshow("VSL Data Collection Engine", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # 2. Recording Phase (Record sequence of frames)
                for frame_num in range(config.SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(frame, results)

                    # Extract landmark vector (258 features)
                    keypoints = extract_landmarks(results)
                    npy_path = os.path.join(
                        config.DATA_DIR, action_dir_name, str(sequence), f"{frame_num}.npy"
                    )
                    np.save(npy_path, keypoints)

                    frame = draw_hud(
                        frame, action, sequence, frame_num, is_capturing=True
                    )

                    cv2.imshow("VSL Data Collection Engine", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print("\n[THÀNH CÔNG] Hoàn tất thu thập dữ liệu toàn bộ ký hiệu!")


if __name__ == "__main__":
    run_collection()
