import os
import sys
import time
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

import config
from models.vsl_model import VSL_BiGRU
from utils.landmark_utils import (
    MediaPipeHolisticDetector,
    draw_styled_landmarks,
    extract_landmarks,
    draw_vietnamese_text
)



class RealtimeVSLRecognizer:
    def __init__(self, model_path=config.BEST_MODEL_PATH):
        self.device = config.DEVICE
        self.actions = config.ACTIONS
        self.sequence_length = config.SEQUENCE_LENGTH
        self.threshold = config.PREDICTION_THRESHOLD

        # Sliding window buffer for sequential frames
        self.sequence_buffer = collections.deque(maxlen=self.sequence_length)
        self.prediction_history = collections.deque(maxlen=config.HISTORY_SMOOTHING_LEN)
        self.sentence = []

        # Load or initialize model
        self.model = VSL_BiGRU(
            input_dim=config.INPUT_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_classes=len(self.actions),
            dropout=config.DROPOUT
        ).to(self.device)

        if os.path.exists(model_path):
            print(f"[INFO] Tải trọng số mô hình từ: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print("[CẢNH BÁO] Chưa tìm thấy checkpoint đã huấn luyện. Đang chạy ở chế độ Demo/Dummy Model.")
        
        self.model.eval()

    def predict_sequence(self):
        """
        Runs neural network inference on the current sliding window buffer.
        Returns:
            predicted_action (str)
            confidence (float)
            probabilities (np.ndarray)
        """
        if len(self.sequence_buffer) < self.sequence_length:
            return None, 0.0, None

        input_tensor = torch.tensor(
            np.array(self.sequence_buffer), dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # Shape: (1, 30, 258)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_idx = np.argmax(probs)
        confidence = probs[best_idx]
        predicted_action = self.actions[best_idx]

        return predicted_action, confidence, probs

    def update_sentence(self, current_prediction, confidence):
        """
        Smooths temporal predictions and updates sentence translation.
        """
        if current_prediction is not None and confidence >= self.threshold:
            self.prediction_history.append(current_prediction)
            
            # Check if all recent predictions agree (Temporal consistency)
            if len(self.prediction_history) == config.HISTORY_SMOOTHING_LEN:
                counts = collections.Counter(self.prediction_history)
                most_common, freq = counts.most_common(1)[0]
                
                if freq >= 4:  # At least 4 of last 5 frames agreed
                    if len(self.sentence) == 0 or self.sentence[-1] != most_common:
                        self.sentence.append(most_common)
                        # Keep only the last 6 words in sentence history
                        if len(self.sentence) > 6:
                            self.sentence = self.sentence[-6:]

    def draw_dashboard(self, image, current_prediction, confidence, probs, fps):
        """
        Draws sleek, modern HUD dashboard on frame.
        """
        h, w, _ = image.shape

        # 1. Top Bar: Live Sentence / Translation Result
        cv2.rectangle(image, (0, 0), (w, 65), (20, 20, 25), -1)
        cv2.line(image, (0, 65), (w, 65), (0, 200, 255), 2)

        sentence_str = " ".join(self.sentence) if self.sentence else "[Đang chờ ký hiệu...]"
        image = draw_vietnamese_text(
            image,
            f"VĂN BẢN DỊCH: {sentence_str}",
            (20, 16),
            font_size=24,
            font_color=(255, 255, 255)
        )

        # 2. Right Sidebar: Top Prediction Confidence Bars
        if probs is not None:
            sidebar_w = 260
            sidebar_x = w - sidebar_w - 20
            sidebar_y = 80
            
            overlay = image.copy()
            cv2.rectangle(overlay, (sidebar_x - 10, sidebar_y - 10), (w - 10, sidebar_y + 190), (15, 15, 20), -1)
            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
            cv2.rectangle(image, (sidebar_x - 10, sidebar_y - 10), (w - 10, sidebar_y + 190), (70, 70, 80), 1)

            image = draw_vietnamese_text(
                image, "XÁC SUẤT DỰ ĐOÁN", (sidebar_x, sidebar_y), font_size=16, font_color=(0, 255, 200)
            )

            # Get Top 3 predictions
            top3_indices = np.argsort(probs)[::-1][:3]
            for i, idx in enumerate(top3_indices):
                act_name = self.actions[idx]
                act_prob = probs[idx]
                bar_y = sidebar_y + 35 + (i * 45)

                # Action name
                image = draw_vietnamese_text(
                    image, f"{act_name}: {act_prob*100:.1f}%", (sidebar_x, bar_y), font_size=15, font_color=(230, 230, 230)
                )
                
                # Progress Bar
                bar_width = int(act_prob * (sidebar_w - 20))
                cv2.rectangle(image, (sidebar_x, bar_y + 22), (sidebar_x + sidebar_w - 20, bar_y + 30), (50, 50, 60), -1)
                bar_color = (0, 255, 100) if i == 0 and act_prob >= self.threshold else (200, 120, 0)
                cv2.rectangle(image, (sidebar_x, bar_y + 22), (sidebar_x + bar_width, bar_y + 30), bar_color, -1)

        # 3. Bottom Status Bar: FPS & Buffer Status
        cv2.rectangle(image, (0, h - 35), (w, h), (15, 15, 20), -1)
        status_info = f"FPS: {fps:.1f}  |  Buffer: {len(self.sequence_buffer)}/{self.sequence_length}  |  [C]: Xóa câu  |  [Q]: Thoát"
        image = draw_vietnamese_text(image, status_info, (15, h - 28), font_size=16, font_color=(180, 180, 180))

        # 4. Center Prediction Badge if confident
        if current_prediction and confidence >= self.threshold:
            badge_text = f"{current_prediction} ({confidence*100:.0f}%)"
            image = draw_vietnamese_text(
                image,
                badge_text,
                (30, 85),
                font_size=28,
                font_color=(0, 255, 100),
                bg_color=(0, 50, 20),
                padding=8
            )

        return image


def run_inference():
    recognizer = RealtimeVSLRecognizer()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    print("\n" + "=" * 60)
    print(" HỆ THỐNG NHẬN DIỆN VSL THEO THỜI GIAN THỰC (REAL-TIME ENGINE)")
    print("=" * 60)
    print(" - Nhấn 'c' để xóa câu dịch hiện tại.")
    print(" - Nhấn 'q' để thoát chương trình.\n")

    detector = MediaPipeHolisticDetector()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[CẢNH BÁO] Không thể đọc luồng camera.")
                break

            # Calculate FPS
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
            recognizer.sequence_buffer.append(keypoints)

            # 3. Model Inference & Temporal Smoothing
            current_pred, conf, probs = recognizer.predict_sequence()
            recognizer.update_sentence(current_pred, conf)

            # 4. Render Dashboard
            frame = recognizer.draw_dashboard(frame, current_pred, conf, probs, fps)

            # 5. Display output
            cv2.imshow("Real-Time Vietnamese Sign Language (VSL) Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                recognizer.sentence.clear()
                recognizer.prediction_history.clear()
                print("[INFO] Đã xóa câu văn bản dịch.")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    run_inference()
