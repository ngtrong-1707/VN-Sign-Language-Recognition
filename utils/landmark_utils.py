"""
Landmark extraction, normalization, drawing, and UI rendering utilities using MediaPipe Tasks and Pillow.
Compatible with MediaPipe 1.0+ on Python 3.12/3.14.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import config

# Paths to MediaPipe model assets
POSE_TASK_PATH = os.path.join(config.MODEL_DIR, "pose_landmarker.task")
HAND_TASK_PATH = os.path.join(config.MODEL_DIR, "hand_landmarker.task")


class MediaPipeHolisticDetector:
    """
    Unified detector wrapper for Pose and Dual-Hand landmark detection.
    """
    def __init__(self):
        # 1. Pose Landmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_TASK_PATH),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # 2. Hand Landmarker (up to 2 hands)
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_TASK_PATH),
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    def process(self, frame_bgr):
        """
        Processes BGR OpenCV frame and returns detected landmarks.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        pose_result = self.pose_detector.detect(mp_image)
        hand_result = self.hand_detector.detect(mp_image)

        return {
            "pose": pose_result,
            "hand": hand_result
        }

    def close(self):
        self.pose_detector.close()
        self.hand_detector.close()


def extract_landmarks(results):
    """
    Extracts structured 1D vector (258 features):
    - Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
    - Left Hand: 21 landmarks * 3 (x, y, z) = 63
    - Right Hand: 21 landmarks * 3 (x, y, z) = 63
    Total: 258 features per frame.
    """
    # 1. Extract Pose (33 points)
    pose_res = results.get("pose")
    if pose_res and len(pose_res.pose_landmarks) > 0:
        landmarks = pose_res.pose_landmarks[0]
        pose_arr = []
        for lm in landmarks:
            vis = getattr(lm, 'visibility', 0.99)
            if vis is None: vis = 0.99
            pose_arr.extend([lm.x, lm.y, lm.z, vis])
        pose = np.array(pose_arr, dtype=np.float32)
        if len(pose) != 132:
            pose = np.pad(pose[:132], (0, max(0, 132 - len(pose))))
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # 2. Extract Left & Right Hands (21 points each)
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)

    hand_res = results.get("hand")
    if hand_res and len(hand_res.hand_landmarks) > 0:
        for idx, hand_lms in enumerate(hand_res.hand_landmarks):
            # Check handedness label if available (Left/Right)
            handedness = "Right"
            if len(hand_res.handedness) > idx and len(hand_res.handedness[idx]) > 0:
                handedness = hand_res.handedness[idx][0].category_name

            h_arr = []
            for lm in hand_lms:
                h_arr.extend([lm.x, lm.y, lm.z])
            h_vec = np.array(h_arr, dtype=np.float32)

            if handedness == "Left":
                lh = h_vec
            else:
                rh = h_vec

    return np.concatenate([pose, lh, rh])


def draw_styled_landmarks(image, results):
    """
    Draws custom glowing joints and bones onto OpenCV BGR image.
    """
    h, w, _ = image.shape

    # Draw Hand Landmarks
    hand_res = results.get("hand")
    if hand_res and len(hand_res.hand_landmarks) > 0:
        for hand_lms in hand_res.hand_landmarks:
            pts = []
            for lm in hand_lms:
                px, py = int(lm.x * w), int(lm.y * h)
                pts.append((px, py))
                cv2.circle(image, (px, py), 4, (0, 240, 255), -1)
                cv2.circle(image, (px, py), 6, (168, 85, 247), 1)

            # Draw finger connection lines
            finger_joints = [
                [0, 1, 2, 3, 4],       # Thumb
                [0, 5, 6, 7, 8],       # Index
                [0, 9, 10, 11, 12],    # Middle
                [0, 13, 14, 15, 16],   # Ring
                [0, 17, 18, 19, 20]    # Pinky
            ]
            for chain in finger_joints:
                for i in range(len(chain) - 1):
                    p1, p2 = pts[chain[i]], pts[chain[i+1]]
                    cv2.line(image, p1, p2, (0, 200, 255), 2)

    # Draw Pose Keypoints (Upper body: shoulders, elbows, wrists)
    pose_res = results.get("pose")
    if pose_res and len(pose_res.pose_landmarks) > 0:
        pose_lms = pose_res.pose_landmarks[0]
        upper_body_indices = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        for idx in upper_body_indices:
            if idx < len(pose_lms):
                lm = pose_lms[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (px, py), 5, (255, 0, 128), -1)


def draw_vietnamese_text(
    image,
    text,
    position,
    font_size=26,
    font_color=(255, 255, 255),
    bg_color=None,
    padding=6
):
    """
    Renders UTF-8 Vietnamese text with full diacritics onto OpenCV BGR image using PIL.
    """
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/segoeui.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox(position, text, font=font)
    if bg_color is not None:
        expanded_bbox = (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        )
        draw.rectangle(expanded_bbox, fill=bg_color)

    draw.text(position, text, font=font, fill=font_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
