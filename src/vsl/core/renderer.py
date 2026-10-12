"""
Visual rendering utilities for OpenCV frames:
- Glowing skeleton keypoint drawing
- Cross-platform UTF-8 Vietnamese text rendering using Pillow and local bundled fonts.
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vsl.config import settings


def draw_styled_landmarks(image: np.ndarray, results: dict) -> np.ndarray:
    """
    Draws modern glowing neon joints and bones onto OpenCV BGR image.
    """
    h, w, _ = image.shape

    # Draw Hand Landmarks
    hand_res = results.get("hand")
    if hand_res and hasattr(hand_res, "hand_landmarks") and len(hand_res.hand_landmarks) > 0:
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
    if pose_res and hasattr(pose_res, "pose_landmarks") and len(pose_res.pose_landmarks) > 0:
        pose_lms = pose_res.pose_landmarks[0]
        upper_body_indices = [11, 12, 13, 14, 15, 16]
        for idx in upper_body_indices:
            if idx < len(pose_lms):
                lm = pose_lms[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (px, py), 5, (255, 0, 128), -1)

    return image


def draw_vietnamese_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 24,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 6
) -> np.ndarray:
    """
    Renders UTF-8 Vietnamese text with full diacritics onto OpenCV BGR image using PIL.
    Uses bundled TrueType font across Linux, macOS, and Windows.
    """
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = settings.DEFAULT_FONT_PATH
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
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
