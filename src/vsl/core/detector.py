"""
Unified MediaPipe Holistic and Vision Tasks detector wrapper.
Provides clean lifecycle handling, frame preprocessing, and safe shutdown.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vsl.config import settings


class MediaPipeHolisticDetector:
    """
    Unified detector wrapper for Pose and Dual-Hand landmark detection.
    Compatible with MediaPipe Tasks API.
    """
    def __init__(self, pose_task_path: str = None, hand_task_path: str = None):
        self.pose_path = pose_task_path or settings.POSE_TASK_PATH
        self.hand_path = hand_task_path or settings.HAND_TASK_PATH

        if not os.path.exists(self.pose_path):
            raise FileNotFoundError(f"Pose landmarker task not found at: {self.pose_path}")
        if not os.path.exists(self.hand_path):
            raise FileNotFoundError(f"Hand landmarker task not found at: {self.hand_path}")

        # 1. Pose Landmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.pose_path),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # 2. Hand Landmarker (Dual-Hand support)
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.hand_path),
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    def process(self, frame_bgr: np.ndarray) -> dict:
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
        """Releases underlying detector resources."""
        if hasattr(self, "pose_detector") and self.pose_detector:
            self.pose_detector.close()
        if hasattr(self, "hand_detector") and self.hand_detector:
            self.hand_detector.close()
