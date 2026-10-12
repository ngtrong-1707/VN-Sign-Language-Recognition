"""
Landmark feature extraction and normalization for VSL sequences.
Constructs a deterministic 258-dimensional vector per video frame:
- Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
- Left Hand: 21 landmarks * 3 (x, y, z) = 63
- Right Hand: 21 landmarks * 3 (x, y, z) = 63
Total: 258 features per frame.
"""

from typing import Dict, Any, Optional
import numpy as np
from vsl.config import settings


def extract_landmarks(results: Dict[str, Any]) -> np.ndarray:
    """
    Extracts structured 1D vector (258 features) from MediaPipe detection results.
    Guarantees fixed length vector even when body parts are occluded or missing.
    """
    # 1. Pose landmarks (33 points * 4 values = 132)
    pose_res = results.get("pose")
    if pose_res and hasattr(pose_res, "pose_landmarks") and len(pose_res.pose_landmarks) > 0:
        landmarks = pose_res.pose_landmarks[0]
        pose_arr = []
        for lm in landmarks:
            vis = getattr(lm, "visibility", 0.99)
            if vis is None:
                vis = 0.99
            pose_arr.extend([float(lm.x), float(lm.y), float(lm.z), float(vis)])
        pose = np.array(pose_arr, dtype=np.float32)
        if len(pose) != 132:
            pose = np.pad(pose[:132], (0, max(0, 132 - len(pose))))
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # 2. Hand landmarks (Left and Right: 21 points * 3 values = 63 each)
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)

    hand_res = results.get("hand")
    if hand_res and hasattr(hand_res, "hand_landmarks") and len(hand_res.hand_landmarks) > 0:
        for idx, hand_lms in enumerate(hand_res.hand_landmarks):
            handedness = "Right"
            if hasattr(hand_res, "handedness") and len(hand_res.handedness) > idx and len(hand_res.handedness[idx]) > 0:
                handedness = hand_res.handedness[idx][0].category_name

            h_arr = []
            for lm in hand_lms:
                h_arr.extend([float(lm.x), float(lm.y), float(lm.z)])
            h_vec = np.array(h_arr, dtype=np.float32)

            if handedness == "Left":
                lh = h_vec
            else:
                rh = h_vec

    output_vector = np.concatenate([pose, lh, rh])
    assert len(output_vector) == settings.INPUT_FEATURE_DIM, f"Expected {settings.INPUT_FEATURE_DIM}, got {len(output_vector)}"
    return output_vector
