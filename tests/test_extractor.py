"""
Unit tests for Landmark Extractor and Normalizer.
"""

import numpy as np
import pytest
from vsl.core.extractor import extract_landmarks
from vsl.config import settings


def test_extract_landmarks_empty_input():
    """Verify that empty/none input yields an exact zero vector of length 258."""
    empty_results = {"pose": None, "hand": None}
    vector = extract_landmarks(empty_results)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (settings.INPUT_FEATURE_DIM,)
    assert np.all(vector == 0.0)


def test_extract_landmarks_with_mock_data():
    """Verify that structured objects are parsed correctly."""
    class MockLandmark:
        def __init__(self, x, y, z, vis=0.99):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = vis

    class MockPoseResult:
        def __init__(self):
            self.pose_landmarks = [[MockLandmark(0.5, 0.5, 0.0, 0.95) for _ in range(33)]]

    class MockCategory:
        def __init__(self, name):
            self.category_name = name

    class MockHandResult:
        def __init__(self):
            self.hand_landmarks = [
                [MockLandmark(0.1, 0.2, 0.3) for _ in range(21)],  # Left Hand
                [MockLandmark(0.7, 0.8, 0.9) for _ in range(21)]   # Right Hand
            ]
            self.handedness = [[MockCategory("Left")], [MockCategory("Right")]]

    results = {
        "pose": MockPoseResult(),
        "hand": MockHandResult()
    }

    vector = extract_landmarks(results)
    assert vector.shape == (258,)
    # Pose first x should be 0.5
    assert np.isclose(vector[0], 0.5)
    # Left hand first x at index 132
    assert np.isclose(vector[132], 0.1)
    # Right hand first x at index 132 + 63 = 195
    assert np.isclose(vector[195], 0.7)
