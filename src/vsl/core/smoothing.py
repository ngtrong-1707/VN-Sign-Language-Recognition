"""
Temporal prediction smoothing and debouncing filters.
Prevents flickering and guarantees stable sentence translation in real-time inference.
"""

from collections import deque, Counter
from typing import Optional, List, Tuple
from vsl.config import settings


class TemporalPredictionSmoother:
    """
    Sliding window consensus filter for noisy temporal predictions.
    Uses majority voting and temporal debouncing.
    """
    def __init__(
        self,
        window_size: int = settings.HISTORY_SMOOTHING_LEN,
        threshold: float = settings.PREDICTION_THRESHOLD,
        agreement_ratio: float = 0.75,
        max_sentence_length: int = 10
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.agreement_threshold = int(window_size * agreement_ratio)
        self.max_sentence_length = max_sentence_length

        self.history = deque(maxlen=window_size)
        self.sentence: List[str] = []
        self.last_added_action: Optional[str] = None

    def update(self, action_name: Optional[str], confidence: float) -> Tuple[bool, List[str]]:
        """
        Updates the smoother with a newly predicted frame.
        Returns:
            (has_new_word, current_sentence)
        """
        if action_name is None or confidence < self.threshold:
            return False, self.sentence

        self.history.append(action_name)

        # Require minimum threshold agreement samples
        if len(self.history) < self.agreement_threshold:
            return False, self.sentence

        counts = Counter(self.history)
        most_common, freq = counts.most_common(1)[0]

        # Check if enough recent frames agreed
        if freq >= self.agreement_threshold:
            if most_common != self.last_added_action:
                self.sentence.append(most_common)
                self.last_added_action = most_common
                if len(self.sentence) > self.max_sentence_length:
                    self.sentence = self.sentence[-self.max_sentence_length:]
                return True, self.sentence

        return False, self.sentence

    def clear(self):
        """Clears both sentence buffer and prediction history."""
        self.history.clear()
        self.sentence.clear()
        self.last_added_action = None

    def get_sentence_text(self) -> str:
        return " ".join(self.sentence)
