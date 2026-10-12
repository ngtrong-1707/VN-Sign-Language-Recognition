"""
Unit tests for Temporal Prediction Smoother.
"""

from vsl.core.smoothing import TemporalPredictionSmoother


def test_smoother_no_threshold():
    smoother = TemporalPredictionSmoother(window_size=4, threshold=0.8, agreement_ratio=0.75)

    # Low confidence should not trigger
    added, sentence = smoother.update("Xin chao", 0.5)
    assert not added
    assert len(sentence) == 0


def test_smoother_majority_vote():
    smoother = TemporalPredictionSmoother(window_size=4, threshold=0.8, agreement_ratio=0.75)

    # 3 consecutive high-confidence predictions of the same sign
    smoother.update("Xin chao", 0.9)
    smoother.update("Xin chao", 0.9)
    added, sentence = smoother.update("Xin chao", 0.9)

    # 3 out of 4 agree -> should append word
    assert added
    assert sentence == ["Xin chao"]

    # Same word again immediately should debounce (not repeat)
    added2, sentence2 = smoother.update("Xin chao", 0.95)
    assert not added2
    assert sentence2 == ["Xin chao"]
