"""
Unit tests for VSL Inference Engine and Model Architectures.
"""

import numpy as np
import pytest
import torch
from vsl.config import settings, ACTIONS
from vsl.models.architectures import VSL_BiGRU
from vsl.models.engine import VSLInferenceEngine


def test_bigru_forward_shape():
    """Verify PyTorch model forward pass produces valid logits."""
    model = VSL_BiGRU(
        input_dim=settings.INPUT_FEATURE_DIM,
        hidden_dim=32,
        num_layers=1,
        num_classes=len(ACTIONS),
        dropout=0.1
    )
    model.eval()

    dummy_input = torch.randn(2, settings.SEQUENCE_LENGTH, settings.INPUT_FEATURE_DIM)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, len(ACTIONS))


def test_inference_engine_prediction():
    """Verify ONNX/PyTorch InferenceEngine produces consistent output structure."""
    engine = VSLInferenceEngine(use_onnx=True)
    dummy_sequence = np.random.randn(settings.SEQUENCE_LENGTH, settings.INPUT_FEATURE_DIM).astype(np.float32)

    pred_name, display_name, icon, conf, top5, latency_ms = engine.predict(dummy_sequence)

    assert isinstance(pred_name, str)
    assert pred_name in ACTIONS
    assert isinstance(display_name, str)
    assert isinstance(icon, str)
    assert 0.0 <= conf <= 1.0
    assert len(top5) == 5
    assert latency_ms > 0.0
