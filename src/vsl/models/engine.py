"""
Unified Inference Engine for VSL Models.
Supports both ONNX Runtime (Ultra-low latency CPU execution) and native PyTorch.
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from vsl.config import settings, ACTIONS, ACTION_DISPLAY_MAP, ACTION_ICON_MAP
from vsl.models.architectures import VSL_BiGRU

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class VSLInferenceEngine:
    """
    High-performance inference engine for sequential gesture classification.
    """
    def __init__(self, use_onnx: bool = True):
        self.device = settings.DEVICE
        self.actions = ACTIONS
        self.display_map = ACTION_DISPLAY_MAP
        self.icon_map = ACTION_ICON_MAP
        self.seq_len = settings.SEQUENCE_LENGTH
        self.use_onnx = use_onnx and HAS_ONNX and os.path.exists(settings.ONNX_MODEL_PATH)

        self.ort_session = None
        self.pytorch_model = None

        if self.use_onnx:
            print(f"[ENGINE] Nạp mô hình ONNX siêu tốc từ: {settings.ONNX_MODEL_PATH}")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2
            self.ort_session = ort.InferenceSession(
                settings.ONNX_MODEL_PATH,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
        else:
            print("[ENGINE] Sử dụng PyTorch Native Inference Engine.")
            self.pytorch_model = VSL_BiGRU(
                input_dim=settings.INPUT_FEATURE_DIM,
                hidden_dim=settings.HIDDEN_DIM,
                num_layers=settings.NUM_LAYERS,
                num_classes=len(self.actions),
                dropout=settings.DROPOUT
            ).to(self.device)

            if os.path.exists(settings.BEST_MODEL_PATH):
                print(f"[ENGINE] Nạp checkpoint PyTorch từ: {settings.BEST_MODEL_PATH}")
                self.pytorch_model.load_state_dict(
                    torch.load(settings.BEST_MODEL_PATH, map_location=self.device, weights_only=True)
                )
            else:
                print("[ENGINE] Checkpoint PyTorch chưa tồn tại. Khởi tạo trọng số mặc định.")

            self.pytorch_model.eval()

    def predict(self, sequence_window: np.ndarray) -> Tuple[str, str, str, float, List[Dict[str, Any]], float]:
        """
        Input: sequence_window: np.ndarray of shape (30, 258)
        Returns:
            (predicted_name, display_name, icon, confidence, top5_list, latency_ms)
        """
        start_t = time.perf_counter()

        # Ensure correct shape (1, 30, 258)
        if sequence_window.ndim == 2:
            input_batch = np.expand_dims(sequence_window, axis=0).astype(np.float32)
        else:
            input_batch = sequence_window.astype(np.float32)

        if self.use_onnx and self.ort_session is not None:
            raw_logits = self.ort_session.run([self.output_name], {self.input_name: input_batch})[0]
            # Softmax calculation
            exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
            probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]
        else:
            tensor_in = torch.tensor(input_batch, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.pytorch_model(tensor_in)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        predicted_name = self.actions[best_idx]
        display_name = self.display_map.get(predicted_name, predicted_name)
        icon = self.icon_map.get(predicted_name, "👋")

        # Top 5 actions
        top5_indices = np.argsort(probs)[::-1][:5]
        top5 = []
        for idx in top5_indices:
            act_name = self.actions[idx]
            top5.append({
                "name": act_name,
                "display": self.display_map.get(act_name, act_name),
                "icon": self.icon_map.get(act_name, "✨"),
                "probability": float(probs[idx])
            })

        return predicted_name, display_name, icon, confidence, top5, latency_ms

    @classmethod
    def export_pytorch_to_onnx(cls, model_path: str = None, onnx_path: str = None) -> str:
        """Utility method to export a trained PyTorch model to ONNX."""
        model_path = model_path or settings.BEST_MODEL_PATH
        onnx_path = onnx_path or settings.ONNX_MODEL_PATH

        model = VSL_BiGRU(
            input_dim=settings.INPUT_FEATURE_DIM,
            hidden_dim=settings.HIDDEN_DIM,
            num_layers=settings.NUM_LAYERS,
            num_classes=len(ACTIONS),
            dropout=settings.DROPOUT
        )

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

        dummy_input = torch.randn(1, settings.SEQUENCE_LENGTH, settings.INPUT_FEATURE_DIM)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input_sequence"],
            output_names=["logits"],
            dynamic_axes={"input_sequence": {0: "batch_size"}, "logits": {0: "batch_size"}}
        )
        print(f"[ONNX] Xuất thành công mô hình tại: {onnx_path}")
        return onnx_path
