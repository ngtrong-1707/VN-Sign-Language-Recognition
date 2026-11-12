"""
Lightweight, torch-free inference core for VSL Vision.

Shared by:
- the full FastAPI server (`vsl.api.server`) for REST prediction / timeline helpers, and
- the serverless Vercel deployment (`vsl.serverless.app`) which only ships
  numpy + onnxruntime (no PyTorch / OpenCV / MediaPipe on the server).

On Vercel, MediaPipe runs inside the browser (WASM) and only the 258-d landmark
vectors are sent to the API, so the function bundle stays small.
"""

import json
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "models" / "checkpoints"
ONNX_MODEL_PATH = MODEL_DIR / "vsl_model.onnx"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
DEMO_SEQUENCES_PATH = ROOT_DIR / "data" / "demo" / "demo_sequences.json"
RAW_SEQUENCES_DIR = ROOT_DIR / "data" / "sequences"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 258
POSE_DIM = 33 * 4
PREDICTION_THRESHOLD = 0.75
MAX_TIMELINE_FRAMES = 1800  # ~90s @ 20 FPS

DEMO_TIMELINE_ACTIONS = ["Xin chao", "Toi", "Yeu", "Viet Nam", "Cam on"]

PredictionTuple = Tuple[str, str, str, float, List[Dict[str, Any]], float]


@lru_cache(maxsize=1)
def load_actions_metadata() -> List[Dict[str, Any]]:
    """Loads the 20-sign vocabulary metadata (same order as the model's output classes)."""
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class LiteInferenceEngine:
    """ONNX Runtime-only inference engine (no PyTorch dependency)."""

    def __init__(self, model_path: Path = ONNX_MODEL_PATH):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        meta = load_actions_metadata()
        self.actions = [m["name"] for m in meta]
        self.display_map = {m["name"]: m["display"] for m in meta}
        self.icon_map = {m["name"]: m["icon"] for m in meta}

    def predict(self, sequence_window: np.ndarray) -> PredictionTuple:
        start_t = time.perf_counter()
        batch = sequence_window.astype(np.float32)
        if batch.ndim == 2:
            batch = np.expand_dims(batch, axis=0)

        logits = self.session.run([self.output_name], {self.input_name: batch})[0]
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = (exp / np.sum(exp, axis=1, keepdims=True))[0]
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        best = int(np.argmax(probs))
        name = self.actions[best]
        top5 = [
            {
                "name": self.actions[i],
                "display": self.display_map.get(self.actions[i], self.actions[i]),
                "icon": self.icon_map.get(self.actions[i], "✨"),
                "probability": float(probs[i]),
            }
            for i in np.argsort(probs)[::-1][:5]
        ]
        return (
            name,
            self.display_map.get(name, name),
            self.icon_map.get(name, "👋"),
            float(probs[best]),
            top5,
            latency_ms,
        )


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def hands_present(keypoints: np.ndarray) -> bool:
    """True when at least one hand has been detected in the 258-d frame vector."""
    return bool(np.any(keypoints[POSE_DIM:] != 0))


def extract_2d_overlay(keypoints: np.ndarray) -> Dict[str, Any]:
    """
    Extracts normalized 2D coordinates (x, y) from the 258-d feature vector
    for real-time browser canvas skeleton rendering.
    """
    hands = []
    for label, start in (("Left", POSE_DIM), ("Right", POSE_DIM + 63)):
        pts = keypoints[start:start + 63].reshape(21, 3)
        if np.any(pts != 0):
            hands.append({
                "label": label,
                "points": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in pts],
            })

    pose_pts = []
    pose = keypoints[:POSE_DIM].reshape(33, 4)
    if np.any(pose != 0):
        # Upper body: shoulders, elbows, wrists
        for idx in (11, 12, 13, 14, 15, 16):
            p = pose[idx]
            if p[0] > 0 or p[1] > 0:
                pose_pts.append([round(float(p[0]), 4), round(float(p[1]), 4)])

    return {"hands": hands, "pose": pose_pts}


def to_window(frames: Sequence[Sequence[float]]) -> np.ndarray:
    """Validates and converts a JSON frame list into a (30, 258) float32 window."""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        raise ValueError(f"Mỗi frame phải có đúng {FEATURE_DIM} giá trị (nhận được shape {arr.shape}).")
    if arr.shape[0] < SEQUENCE_LENGTH:
        raise ValueError(f"Cần tối thiểu {SEQUENCE_LENGTH} frames (nhận được {arr.shape[0]}).")
    return arr[-SEQUENCE_LENGTH:]


def prediction_payload(pred: PredictionTuple, threshold: float = PREDICTION_THRESHOLD) -> Dict[str, Any]:
    name, display, icon, conf, top5, latency_ms = pred
    return {
        "type": "prediction",
        "name": name,
        "display": display,
        "icon": icon,
        "confidence": conf,
        "is_confident": conf >= threshold,
        "top5": top5,
        "latency_ms": round(latency_ms, 2),
        "buffer_size": SEQUENCE_LENGTH,
    }


def _fmt_timestamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def build_timeline(
    frames: Iterable[np.ndarray],
    fps: float,
    engine: Any,
    threshold: float = PREDICTION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Converts a chronological list of 258-d frame vectors into a sign timeline.
    - Frames without hands are skipped; a pause of >= 0.5s resets the buffer so
      the same sign can be recognised again later.
    - Inference runs roughly 6 times per second on a 30-frame sliding window.
    """
    fps = float(fps) if fps and fps > 0 else 30.0
    stride = max(1, int(round(fps / 6)))
    idle_reset = max(3, int(round(fps * 0.5)))

    buffer: deque = deque(maxlen=SEQUENCE_LENGTH)
    timeline: List[Dict[str, Any]] = []
    last_detected: Optional[str] = None
    idle = 0
    total = 0

    for idx, vec in enumerate(frames, start=1):
        total = idx
        vec = np.asarray(vec, dtype=np.float32)
        if not hands_present(vec):
            idle += 1
            if idle >= idle_reset:
                buffer.clear()
                last_detected = None
            continue

        idle = 0
        buffer.append(vec)
        if len(buffer) == SEQUENCE_LENGTH and idx % stride == 0:
            name, display, icon, conf, _top5, _lat = engine.predict(np.stack(buffer))
            if conf >= threshold and name != last_detected:
                sec = round(idx / fps, 2)
                timeline.append({
                    "timestamp": _fmt_timestamp(sec),
                    "second": sec,
                    "name": name,
                    "display": display,
                    "icon": icon,
                    "confidence": round(conf * 100, 1),
                })
                last_detected = name

    sentence = " ".join(item["display"] for item in timeline)
    return {
        "status": "success",
        "total_frames": total,
        "duration_sec": round(total / fps, 2),
        "sentence": sentence or "Không phát hiện cử chỉ VSL rõ ràng",
        "timeline": timeline,
    }


# ---------------------------------------------------------------------------
# Demo sequences (bundled compact JSON, falls back to raw .npy dataset)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_demo_bundle() -> Dict[str, List[List[float]]]:
    if DEMO_SEQUENCES_PATH.exists():
        with open(DEMO_SEQUENCES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_demo_sequence(action_name: str, sample_idx: int = 0) -> Optional[np.ndarray]:
    """Returns a (30, 258) landmark sequence for the requested sign, or None."""
    if sample_idx == 0:
        bundle = _load_demo_bundle()
        if action_name in bundle:
            return np.asarray(bundle[action_name], dtype=np.float32)

    seq_dir = RAW_SEQUENCES_DIR / action_name.replace(" ", "_") / str(sample_idx)
    files = [seq_dir / f"{i}.npy" for i in range(SEQUENCE_LENGTH)]
    if not all(p.exists() for p in files):
        return None
    return np.stack([np.load(str(p)).astype(np.float32) for p in files])


def demo_sequence_payload(action_name: str, engine: Any, sample_idx: int = 0) -> Optional[Dict[str, Any]]:
    seq = load_demo_sequence(action_name, sample_idx)
    if seq is None:
        return None
    return {
        "status": "success",
        "action": action_name,
        "frames": np.round(seq, 5).tolist(),
        "overlays": [extract_2d_overlay(v) for v in seq],
        "prediction": prediction_payload(engine.predict(seq)),
    }


def run_demo_timeline(engine: Any, fps: float = 30.0) -> Dict[str, Any]:
    """Runs the 5-sign demo sentence through the real ONNX model."""
    timeline = []
    total_frames = 0
    for act_name in DEMO_TIMELINE_ACTIONS:
        seq = load_demo_sequence(act_name)
        if seq is None:
            continue
        total_frames += SEQUENCE_LENGTH
        name, display, icon, conf, _top5, latency_ms = engine.predict(seq)
        sec = round(total_frames / fps, 2)
        timeline.append({
            "timestamp": _fmt_timestamp(sec),
            "second": sec,
            "name": name,
            "display": display,
            "icon": icon,
            "confidence": round(conf * 100, 1),
            "latency_ms": round(latency_ms, 2),
        })

    return {
        "status": "success",
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2),
        "sentence": " ".join(item["display"] for item in timeline),
        "timeline": timeline,
    }

