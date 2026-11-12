"""
Builds the compact demo bundle `data/demo/demo_sequences.json` (sample #0 of every sign)
from the raw dataset in `data/sequences/`. The bundle powers the in-browser
"AI simulation" and the demo timeline, including on Vercel where the 24k raw .npy
files are not deployed.

Usage:
    python scripts/build_demo_bundle.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vsl.serverless.core import (  # noqa: E402
    DEMO_SEQUENCES_PATH,
    RAW_SEQUENCES_DIR,
    SEQUENCE_LENGTH,
    load_actions_metadata,
)


def main() -> None:
    bundle = {}
    for item in load_actions_metadata():
        name = item["name"]
        seq_dir = RAW_SEQUENCES_DIR / name.replace(" ", "_") / "0"
        frames = [np.load(str(seq_dir / f"{i}.npy")).astype(np.float32) for i in range(SEQUENCE_LENGTH)]
        bundle[name] = np.round(np.stack(frames), 5).tolist()

    DEMO_SEQUENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_SEQUENCES_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[OK] {len(bundle)} sequences -> {DEMO_SEQUENCES_PATH} ({DEMO_SEQUENCES_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

