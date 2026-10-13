"""
Legacy entrypoint for Real-time OpenCV HUD inference.
Redirects to the modernized modular CLI in src/vsl/cli/hud.py.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vsl.cli.hud import main

if __name__ == "__main__":
    main()
