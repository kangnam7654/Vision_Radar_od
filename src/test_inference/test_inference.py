"""Legacy batch-inference script over ``src/test_inference/test_data`` (not a unit test).

Runs the current pipeline (``src/inference.py``) with the committed checkpoint
``runs/detect/train4/weights/best.pt`` and writes to the git-ignored
``outputs/test_inference/`` folder. The committed ``test_results/`` were produced by an
earlier model with different classes and are intentionally left untouched; results from
this script are not expected to match them.

Usage (from the repository root)::

    uv run python src/test_inference/test_inference.py
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import inference  # noqa: E402


def main(argv=None):
    defaults = [
        "--source", "src/test_inference/test_data",
        "--circle-json", "src/test_inference/test_circle_coordinates_old.json",
        "--output-dir", "outputs/test_inference",
    ]
    return inference.main(defaults + list(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
