"""CLI shim for :mod:`training.positive_transfer`."""
from __future__ import annotations

import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.positive_transfer import main


if __name__ == "__main__":
    main()
