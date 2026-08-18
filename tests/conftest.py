"""Pytest configuration for test suite import path resolution."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "scripts", ROOT / "src"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
