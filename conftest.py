"""
Root conftest.py — ensures the repo root is on sys.path so `from src.common
import ...` style imports resolve consistently whether pytest is invoked as
`pytest tests/`, `python -m pytest`, from a different cwd, or by an IDE's
test runner. Without this, import behavior can differ between a local dev
machine and CI (different pytest import modes / invocation styles).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
