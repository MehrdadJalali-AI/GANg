"""Validity–utility gap analysis (npj script entry point)."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "14_validity_utility_gap_analysis.py"), run_name="__main__")
