"""Claim-safety audit for manuscript sources."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "17_claim_safety_audit.py"), run_name="__main__")
