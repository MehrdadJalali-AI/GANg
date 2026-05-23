from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from qmof_galaxy_gan.generation.generate_nodes import generate_nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qmof_galaxy_1000.yaml")
    args = parser.parse_args()
    generate_nodes(args.config)


if __name__ == "__main__":
    main()
