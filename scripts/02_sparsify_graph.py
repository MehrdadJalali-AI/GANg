from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmof_galaxy_gan.graph.sparsify_similarity_graph import sparsify_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qmof_galaxy_1000.yaml")
    args = parser.parse_args()
    sparsify_graph(args.config)


if __name__ == "__main__":
    main()
