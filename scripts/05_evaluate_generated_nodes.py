from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from qmof_galaxy_gan.evaluation.distribution_alignment import evaluate_distribution_alignment
from qmof_galaxy_gan.evaluation.graph_topology import evaluate_topology
from qmof_galaxy_gan.evaluation.nearest_neighbor_interpretation import nearest_neighbor_interpretation
from qmof_galaxy_gan.evaluation.novelty import evaluate_novelty
from qmof_galaxy_gan.evaluation.validity import evaluate_validity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qmof_galaxy_1000.yaml")
    args = parser.parse_args()
    evaluate_validity(args.config)
    evaluate_novelty(args.config)
    evaluate_distribution_alignment(args.config)
    evaluate_topology(args.config)
    nearest_neighbor_interpretation(args.config)


if __name__ == "__main__":
    main()
