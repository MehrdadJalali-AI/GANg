from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-qmof-galaxy")
os.environ.setdefault("MPLBACKEND", "Agg")

from qmof_galaxy_gan.plotting.plot_claim_boundary import plot_claim_boundary
from qmof_galaxy_gan.plotting.plot_distribution_alignment import plot_distribution_alignment
from qmof_galaxy_gan.plotting.plot_downstream_results import plot_downstream_results
from qmof_galaxy_gan.plotting.plot_embedding_space import plot_embedding_space
from qmof_galaxy_gan.plotting.plot_similarity_graph import plot_similarity_graph
from qmof_galaxy_gan.plotting.plot_training_curves import plot_training_curves
from qmof_galaxy_gan.plotting.plot_workflow import plot_workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qmof_galaxy_1000.yaml")
    args = parser.parse_args()
    for fn in [plot_workflow, plot_similarity_graph, plot_training_curves, plot_distribution_alignment,
               plot_embedding_space, plot_downstream_results, plot_claim_boundary]:
        fn(args.config)


if __name__ == "__main__":
    main()
