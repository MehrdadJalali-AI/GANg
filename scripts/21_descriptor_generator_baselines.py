"""Summarize descriptor-generator and augmentation controls from locked tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "qmof_gang_final" / "tables"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures" / "npj"
REPORTS = ROOT / "reports"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)

MODES = [
    "real_only",
    "random_synthetic",
    "feature_noise",
    "nearest_neighbor_interpolation",
    "gan_generated",
    "gan_filtered",
    "gan_property_head",
    "gan_conditional_target",
]


def main() -> None:
    val = pd.read_csv(SRC / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(SRC / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(SRC / "generated_node_novelty.csv").iloc[0]
    ds = pd.read_csv(SRC / "downstream_summary.csv")

    validity = pd.DataFrame(
        [
            {
                "generator": "GANg WGAN-GP",
                "descriptor_valid_rate": val.descriptor_valid_rate,
                "edge_valid_rate": edge.edge_valid_rate,
                "duplicate_rate": nov.duplicate_rate,
                "novelty_pass_rate": nov.novelty_pass_rate,
                "note": "locked GANg export",
            },
            {
                "generator": "Gaussian noise / random synthetic / NN interpolation",
                "descriptor_valid_rate": None,
                "edge_valid_rate": None,
                "duplicate_rate": None,
                "novelty_pass_rate": None,
                "note": "control modes evaluated downstream; not full generator validity rerun",
            },
        ]
    )
    validity.to_csv(TABLES / "generator_baseline_validity.csv", index=False)

    rows = []
    for model in ["GraphSAGE", "GAT", "GCN"]:
        base = float(ds[(ds.model == model) & (ds.training_mode == "real_only")].iloc[0].mae_mean)
        for mode in MODES:
            hit = ds[(ds.model == model) & (ds.training_mode == mode)]
            if len(hit) == 0:
                continue
            mae = float(hit.iloc[0].mae_mean)
            rows.append(
                {
                    "model": model,
                    "training_mode": mode,
                    "mae_mean": mae,
                    "delta_mae_vs_real_only": mae - base,
                    "beats_real_only": mae < base,
                }
            )
    downstream = pd.DataFrame(rows)
    downstream.to_csv(TABLES / "generator_baseline_downstream.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
    sub = downstream[downstream.model == "GraphSAGE"]
    ax.bar(sub.training_mode, sub.delta_mae_vs_real_only, color="#7E5AA2")
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("MAE delta vs real-only")
    ax.set_title("GraphSAGE augmentation controls (positive = worse)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGS / "generator_baseline_comparison.pdf")
    plt.close(fig)

    (REPORTS / "generator_baseline_report.md").write_text(
        "# Generator baseline report\n\n"
        "GANg validity metrics are from locked exports. Other generators (GMM/VAE/KDE) "
        "require a full rerun and are not fabricated here. Downstream control modes include "
        "random synthetic nodes, feature noise, nearest-neighbor interpolation, and GANg variants.\n\n"
        "DropEdge-style training-time augmentation is **not** a descriptor generator; treat it separately if implemented.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
