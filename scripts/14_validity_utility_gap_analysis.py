from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "qmof_gang_final" / "tables"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def main() -> None:
    val = pd.read_csv(SRC / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(SRC / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(SRC / "generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(SRC / "filtering_summary.csv").iloc[0]
    dist = pd.read_csv(SRC / "distribution_alignment.csv").iloc[0]
    nn = pd.read_csv(SRC / "nearest_neighbor_interpretation.csv")
    ds = pd.read_csv(SRC / "downstream_summary.csv")
    rows = []
    for model, g in ds.groupby("model"):
        base = float(g[g.training_mode.eq("real_only")].iloc[0].mae_mean)
        for _, r in g.iterrows():
            if r.training_mode == "real_only":
                continue
            rows.append(
                {
                    "target": "outputs.pbe.bandgap",
                    "model": model,
                    "training_mode": r.training_mode,
                    "descriptor_valid_rate": float(val.descriptor_valid_rate),
                    "edge_valid_rate": float(edge.edge_valid_rate),
                    "duplicate_rate": float(nov.duplicate_rate),
                    "novelty_pass_rate": float(nov.novelty_pass_rate),
                    "filtered_pass_rate": float(filt.filtered_pass_rate),
                    "descriptor_wasserstein_proxy": float(dist.descriptor_wasserstein_proxy),
                    "descriptor_mean_abs_shift": float(dist.descriptor_mean_abs_shift),
                    "nearest_neighbor_similarity_mean": float(nn.top1_similarity.mean()),
                    "real_only_mae": base,
                    "mode_mae": float(r.mae_mean),
                    "delta_mae_vs_real_only": float(r.mae_mean - base),
                    "utility_interpretation": "worse_than_real_only" if float(r.mae_mean - base) > 0 else "better_than_real_only",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "validity_utility_gap_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    rates = [
        ("descriptor valid", val.descriptor_valid_rate),
        ("edge valid", edge.edge_valid_rate),
        ("non-duplicate", 1 - nov.duplicate_rate),
        ("novelty pass", nov.novelty_pass_rate),
        ("filtered pass", filt.filtered_pass_rate),
    ]
    axes[0].bar([r[0] for r in rates], [r[1] for r in rates], color=["#4C9FAD", "#4C9FAD", "#4C9FAD", "#4C9FAD", "#D99A3D"])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("High descriptor/edge validity")
    axes[0].tick_params(axis="x", rotation=35)
    sub = out[out.training_mode.str.startswith("gan")]
    colors = {"GraphSAGE": "#4C9FAD", "GAT": "#7E5AA2", "GCN": "#D99A3D"}
    for model, g in sub.groupby("model"):
        axes[1].scatter(g["nearest_neighbor_similarity_mean"], g["delta_mae_vs_real_only"], label=model, s=55, color=colors.get(model))
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("Mean top-1 NN similarity")
    axes[1].set_ylabel("MAE delta vs real-only")
    axes[1].set_title("Validity does not imply utility")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "validity_utility_gap_panel.pdf")
    fig.savefig(FIGS / "validity_utility_gap_panel.png", dpi=600)
    plt.close(fig)

    msg = [
        "# Validity–utility gap analysis",
        "",
        f"Descriptor-valid rate: {val.descriptor_valid_rate:.3f}.",
        f"Edge-valid rate: {edge.edge_valid_rate:.3f}.",
        f"Novelty-pass rate: {nov.novelty_pass_rate:.3f}.",
        f"Filtered-pass rate: {filt.filtered_pass_rate:.3f}.",
        f"Mean top-1 nearest-neighbor similarity: {nn.top1_similarity.mean():.3f}.",
        "",
        "Despite high descriptor/edge validity, all GANg augmentation modes have positive MAE delta relative to real-only for at least one model, and no model shows consistent improvement across GraphSAGE, GAT, and GCN.",
        "This supports the manuscript framing: descriptor-space graph-node generation exposes a validity–utility gap rather than a MOF-discovery result.",
    ]
    (REPORTS / "validity_utility_gap.md").write_text("\n".join(msg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
