from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "qmof_gang_final" / "tables"
OLD_FIGS = ROOT / "experiments" / "qmof_gang_final" / "figures"
FIGS = ROOT / "figures" / "main"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TEAL = "#4C9FAD"
BLUE = "#8DB6D9"
PURPLE = "#7E5AA2"
ORANGE = "#D99A3D"
GRAY = "#555555"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig1() -> None:
    import shutil
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module

    import_module("13_replace_fig1_from_attached").draw()
    for ext in ["pdf", "png"]:
        shutil.copyfile(OLD_FIGS / f"GANg_workflow_nature_style.{ext}", FIGS / f"Figure_1_workflow.{ext}")


def fig2() -> None:
    before = pd.read_csv(SRC / "graph_summary_before_sparsification.csv").iloc[0]
    after = pd.read_csv(SRC / "final_sparsification_summary.csv").iloc[0]
    grid = pd.read_csv(SRC / "sparsification_grid.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = ["before", "top-k=10"]
    edges = [before.edges, after.edges]
    ax.bar(labels, edges, color=[BLUE, TEAL])
    ax.set_ylabel("Edges")
    ax.set_title("Sparsification preserves connectivity")
    for i, v in enumerate(edges):
        ax.text(i, v + max(edges) * 0.03, f"{int(v):,}", ha="center")
    txt = (
        f"edge reduction: {after.edge_reduction_percent:.2f}%\n"
        f"components: {int(after.connected_components)}\n"
        f"homophily: {before.target_bandgap_homophily:.3f} → {after.target_bandgap_homophily:.3f}"
    )
    ax.text(0.55, 0.78, txt, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor="white", edgecolor="#dddddd"))
    save(fig, "Figure_2_sparsification")


def fig3() -> None:
    val = pd.read_csv(SRC / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(SRC / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(SRC / "generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(SRC / "filtering_summary.csv").iloc[0]
    dist = pd.read_csv(SRC / "distribution_alignment.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    rates = [val.descriptor_valid_rate, edge.edge_valid_rate, 1 - nov.duplicate_rate, nov.novelty_pass_rate, filt.filtered_pass_rate]
    labels = ["descriptor\nvalid", "edge\nvalid", "non-\nduplicate", "novelty\npass", "filtered\npass"]
    axes[0].bar(labels, rates, color=[TEAL, TEAL, TEAL, TEAL, ORANGE])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Generated-node screens")
    metrics = ["descriptor\nshift", "Wasserstein\nproxy", "edge-weight\nmean"]
    values = [dist.descriptor_mean_abs_shift, dist.descriptor_wasserstein_proxy, dist.edge_weight_mean_generated]
    axes[1].bar(metrics, values, color=[PURPLE, PURPLE, TEAL])
    axes[1].set_title("Distribution diagnostics")
    axes[1].set_ylabel("Value")
    axes[1].text(0.02, 0.92, "nearby but distinguishable\ndescriptor space", transform=axes[1].transAxes, fontsize=10)
    save(fig, "Figure_3_validity_distribution")


def fig4() -> None:
    nn = pd.read_csv(SRC / "nearest_neighbor_interpretation.csv")
    examples = nn[nn.generated_node_id.isin(["gen-118", "gen-303", "gen-926", "gen-52", "gen-899"])].copy()
    if examples.empty:
        examples = nn.head(5)
    fig, axes = plt.subplots(len(examples), 1, figsize=(8, 1.45 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, examples.iterrows()):
        ax.axis("off")
        ax.text(0.01, 0.65, r.generated_node_id, fontweight="bold", fontsize=11)
        ax.text(0.18, 0.65, f"{r.top1_qmof_id} ({r.top1_similarity:.3f})", fontsize=10)
        ax.text(0.42, 0.65, f"{r.top2_qmof_id} ({r.top2_similarity:.3f})", fontsize=10)
        ax.text(0.66, 0.65, f"{r.top3_qmof_id} ({r.top3_similarity:.3f})", fontsize=10)
        ax.text(0.18, 0.25, f"top-5 bandgap {r.top5_bandgap_mean:.3f} ± {r.top5_bandgap_std:.3f} eV", fontsize=9)
        ax.text(0.66, 0.25, "descriptor-space hypothesis; not generated CIF", fontsize=9, color=PURPLE)
    save(fig, "Figure_4_nearest_neighbor_interpretation")


def fig5() -> None:
    ds = pd.read_csv(SRC / "downstream_summary.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    models = ["GraphSAGE", "GAT", "GCN"]
    x = np.arange(len(models))
    width = 0.34
    real = []
    best_gan = []
    best_names = []
    for m in models:
        g = ds[ds.model.eq(m)]
        real.append(float(g[g.training_mode.eq("real_only")].iloc[0].mae_mean))
        gg = g[g.training_mode.str.startswith("gan")].sort_values("mae_mean").iloc[0]
        best_gan.append(float(gg.mae_mean))
        best_names.append(gg.training_mode.replace("gan_", "GANg "))
    ax.bar(x - width / 2, real, width, label="real-only", color=BLUE)
    ax.bar(x + width / 2, best_gan, width, label="best GANg augmentation", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("MAE on held-out real QMOF nodes")
    ax.set_title("Validity–utility gap: valid nodes do not improve prediction")
    ax.legend(frameon=False)
    ax.text(0.02, 0.93, "lower is better", transform=ax.transAxes, fontsize=10)
    for i, (r, g) in enumerate(zip(real, best_gan)):
        ax.text(i - width / 2, r + 0.015, f"{r:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, g + 0.015, f"{g:.3f}", ha="center", fontsize=9)
    save(fig, "Figure_5_validity_utility_gap")


def main() -> None:
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()


if __name__ == "__main__":
    main()
