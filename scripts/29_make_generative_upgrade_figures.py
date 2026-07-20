#!/usr/bin/env python3
"""Make generative-upgrade figures from completed CSV outputs."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIG = ROOT / "figures" / "generative_upgrade"

REAL_BLUE = "#4C72B0"
GANG_AMBER = "#DD8452"
MUTED_PURPLE = "#8172B2"
MUTED_TEAL = "#64A88A"
GRID = "#E5E5E5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def baseline_comparison() -> None:
    candidate = TABLES / "remaining_issues" / "generator_baseline_validity_with_gang.csv"
    df = pd.read_csv(candidate if candidate.exists() else TABLES / "generator_baseline_validity.csv")
    df = df[pd.to_numeric(df["generated_count"], errors="coerce").notna()].copy()
    label_map = {
        "gaussian_noise": "Gaussian/\nnoise",
        "nearest_neighbor_interpolation": "nearest-neighbour\ninterpolation",
        "kde_bootstrap": "KDE/\nbootstrap",
        "gmm": "GMM",
        "vae": "VAE",
        "GANg": "GANg",
    }
    labels = df["generator"].map(label_map).fillna(df["generator"].str.replace("_", "\n"))
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    width = 0.32
    ax = axes[0]
    ax.bar(x - width / 2, df["descriptor_valid_rate"], width, label="descriptor valid", color=MUTED_TEAL)
    ax.bar(x + width / 2, df["edge_valid_rate"], width, label="edge valid", color=REAL_BLUE)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("A  Validity and edge checks", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    ax.bar(x - width / 2, df["novelty_pass_rate"], width, label="full-reference-pool novelty pass", color=GANG_AMBER)
    ax.bar(x + width / 2, df["diversity_score"], width, label="diversity score", color=MUTED_PURPLE)
    ymax = max(0.08, float(max(df["novelty_pass_rate"].max(), df["diversity_score"].max())) * 1.25)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Rate or mean pairwise cosine distance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("B  Novelty and descriptor diversity", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "generator_baseline_comparison")


def target_hits() -> None:
    df = pd.read_csv(TABLES / "conditional_generation_target_hits.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = df["target_window"].str.replace("_", "\n")
    colors = [MUTED_TEAL if "bandgap" in t else REAL_BLUE if "CBM" in t else MUTED_PURPLE if "VBM" in t else GANG_AMBER for t in df["target_window"]]
    ax.bar(np.arange(len(df)), df["nearest_neighbor_target_hit_rate"], color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Nearest-analogue target-window hit rate")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Target-conditioned descriptor-space hypotheses")
    ax.text(0.99, 0.96, "nearest real QMOF analogue estimate", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "target_conditioned_hit_rates")


def nearest_analogues() -> None:
    cards = pd.read_csv(TABLES / "generative_hypothesis_cards.csv").head(20)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sc = ax.scatter(cards["top1_similarity"], cards["top5_bandgap_mean"], c=cards["top5_bandgap_std"], cmap="viridis", s=60, edgecolor="white")
    ax.set_xlabel("Top-1 nearest real QMOF similarity")
    ax.set_ylabel("Top-5 analogue bandgap mean (eV)")
    ax.set_title("Generated descriptor-space candidates and nearest real analogues")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Top-5 bandgap std (eV)")
    ax.text(0.02, 0.96, "not generated CIFs", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "generated_candidates_nearest_analogues")


def hypothesis_card() -> None:
    card = pd.read_csv(TABLES / "generative_hypothesis_cards.csv").iloc[0]
    rows = pd.read_csv(TABLES / "generated_node_nearest_cif_analogues.csv")
    rows = rows[rows["generated_node_id"].eq(card["generated_node_id"])].head(5)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axis("off")
    global_id = f"rank1_bandgap_analogue_{str(card['generated_node_id']).replace('-', '_')}"
    title = f"Generative hypothesis card: {global_id}"
    ax.text(0.02, 0.94, title, fontsize=15, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.85, "Descriptor-space hypothesis from rank-1 analogue retrieval; nearest structures are real QMOF analogue CIFs", fontsize=10, transform=ax.transAxes)
    y = 0.72
    ax.text(0.02, y, "Nearest real QMOF analogues", fontsize=11, weight="bold", transform=ax.transAxes)
    y -= 0.08
    for _, r in rows.iterrows():
        ax.text(0.04, y, f"rank {int(r['rank'])}: {r['qmof_id']}   similarity={r['similarity']:.4f}   bandgap={r['target_bandgap']:.3f} eV", fontsize=10, transform=ax.transAxes)
        y -= 0.08
    ax.text(0.02, 0.14, f"Top-5 bandgap mean = {card['top5_bandgap_mean']:.3f} eV; std = {card['top5_bandgap_std']:.3f} eV", fontsize=10, transform=ax.transAxes)
    ax.text(0.02, 0.06, "No generated CIF, no DFT validation, no force-field relaxation, no experimental validation.", fontsize=9, color="#8a1c1c", transform=ax.transAxes)
    save(fig, "generative_hypothesis_card")


def main() -> None:
    baseline_comparison()
    target_hits()
    nearest_analogues()
    hypothesis_card()


if __name__ == "__main__":
    main()
