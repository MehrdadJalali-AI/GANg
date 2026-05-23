"""Paired statistical testing for downstream augmentation vs real-only."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "qmof_gang_final" / "tables" / "downstream_seedwise.csv"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures" / "npj"
REPORTS = ROOT / "reports"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def wilcoxon_p(delta: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        if np.allclose(delta, 0):
            return float("nan")
        return float(wilcoxon(delta).pvalue)
    except Exception:
        return float("nan")


def bootstrap_ci(delta: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cohen_dz(delta: np.ndarray) -> float:
    if len(delta) < 2 or np.std(delta, ddof=1) == 0:
        return float("nan")
    return float(np.mean(delta) / np.std(delta, ddof=1))


def main() -> None:
    df = pd.read_csv(SRC)
    rows = []
    for model, g in df.groupby("model"):
        base = g[g.training_mode.eq("real_only")].set_index("seed")["mae"]
        for mode, h in g.groupby("training_mode"):
            if mode == "real_only":
                continue
            aug = h.set_index("seed")["mae"]
            seeds = sorted(set(base.index) & set(aug.index))
            if not seeds:
                continue
            delta = aug.loc[seeds].to_numpy() - base.loc[seeds].to_numpy()
            lo, hi = bootstrap_ci(delta)
            mean_d = float(delta.mean())
            if hi < 0:
                direction = "improves"
            elif lo > 0:
                direction = "worsens"
            else:
                direction = "indistinguishable"
            rows.append(
                {
                    "target": "outputs.pbe.bandgap",
                    "dataset_size": 1000,
                    "model": model,
                    "training_mode": mode,
                    "n_paired_seeds": len(seeds),
                    "paired_delta_mae_mean": mean_d,
                    "bootstrap95_ci_low": lo,
                    "bootstrap95_ci_high": hi,
                    "wilcoxon_p_value": wilcoxon_p(delta),
                    "cohen_dz": cohen_dz(delta),
                    "direction_vs_real_only": direction,
                }
            )
    out = pd.DataFrame(rows).sort_values(["model", "paired_delta_mae_mean"])
    out.to_csv(TABLES / "statistical_testing_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    sub = out[out.training_mode.str.contains("gan", case=False) | out.training_mode.isin(["nearest-neighbor interpolation", "feature-noise"])]
    colors = {"GraphSAGE": "#4C9FAD", "GAT": "#7E5AA2", "GCN": "#D99A3D"}
    for model, g in sub.groupby("model"):
        ax.scatter(g.training_mode, g.paired_delta_mae_mean, label=model, color=colors.get(model), s=60)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Paired MAE delta vs real-only")
    ax.set_title("Effect direction by augmentation mode")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "statistical_effect_sizes.pdf")
    plt.close(fig)

    lines = ["# Statistical testing", "", "Positive paired delta MAE means worse than real-only.", ""]
    for model, g in out.groupby("model"):
        best = g.loc[g.paired_delta_mae_mean.idxmin()]
        lines.append(
            f"- **{model}** best mode `{best.training_mode}`: mean delta {best.paired_delta_mae_mean:.3f}, "
            f"95% CI [{best.bootstrap95_ci_low:.3f}, {best.bootstrap95_ci_high:.3f}], direction `{best.direction_vs_real_only}`."
        )
    (REPORTS / "statistical_testing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
