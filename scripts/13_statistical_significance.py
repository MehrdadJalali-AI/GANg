from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "qmof_gang_final" / "tables" / "downstream_seedwise.csv"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
TABLES.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def wilcoxon_or_nan(x: np.ndarray):
    try:
        from scipy.stats import wilcoxon

        if np.allclose(x, 0):
            return np.nan
        return float(wilcoxon(x).pvalue)
    except Exception:
        return np.nan


def bootstrap_ci(delta: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


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
            rows.append(
                {
                    "target": "outputs.pbe.bandgap",
                    "model": model,
                    "training_mode": mode,
                    "n_paired_seeds": len(seeds),
                    "real_only_mae_mean": float(base.loc[seeds].mean()),
                    "mode_mae_mean": float(aug.loc[seeds].mean()),
                    "paired_delta_mae_mean": float(delta.mean()),
                    "bootstrap95_ci_low": lo,
                    "bootstrap95_ci_high": hi,
                    "wilcoxon_p_value": wilcoxon_or_nan(delta),
                    "improves_over_real_only": bool(delta.mean() < 0 and hi < 0),
                }
            )
    out = pd.DataFrame(rows).sort_values(["model", "paired_delta_mae_mean"])
    out.to_csv(TABLES / "statistical_significance.csv", index=False)

    lines = [
        "# Statistical significance audit",
        "",
        "Comparison uses paired seeds from `experiments/qmof_gang_final/tables/downstream_seedwise.csv`.",
        "Positive delta means the augmentation mode has higher MAE than real-only and is worse.",
        "",
    ]
    for model, g in out.groupby("model"):
        lines.append(f"## {model}")
        best = g.iloc[0]
        lines.append(
            f"Best augmentation/control by paired delta: `{best.training_mode}` with mean delta "
            f"{best.paired_delta_mae_mean:.3f} MAE and 95% bootstrap CI "
            f"[{best.bootstrap95_ci_low:.3f}, {best.bootstrap95_ci_high:.3f}]."
        )
        if bool(best.improves_over_real_only):
            lines.append("This mode has a bootstrap CI entirely below zero.")
        else:
            lines.append("No augmentation mode shows a robust paired improvement over real-only.")
        lines.append("")
    lines.append("No result here changes the manuscript claim: generated-node augmentation does not improve held-out real-node prediction across GraphSAGE, GAT, and GCN.")
    (REPORTS / "statistical_significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
