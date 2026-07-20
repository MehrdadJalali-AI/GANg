#!/usr/bin/env python3
"""Final scientific correction pass, excluding Figure 5.

This script deliberately does not read, write, or regenerate any Figure 5 file.
It recomputes downstream GANg-mode summaries using validation-only cross-mode
selection and held-out test evaluation from the existing result tables.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gang")

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "tables" / "audit_reports"
MAIN_FIG = ROOT / "figures" / "main" / "Figure_5_validity_utility_gap"
MT_FIG = ROOT / "figures" / "high_rank_5000" / "multi_target_summary_5000"

GAN_MODES = [
    "gan_generated",
    "gan_filtered",
    "gan_property_head",
    "gan_conditional_target",
]

MODE_LABELS = {
    "real_only": "real-only",
    "gan_generated": "GANg generated",
    "gan_filtered": "filtered GANg",
    "gan_property_head": "GANg property-head",
    "gan_conditional_target": "GANg conditional",
    "nearest_neighbor_interpolation": "NN interpolation",
    "feature_noise": "feature-noise",
    "random_synthetic": "random synthetic",
}

DATASETS = [
    {
        "nodes": 1000,
        "target": "bandgap",
        "target_label": "bandgap",
        "path": ROOT / "experiments/qmof_galaxy_final/tables",
    },
    {
        "nodes": 5000,
        "target": "bandgap",
        "target_label": "bandgap",
        "path": ROOT / "experiments/qmof_gang_5000_outputs_pbe_bandgap/tables",
    },
    {
        "nodes": 5000,
        "target": "cbm",
        "target_label": "CBM",
        "path": ROOT / "experiments/qmof_gang_5000_outputs_pbe_cbm/tables",
    },
    {
        "nodes": 5000,
        "target": "vbm",
        "target_label": "VBM",
        "path": ROOT / "experiments/qmof_gang_5000_outputs_pbe_vbm/tables",
    },
    {
        "nodes": 5000,
        "target": "energy_total",
        "target_label": "total energy",
        "path": ROOT / "experiments/qmof_gang_5000_outputs_pbe_energy_total/tables",
    },
]


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def ci_t(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= 1:
        return float("nan"), float("nan")
    mean = values.mean()
    sem = values.std(ddof=1) / math.sqrt(n)
    if stats is None:
        q = 1.96
    else:
        q = float(stats.t.ppf(0.975, n - 1))
    return mean - q * sem, mean + q * sem


def paired_stats(delta: np.ndarray) -> dict[str, float | int]:
    delta = np.asarray(delta, dtype=float)
    lo, hi = ci_t(delta)
    sd = delta.std(ddof=1) if len(delta) > 1 else float("nan")
    dz = delta.mean() / sd if sd and not math.isnan(sd) else float("nan")
    if stats is not None and len(delta) > 1:
        try:
            p = float(stats.wilcoxon(delta).pvalue)
        except Exception:
            p = float("nan")
    else:
        p = float("nan")
    return {
        "n_paired_seeds": int(len(delta)),
        "delta_mae_mean": float(delta.mean()),
        "delta_mae_std": float(sd),
        "delta_mae_ci95_low": float(lo),
        "delta_mae_ci95_high": float(hi),
        "paired_cohen_dz": float(dz),
        "wilcoxon_p": p,
        "seeds_gang_lower_mae": int((delta < 0).sum()),
        "seeds_real_lower_mae": int((delta > 0).sum()),
        "seeds_tied": int((delta == 0).sum()),
    }


def read_dataset(ds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(ds["path"] / "downstream_gnn_summary.csv")
    seedwise = pd.read_csv(ds["path"] / "downstream_gnn_seedwise.csv")
    return summary, seedwise


def compute_validation_selected() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    seed_rows = []
    stat_rows = []
    for ds in DATASETS:
        summary, seedwise = read_dataset(ds)
        for model in ["GraphSAGE", "GAT", "GCN"]:
            sub = summary.loc[summary["model"].eq(model)].copy()
            real = sub.loc[sub["training_mode"].eq("real_only")]
            if len(real) != 1:
                raise ValueError(f"Expected one real-only row for {ds['target']} {model}.")
            real = real.iloc[0]
            cand = sub.loc[sub["training_mode"].isin(GAN_MODES)].copy()
            missing = sorted(set(GAN_MODES) - set(cand["training_mode"]))
            if missing:
                raise ValueError(f"Missing GANg candidate modes for {ds['target']} {model}: {missing}")
            val = cand.sort_values(["best_val_mae_mean", "training_mode"], kind="mergesort").iloc[0]
            old = cand.sort_values(["mae_mean", "training_mode"], kind="mergesort").iloc[0]
            delta = float(val["mae_mean"] - real["mae_mean"])
            rows.append(
                {
                    "target": ds["target"],
                    "target_label": ds["target_label"],
                    "nodes": ds["nodes"],
                    "architecture": model,
                    "real_only_test_mae": float(real["mae_mean"]),
                    "real_only_test_mae_std": float(real["mae_std"]),
                    "validation_selected_mode": val["training_mode"],
                    "validation_selected_mode_label": MODE_LABELS.get(val["training_mode"], val["training_mode"]),
                    "validation_mae_of_selected_mode": float(val["best_val_mae_mean"]),
                    "selected_mode_test_mae": float(val["mae_mean"]),
                    "selected_mode_test_mae_std": float(val["mae_std"]),
                    "selected_mode_test_rmse": float(val["rmse_mean"]),
                    "selected_mode_test_r2": float(val["r2_mean"]),
                    "selected_mode_test_spearman": float(val["spearman_mean"]),
                    "selected_mode_test_kendall": float(val["kendall_mean"]),
                    "delta_mae": delta,
                    "old_test_minimum_mode": old["training_mode"],
                    "old_test_minimum_test_mae": float(old["mae_mean"]),
                    "correction_magnitude": float(val["mae_mean"] - old["mae_mean"]),
                    "selected_mode_changed": bool(val["training_mode"] != old["training_mode"]),
                    "candidate_gang_modes": ";".join(GAN_MODES),
                    "source_summary_table": str((ds["path"] / "downstream_gnn_summary.csv").relative_to(ROOT)),
                    "source_seedwise_table": str((ds["path"] / "downstream_gnn_seedwise.csv").relative_to(ROOT)),
                }
            )
            real_sw = seedwise.loc[(seedwise["model"].eq(model)) & (seedwise["training_mode"].eq("real_only"))].copy()
            sel_sw = seedwise.loc[(seedwise["model"].eq(model)) & (seedwise["training_mode"].eq(val["training_mode"]))].copy()
            merged = real_sw.merge(sel_sw, on=["model", "seed"], suffixes=("_real_only", "_selected"))
            if len(merged) == 0:
                continue
            merged["delta_mae"] = merged["mae_selected"] - merged["mae_real_only"]
            for _, r in merged.iterrows():
                seed_rows.append(
                    {
                        "target": ds["target"],
                        "target_label": ds["target_label"],
                        "nodes": ds["nodes"],
                        "architecture": model,
                        "seed": int(r["seed"]),
                        "real_only_mae": float(r["mae_real_only"]),
                        "validation_selected_mode": val["training_mode"],
                        "selected_mode_mae": float(r["mae_selected"]),
                        "delta_mae": float(r["delta_mae"]),
                        "selected_mode_rmse": float(r["rmse_selected"]),
                        "selected_mode_r2": float(r["r2_selected"]),
                        "selected_mode_spearman": float(r["spearman_selected"]),
                        "selected_mode_kendall": float(r["kendall_selected"]),
                        "selected_mode_best_val_mae": float(r["best_val_mae_selected"]),
                    }
                )
            st = paired_stats(merged["delta_mae"].to_numpy())
            st.update(
                {
                    "target": ds["target"],
                    "target_label": ds["target_label"],
                    "nodes": ds["nodes"],
                    "architecture": model,
                    "validation_selected_mode": val["training_mode"],
                    "real_only_test_mae_mean": float(real["mae_mean"]),
                    "selected_mode_test_mae_mean": float(val["mae_mean"]),
                }
            )
            stat_rows.append(st)
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), pd.DataFrame(stat_rows)


def write_csvs(results: pd.DataFrame, seedwise: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    TABLES.mkdir(exist_ok=True)
    results.to_csv(TABLES / "validation_selected_results.csv", index=False)
    seedwise.to_csv(TABLES / "validation_selected_seedwise_results.csv", index=False)
    stats_df.to_csv(TABLES / "validation_selected_statistics.csv", index=False)
    # Backwards-compatible aliases used by earlier revision artefacts.
    results.to_csv(TABLES / "validation_selected_downstream_summary.csv", index=False)
    seedwise.to_csv(TABLES / "validation_selected_downstream_seedwise_5000.csv", index=False)


def write_latex_tables(results: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    # Public GitHub copy: write LaTeX table snippets under tables/latex rather
    # than creating manuscript/internal directories.
    outdir = TABLES / "latex"
    outdir.mkdir(parents=True, exist_ok=True)
    one = results.loc[(results["nodes"].eq(1000)) & (results["target"].eq("bandgap"))]
    lines = [
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Architecture & Validation-selected mode & Real-only MAE & Selected MAE & $\\Delta$MAE & Val. MAE \\\\",
        "\\midrule",
    ]
    for _, r in one.iterrows():
        lines.append(
            f"{r.architecture} & {MODE_LABELS[r.validation_selected_mode]} & "
            f"{r.real_only_test_mae:.3f} $\\pm$ {r.real_only_test_mae_std:.3f} & "
            f"{r.selected_mode_test_mae:.3f} $\\pm$ {r.selected_mode_test_mae_std:.3f} & "
            f"{r.delta_mae:.3f} & {r.validation_mae_of_selected_mode:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (outdir / "validation_selected_1000_bandgap.tex").write_text("\n".join(lines) + "\n")

    mt = results.loc[results["nodes"].eq(5000)].copy()
    lines = [
        "\\begin{longtable}{@{}lllrrrrr@{}}",
        "\\caption{5000-node validation-selected downstream augmentation summary. The selected GANg mode is chosen by validation MAE within each target--architecture pair and then evaluated once on held-out real QMOF test nodes. Delta is selected-mode test MAE minus real-only test MAE; negative values indicate lower mean MAE for the validation-selected GANg mode. Total-energy MAE values are raw extensive-target values and are interpreted with the caveats in the text.}\\\\",
        "\\toprule",
        "Target & Model & Selected mode & Real-only MAE & Selected MAE & $\\Delta$MAE & Val. MAE & Old test-min. MAE \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Target & Model & Selected mode & Real-only MAE & Selected MAE & $\\Delta$MAE & Val. MAE & Old test-min. MAE \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for _, r in mt.iterrows():
        digits = 3 if r.target != "energy_total" else 1
        lines.append(
            f"{r.target_label} & {r.architecture} & {MODE_LABELS[r.validation_selected_mode]} & "
            f"{r.real_only_test_mae:.{digits}f} & {r.selected_mode_test_mae:.{digits}f} & "
            f"{r.delta_mae:.{digits}f} & {r.validation_mae_of_selected_mode:.{digits}f} & "
            f"{r.old_test_minimum_test_mae:.{digits}f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{longtable}"]
    (outdir / "validation_selected_5000_table.tex").write_text("\n".join(lines) + "\n")

    lines = [
        "\\begin{tabular}{@{}lllrrrr@{}}",
        "\\toprule",
        "Target & Model & Selected mode & Mean $\\Delta$MAE & 95\\% CI & Cohen $d_z$ & GANg-lower seeds \\\\",
        "\\midrule",
    ]
    for _, r in stats_df.iterrows():
        if int(r.nodes) == 1000 or r.target in {"bandgap", "cbm", "vbm"}:
            lines.append(
                f"{r.target_label} ({int(r.nodes)}) & {r.architecture} & {MODE_LABELS[r.validation_selected_mode]} & "
                f"{r.delta_mae_mean:.3f} & [{r.delta_mae_ci95_low:.3f}, {r.delta_mae_ci95_high:.3f}] & "
                f"{r.paired_cohen_dz:.2f} & {int(r.seeds_gang_lower_mae)}/{int(r.n_paired_seeds)} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (outdir / "validation_selected_statistics_table.tex").write_text("\n".join(lines) + "\n")


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )


def plot_figure7(results: pd.DataFrame) -> None:
    df = results.loc[(results["nodes"].eq(1000)) & (results["target"].eq("bandgap"))].copy()
    order = ["GraphSAGE", "GAT", "GCN"]
    df["architecture"] = pd.Categorical(df["architecture"], order, ordered=True)
    df = df.sort_values("architecture")
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=300)
    x = np.arange(len(df))
    w = 0.34
    ax.bar(x - w / 2, df["real_only_test_mae"], w, color="#4C72B0", label="real-only")
    ax.bar(x + w / 2, df["selected_mode_test_mae"], w, color="#DD8452", label="validation-selected GANg mode")
    ax.errorbar(x - w / 2, df["real_only_test_mae"], yerr=df["real_only_test_mae_std"], fmt="none", ecolor="#333333", lw=0.8, capsize=2)
    ax.errorbar(x + w / 2, df["selected_mode_test_mae"], yerr=df["selected_mode_test_mae_std"], fmt="none", ecolor="#333333", lw=0.8, capsize=2)
    for i, r in enumerate(df.itertuples()):
        ax.text(i, max(r.real_only_test_mae, r.selected_mode_test_mae) + 0.035, f"Δ={r.delta_mae:+.3f}\n{MODE_LABELS[r.validation_selected_mode]}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, df["architecture"])
    ax.set_ylabel("Held-out test MAE")
    ax.set_title("1000-node bandgap: validation-selected GANg mode")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_ylim(0, max(df["selected_mode_test_mae"].max(), df["real_only_test_mae"].max()) + 0.16)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in [".pdf", ".png", ".svg"]:
        fig.savefig(MAIN_FIG.with_suffix(ext), dpi=600 if ext == ".png" else None)
    plt.close(fig)
    pd.DataFrame(
        {
            "source_table": ["tables/validation_selected_results.csv"],
            "selection_rule": ["minimum validation MAE among GANg modes, per architecture"],
            "figure": ["figures/main/Figure_5_validity_utility_gap.pdf"],
        }
    ).to_csv(MAIN_FIG.with_name(MAIN_FIG.name + "_provenance.csv"), index=False)


def plot_figure8(results: pd.DataFrame) -> None:
    df = results.loc[results["nodes"].eq(5000)].copy()
    electronic = df.loc[df["target"].isin(["bandgap", "cbm", "vbm"])].copy()
    energy = df.loc[df["target"].eq("energy_total")].copy()
    fig = plt.figure(figsize=(7.4, 3.9), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.45], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    pivot = electronic.pivot(index="target_label", columns="architecture", values="delta_mae").loc[["bandgap", "CBM", "VBM"], ["GraphSAGE", "GAT", "GCN"]]
    im = ax.imshow(pivot.values, cmap=mpl.colors.LinearSegmentedColormap.from_list("div", ["#4C72B0", "#F7F7F7", "#DD8452"]), vmin=-0.02, vmax=0.02, aspect="auto")
    ax.set_xticks(np.arange(3), pivot.columns)
    ax.set_yticks(np.arange(3), pivot.index)
    ax.set_title("A  Electronic targets", loc="left", fontweight="bold", fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=8, color="#222222")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("ΔMAE", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    energy = energy.set_index("architecture").loc[["GraphSAGE", "GAT", "GCN"]].reset_index()
    colours = ["#4C72B0" if v < 0 else "#DD8452" for v in energy["delta_mae"]]
    y = np.arange(len(energy))
    ax2.barh(y, energy["delta_mae"], color=colours)
    ax2.axvline(0, color="#333333", lw=0.8)
    ax2.set_yticks(y, energy["architecture"])
    ax2.invert_yaxis()
    ax2.set_xlabel("Raw ΔMAE")
    ax2.set_title("B  Total energy", loc="left", fontweight="bold", fontsize=10)
    xmin = min(float(energy["delta_mae"].min()) * 1.18, -1.0)
    ax2.set_xlim(xmin, 0)
    for yi, r in enumerate(energy.itertuples()):
        pct = 100 * (r.real_only_test_mae - r.selected_mode_test_mae) / r.real_only_test_mae
        label = f"{r.delta_mae:+.1f}\n({pct:+.1f}%)"
        if abs(r.delta_mae) < 40:
            ax2.text(r.delta_mae - 5, yi, label, va="center", ha="right", fontsize=7, color="#333333")
        else:
            ax2.text(r.delta_mae / 2, yi, label, va="center", ha="center", fontsize=7, color="white")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in [".pdf", ".png", ".svg"]:
        fig.savefig(MT_FIG.with_suffix(ext), dpi=600 if ext == ".png" else None)
    plt.close(fig)
    pd.DataFrame(
        {
            "source_table": ["tables/validation_selected_results.csv"],
            "selection_rule": ["minimum validation MAE among GANg modes, per target and architecture"],
            "figure": ["figures/high_rank_5000/multi_target_summary_5000.pdf"],
        }
    ).to_csv(MT_FIG.with_name(MT_FIG.name + "_provenance.csv"), index=False)


def write_mode_audit(results: pd.DataFrame) -> None:
    lines = [
        "# Mode-selection audit",
        "",
        "This audit replaces the old manuscript-facing `best_gang_*` aggregation with leakage-safe validation-selected GANg-mode summaries.",
        "",
        "## Code locations",
        "",
        "- Old test-minimum aggregation: `scripts/34_aggregate_5000_package.py::validity_utility`, which sorted GANg modes by held-out test `mae_mean`.",
        "- Old 1000-node manuscript summaries: `scripts/11_gang_audit_pass.py` and `scripts/12_submission_polish.py`, which also chose manuscript-facing generated modes from test MAE summaries.",
        "- Corrected selection in this pass: `scripts/46_address_remaining_issues_except_figure5.py::compute_validation_selected`.",
        "",
        "## Candidate GANg modes",
        "",
        ", ".join(f"`{m}`" for m in GAN_MODES),
        "",
        "Checkpoint selection within a single mode uses the recorded `best_val_mae`/`best_val_mae_mean` from each training run. Cross-mode selection is a separate step: for each target and architecture, this correction selects the GANg mode with the lowest validation MAE summary and then reports that locked mode's held-out test metrics.",
        "",
        "Validation metric: mean validation MAE (`best_val_mae_mean`).",
        "",
        "Test-evaluation rule: after the cross-mode validation selection is locked, report the held-out real-QMOF test metrics for that selected mode. Test labels are not used for selecting among GANg modes.",
        "",
        "## Old versus corrected modes",
        "",
        "| Target | Nodes | Architecture | Old test-minimum mode | New validation-selected mode | Changed? | Correction magnitude |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for _, r in results.iterrows():
        lines.append(
            f"| {r.target_label} | {int(r.nodes)} | {r.architecture} | `{r.old_test_minimum_mode}` | "
            f"`{r.validation_selected_mode}` | {str(bool(r.selected_mode_changed)).lower()} | {r.correction_magnitude:.6g} |"
        )
    (REPORTS / "mode_selection_audit.md").write_text("\n".join(lines) + "\n")


def write_other_reports(results: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    REPORTS.mkdir(exist_ok=True)
    changed = int(results["selected_mode_changed"].sum())
    (REPORTS / "statistical_design_audit.md").write_text(
        "\n".join(
            [
                "# Statistical design audit",
                "",
                "The current downstream analyses use five paired initialisation seeds on one fixed train/validation/test split. This quantifies initialisation sensitivity but not split-to-split uncertainty.",
                "",
                "No additional split reruns were found or performed in this pass. Therefore, the manuscript retains the exploratory framing and does not make significance, equivalence, or universal augmentation-gain claims.",
                "",
                "After validation-selected mode correction, paired delta MAE, 95% t confidence intervals, paired Cohen dz, Wilcoxon p-values, and seedwise consistency counts were recomputed in `tables/validation_selected_statistics.csv` from `tables/validation_selected_seedwise_results.csv`.",
            ]
        )
        + "\n"
    )
    (REPORTS / "generator_downstream_parity_summary.md").write_text(
        "\n".join(
            [
                "# Generator downstream parity summary",
                "",
                "Full downstream parity for KDE/bootstrap and GMM across GraphSAGE, GAT, and GCN was not completed in the available result tables.",
                "",
                "Available evidence remains limited to compact baseline/downstream tables already present in the repository. The manuscript therefore treats KDE/bootstrap and GMM comparisons as generator-quality diagnostics, not as downstream-utility or GANg-superiority evidence.",
            ]
        )
        + "\n"
    )
    (TABLES / "generator_downstream_parity_results.csv").write_text(
        "status,source,interpretation\nnot_completed,available_repository_tables,Full KDE/GMM downstream parity across all three architectures is unavailable; no GANg superiority claim is made.\n"
    )
    (REPORTS / "largest_component_control.md").write_text(
        "\n".join(
            [
                "# Largest-component control",
                "",
                "The 5000-node sparsified graph has components of sizes 4923, 29, 18, 17, and 13.",
                "",
                "No 4923-node largest-component-only downstream rerun result file was found or generated in this pass. The manuscript therefore keeps this as an explicit limitation and retains the connected 1000-node bandgap study as the connectivity-controlled reference.",
            ]
        )
        + "\n"
    )
    (REPORTS / "final_scientific_consistency_report.md").write_text(
        "\n".join(
            [
                "# Final scientific consistency report",
                "",
                f"Validation-selected principal result rows: {len(results)}.",
                f"Rows where validation-selected mode differs from old test-minimum mode: {changed}.",
                "",
                "Test leakage status: the principal GANg-mode summaries in the regenerated source tables and downstream figures use validation-only cross-mode selection followed by held-out test evaluation. Historical test-minimum values are retained only in audit columns/tables.",
                "",
                "Figure/table match: Figure 7 and Figure 8 were regenerated from `tables/validation_selected_results.csv`. Figure 5 was not modified by this pass.",
                "",
                "Remaining limitations: fixed split with five seeds only; no largest-component-only 4923-node control; no full KDE/GMM downstream parity; no normalised-energy GNN rerun; no DOI archive created in this local pass.",
            ]
        )
        + "\n"
    )
    (REPORTS / "remaining_unresolved_items.md").write_text(
        "\n".join(
            [
                "# Remaining unresolved items",
                "",
                "- Additional split reruns were not performed.",
                "- Largest-component-only 4923-node downstream control was not performed.",
                "- Full downstream parity for KDE/bootstrap and GMM across GraphSAGE, GAT, and GCN was not completed.",
                "- Normalised-energy GNN rerun was not completed; raw total-energy results remain extensive-target diagnostics.",
                "- External DOI/publisher reference verification and Zenodo archival require online/manual steps.",
            ]
        )
        + "\n"
    )
    (REPORTS / "revision_change_log.md").write_text(
        "\n".join(
            [
                "# Revision change log",
                "",
                "- Replaced principal downstream GANg-mode summaries with validation-selected test results.",
                "- Regenerated Figure 7 and Figure 8 from corrected validation-selected source tables.",
                "- Left Figure 5 and all Figure 5 source files unchanged.",
                "- Recomputed paired seedwise statistics and wrote audit tables.",
                "- Reframed Supplementary S29A as scientific selection/evaluation controls rather than an internal audit log.",
            ]
        )
        + "\n"
    )


def write_sparsification_audit() -> None:
    grid = ROOT / "experiments/qmof_galaxy_final/tables/sparsification_grid.csv"
    final = ROOT / "experiments/qmof_galaxy_final/tables/final_sparsification_summary.csv"
    lines = [
        "# Sparsification-selection audit",
        "",
        f"Candidate-grid source: `{grid.relative_to(ROOT)}`",
        f"Final-summary source: `{final.relative_to(ROOT)}`",
        "",
        "The manuscript-facing top-k=10 statement was compared against the available code and tables. `scripts/09_build_final_package.py` contains a historical candidate-selection expression that sorted valid candidates by edge reduction and then bandgap homophily. That means target-bin homophily appears in the historical code path as a possible tie-breaker.",
        "",
        "For the recorded 1000-node package, top-k=10 is still defensible using unsupervised graph criteria alone: it preserves one connected component and all real nodes while reducing edge count from 9998 to 6651 and giving average degree 13.302. The manuscript should therefore describe top-k=10 as an audited locked setting with homophily reported post hoc, not as a fully automated leakage-safe selection if the historical code path is cited.",
        "",
        "Test labels did not affect the recorded sparsification decision. Target labels were accessed for homophily diagnostics and potentially as a tie-breaker in historical code, so the audit avoids an unqualified claim that the original code was entirely target-label-free.",
    ]
    (REPORTS / "sparsification_selection_audit.md").write_text("\n".join(lines) + "\n")


def copy_existing_or_write_release_reports() -> None:
    # Preserve existing richer audits when present; otherwise write conservative status files.
    if not (REPORTS / "total_energy_normalization_audit.md").exists():
        (REPORTS / "total_energy_normalization_audit.md").write_text("# Total-energy normalisation audit\n\nNo normalised-energy GNN rerun was completed.\n")
    if not (TABLES / "total_energy_baselines.csv").exists():
        (TABLES / "total_energy_baselines.csv").write_text("status\nnot_completed\n")
    if not (TABLES / "normalized_energy_results.csv").exists():
        (TABLES / "normalized_energy_results.csv").write_text("status\nnot_completed\n")
    (REPORTS / "archive_release_checklist.md").write_text(
        "\n".join(
            [
                "# Archive release checklist",
                "",
                "- Code: prepared in repository.",
                "- Derived tables: present under `tables/` and experiment table folders.",
                "- Manuscript sources are not included in the public GitHub code package.",
                "- Audit logs are written under `tables/audit_reports/`.",
                "- README/licence/CITATION: verify before public release.",
                "- Checksums/Zenodo DOI: not created in this local pass; no fake DOI should be inserted.",
            ]
        )
        + "\n"
    )
    manifest_rows = []
    for rel in ["tables/validation_selected_results.csv", "tables/validation_selected_statistics.csv", "tables/audit_reports/mode_selection_audit.md"]:
        p = ROOT / rel
        if p.exists():
            manifest_rows.append({"path": rel, "bytes": p.stat().st_size})
    with (TABLES / "release_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    configure_plotting()
    results, seedwise, stats_df = compute_validation_selected()
    write_csvs(results, seedwise, stats_df)
    write_latex_tables(results, stats_df)
    write_mode_audit(results)
    write_sparsification_audit()
    write_other_reports(results, stats_df)
    copy_existing_or_write_release_reports()
    plot_figure7(results)
    plot_figure8(results)
    print("Validation-selected principal rows:", len(results))
    print("Changed modes:", int(results["selected_mode_changed"].sum()))
    print(results[["nodes", "target", "architecture", "validation_selected_mode", "selected_mode_test_mae", "delta_mae", "old_test_minimum_mode", "old_test_minimum_test_mae"]].to_string(index=False))
    print("Regenerated non-Figure-5 figures:")
    print(MAIN_FIG.with_suffix(".pdf").relative_to(ROOT))
    print(MT_FIG.with_suffix(".pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
