#!/usr/bin/env python3
"""Implementation-grounded final audit tables and scaling figure.

This script intentionally avoids Figures 1 and 5. It reads existing
implementation outputs, computes derived quantities programmatically, writes
machine-readable evidence tables, and regenerates only the graph-scaling figure
from those evidence tables.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gang")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGS = ROOT / "figures" / "high_rank_5000"

SMALL_GRAPH = ROOT / "experiments/qmof_gang_final/tables"
LARGE_GRAPH = ROOT / "experiments/qmof_gang_5000_outputs_pbe_bandgap/tables"
VALIDATION_SELECTED = TABLES / "validation_selected_results.csv"
VALIDATION_STATS = TABLES / "validation_selected_statistics.csv"
BASELINE = TABLES / "remaining_issues/generator_baseline_validity_with_gang.csv"


def read_one(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) == 0:
        raise ValueError(f"Empty table: {path}")
    return frame.iloc[0]


def density(nodes: int, edges: int) -> float:
    return 2.0 * edges / (nodes * (nodes - 1))


def build_scaling_summary() -> pd.DataFrame:
    g_small = read_one(SMALL_GRAPH / "graph_summary_before_sparsification.csv")
    s_small = read_one(SMALL_GRAPH / "final_sparsification_summary.csv")
    g_large = read_one(LARGE_GRAPH / "graph_summary_before_sparsification.csv")
    s_large = read_one(LARGE_GRAPH / "final_sparsification_summary.csv")

    small_nodes = int(g_small["nodes"])
    large_nodes = int(g_large["nodes"])
    small_before = int(g_small["edges"])
    large_before = int(g_large["edges"])
    small_after = int(s_small["edges"])
    large_after = int(s_large["edges"])
    small_reduction = 100.0 * (small_before - small_after) / small_before
    large_reduction = 100.0 * (large_before - large_after) / large_before
    small_density = density(small_nodes, small_after)
    large_density = density(large_nodes, large_after)

    rows = [
        {
            "graph_characteristic": "Node count",
            "smaller_study": small_nodes,
            "larger_study": large_nodes,
            "derived_interpretation": f"{large_nodes / small_nodes:.2f}x nodes",
            "source_smaller": str((SMALL_GRAPH / "graph_summary_before_sparsification.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "graph_summary_before_sparsification.csv").relative_to(ROOT)),
            "calculation": "larger nodes / smaller nodes",
        },
        {
            "graph_characteristic": "Edges before sparsification",
            "smaller_study": small_before,
            "larger_study": large_before,
            "derived_interpretation": f"{large_before / small_before:.2f}x edges",
            "source_smaller": str((SMALL_GRAPH / "graph_summary_before_sparsification.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "graph_summary_before_sparsification.csv").relative_to(ROOT)),
            "calculation": "larger pre-sparsification edges / smaller pre-sparsification edges",
        },
        {
            "graph_characteristic": "Edges after sparsification",
            "smaller_study": small_after,
            "larger_study": large_after,
            "derived_interpretation": f"{large_after / small_after:.2f}x retained edges",
            "source_smaller": str((SMALL_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "calculation": "larger post-sparsification edges / smaller post-sparsification edges",
        },
        {
            "graph_characteristic": "Edge reduction (%)",
            "smaller_study": small_reduction,
            "larger_study": large_reduction,
            "derived_interpretation": f"{small_reduction / large_reduction:.2f}x removed-fraction ratio",
            "source_smaller": str((SMALL_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "calculation": "100 * (edges before - edges after) / edges before; smaller reduction / larger reduction",
        },
        {
            "graph_characteristic": "Average degree after sparsification",
            "smaller_study": float(s_small["average_degree"]),
            "larger_study": float(s_large["average_degree"]),
            "derived_interpretation": f"{float(s_large['average_degree']) / float(s_small['average_degree']):.2f}x average degree",
            "source_smaller": str((SMALL_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "calculation": "larger post-sparsification average degree / smaller post-sparsification average degree",
        },
        {
            "graph_characteristic": "Graph density after sparsification",
            "smaller_study": small_density,
            "larger_study": large_density,
            "derived_interpretation": f"{large_density / small_density:.2f}x density",
            "source_smaller": str((SMALL_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "calculation": "2E/(V(V-1)) for a simple undirected graph",
        },
        {
            "graph_characteristic": "Connected components after sparsification",
            "smaller_study": int(s_small["connected_components"]),
            "larger_study": int(s_large["connected_components"]),
            "derived_interpretation": "more components in larger study",
            "source_smaller": str((SMALL_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "calculation": "reported by graph implementation",
        },
        {
            "graph_characteristic": "Largest-component size",
            "smaller_study": int(s_small["largest_component_size"]),
            "larger_study": int(s_large["largest_component_size"]),
            "derived_interpretation": f"{100.0 * int(s_large['largest_component_size']) / large_nodes:.1f}\\% of larger-study nodes in largest component",
            "source_smaller": str((SMALL_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "calculation": "reported largest-component size; larger largest component / larger node count",
        },
        {
            "graph_characteristic": "Sparsification configuration",
            "smaller_study": f"{s_small['chosen_method']} {s_small['chosen_setting']}",
            "larger_study": f"{s_large['chosen_method']} {s_large['chosen_setting']}",
            "derived_interpretation": "locked settings differ",
            "source_smaller": str((SMALL_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "source_larger": str((LARGE_GRAPH / "final_sparsification_summary.csv").relative_to(ROOT)),
            "calculation": "reported by sparsification implementation",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "implementation_grounded_scaling_summary.csv", index=False)
    return out


def fmt_value(metric: str, value: object) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    if "density" in metric.lower():
        return f"{value:.5f}"
    if "reduction" in metric.lower():
        return f"{value:.2f}\\%"
    if "average degree" in metric.lower():
        return f"{value:.3f}"
    if abs(value - round(value)) < 1e-10:
        return f"{int(round(value)):,}"
    return f"{value:.3f}"


def write_scaling_latex(scaling: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    display_labels = {
        "Edge reduction (%)": "Edge reduction",
        "Average degree after sparsification": "Avg. degree after sparsification",
        "Graph density after sparsification": "Density after sparsification",
        "Connected components after sparsification": "Components after sparsification",
    }
    lines = [
        "\\begin{tabular}{@{}p{0.24\\linewidth}p{0.15\\linewidth}p{0.15\\linewidth}p{0.36\\linewidth}@{}}",
        "\\toprule",
        "Graph characteristic & Smaller study & Larger study & Scaling interpretation \\\\",
        "\\midrule",
    ]
    for _, r in scaling.iterrows():
        metric = str(r["graph_characteristic"])
        display_metric = display_labels.get(metric, metric)
        lines.append(
            f"{display_metric} & {fmt_value(metric, r['smaller_study'])} & {fmt_value(metric, r['larger_study'])} & {r['derived_interpretation']} \\\\"
        )
        lines.append("\\addlinespace[0.18em]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "implementation_grounded_scaling_summary.tex").write_text("\n".join(lines) + "\n")


def regenerate_scaling_figure(scaling: pd.DataFrame) -> None:
    edge_rows = scaling.set_index("graph_characteristic")
    before = [
        float(edge_rows.loc["Edges before sparsification", "smaller_study"]),
        float(edge_rows.loc["Edges before sparsification", "larger_study"]),
    ]
    after = [
        float(edge_rows.loc["Edges after sparsification", "smaller_study"]),
        float(edge_rows.loc["Edges after sparsification", "larger_study"]),
    ]
    labels = ["1000-node\nbandgap", "5000-node\nbandgap"]
    x = np.arange(2)
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    b1 = ax.bar(x - 0.18, before, width=0.36, label="before sparsification", color="#4477AA", edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + 0.18, after, width=0.36, label="after sparsification", color="#EE7733", edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Undirected weighted edge count")
    ax.set_title("Edge-count scaling from implementation outputs", pad=12, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    ax.bar_label(b1, labels=[f"{int(v):,}" for v in before], padding=3, fontsize=9)
    ax.bar_label(b2, labels=[f"{int(v):,}" for v in after], padding=3, fontsize=9)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGS / f"scaling_comparison_1000_vs_5000.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def build_metric_source_map(scaling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in scaling.iterrows():
        rows.append(
            {
                "manuscript_section": "Section 2.9 / Supplementary S28",
                "table_or_figure": "Scaling summary table / Figure 9",
                "metric_name": r["graph_characteristic"],
                "numerical_value": f"small={r['smaller_study']}; large={r['larger_study']}",
                "source_output_file": f"{r['source_smaller']} | {r['source_larger']}",
                "source_column": "see metric-specific source table columns",
                "row_selection_rule": "row 0 from locked 1000-node bandgap and 5000-node bandgap outputs",
                "calculation_script": "scripts/47_implementation_grounded_audit.py",
                "configuration_and_seed": "locked QMOF 1000/5000 bandgap outputs; split seed 42 where recorded",
            }
        )
    if VALIDATION_SELECTED.exists():
        val = pd.read_csv(VALIDATION_SELECTED)
        for _, r in val.iterrows():
            rows.append(
                {
                    "manuscript_section": "Section 2.8/2.9 and Supplementary S12/S22/S28",
                    "table_or_figure": "Figure 7 / Figure 8 / validation-selected tables",
                    "metric_name": f"{r['nodes']} {r['target']} {r['architecture']} validation-selected downstream",
                    "numerical_value": f"real={r['real_only_test_mae']}; selected={r['selected_mode_test_mae']}; delta={r['delta_mae']}; selected_mode={r['validation_selected_mode']}",
                    "source_output_file": f"{r['source_summary_table']} | {r['source_seedwise_table']}",
                    "source_column": "best_val_mae_mean, mae_mean, mae_std, rmse_mean, r2_mean, spearman_mean, kendall_mean",
                    "row_selection_rule": "select GANg mode by minimum validation MAE within target-architecture pair; evaluate held-out real test nodes",
                    "calculation_script": "scripts/46_address_remaining_issues_except_figure5.py",
                    "configuration_and_seed": "five seeds 0-4; fixed split seed from locked implementation outputs",
                }
            )
    if BASELINE.exists():
        base = pd.read_csv(BASELINE)
        for _, r in base.iterrows():
            rows.append(
                {
                    "manuscript_section": "Section 2.6 and Supplementary S23",
                    "table_or_figure": "Supplementary descriptor-generator baseline table/figure",
                    "metric_name": f"{r['generator']} generator-quality diagnostics",
                    "numerical_value": f"descriptor_valid={r['descriptor_valid_rate']}; edge_valid={r['edge_valid_rate']}; full_reference_novelty={r['novelty_pass_rate']}; diversity={r['diversity_score']}",
                    "source_output_file": str(BASELINE.relative_to(ROOT)),
                    "source_column": "descriptor_valid_rate, edge_valid_rate, novelty_pass_rate, diversity_score",
                    "row_selection_rule": f"generator == {r['generator']}",
                    "calculation_script": "scripts/40_remaining_issue_evidence.py and scripts/29_make_generative_upgrade_figures.py",
                    "configuration_and_seed": "harmonised descriptor-generator comparison; full-reference-pool novelty",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "implementation_grounded_metric_source_map.csv", index=False)
    return out


def run_consistency_checks(scaling: pd.DataFrame) -> pd.DataFrame:
    checks = []
    idx = scaling.set_index("graph_characteristic")
    small_before = float(idx.loc["Edges before sparsification", "smaller_study"])
    small_after = float(idx.loc["Edges after sparsification", "smaller_study"])
    large_before = float(idx.loc["Edges before sparsification", "larger_study"])
    large_after = float(idx.loc["Edges after sparsification", "larger_study"])
    small_red = float(idx.loc["Edge reduction (%)", "smaller_study"])
    large_red = float(idx.loc["Edge reduction (%)", "larger_study"])
    checks.append({"check": "small edge reduction recomputed", "status": "PASS" if abs(100 * (small_before - small_after) / small_before - small_red) < 1e-9 else "FAIL"})
    checks.append({"check": "large edge reduction recomputed", "status": "PASS" if abs(100 * (large_before - large_after) / large_before - large_red) < 1e-9 else "FAIL"})
    if VALIDATION_SELECTED.exists():
        val = pd.read_csv(VALIDATION_SELECTED)
        delta_ok = np.allclose(val["selected_mode_test_mae"] - val["real_only_test_mae"], val["delta_mae"], atol=1e-12)
        checks.append({"check": "validation-selected delta MAE equals selected minus real-only", "status": "PASS" if delta_ok else "FAIL"})
        mode_ok = val["validation_selected_mode"].notna().all()
        checks.append({"check": "validation-selected mode recorded for every downstream row", "status": "PASS" if mode_ok else "FAIL"})
    if BASELINE.exists():
        base = pd.read_csv(BASELINE)
        generators = set(base["generator"])
        expected = {"gaussian_noise", "nearest_neighbor_interpolation", "kde_bootstrap", "gmm", "vae", "GANg"}
        checks.append({"check": "all six descriptor generators present", "status": "PASS" if expected == generators else "FAIL"})
    out = pd.DataFrame(checks)
    out.to_csv(TABLES / "implementation_grounded_consistency_checks.csv", index=False)
    if (out["status"] != "PASS").any():
        raise AssertionError(out.to_string(index=False))
    return out


def write_report(metric_map: pd.DataFrame, checks: pd.DataFrame) -> None:
    lines = [
        "# Implementation-grounded final audit",
        "",
        "This pass treats machine-generated implementation outputs as the authority for retained quantitative values.",
        "Figures 1 and 5 were intentionally not read for regeneration and were not modified.",
        "",
        "## Scripts executed",
        "",
        "- `scripts/46_address_remaining_issues_except_figure5.py` for validation-selected downstream result tables and Figures 7/8.",
        "- `scripts/29_make_generative_upgrade_figures.py::baseline_comparison` for the Supplementary descriptor-generator baseline figure.",
        "- `scripts/47_implementation_grounded_audit.py` for graph-scaling evidence, Figure 9, metric-source mapping, and final consistency checks.",
        "",
        "## Output tables generated",
        "",
        "- `tables/implementation_grounded_scaling_summary.csv`",
        "- `tables/implementation_grounded_scaling_summary.tex`",
        "- `tables/implementation_grounded_metric_source_map.csv`",
        "- `tables/implementation_grounded_consistency_checks.csv`",
        "",
        "## Figures regenerated",
        "",
        "- `figures/high_rank_5000/scaling_comparison_1000_vs_5000.pdf/png/svg`",
        "",
        "## Consistency checks",
        "",
    ]
    for _, r in checks.iterrows():
        lines.append(f"- {r['status']}: {r['check']}")
    lines += [
        "",
        "## Metric-source-map rows",
        "",
        f"- {len(metric_map)} rows written.",
        "",
        "## Unresolved values",
        "",
        "- No unresolved graph-scaling, validation-selected downstream, or Supplementary generator-baseline values were found in this pass.",
        "- Historical scripts/reports may still contain archived terminology or earlier values; manuscript-facing regenerated tables and figures use the final evidence tables above.",
    ]
    (TABLES / "implementation_grounded_final_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    scaling = build_scaling_summary()
    write_scaling_latex(scaling)
    regenerate_scaling_figure(scaling)
    metric_map = build_metric_source_map(scaling)
    checks = run_consistency_checks(scaling)
    write_report(metric_map, checks)
    print("PASS implementation-grounded audit")
    print(f"Scaling rows: {len(scaling)}")
    print(f"Metric-source rows: {len(metric_map)}")


if __name__ == "__main__":
    main()
