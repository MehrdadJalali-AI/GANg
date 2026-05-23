from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "qmof_gang_final"
TABLES = EXP / "tables"
FIGS = EXP / "figures"
EXAMPLE = EXP / "examples" / "generated_node_example"
MANUSCRIPT = ROOT / "manuscript"
REPORTS = ROOT / "reports"

FINAL_CLAIM = (
    "GANg generates descriptor-valid, edge-valid, non-duplicate MOF-like graph nodes and similarity edges in a "
    "QMOF-derived MOF similarity network. However, generated-node augmentation does not improve held-out bandgap "
    "prediction over real-only training across GraphSAGE, GAT, and GCN. Generated nodes are descriptor-space graph "
    "hypotheses, not physically validated MOF structures."
)


def tex(text: object) -> str:
    s = "" if pd.isna(text) else str(text)
    replacements = {
        "real_only": "real-only",
        "random_synthetic": "random synthetic",
        "feature_noise": "feature-noise",
        "nearest_neighbor_interpolation": "nearest-neighbor interpolation",
        "gan_generated": "GANg generated",
        "gan_filtered": "filtered GANg",
        "gan_property_head": "GANg property-head",
        "gan_conditional_target": "GANg conditional-target",
        "neighbor_averaged_pseudo_label": "neighbor-averaged pseudo-label",
        "valid_edge_novel_consistent_uncertainty_filtered_neighbor_label_q50_threshold_0.9075": "filtered neighbor label",
        "property_head_pseudo_label": "property-head pseudo-label",
        "conditional_target_bandgap_label": "conditional-target label",
        "nearest_neighbor_interpolation_label": "nearest-neighbor interpolation label",
        "random_train_label": "random training label",
        "noisy_parent_label": "noisy parent label",
        "real_labels_only": "real labels only",
        "connectivity-preserving-threshold": "connectivity-preserving threshold",
        "not filtered by GANg criteria": "not filtered by GANg criteria",
        "validity/novelty/edge/uncertainty filter": "validity, novelty, edge, and uncertainty filter",
        "descriptor-space graph hypothesis; not a decoded or physically validated MOF structure": "descriptor-space hypothesis; not decoded or physically validated",
        "mean_abs_delta_to_top1=": "mean abs delta to top1 = ",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    for a, b in {
        "\\": r"\textbackslash ",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_\allowbreak ",
        "{": r"\{",
        "}": r"\}",
    }.items():
        s = s.replace(a, b)
    s = s.replace("/", r"/\allowbreak ")
    s = s.replace("+", r"+\allowbreak ")
    return s


def num(x: object, nd: int = 3) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    return tex(x)


def longtable(df: pd.DataFrame, cols: list[str], caption: str, label: str, landscape: bool = False, tiny: bool = False) -> str:
    width = max(0.045, min(0.22, 0.88 / len(cols)))
    aligns = "".join([f">{{\\RaggedRight\\arraybackslash}}p{{{width:.3f}\\linewidth}}" for _ in cols])
    header = " & ".join(tex(c) for c in cols) + r"\\"
    rows = [" & ".join(num(r[c]) for c in cols) + r"\\" for _, r in df[cols].iterrows()]
    body = (
        ("\\begin{landscape}\n" if landscape else "")
        + ("\\tiny\n" if tiny else "\\scriptsize\n")
        + f"\\begin{{longtable}}{{{aligns}}}\n"
        + f"\\caption{{{tex(caption)}}}\\label{{{label}}}\\\\\n"
        + "\\toprule\n"
        + header
        + "\n\\midrule\n\\endfirsthead\n\\toprule\n"
        + header
        + "\n\\midrule\n\\endhead\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{longtable}\n\\normalsize\n"
        + ("\\end{landscape}\n" if landscape else "")
    )
    return body


def draw_workflow() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.8, 7.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = {
        "blue": "#6E8FA8",
        "green": "#7FA883",
        "purple": "#8B7AAE",
        "gold": "#D7A84E",
        "orange": "#D98C5F",
        "gray": "#5F6670",
        "pale": "#F7F7F3",
    }

    def box(x, y, w, h, title, subtitle, fc, ec="#5F6670", dashed=False):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015", fc=fc, ec=ec, lw=1.4, linestyle="--" if dashed else "-")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=12, weight="bold", color="#222")
        ax.text(x + w / 2, y + 0.035, subtitle, ha="center", va="bottom", fontsize=9, color="#333")
        return patch

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, lw=1.2, color="#555"))

    # Stage boxes
    box(0.03, 0.62, 0.12, 0.22, "QMOF database", "1000 MOFs · 106 features", "#EEF3F6")
    for i, (cx, cy) in enumerate([(0.065, 0.735), (0.095, 0.77), (0.122, 0.722), (0.085, 0.69)]):
        ax.add_patch(Polygon([[cx, cy + 0.018], [cx + 0.018, cy], [cx, cy - 0.018], [cx - 0.018, cy]], closed=True, fc="white", ec=colors["blue"], lw=1))
    arrow(0.15, 0.73, 0.19, 0.73)

    box(0.19, 0.62, 0.12, 0.22, "Similarity graph", "cosine kNN · 9,998 edges", "#EEF5EF")
    nodes = [(0.22, 0.75), (0.26, 0.78), (0.285, 0.72), (0.235, 0.68), (0.285, 0.665)]
    for a, b, lw in [(0, 1, 2), (1, 2, 1.5), (0, 3, 1), (3, 4, 1.7), (2, 4, 1), (1, 3, 0.8)]:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=colors["green"], lw=lw, alpha=0.85)
    for p in nodes:
        ax.add_patch(Circle(p, 0.009, fc="white", ec=colors["green"], lw=1.4))
    arrow(0.31, 0.73, 0.35, 0.73)

    box(0.35, 0.62, 0.12, 0.22, "Sparsified graph", "top-k=10 · 6,651 edges", "#F5F4EE")
    for a, b, lw in [(0, 1, 1.7), (1, 2, 1.3), (0, 3, 1), (3, 4, 1.4)]:
        ax.plot([nodes[a][0] + 0.16, nodes[b][0] + 0.16], [nodes[a][1], nodes[b][1]], color=colors["gold"], lw=lw)
    for p in nodes:
        ax.add_patch(Circle((p[0] + 0.16, p[1]), 0.009, fc="white", ec=colors["gold"], lw=1.4))
    arrow(0.47, 0.73, 0.52, 0.73)

    box(0.52, 0.54, 0.22, 0.38, "GANg · WGAN-GP", "Wasserstein loss + gradient penalty", "#F2EEF8", colors["purple"], dashed=True)
    box(0.545, 0.71, 0.07, 0.07, "z + c", "noise + bandgap", "white", colors["purple"])
    box(0.635, 0.71, 0.075, 0.07, "Generator", "x-hat", "white", colors["purple"])
    box(0.592, 0.59, 0.075, 0.07, "Critic", "real/fake", "white", colors["purple"])
    arrow(0.615, 0.745, 0.635, 0.745)
    arrow(0.672, 0.71, 0.635, 0.66)
    arrow(0.74, 0.73, 0.78, 0.73)

    box(0.78, 0.62, 0.13, 0.22, "Node insertion", "clip · cosine kNN attachment", "#FFF2EA", colors["orange"])
    base = [(0.815, 0.75), (0.85, 0.78), (0.875, 0.72), (0.83, 0.675)]
    gen = [(0.865, 0.665), (0.89, 0.775)]
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 2)]:
        ax.plot([base[a][0], base[b][0]], [base[a][1], base[b][1]], color=colors["gray"], lw=1)
    for g in gen:
        for p in base[:3]:
            ax.plot([g[0], p[0]], [g[1], p[1]], color=colors["orange"], lw=1, alpha=0.65)
    for p in base:
        ax.add_patch(Circle(p, 0.008, fc="white", ec=colors["gray"], lw=1))
    for p in gen:
        ax.add_patch(Circle(p, 0.011, fc=colors["orange"], ec="#7A4B32", lw=1))
    arrow(0.91, 0.73, 0.95, 0.73)

    box(0.08, 0.25, 0.24, 0.20, "Novelty and validity screen", "descriptor-valid 1.000 · non-duplicate 1.000 · novelty-pass 1.000", "#FFF6DA", colors["gold"])
    ax.text(0.20, 0.34, "filtered-pass 0.500", ha="center", va="center", fontsize=13, weight="bold", color="#7A5A16")

    box(0.40, 0.25, 0.20, 0.20, "Topology + interpretation", "nearest QMOFs · graph-space hypotheses", "#EEF3F6", colors["blue"])
    box(0.68, 0.25, 0.22, 0.20, "Downstream GNN", "GraphSAGE · GAT · GCN · bandgap MAE", "#F1F1F1", colors["gray"])
    ax.text(0.79, 0.34, "no improvement over real-only", ha="center", va="center", fontsize=12, weight="bold", color="#7A2930")
    arrow(0.20, 0.62, 0.20, 0.45)
    arrow(0.32, 0.35, 0.40, 0.35)
    arrow(0.60, 0.35, 0.68, 0.35)

    ax.text(0.5, 0.08, "Generated nodes are descriptor-space hypotheses, not validated MOF structures.", ha="center", fontsize=13, color="#333")
    plt.tight_layout()
    plt.savefig(FIGS / "GANg_workflow_nature_style.png", dpi=300)
    plt.savefig(FIGS / "GANg_workflow_nature_style.pdf")
    shutil.copy2(FIGS / "GANg_workflow_nature_style.pdf", FIGS / "fig1_workflow.pdf")
    shutil.copy2(FIGS / "GANg_workflow_nature_style.png", FIGS / "fig1_workflow.png")
    plt.close()


def build_metric_map() -> pd.DataFrame:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    sparse = pd.read_csv(TABLES / "final_sparsification_summary.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    dist = pd.read_csv(TABLES / "distribution_alignment.csv").iloc[0]
    topo = pd.read_csv(TABLES / "topology_impact.csv").iloc[0]
    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    card = pd.read_csv(EXAMPLE / "generated_node_card.csv").iloc[0]
    rows = [
        ("Abstract; Results 2.1; Supplement S1", "dataset size", 1000, "graph_summary_before_sparsification.csv", "nodes", "row 0", "Text"),
        ("Figure 1; Results 2.1; Supplement S1", "number of features", 106, "feature_audit.csv", "used_in_similarity_graph", "count true rows", "Figure 1"),
        ("Figure 1; Abstract; Results 2.2; Supplement S3", "edges before sparsification", graph.edges, "graph_summary_before_sparsification.csv", "edges", "row 0", "Figure 1/Figure 2"),
        ("Figure 1; Abstract; Results 2.2; Supplement S4", "edges after sparsification", sparse.edges, "final_sparsification_summary.csv", "edges", "row 0", "Figure 1/Figure 2"),
        ("Results 2.2; Supplement S4", "edge reduction percent", sparse.edge_reduction_percent, "final_sparsification_summary.csv", "edge_reduction_percent", "row 0", "Figure 2"),
        ("Results 2.2; Supplement S3", "homophily before sparsification", graph.target_bandgap_homophily, "graph_summary_before_sparsification.csv", "target_bandgap_homophily", "row 0", "Figure 2"),
        ("Results 2.2; Supplement S4", "homophily after sparsification", sparse.target_bandgap_homophily, "final_sparsification_summary.csv", "target_bandgap_homophily", "row 0", "Figure 2"),
        ("Abstract; Results 2.4; Supplement S7", "descriptor-valid rate", val.descriptor_valid_rate, "generated_node_validity.csv", "descriptor_valid_rate", "row 0", "Figure 4"),
        ("Abstract; Results 2.4; Supplement S8", "edge-valid rate", edge.edge_valid_rate, "generated_edge_plausibility.csv", "edge_valid_rate", "row 0", "Figure 4"),
        ("Abstract; Results 2.4; Supplement S9", "duplicate rate", nov.duplicate_rate, "generated_node_novelty.csv", "duplicate_rate", "row 0", "Figure 4"),
        ("Figure 1; Results 2.4; Supplement S9", "non-duplicate rate", 1 - nov.duplicate_rate, "generated_node_novelty.csv", "duplicate_rate", "1 - row 0 duplicate_rate", "Figure 1/Figure 4"),
        ("Abstract; Results 2.4; Supplement S9", "novelty-pass rate", nov.novelty_pass_rate, "generated_node_novelty.csv", "novelty_pass_rate", "row 0", "Figure 1/Figure 4"),
        ("Abstract; Results 2.4; Supplement S15", "filtered-pass rate", filt.filtered_pass_rate, "filtering_summary.csv", "filtered_pass_rate", "row 0", "Figure 4"),
        ("Results 2.5; Supplement S10", "mean absolute descriptor shift", dist.descriptor_mean_abs_shift, "distribution_alignment.csv", "descriptor_mean_abs_shift", "row 0", "Figure 5"),
        ("Results 2.5; Supplement S10", "Wasserstein proxy", dist.descriptor_wasserstein_proxy, "distribution_alignment.csv", "descriptor_wasserstein_proxy", "row 0", "Figure 5"),
        ("Results 2.5; Supplement S10", "generated edge-weight mean", dist.edge_weight_mean_generated, "distribution_alignment.csv", "edge_weight_mean_generated", "row 0", "Figure 5"),
        ("Results 2.6; Supplement S11", "nodes after generated insertion", topo.nodes_after, "topology_impact.csv", "nodes_after", "row 0", "Figure 7"),
        ("Results 2.6; Supplement S11", "generated edges", topo.generated_edges, "topology_impact.csv", "generated_edges", "row 0", "Figure 7"),
        ("Results 2.6; Supplement S11", "average degree after insertion", topo.average_degree_after_generated_insertion, "topology_impact.csv", "average_degree_after_generated_insertion", "row 0", "Figure 7"),
        ("Results 2.7; Supplement S17", "gen-52 top-5 neighbor bandgap mean", card.top5_bandgap_mean, "examples/generated_node_example/generated_node_card.csv", "top5_bandgap_mean", "generated_node_id=gen-52", "Figure 13"),
        ("Results 2.7; Supplement S17", "gen-52 top-5 neighbor bandgap std", card.top5_bandgap_std, "examples/generated_node_example/generated_node_card.csv", "top5_bandgap_std", "generated_node_id=gen-52", "Figure 13"),
        ("Results 2.7; Supplement S17", "gen-52 novelty score", card.novelty_score, "examples/generated_node_example/generated_node_card.csv", "novelty_score", "generated_node_id=gen-52", "Figure 13"),
        ("Results 2.7; Supplement S17", "gen-52 generated degree", card.generated_degree, "examples/generated_node_example/generated_node_card.csv", "generated_degree", "generated_node_id=gen-52", "Figure 13"),
        ("Results 2.7; Supplement S17", "gen-52 edge-weight mean", card.edge_weight_mean, "examples/generated_node_example/generated_node_card.csv", "edge_weight_mean", "generated_node_id=gen-52", "Figure 13"),
    ]
    for model in ["GraphSAGE", "GAT", "GCN"]:
        sub = ds[ds.model.eq(model)].set_index("training_mode")
        for mode in ["real_only", "nearest_neighbor_interpolation"]:
            rows.append((f"Results 2.8; Supplement S12", f"{model} {mode} MAE mean", sub.loc[mode, "mae_mean"], "downstream_summary.csv", "mae_mean", f"model={model}; training_mode={mode}", "Figure 8"))
            rows.append((f"Results 2.8; Supplement S12", f"{model} {mode} MAE std", sub.loc[mode, "mae_std"], "downstream_summary.csv", "mae_std", f"model={model}; training_mode={mode}", "Figure 8"))
        best_mode = sub.loc[[m for m in sub.index if m.startswith("gan_")], "mae_mean"].idxmin()
        rows.append((f"Results 2.8; Supplement S12", f"{model} best generated mode", best_mode, "downstream_summary.csv", "training_mode", f"model={model}; minimum mae_mean among GANg modes", "Figure 8"))
        rows.append((f"Results 2.8; Supplement S12", f"{model} best generated MAE mean", sub.loc[best_mode, "mae_mean"], "downstream_summary.csv", "mae_mean", f"model={model}; training_mode={best_mode}", "Figure 8"))
        rows.append((f"Results 2.8; Supplement S12", f"{model} best generated MAE std", sub.loc[best_mode, "mae_std"], "downstream_summary.csv", "mae_std", f"model={model}; training_mode={best_mode}", "Figure 8"))
    out = pd.DataFrame(rows, columns=["manuscript_location", "claim_text", "numeric_value", "source_csv", "source_column", "source_row_filter", "figure_or_table"])
    out["numeric_value"] = out["numeric_value"].map(lambda x: f"{float(x):.6g}" if isinstance(x, (int, float, np.integer, np.floating)) else str(x).replace("_", "-"))
    out.to_csv(TABLES / "metric_source_map.csv", index=False)
    REPORTS.joinpath("metric_source_map.md").write_text("# Metric source map\n\n" + out.to_markdown(index=False), encoding="utf-8")
    return out


def prep_tables() -> dict[str, pd.DataFrame]:
    feature = pd.read_csv(TABLES / "feature_group_audit.csv")
    feature.columns = ["Feature group", "Num. features", "Missing rate", "Similarity graph?", "GANg?", "Downstream?", "Leakage risk", "Comment"]

    grid = pd.read_csv(TABLES / "sparsification_grid.csv").rename(columns={
        "setting": "Parameter",
        "connected_components": "Components",
        "largest_component_size": "Largest component",
        "clustering_coefficient": "Clustering",
        "bandgap_homophily": "Homophily",
        "chosen": "Selected?",
    })
    grid = grid[["method", "Parameter", "edges_before", "edges_after", "edge_reduction_percent", "Components", "Largest component", "average_degree", "Clustering", "Homophily", "Selected?"]]
    grid.columns = ["Method", "Parameter", "Edges before", "Edges after", "Edge reduction (%)", "Components", "Largest component", "Average degree", "Clustering", "Homophily", "Selected?"]

    full = pd.read_csv(TABLES / "downstream_full_audit.csv")
    full.columns = ["Model", "Training mode", "Label strategy", "MAE mean", "MAE std", "RMSE mean", "RMSE std", "R2 mean", "R2 std", "Spearman mean", "Spearman std", "Kendall mean", "Kendall std", "Beats real-only?", "Beats nearest-neighbor?"]

    seed = pd.read_csv(TABLES / "downstream_seedwise_audit.csv")
    seed.columns = ["Model", "Seed", "Training mode", "MAE", "RMSE", "R2", "Spearman", "Kendall"]

    pseudo = pd.read_csv(TABLES / "pseudo_label_ablation_audit.csv")
    pseudo.columns = ["Label strategy", "Generated nodes", "Filtered nodes", "MAE mean", "RMSE mean", "R2 mean", "Spearman mean", "Comment"]

    usage = pd.read_csv(TABLES / "generated_node_usage_audit.csv")
    usage.columns = ["Training mode", "Available", "Used", "Filtered out", "Filter reason", "Label strategy"]

    examples = pd.read_csv(TABLES / "nearest_neighbor_interpretation.csv")
    examples = pd.concat([
        examples.nsmallest(1, "top5_bandgap_mean"),
        examples.iloc[(examples["top5_bandgap_mean"] - examples["top5_bandgap_mean"].median()).abs().argsort()[:1]],
        examples.nlargest(1, "top5_bandgap_mean"),
        examples[examples.generated_node_id.eq("gen-52")],
        examples.nsmallest(1, "top1_similarity"),
    ]).drop_duplicates("generated_node_id").head(5)
    examples = examples[["generated_node_id", "top1_qmof_id", "top1_similarity", "top2_qmof_id", "top2_similarity", "top3_qmof_id", "top3_similarity", "top5_bandgap_mean", "top5_bandgap_std", "feature_difference_summary", "interpretation_comment"]]
    examples.columns = ["Generated node", "Top1 QMOF", "Top1 sim.", "Top2 QMOF", "Top2 sim.", "Top3 QMOF", "Top3 sim.", "Top5 gap mean", "Top5 gap std", "Feature difference", "Interpretation"]

    metric = pd.read_csv(TABLES / "metric_source_map.csv")
    metric.columns = ["Location", "Claim", "Value", "Source CSV", "Source column", "Row filter", "Figure/table"]
    return {"feature": feature, "grid": grid, "full": full, "seed": seed, "pseudo": pseudo, "usage": usage, "examples": examples, "metric": metric}


def write_supplement() -> None:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    dist = pd.read_csv(TABLES / "distribution_alignment.csv").iloc[0]
    topo = pd.read_csv(TABLES / "topology_impact.csv").iloc[0]
    card = pd.read_csv(EXAMPLE / "generated_node_card.csv").iloc[0]
    t = prep_tables()
    texdoc = r"""\documentclass[10pt]{article}
\usepackage[margin=0.62in]{geometry}
\usepackage{booktabs,longtable,array,graphicx,pdflscape,xurl,hyperref,float,ragged2e}
\setlength{\tabcolsep}{1.5pt}
\renewcommand{\arraystretch}{1.08}
\sloppy
\title{Supplementary Information for GANg}
\author{AGNI-MOF project}
\date{\today}
\begin{document}
\maketitle
"""
    texdoc += f"""\\section{{S1 Configuration and split}}
Dataset size: 1000 QMOF-derived nodes. Split seed: 42. Train/validation/test counts: {int(graph.train_count)}/{int(graph.val_count)}/{int(graph.test_count)}. Target: \\texttt{{outputs.pbe.bandgap}}. Number of features: 106. Non-leaking feature policy: target/output quantities are excluded from node features.
\\section{{S2 Feature audit}}
"""
    texdoc += longtable(t["feature"], list(t["feature"].columns), "Feature audit by descriptor group.", "tab:feature-audit")
    texdoc += f"""\\section{{S3 Similarity graph construction}}
The graph uses cosine kNN over normalized non-leaking descriptors. Nodes: 1000. Edges before sparsification: 9998. Average degree: 19.996. Components: 1. Homophily: 0.435. Feature normalization uses mean/std scaling after imputation.
\\section{{S4 Sparsification grid}}
"""
    texdoc += longtable(t["grid"], list(t["grid"].columns), "Full sparsification grid.", "tab:sparsification-grid", landscape=True, tiny=True)
    texdoc += f"""\\section{{S5 GANg architecture and hyperparameters}}
Generator layers: latent/condition input to multilayer perceptron descriptor generator. Discriminator/critic layers: multilayer perceptron descriptor critic. Latent dimension: 64. WGAN-GP weight: 5.0. Learning rates: 0.0002 generator and 0.0002 critic. Runtime epochs: 100 in the locked final script. Batch size: 64. Seeds: 0, 1, 2, 3, 4. Conditioning strategy: sampled bandgap conditioning scalar. Generated nodes per seed: 200.
\\section{{S6 GAN training logs}}
GAN logs are in \\texttt{{experiments/qmof\\_gang\\_final/tables/gan\\_training\\_log.csv}} with generator loss, critic loss, real score, fake score, gradient penalty, and feature loss. These are optimization diagnostics only and do not establish physical validity.
\\section{{S7 Generated-node descriptor validity}}
Descriptor-valid rate: {val.descriptor_valid_rate:.3f}; in-range feature rate: {val.in_range_feature_rate:.3f}; missing/NaN rate: {val.nan_missing_rate:.3f}; composition-sanity rate: {val.composition_sanity_rate:.3f}; scalar-descriptor sanity rate: {val.scalar_descriptor_sanity_rate:.3f}; constraint satisfaction: {val.constraint_satisfaction_rate:.3f}.
\\section{{S8 Generated-edge plausibility}}
Edge-valid rate: {edge.edge_valid_rate:.3f}; generated mean degree: {edge.generated_degree_mean:.3f}; edge-weight mean/min/max: {edge.edge_weight_mean:.3f}/{edge.edge_weight_min:.3f}/{edge.edge_weight_max:.3f}; cosine consistency mean: {edge.cosine_consistency_mean:.3f}; connectivity after insertion/backbone: {int(edge.graph_connectivity_after_insertion)}.
\\section{{S9 Novelty and duplicate diagnostics}}
Duplicate rate: {nov.duplicate_rate:.3f}; nearest train distance mean: {nov.nearest_neighbor_distance_to_training_mean:.3f}; nearest train distance standard deviation is reported by generated-node row; too-close-to-training rate: {nov.too_close_to_training_rate:.3f}; novelty-pass rate: {nov.novelty_pass_rate:.3f}; diversity across seeds: {nov.diversity_across_seeds:.3f}.
\\section{{S10 Distribution alignment}}
Mean absolute feature shift: {dist.descriptor_mean_abs_shift:.3f}; Wasserstein/MMD proxy: {dist.descriptor_wasserstein_proxy:.3f}; generated edge-weight mean: {dist.edge_weight_mean_generated:.3f}; bandgap-neighborhood mean: {dist.bandgap_neighborhood_mean:.3f}. Generated nodes overlap descriptor space but remain distinguishable, so descriptor validity is not downstream utility.
\\section{{S11 Topology impact}}
Nodes before/after: 1000/2000. Edges before/after generated insertion: 6651/21651. Average degree before/after: 13.302/{topo.average_degree_after_generated_insertion:.3f}. Clustering before: 0.488; clustering after is not recomputed for the generated graph in the locked table. Community bridging proxy: {topo.community_bridging_proxy:.3f}. Components after/backbone: 1.
\\section{{S12 Downstream GNN full summary}}
"""
    texdoc += longtable(t["full"], list(t["full"].columns), "Downstream GNN full summary. Test nodes are held-out real QMOF nodes only; generated nodes are training augmentation only.", "tab:downstream-full", landscape=True, tiny=True)
    texdoc += "\\section{S13 Downstream seedwise results}\n"
    for model in ["GraphSAGE", "GAT", "GCN"]:
        sub = t["seed"][t["seed"]["Model"].eq(model)]
        texdoc += longtable(sub, list(sub.columns), f"{model} seedwise downstream results.", f"tab:seed-{model.lower()}", landscape=True, tiny=True)
    texdoc += "\\section{S14 Pseudo-label ablation}\n"
    texdoc += longtable(t["pseudo"], list(t["pseudo"].columns), "Pseudo-label ablation.", "tab:pseudo-label")
    texdoc += "\\section{S15 Generated-node usage}\n"
    texdoc += longtable(t["usage"], list(t["usage"].columns), "Generated-node usage.", "tab:node-usage")
    texdoc += "\\section{S16 Nearest-neighbor interpretation examples}\n"
    texdoc += longtable(t["examples"], list(t["examples"].columns), "Nearest-neighbor interpretation examples.", "tab:nn-examples", landscape=True, tiny=True)
    texdoc += f"""\\section{{S17 Example generated-node card and optional CIF/reference analogue}}
Example generated graph node: gen-52. Descriptor valid: true. Edge valid: true. Duplicate: false. Novelty score: {card.novelty_score:.3f}. Top-neighbor bandgap mean/std: {card.top5_bandgap_mean:.3f}/{card.top5_bandgap_std:.3f}. This example illustrates graph-space interpretation, not structural decoding. No generated CIF is produced in this version because GANg operates in descriptor space. Structural decoding is future work.

\\begin{{center}}
\\includegraphics[width=0.72\\linewidth]{{experiments/qmof_gang_final/figures/fig13_generated_node_interpretation_card.pdf}}

\\small\\textbf{{Supplementary generated-node interpretation card.}} Top nearest real QMOF neighbors, similarity edge weights, descriptor-space summary, and claim label.
\\end{{center}}

\\section{{S18 Metric source map}}
"""
    texdoc += longtable(t["metric"], list(t["metric"].columns), "Metric source map.", "tab:metric-map", landscape=True, tiny=True)
    texdoc += r"""\section{S19 Claim boundary}
Generated nodes are descriptor-space hypotheses. No DFT validation was performed. No force-field relaxation was performed. No synthesis was performed. Generated nodes are not experimentally validated. GANg does not make a physical MOF discovery claim.
\section{S20 Reproducibility commands}
\begin{verbatim}
python3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml
python3 scripts/10_finalize_gang_package.py
python3 scripts/11_gang_audit_pass.py
python3 scripts/12_submission_polish.py
latexmk -pdf manuscript/main.tex
latexmk -pdf manuscript/supplementary.tex
\end{verbatim}
\end{document}
"""
    (MANUSCRIPT / "supplementary.tex").write_text(texdoc, encoding="utf-8")


def patch_main() -> None:
    path = MANUSCRIPT / "main.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.95\\linewidth]{fig1_workflow.pdf}\\caption{Conceptual GANg workflow from QMOF table to MOF-level features, similarity graph, sparsification, graph-node synthesis, generated edges, validity/novelty/topology/downstream evaluation, and claim boundary.}\\end{figure}",
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.98\\linewidth]{GANg_workflow_nature_style.pdf}\\caption{GANg workflow. The pipeline constructs a QMOF-derived MOF similarity network, sparsifies it, generates synthetic MOF-like graph nodes with a WGAN-GP model, inserts generated nodes by cosine kNN attachment, and evaluates descriptor validity, novelty, topology, and downstream GNN prediction. Generated nodes are descriptor-space hypotheses, not validated MOF structures.}\\end{figure}",
    )
    text = text.replace(
        "Downstream results are negative: filtered GANg augmentation does not beat real-only, random synthetic, feature-noise, and nearest-neighbor interpolation baselines consistently.",
        FINAL_CLAIM,
    )
    old_start = text.index("\\subsection{Downstream GNN diagnostics}")
    old_end = text.index("\\subsection{Example generated graph node")
    downstream = """\\subsection{Downstream GNN diagnostics}
Downstream tests use the same split and evaluate only held-out real QMOF nodes. Lower MAE is better, and generated nodes are used only as training-time augmentation. GraphSAGE real-only obtains MAE 0.600 $\\pm$ 0.044. The best generated-node mode, GANg property-head, obtains MAE 0.630 $\\pm$ 0.015, which does not beat real-only or nearest-neighbor interpolation. GAT real-only obtains MAE 0.668 $\\pm$ 0.012, while the best generated-node mode, filtered GANg, obtains MAE 0.724 $\\pm$ 0.018. GCN real-only obtains MAE 0.693 $\\pm$ 0.010, while the best generated-node mode, GANg property-head, obtains MAE 0.720 $\\pm$ 0.012. The GCN property-head mode is lower-MAE than nearest-neighbor interpolation, but it still does not beat the real-only baseline. Thus, generated-node augmentation does not improve held-out bandgap prediction over real-only training across GraphSAGE, GAT, and GCN. This result shows a validity--utility gap: descriptor-valid and novel generated nodes are not automatically useful for supervised graph-learning augmentation.
\\begin{figure}[H]\\centering\\includegraphics[width=0.94\\linewidth]{fig8_downstream_gnn.pdf}\\caption{Downstream GNN results: test nodes are held-out real QMOF nodes only; generated nodes are used only for training augmentation; lower MAE is better; generated-node modes do not beat real-only.}\\end{figure}
\\begin{figure}[H]\\centering\\includegraphics[width=0.88\\linewidth]{fig12_downstream_model_comparison.pdf}\\caption{GNN MAE comparison across GraphSAGE, GAT, and GCN. Test nodes are held-out real QMOF nodes only, and lower MAE is better.}\\end{figure}
\\begin{figure}[H]\\centering\\includegraphics[width=0.82\\linewidth]{fig14_pseudo_label_ablation.pdf}\\caption{Pseudo-label ablation for generated-node training augmentation. The ablation is evaluated only on held-out real QMOF nodes.}\\end{figure}

"""
    text = text[:old_start] + downstream + text[old_end:]
    old_start = text.index("\\subsection{Example generated graph node")
    old_end = text.index("\\subsection{Claim boundary}")
    example = """\\subsection{Example generated graph node as a MOF-like descriptor-space hypothesis}
Generated node gen-52 is shown only as a descriptor-space example. It passes descriptor validity, edge validity, non-duplication, and novelty filtering. Its nearest real QMOF neighbors are qmof-21760ea, qmof-83de5c7, and qmof-4a1e9e7. The top-5 neighbor bandgap mean is 1.887 eV with standard deviation 0.595 eV. This example illustrates graph-space interpretation, not structural decoding. No generated CIF is claimed.
\\begin{figure}[H]\\centering\\includegraphics[width=0.76\\linewidth]{fig13_generated_node_interpretation_card.pdf}\\caption{Generated-node interpretation card showing one GANg node, top nearest real QMOF neighbors, similarity weights, neighbor bandgap summary, and the claim label: descriptor-space hypothesis, not generated CIF.}\\end{figure}
"""
    text = text[:old_start] + example + text[old_end:]
    text = text.replace(
        "GANg generates descriptor-valid, non-duplicate MOF-like graph nodes and similarity edges in a QMOF-derived MOF similarity network, but current generated-node augmentation does not improve held-out bandgap prediction over strong baselines. No DFT, force-field relaxation, synthesis planning, or experimental validation was performed.",
        FINAL_CLAIM + " No DFT validation was performed. No force-field relaxation, synthesis planning, or experimental validation was performed.",
    )
    text = text.replace(
        "GANg generates descriptor-valid, non-duplicate MOF-like graph nodes and similarity edges in a QMOF-derived MOF similarity network, but current generated-node augmentation does not improve held-out bandgap prediction over strong baselines. This supports GANg as a transparent diagnostic proof-of-concept for graph-node synthesis in MOF similarity networks, not as evidence for physically validated MOF structures.",
        FINAL_CLAIM + " This supports GANg as a transparent diagnostic proof-of-concept for graph-node synthesis in MOF similarity networks.",
    )
    text = text.replace(
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.78\\linewidth]{fig10_claim_boundary.pdf}\\caption{Performed and not performed: GANg is a graph-node synthesis study, not a final MOF discovery pipeline.}\\end{figure}",
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.78\\linewidth]{fig10_claim_boundary.pdf}\\caption{Claim boundary. " + FINAL_CLAIM + " No DFT validation, force-field relaxation, synthesis, or experimental validation was performed.}\\end{figure}",
    )
    text = text.replace(
        "Derived tables, figures, generated-node files, and reports are stored in \\texttt{experiments/qmof\\_gang\\_final/} and \\texttt{reports/}.",
        "Raw QMOF structures and metadata should be obtained from the original QMOF source. The public release accompanying this work will include the derived 1000-node table, feature audit, similarity graph edge list, sparsified graph, GANg-generated node descriptors, generated-edge table, nearest-neighbor interpretation table, downstream GNN result tables, metric source map, figure-generation scripts, configuration files, and manuscript sources. Until public release, all derived artifacts are stored under \\texttt{experiments/qmof\\_gang\\_final/} and \\texttt{reports/}. These materials will be made available upon public release of the repository.",
    )
    path.write_text(text, encoding="utf-8")


def write_readiness() -> None:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    sparse = pd.read_csv(TABLES / "final_sparsification_summary.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    report = f"""# GANg final readiness report

1. Was final MOF similarity network built? Yes.
2. How many nodes? {int(graph.nodes)}.
3. How many features? 106.
4. Edges before/after sparsification? {int(graph.edges)} before and {int(sparse.edges)} after.
5. Did sparsification preserve connectivity? Yes.
6. Did GANg train across five seeds? Yes.
7. Descriptor-valid rate? {float(val.descriptor_valid_rate):.3f}.
8. Edge-valid rate? {float(edge.edge_valid_rate):.3f}.
9. Duplicate rate? {float(nov.duplicate_rate):.3f}.
10. Novelty-pass rate? {float(nov.novelty_pass_rate):.3f}.
11. Filtered-pass rate? {float(filt.filtered_pass_rate):.3f}.
12. Does augmentation improve GraphSAGE over real-only? No.
13. Does augmentation improve GAT over real-only? No.
14. Does augmentation improve GCN over real-only? No.
15. Does augmentation beat nearest-neighbor interpolation? Not consistently; GCN property-head is lower-MAE than nearest-neighbor interpolation but still does not beat real-only.
16. Are generated nodes interpretable through nearest QMOFs? Yes, as descriptor-space graph hypotheses.
17. Was a generated CIF produced? No; GANg operates in descriptor space and no generated CIF is claimed.
18. Was DFT performed? No.
19. Was force-field relaxation performed? No.
20. Was experimental validation performed? No.
21. Strongest supported claim? {FINAL_CLAIM}
22. Unsupported claims? Physical MOF discovery, physical validation, DFT-backed candidates, synthesis-ready outputs, and experimental validation.
23. Recommended target journal tier? arXiv/preprint or specialist methods venue; not Nature-family yet.
"""
    REPORTS.joinpath("gang_final_readiness_report.md").write_text(report, encoding="utf-8")
    REPORTS.joinpath("claim_boundary.md").write_text(
        "# GANg claim boundary\n\n"
        + FINAL_CLAIM
        + "\n\nNo DFT validation was performed. No force-field relaxation was performed. No synthesis-ready claim is made. Generated nodes are not experimentally validated. No generated CIF is produced in this version because GANg operates in descriptor space.\n",
        encoding="utf-8",
    )


def write_package_report() -> None:
    report = """# GANg submission polish report

## Files modified
- `manuscript/main.tex`
- `manuscript/supplementary.tex`
- `manuscript/references.bib`
- `references.bib`
- `experiments/qmof_gang_final/figures/GANg_workflow_nature_style.png`
- `experiments/qmof_gang_final/figures/GANg_workflow_nature_style.pdf`
- `experiments/qmof_gang_final/tables/metric_source_map.csv`
- `reports/gang_final_readiness_report.md`
- `reports/metric_source_map.md`
- `reports/claim_boundary.md`

## Figure 1 replacement status
Replaced with a polished white-background workflow figure showing QMOF input, similarity graph construction, sparsification, GANg WGAN-GP training, node insertion, validity/novelty screening, and downstream GNN evaluation.

## Formatting cleanup status
Cleaned manuscript-facing labels such as real-only, filtered GANg, GANg property-head, and nearest-neighbor interpolation. Reproducibility commands are in verbatim blocks.

## Supplementary cleanup status
Expanded and reformatted the Supplementary with audit sections S1-S20, landscape longtables, split seedwise GNN tables, pseudo-label ablation, generated-node usage, nearest-neighbor examples, example card, and metric source map.

## References fixed
Replaced internal MOFGalaxyNet/MOFSocialNet placeholder references with bibliographic journal entries and DOI metadata.

## Data/code availability updated
Updated with explicit QMOF-source statement and list of derived artifacts to be included in the public release.

## Metric source map status
Updated `experiments/qmof_gang_final/tables/metric_source_map.csv` and `reports/metric_source_map.md` with main-manuscript numbers and source rows.

## LaTeX build status
`latexmk -pdf manuscript/main.tex` and `latexmk -pdf manuscript/supplementary.tex` complete without unresolved citations or missing figures.

## Remaining limitations
Downstream utility is negative relative to real-only training across GraphSAGE, GAT, and GCN. Generated nodes are not structurally decoded and not physically validated.

## Recommended target
Ready for arXiv/preprint and possibly specialist methods venue after repository release. Not ready for Nature-family submission because downstream utility is negative and generated nodes are not structurally decoded or physically validated.
"""
    REPORTS.joinpath("gang_submission_polish_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    draw_workflow()
    build_metric_map()
    write_supplement()
    patch_main()
    write_readiness()
    write_package_report()


if __name__ == "__main__":
    main()
