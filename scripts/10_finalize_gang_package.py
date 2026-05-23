from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "qmof_galaxy_final"
EXP = ROOT / "experiments" / "qmof_gang_final"
TABLES = EXP / "tables"
FIGS = EXP / "figures"
GEN = EXP / "generated_nodes"


def copy_locked_artifacts() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing locked experiment directory: {SRC}")
    for sub in ["tables", "figures", "generated_nodes", "graphs", "models", "logs"]:
        src = SRC / sub
        dst = EXP / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)


def build_filtering_outputs() -> dict[str, float]:
    nodes = pd.read_csv(TABLES / "generated_nodes.csv")
    edges = pd.read_csv(TABLES / "generated_edges.csv")
    validity = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edgep = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    novelty = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]

    degree = edges.groupby("generated_node_id").size()
    weight_consistency = edges.groupby("generated_node_id")["edge_weight"].mean()
    uncertainty_threshold = nodes["uncertainty_score"].median()
    degree_min, degree_max = 1, 25
    nearest = nodes["nearest_train_similarity"]
    keep = (
        nodes["descriptor_valid"].astype(bool)
        & nodes["constraint_satisfied"].astype(bool)
        & (~nodes["duplicate"].astype(bool))
        & (nodes["novelty_score"] > 0.01)
        & (nearest >= 0.20)
        & (nearest <= 0.99)
        & (nodes["generated_node_id"].map(weight_consistency) >= -1.0)
        & (nodes["generated_node_id"].map(degree).between(degree_min, degree_max))
        & (nodes["uncertainty_score"] <= uncertainty_threshold)
    )
    filtered = nodes.loc[keep].copy()
    filtered_edges = edges[edges["generated_node_id"].isin(filtered["generated_node_id"])].copy()

    nodes.to_csv(GEN / "generated_nodes_all.csv", index=False)
    filtered.to_csv(GEN / "generated_nodes_filtered.csv", index=False)
    filtered_edges.to_csv(GEN / "generated_edges_filtered.csv", index=False)

    usage = pd.read_csv(TABLES / "generated_node_usage.csv")
    downstream_use_count = int(usage.loc[usage["training_mode"].eq("gan_filtered"), "generated_nodes_used"].max())
    summary = {
        "generated_nodes_before_filtering": int(len(nodes)),
        "generated_nodes_retained": int(len(filtered)),
        "descriptor_valid_rate": float(validity["descriptor_valid_rate"]),
        "edge_valid_rate": float(edgep["edge_valid_rate"]),
        "duplicate_rate": float(novelty["duplicate_rate"]),
        "novelty_pass_rate": float(novelty["novelty_pass_rate"]),
        "filtered_pass_rate": float(len(filtered) / max(len(nodes), 1)),
        "downstream_use_count": downstream_use_count,
        "uncertainty_threshold": float(uncertainty_threshold),
    }
    pd.DataFrame([summary]).to_csv(TABLES / "filtering_summary.csv", index=False)
    return summary


def expand_nearest_neighbor_table() -> None:
    nn = pd.read_csv(TABLES / "nearest_neighbor_interpretation.csv")
    nodes = pd.read_csv(TABLES / "generated_nodes.csv").set_index("generated_node_id")
    rows = []
    for r in nn.itertuples(index=False):
        ids = str(r.top_5_nearest_qmof_ids).split(";")
        sims = [float(x) for x in str(r.similarity_scores).split(";")]
        gid = r.generated_node_id
        rows.append(
            {
                "generated_node_id": gid,
                "top1_qmof_id": ids[0] if len(ids) > 0 else "",
                "top1_similarity": sims[0] if len(sims) > 0 else np.nan,
                "top2_qmof_id": ids[1] if len(ids) > 1 else "",
                "top2_similarity": sims[1] if len(sims) > 1 else np.nan,
                "top3_qmof_id": ids[2] if len(ids) > 2 else "",
                "top3_similarity": sims[2] if len(sims) > 2 else np.nan,
                "top5_bandgap_mean": float(r.neighbor_bandgap_mean),
                "top5_bandgap_std": float(r.neighbor_bandgap_std),
                "nearest_train_distance": float(1.0 - nodes.loc[gid, "nearest_train_similarity"]),
                "nearest_test_distance": "",
                "feature_difference_summary": r.feature_difference_summary,
                "interpretation_comment": "descriptor-space graph hypothesis; not a decoded or physically validated MOF structure",
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "nearest_neighbor_interpretation.csv", index=False)


def alias_requested_downstream_tables() -> None:
    shutil.copy2(TABLES / "downstream_gnn_summary.csv", TABLES / "downstream_summary.csv")
    shutil.copy2(TABLES / "downstream_gnn_seedwise.csv", TABLES / "downstream_seedwise.csv")


def final_claim(ds: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    flags = {}
    for model in ["GraphSAGE", "GAT", "GCN"]:
        sub = ds[ds.model == model].set_index("training_mode")
        flags[model] = bool(
            sub.loc["gan_filtered", "mae_mean"]
            < min(
                sub.loc["real_only", "mae_mean"],
                sub.loc["random_synthetic", "mae_mean"],
                sub.loc["feature_noise", "mae_mean"],
                sub.loc["nearest_neighbor_interpolation", "mae_mean"],
            )
        )
    if all(flags.values()):
        claim = (
            "GANg generates descriptor-valid, non-duplicate MOF-like graph nodes and similarity edges in a "
            "QMOF-derived MOF similarity network, and filtered generated-node augmentation improves held-out "
            "MOF-level bandgap prediction over real-only and synthetic-node baselines."
        )
    elif any(flags.values()):
        winners = ", ".join(k for k, v in flags.items() if v)
        claim = (
            "GANg generates descriptor-valid, non-duplicate MOF-like graph nodes and similarity edges in a "
            f"QMOF-derived MOF similarity network. Downstream benefits are model-dependent and observed only for {winners}, "
            "not consistently across GraphSAGE, GAT, and GCN."
        )
    else:
        claim = (
            "GANg generates descriptor-valid, non-duplicate MOF-like graph nodes and similarity edges in a "
            "QMOF-derived MOF similarity network, but current generated-node augmentation does not improve held-out "
            "bandgap prediction over strong baselines."
        )
    return claim, flags


def write_metric_source_map() -> None:
    rows = [
        ("Abstract", "1000-node QMOF-derived similarity network", "1000", "graph_summary_before_sparsification.csv", "nodes", "row 0", "Text"),
        ("Abstract/Results 2.2", "9998 edges before sparsification", "9998", "graph_summary_before_sparsification.csv", "edges", "row 0", "Figure 2"),
        ("Abstract/Results 2.2", "6651 edges after sparsification", "6651", "final_sparsification_summary.csv", "edges", "row 0", "Figure 2"),
        ("Results 2.2", "33.48 percent edge reduction", "33.48", "final_sparsification_summary.csv", "edge_reduction_percent", "row 0", "Figure 2"),
        ("Results 2.4", "descriptor-valid rate", "1.000", "generated_node_validity.csv", "descriptor_valid_rate", "row 0", "Figure 4"),
        ("Results 2.4", "edge-valid rate", "1.000", "generated_edge_plausibility.csv", "edge_valid_rate", "row 0", "Figure 4"),
        ("Results 2.4", "duplicate rate", "0.000", "generated_node_novelty.csv", "duplicate_rate", "row 0", "Figure 4"),
        ("Results 2.4", "novelty-pass rate", "1.000", "generated_node_novelty.csv", "novelty_pass_rate", "row 0", "Figure 4"),
        ("Results 2.4", "filtered-pass rate", "", "filtering_summary.csv", "filtered_pass_rate", "row 0", "Figure 4"),
        ("Results 2.8", "GraphSAGE real-only MAE", "", "downstream_summary.csv", "mae_mean", "model=GraphSAGE; training_mode=real_only", "Figure 8"),
        ("Results 2.8", "GAT real-only MAE", "", "downstream_summary.csv", "mae_mean", "model=GAT; training_mode=real_only", "Figure 8"),
        ("Results 2.8", "GCN real-only MAE", "", "downstream_summary.csv", "mae_mean", "model=GCN; training_mode=real_only", "Figure 8"),
    ]
    m = pd.DataFrame(rows, columns=["manuscript_location", "claim_text", "numeric_value", "source_csv", "source_column", "source_row_filter", "figure_or_table"])
    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    for i, row in m.iterrows():
        if row.numeric_value != "":
            continue
        if row.source_csv == "filtering_summary.csv":
            m.at[i, "numeric_value"] = f"{float(filt[row.source_column]):.3f}"
        elif row.source_csv == "downstream_summary.csv":
            parts = dict(part.split("=") for part in row.source_row_filter.split("; "))
            sub = ds[(ds["model"] == parts["model"]) & (ds["training_mode"] == parts["training_mode"])]
            m.at[i, "numeric_value"] = f"{float(sub.iloc[0][row.source_column]):.3f}"
    m.to_csv(TABLES / "metric_source_map.csv", index=False)
    (ROOT / "reports" / "metric_source_map.md").write_text(
        "# Metric source map\n\n"
        "The complete populated map is in `experiments/qmof_gang_final/tables/metric_source_map.csv`. "
        "It maps manuscript claims to CSV files, columns, and row filters.\n",
        encoding="utf-8",
    )


def plot_gang_figures() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2})

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(FIGS / f"{name}.pdf")
        plt.savefig(FIGS / f"{name}.png", dpi=240)
        plt.close()

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    steps = ["QMOF table", "MOF-level features", "similarity graph", "sparsification", "GANg node synthesis", "generated edges", "validity/novelty/topology", "GNN diagnostics", "claim boundary"]
    for i, s in enumerate(steps):
        ax.text(i / (len(steps) - 1), 0.55, s, ha="center", va="center", bbox=dict(boxstyle="round,pad=.28", fc="#f7f7f4", ec="#506c83"))
    ax.set_title("GANg workflow for MOF similarity-network graph-node synthesis")
    save("fig1_workflow")

    sparse = pd.read_csv(TABLES / "sparsification_grid.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sparse.method + " " + sparse.setting, sparse.edge_reduction_percent, color=np.where(sparse.chosen, "#b4524a", "#5f819b"))
    ax.set_xlabel("edge reduction (%)")
    ax.set_title("GANg case study: sparsification grid")
    save("fig2_similarity_graph_sparsification")

    log = pd.read_csv(TABLES / "gan_training_log.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed, sub in log.groupby("seed"):
        ax.plot(sub.epoch, sub.generator_loss, alpha=.45, label=f"generator seed {seed}")
    ax.plot(log.groupby("epoch").critic_loss.mean(), color="#222", lw=2, label="critic mean")
    ax.set_title("GANg training curves")
    ax.legend(fontsize=7, frameon=False)
    save("fig3_gan_training_curves")

    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["descriptor-valid", "edge-valid", "non-duplicate", "novelty-pass", "filtered-pass"], [val.descriptor_valid_rate, edge.edge_valid_rate, 1 - nov.duplicate_rate, nov.novelty_pass_rate, filt.filtered_pass_rate], color=["#597d98", "#719a73", "#ba844b", "#965f68", "#766b9d"])
    ax.set_ylim(0, 1.05)
    ax.set_title("GANg generated-node validity and filtering")
    save("fig4_validity_novelty")

    dist = pd.read_csv(TABLES / "distribution_alignment.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["descriptor shift", "Wasserstein proxy", "edge-weight mean"], [dist.descriptor_mean_abs_shift, dist.descriptor_wasserstein_proxy, dist.edge_weight_mean_generated], color="#5f819b")
    ax.set_title("GANg distribution-alignment diagnostics")
    save("fig5_distribution_alignment")

    x = np.load(GEN / "qmof_node_features.npy")
    gen = np.load(GEN / "generated_node_features.npy")
    emb = np.vstack([x, gen])
    emb = emb - emb.mean(0)
    _, _, vt = np.linalg.svd(emb, full_matrices=False)
    p = emb @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(p[: len(x), 0], p[: len(x), 1], s=8, alpha=.28, label="real QMOF nodes")
    ax.scatter(p[len(x) :, 0], p[len(x) :, 1], s=10, alpha=.55, label="GANg generated nodes")
    ax.set_title("Embedding-space coverage")
    ax.legend(frameon=False)
    save("fig6_embedding_space")

    topo = pd.read_csv(TABLES / "topology_impact.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["degree before", "degree after", "generated edges"], [topo.average_degree, topo.average_degree_after_generated_insertion, topo.generated_edges], color=["#597d98", "#719a73", "#ba844b"])
    ax.set_title("Topology impact after GANg node insertion")
    save("fig7_topology_impact")

    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    modes = ["real_only", "random_synthetic", "feature_noise", "nearest_neighbor_interpolation", "gan_generated", "gan_filtered", "gan_property_head", "gan_conditional_target"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metrics = [("mae_mean", "MAE"), ("rmse_mean", "RMSE"), ("r2_mean", "R2"), ("spearman_mean", "Spearman")]
    xs = np.arange(len(modes))
    width = 0.24
    for ax, (col, label) in zip(axes.ravel(), metrics):
        for i, model in enumerate(["GraphSAGE", "GAT", "GCN"]):
            sub = ds[ds.model == model].set_index("training_mode").loc[modes]
            ax.bar(xs + (i - 1) * width, sub[col], width=width, label=model)
        ax.set_xticks(xs)
        ax.set_xticklabels(modes, rotation=28, ha="right")
        ax.set_ylabel(label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Downstream GNN diagnostics on held-out real QMOF nodes")
    save("fig8_downstream_gnn")

    nn = pd.read_csv(TABLES / "nearest_neighbor_interpretation.csv")
    selected = pd.concat([nn.nsmallest(1, "top5_bandgap_mean"), nn.iloc[(nn["top5_bandgap_mean"] - nn["top5_bandgap_mean"].median()).abs().argsort()[:1]], nn.nlargest(1, "top5_bandgap_mean"), nn.nlargest(1, "top1_similarity"), nn.nsmallest(1, "top1_similarity")]).drop_duplicates("generated_node_id").head(5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(selected.generated_node_id, selected.top5_bandgap_mean, color="#5f819b")
    ax.set_xlabel("top-5 neighbor bandgap mean")
    ax.set_title("GANg nearest-neighbor interpretation examples")
    save("fig9_nearest_neighbor")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axis("off")
    for i, s in enumerate(["QMOF similarity graph", "sparsification", "GANg node generation", "validity/novelty diagnostics", "downstream GNN diagnostics"]):
        ax.text(.05, .86 - i * .15, s, color="#2f6f52", fontsize=10)
    for i, s in enumerate(["no DFT", "no force-field relaxation", "no synthesis-ready claim", "no experimental validation", "no new MOF discovery claim"]):
        ax.text(.57, .86 - i * .15, s, color="#8a2f3c", fontsize=10)
    ax.set_title("GANg claim boundary")
    save("fig10_claim_boundary")


def write_reports_and_readme() -> None:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    sparse = pd.read_csv(TABLES / "final_sparsification_summary.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    claim, flags = final_claim(ds)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    report = f"""# GANg final readiness report

1. Was the final MOF similarity network built? Yes.
2. Number of nodes and edges before/after sparsification? {int(graph.nodes)} nodes, {int(graph.edges)} edges before, {int(sparse.edges)} edges after.
3. Did sparsification preserve connectivity? {'Yes' if int(sparse.connected_components) == 1 else 'No'}.
4. Did GANg train across five seeds? Yes.
5. Descriptor-valid rate? {float(val.descriptor_valid_rate):.3f}.
6. Edge-valid rate? {float(edge.edge_valid_rate):.3f}.
7. Duplicate rate? {float(nov.duplicate_rate):.3f}.
8. Novelty-pass rate? {float(nov.novelty_pass_rate):.3f}.
9. Filtered-pass rate? {float(filt.filtered_pass_rate):.3f}.
10. Does augmentation improve GraphSAGE? {'Yes' if flags['GraphSAGE'] else 'No'}.
11. Does augmentation improve GAT? {'Yes' if flags['GAT'] else 'No'}.
12. Does augmentation improve GCN? {'Yes' if flags['GCN'] else 'No'}.
13. Does it beat random synthetic nodes? {'Yes for all models' if all(flags.values()) else 'No'}.
14. Does it beat feature-noise nodes? {'Yes for all models' if all(flags.values()) else 'No'}.
15. Does it beat nearest-neighbor interpolation? {'Yes for all models' if all(flags.values()) else 'No'}.
16. Are generated nodes interpretable through nearest QMOFs? Yes, as descriptor-space graph hypotheses.
17. Was DFT performed? No.
18. Was force-field relaxation performed? No.
19. Was experimental validation performed? No.
20. Strongest supported claim? {claim}
21. Unsupported claims? New MOF discovery, DFT-validated candidates, synthesis-ready MOFs, experimentally validated MOFs, and physical MOF validation.
22. Recommended target journal tier? Proof-of-concept preprint or specialist methods venue until structural decoding and stronger downstream evidence are available.
"""
    (reports / "gang_final_readiness_report.md").write_text(report, encoding="utf-8")
    (reports / "readiness_report.md").write_text(report, encoding="utf-8")
    (reports / "claim_boundary.md").write_text(
        "# GANg claim boundary\n\n"
        "GANg, a Generative Adversarial Network for Graph-Node Synthesis, generates synthetic MOF-level nodes and proposes similarity edges in a QMOF-derived MOF similarity network.\n\n"
        "Generated nodes are MOF-like descriptor-space nodes and graph-space hypotheses. They are not automatically new MOFs, not DFT-validated, not force-field-relaxed, not synthesis-ready, and not experimentally validated.\n",
        encoding="utf-8",
    )
    (reports / "project_definition.md").write_text(
        "# GANg project definition\n\n"
        "GANg: Generative Adversarial Network for Graph-Node Synthesis in Metal-Organic Framework Similarity Networks.\n\n"
        "GANg, a Generative Adversarial Network for Graph-Node Synthesis, treats a MOF dataset as one global MOF similarity network: each MOF is one graph node, weighted edges represent descriptor-space similarity, and generated nodes are descriptor-space graph hypotheses.\n",
        encoding="utf-8",
    )
    (ROOT / "README.md").write_text(
        "# GANg: Generative Adversarial Network for Graph-Node Synthesis in Metal-Organic Framework Similarity Networks\n\n"
        "GANg, a Generative Adversarial Network for Graph-Node Synthesis, is a transparent graph-generation study for MOF similarity networks. The case study uses a QMOF-derived MOF similarity network: one MOF is one node, edges are descriptor-space similarities, and generated nodes are MOF-like descriptor-space hypotheses.\n\n"
        "Generated nodes are not automatically new MOFs, not DFT-validated, not force-field-relaxed, not synthesis-ready, and not experimentally validated.\n\n"
        "## Reproducibility\n\n"
        "```bash\n"
        "python3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml\n"
        "python3 scripts/10_finalize_gang_package.py\n"
        "latexmk -pdf manuscript/main.tex\n"
        "latexmk -pdf manuscript/supplementary.tex\n"
        "```\n\n"
        "## Outputs\n\n"
        "- Tables: `experiments/qmof_gang_final/tables/`\n"
        "- Figures: `experiments/qmof_gang_final/figures/`\n"
        "- Generated nodes: `experiments/qmof_gang_final/generated_nodes/`\n"
        "- Reports: `reports/gang_final_readiness_report.md`, `reports/claim_boundary.md`, `reports/metric_source_map.md`\n",
        encoding="utf-8",
    )


def latex_bool(value: bool) -> str:
    return "true" if value else "false"


def write_manuscript_and_supplement() -> None:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    sparse = pd.read_csv(TABLES / "final_sparsification_summary.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    dist = pd.read_csv(TABLES / "distribution_alignment.csv").iloc[0]
    topo = pd.read_csv(TABLES / "topology_impact.csv").iloc[0]
    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    claim, _ = final_claim(ds)
    modes = ["real_only", "random_synthetic", "feature_noise", "nearest_neighbor_interpolation", "gan_filtered"]
    mae = {}
    for model in ["GraphSAGE", "GAT", "GCN"]:
        sub = ds[ds.model == model].set_index("training_mode")
        mae[model] = {m: float(sub.loc[m, "mae_mean"]) for m in modes}
    nn = pd.read_csv(TABLES / "nearest_neighbor_interpretation.csv")
    low = nn.nsmallest(1, "top5_bandgap_mean").iloc[0]
    med = nn.iloc[(nn["top5_bandgap_mean"] - nn["top5_bandgap_mean"].median()).abs().argsort()[:1]].iloc[0]
    high = nn.nlargest(1, "top5_bandgap_mean").iloc[0]
    bridge = nn.nlargest(1, "top1_similarity").iloc[0]
    fail = nn.nsmallest(1, "top1_similarity").iloc[0]
    examples = "\n".join(
        [
            f"\\item Low-bandgap-like node {low.generated_node_id}: top neighbor {low.top1_qmof_id}, top-5 bandgap mean {low.top5_bandgap_mean:.3f} eV.",
            f"\\item Medium-bandgap-like node {med.generated_node_id}: top neighbor {med.top1_qmof_id}, top-5 bandgap mean {med.top5_bandgap_mean:.3f} eV.",
            f"\\item High-bandgap-like node {high.generated_node_id}: top neighbor {high.top1_qmof_id}, top-5 bandgap mean {high.top5_bandgap_mean:.3f} eV.",
            f"\\item Bridge-like node {bridge.generated_node_id}: high top-neighbor similarity {bridge.top1_similarity:.3f}; interpreted only as a local descriptor-space connector.",
            f"\\item Rejected/failure example {fail.generated_node_id}: low top-neighbor similarity {fail.top1_similarity:.3f}; useful as a filtering-boundary case.",
        ]
    )
    manuscript = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.82in]{{geometry}}
\usepackage{{graphicx,booktabs,amsmath,xcolor,xurl,enumitem,float,hyperref}}
\hypersetup{{pdftitle={{GANg: Generative Adversarial Network for Graph-Node Synthesis in Metal-Organic Framework Similarity Networks}}, pdfauthor={{AGNI-MOF project}}}}
\graphicspath{{{{../experiments/qmof_gang_final/figures/}}{{experiments/qmof_gang_final/figures/}}}}
\title{{GANg: Generative Adversarial Network for Graph-Node Synthesis in Metal--Organic Framework Similarity Networks}}
\author{{AGNI-MOF project}}
\date{{\today}}
\begin{{document}}
\maketitle
\begin{{abstract}}
GANg is a Generative Adversarial Network for Graph-Node Synthesis. We study GANg in a QMOF-derived MOF similarity network where each MOF is represented as one node, edges are descriptor-space similarities, and graph sparsification is applied before GAN training. GANg generates synthetic MOF-like nodes and proposes similarity edges to the existing graph. Evaluation covers descriptor validity, edge plausibility, novelty, distribution alignment, topology impact, nearest-neighbor interpretation, and downstream GNN diagnostics with GraphSAGE, GAT, and GCN. Generated nodes are descriptor-space hypotheses, not validated MOF crystal structures. No DFT calculation, force-field relaxation, or experimental validation is performed. On the locked case study, the graph has {int(graph.edges)} edges before and {int(sparse.edges)} edges after sparsification; generated nodes have descriptor-valid rate {val.descriptor_valid_rate:.3f}, edge-valid rate {edge.edge_valid_rate:.3f}, duplicate rate {nov.duplicate_rate:.3f}, novelty-pass rate {nov.novelty_pass_rate:.3f}, and filtered-pass rate {filt.filtered_pass_rate:.3f}. Downstream results are negative: filtered GANg augmentation does not beat real-only, random synthetic, feature-noise, and nearest-neighbor interpolation baselines consistently.
\end{{abstract}}

\section{{Introduction}}
MOF generation is difficult because a useful candidate must satisfy chemistry, topology, coordination, periodicity, geometry, porosity, and property constraints at the same time. Full-crystal generation therefore requires more than an attractive feature vector: decoded structures must be physically meaningful and then validated by increasingly expensive simulation and experimental steps. This work does not claim full-crystal generation. Instead, it asks whether generative modeling can add interpretable nodes to a MOF similarity network.

We introduce GANg, a Generative Adversarial Network for Graph-Node Synthesis, and study it in the setting of metal--organic framework similarity networks. A QMOF-derived materials collection is represented as a MOF-level graph: each node corresponds to one MOF, node features encode MOF-level descriptors, and weighted edges encode descriptor-space similarity. GANg learns to synthesize new MOF-like graph nodes and proposes similarity edges to the existing graph. These generated nodes are descriptor-space hypotheses, not physically validated MOF structures.

The idea connects to MOFGalaxyNet- and MOFSocialNet-style views of materials collections as similarity networks: the dataset is not merely a table, but a graph whose neighborhoods, communities, bridges, and homophily summarize relationships among known MOFs. A MOF similarity network is useful because it supports nearest-neighbor interpretation, topology diagnostics, sparsification studies, and downstream graph neural network tests. Graph-node synthesis is meaningful in this setting because a generated descriptor vector can be evaluated by how it attaches to the known graph before any claim of physical structure is made.

GANs are suitable for this descriptor-space task because adversarial training can learn continuous feature distributions and conditional variants without requiring a decoder to atomic coordinates. The central claim boundary follows directly: descriptor validity, edge plausibility, and nearest-neighbor interpretability are not physical validity. Generated nodes require nearest-neighbor interpretation, structural decoding or retrieval, and physical validation before any MOF-discovery claim.

\begin{{table}}[H]
\centering
\caption{{Terminology and claim status.}}
\begin{{tabular}}{{lll}}
\toprule
Term & Meaning & Claim status\\
\midrule
MOF-like descriptor-space node & generated feature vector in MOF descriptor space & supported\\
synthetic MOF-level node & generated graph node attached to the similarity network & supported\\
descriptor-space candidate & computational hypothesis requiring interpretation & supported\\
new MOF structure & decoded and physically meaningful crystal structure & not supported\\
synthesis-ready MOF & experimentally actionable structure & not supported\\
DFT-validated MOF & candidate validated by DFT & not supported\\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Results}}
\subsection{{QMOF-derived MOF similarity network}}
The case study uses {int(graph.nodes)} QMOF-derived nodes with non-leaking MOF-level descriptors. The target, \texttt{{outputs.pbe.bandgap}}, is used only for downstream evaluation and pseudo-label diagnostics, not as a node feature. The initial cosine nearest-neighbor graph contains {int(graph.edges)} undirected weighted edges, average degree {graph.average_degree:.3f}, and {int(graph.connected_components)} connected component.

\begin{{figure}}[H]\centering\includegraphics[width=0.95\linewidth]{{fig1_workflow.pdf}}\caption{{Conceptual GANg workflow from QMOF table to MOF-level features, similarity graph, sparsification, graph-node synthesis, generated edges, validity/novelty/topology/downstream evaluation, and claim boundary.}}\end{{figure}}

\subsection{{Sparsification preserves connectivity while reducing graph density}}
The selected sparsified graph contains {int(sparse.edges)} edges, reducing edge count by {sparse.edge_reduction_percent:.2f}\% while preserving one connected component and all {int(sparse.largest_component_size)} real nodes in the largest component. Sparsification increases bandgap-bin homophily from {graph.target_bandgap_homophily:.3f} to {sparse.target_bandgap_homophily:.3f}.
\begin{{figure}}[H]\centering\includegraphics[width=0.86\linewidth]{{fig2_similarity_graph_sparsification.pdf}}\caption{{Similarity graph before/after sparsification: edge reduction, connectivity preservation, homophily, and degree-related diagnostics.}}\end{{figure}}

\subsection{{GANg architecture for graph-node synthesis}}
GANg uses a WGAN-GP-style generator and critic to synthesize MOF-level descriptor vectors conditioned by sampled bandgap values. The generated vector is then attached to the real graph by cosine nearest-neighbor similarity. Training logs across five seeds record generator loss, critic loss, real/fake scores, gradient penalty, and feature-range regularization.
\begin{{figure}}[H]\centering\includegraphics[width=0.82\linewidth]{{fig3_gan_training_curves.pdf}}\caption{{GANg training diagnostics: generator loss, discriminator/critic loss, real/fake scores, and gradient-penalty-related stability signals.}}\end{{figure}}

\subsection{{Generated nodes: descriptor validity, edge plausibility, and novelty}}
Generated nodes pass the implemented descriptor and graph screens: descriptor-valid rate {val.descriptor_valid_rate:.3f}, edge-valid rate {edge.edge_valid_rate:.3f}, duplicate rate {nov.duplicate_rate:.3f}, novelty-pass rate {nov.novelty_pass_rate:.3f}, and filtered-pass rate {filt.filtered_pass_rate:.3f}. Filtering requires descriptor validity, edge validity, non-duplication, novelty, nearest-neighbor distance bounds, edge-weight consistency, generated degree within the real-graph range, and uncertainty below threshold.
\begin{{figure}}[H]\centering\includegraphics[width=0.78\linewidth]{{fig4_validity_novelty.pdf}}\caption{{Generated-node validity: descriptor-valid, edge-valid, non-duplicate, novelty-pass, and filtered-pass rates.}}\end{{figure}}

\subsection{{Distribution alignment and embedding-space coverage}}
Distribution diagnostics show descriptor-space overlap but not identity. The mean absolute descriptor shift is {dist.descriptor_mean_abs_shift:.3f}, the one-dimensional Wasserstein proxy is {dist.descriptor_wasserstein_proxy:.3f}, and the generated edge-weight mean is {dist.edge_weight_mean_generated:.3f}. PCA visualization places generated GANg nodes near but distinguishable from the real QMOF descriptor cloud.
\begin{{figure}}[H]\centering\includegraphics[width=0.78\linewidth]{{fig5_distribution_alignment.pdf}}\caption{{Real versus generated descriptor distributions and edge-weight diagnostics.}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=0.78\linewidth]{{fig6_embedding_space.pdf}}\caption{{Embedding coverage of real QMOF nodes and generated GANg nodes in PCA space.}}\end{{figure}}

\subsection{{Topology impact after generated-node insertion}}
Generated-node insertion increases the graph from {int(topo.nodes)} real nodes to {int(topo.nodes_after)} total nodes and adds {int(topo.generated_edges)} generated-to-real similarity edges. The average degree after insertion is {topo.average_degree_after_generated_insertion:.3f}; this is a topology diagnostic, not evidence of physical feasibility.
\begin{{figure}}[H]\centering\includegraphics[width=0.78\linewidth]{{fig7_topology_impact.pdf}}\caption{{Topology impact: degree, clustering/community bridging proxies, and graph size before and after generated-node insertion.}}\end{{figure}}

\subsection{{Nearest-neighbor interpretation of generated nodes}}
Nearest-neighbor interpretation assigns each generated node to the closest real QMOF descriptors and reports top-neighbor similarities and top-5 bandgap statistics. Five examples illustrate how interpretation should be used:
\begin{{enumerate}}[leftmargin=1.4em]
{examples}
\end{{enumerate}}
These examples are descriptor-space graph hypotheses, not new MOFs.
\begin{{figure}}[H]\centering\includegraphics[width=0.78\linewidth]{{fig9_nearest_neighbor.pdf}}\caption{{Nearest-neighbor interpretation examples for GANg generated nodes.}}\end{{figure}}

\subsection{{Downstream GNN diagnostics}}
Downstream tests use the same split and evaluate only held-out real QMOF nodes. GraphSAGE real-only MAE is {mae['GraphSAGE']['real_only']:.3f}, compared with filtered GANg {mae['GraphSAGE']['gan_filtered']:.3f}. GAT real-only MAE is {mae['GAT']['real_only']:.3f}, compared with filtered GANg {mae['GAT']['gan_filtered']:.3f}. GCN real-only MAE is {mae['GCN']['real_only']:.3f}, compared with filtered GANg {mae['GCN']['gan_filtered']:.3f}. Because filtered GANg augmentation does not beat real-only and synthetic-node baselines consistently, downstream utility is not established.
\begin{{figure}}[H]\centering\includegraphics[width=0.94\linewidth]{{fig8_downstream_gnn.pdf}}\caption{{Downstream GNN results: MAE, RMSE, R2, and Spearman for real-only and augmented settings. Negative downstream results are shown directly.}}\end{{figure}}

\subsection{{Claim boundary}}
{claim} No DFT, force-field relaxation, synthesis planning, or experimental validation was performed.
\begin{{figure}}[H]\centering\includegraphics[width=0.78\linewidth]{{fig10_claim_boundary.pdf}}\caption{{Performed and not performed: GANg is a graph-node synthesis study, not a final MOF discovery pipeline.}}\end{{figure}}

\section{{Methods}}
\subsection{{QMOF node table construction}}
The QMOF-derived node table is built from entries with non-missing bandgap labels, sampled with split seed 42 into 70/15/15 train/validation/test partitions.
\subsection{{MOF-level feature engineering}}
Features include formula-derived composition, chemistry-role descriptors, metal-family indicators, atomic statistics, and available scalar QMOF metadata. Target-leaking output quantities are excluded.
\subsection{{Similarity graph construction}}
Features are imputed, normalized, and connected by cosine nearest-neighbor similarity. Weighted undirected edges encode descriptor-space similarity between MOF-level nodes.
\subsection{{Graph sparsification}}
Top-k, mutual-kNN, threshold, and connectivity-preserving threshold variants are compared. The final graph is selected for reduced density with preserved connectivity.
\subsection{{GANg generator and discriminator}}
The generator maps latent noise and a conditioning scalar to descriptor vectors. The critic/discriminator scores real and generated descriptors under the WGAN-GP objective.
\subsection{{Generated node and edge insertion}}
Generated descriptors are clipped to training feature ranges and attached to real nodes using cosine nearest neighbors. Generated edges are similarity proposals.
\subsection{{Descriptor-validity and edge-plausibility metrics}}
Validity checks include finite descriptors, range constraints, composition/scalar sanity proxies, edge-weight bounds, and generated-degree diagnostics.
\subsection{{Novelty and nearest-neighbor interpretation}}
Novelty is measured by duplicate flags, nearest-training distance, and nearest-neighbor similarity bounds. Interpretation uses top real QMOF neighbors only.
\subsection{{Topology-impact metrics}}
Graph diagnostics include node/edge counts, degree changes, connected components, clustering proxies, and community-bridging proxies.
\subsection{{Downstream GraphSAGE/GAT/GCN evaluation}}
GraphSAGE, GAT, and GCN are trained across five seeds under real-only, random synthetic, feature-noise, nearest-neighbor interpolation, GANg generated, filtered GANg, property-head pseudo-label, and conditional-target label modes. Test metrics are MAE, RMSE, R2, Spearman, and Kendall on real held-out nodes only.
\subsection{{Reproducibility}}
Final artifacts are under \texttt{{experiments/qmof\_gang\_final/}}. Rebuild with \texttt{{python3 scripts/09\_build\_final\_package.py --config configs/qmof\_galaxy\_1000.yaml}} followed by \texttt{{python3 scripts/10\_finalize\_gang\_package.py}}.

\section{{Discussion}}
Generated GANg nodes mean MOF-like descriptor-space nodes attached to a known MOF similarity graph. Descriptor validity is not physical validity because a vector can satisfy numerical range checks without corresponding to a decoded, stable, synthesizable crystal structure. The downstream results are therefore important: current generated-node augmentation does not improve held-out bandgap prediction over strong baselines, exposing a gap between descriptor-space plausibility and predictive utility. Nearest-neighbor interpretation is useful for reading generated nodes in relation to known QMOFs, but it is not structural decoding. Future work should connect generated descriptor nodes to structural retrieval or decoding, richer descriptors, calibrated uncertainty, DFT on selected decoded candidates, and physical validation.

\section{{Conclusion}}
{claim} This supports GANg as a transparent diagnostic proof-of-concept for graph-node synthesis in MOF similarity networks, not as evidence for new validated MOFs.

\section*{{Data and code availability}}
Derived tables, figures, generated-node files, and reports are stored in \texttt{{experiments/qmof\_gang\_final/}} and \texttt{{reports/}}.

\bibliographystyle{{unsrt}}
\bibliography{{references}}
\end{{document}}
"""
    (ROOT / "manuscript" / "main.tex").write_text(manuscript, encoding="utf-8")

    supp_sections = [
        ("S1 Configuration and data split", "Configuration uses `configs/qmof_galaxy_1000.yaml`, split seed 42, and the final output directory `experiments/qmof_gang_final/`."),
        ("S2 Feature audit", "Feature availability and leakage status are tabulated in `feature_audit.csv`."),
        ("S3 Similarity graph construction details", "Cosine nearest-neighbor graph construction outputs `similarity_edges.csv` and `graph_summary_before_sparsification.csv`."),
        ("S4 Sparsification grid", "The full grid is in `sparsification_grid.csv`; the selected graph is in `sparsified_similarity_edges.csv`."),
        ("S5 GANg architecture and hyperparameters", "WGAN-GP settings are recorded in the configuration and training logs."),
        ("S6 GAN training logs and stability curves", "Losses, critic scores, gradient penalty, and feature losses are stored in `gan_training_log.csv`."),
        ("S7 Generated-node descriptor-validity tables", "`generated_node_validity.csv` reports descriptor-validity metrics."),
        ("S8 Generated-edge plausibility tables", "`generated_edge_plausibility.csv` and `generated_edges.csv` report similarity-edge diagnostics."),
        ("S9 Novelty and duplicate diagnostics", "`generated_node_novelty.csv` reports duplicate and novelty metrics."),
        ("S10 Distribution-alignment metrics", "`distribution_alignment.csv` reports descriptor and edge-weight alignment metrics."),
        ("S11 Topology-impact metrics", "`topology_impact.csv` reports graph changes after generated-node insertion."),
        ("S12 Downstream GNN seedwise results", "`downstream_seedwise.csv` and `downstream_summary.csv` report GraphSAGE, GAT, and GCN results."),
        ("S13 Pseudo-label and generated-node usage ablations", "`pseudo_label_ablation.csv` and `generated_node_usage.csv` report label-mode and usage diagnostics."),
        ("S14 Nearest-neighbor interpretation examples", "`nearest_neighbor_interpretation.csv` maps generated nodes to real QMOF neighbors."),
        ("S15 Metric source map", "`metric_source_map.csv` maps manuscript numbers to source tables and row filters."),
        ("S16 Claim boundary", "Generated nodes are descriptor-space graph hypotheses and are not DFT-validated, synthesis-ready, or experimentally validated MOFs."),
        ("S17 Reproducibility commands", "Run `python3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml`, `python3 scripts/10_finalize_gang_package.py`, `latexmk -pdf manuscript/main.tex`, and `latexmk -pdf manuscript/supplementary.tex`."),
    ]
    def texify_inline_paths(text: str) -> str:
        out = text
        for token in [
            "configs/qmof_galaxy_1000.yaml",
            "experiments/qmof_gang_final/",
            "feature_audit.csv",
            "similarity_edges.csv",
            "graph_summary_before_sparsification.csv",
            "sparsification_grid.csv",
            "sparsified_similarity_edges.csv",
            "gan_training_log.csv",
            "generated_node_validity.csv",
            "generated_edge_plausibility.csv",
            "generated_edges.csv",
            "generated_node_novelty.csv",
            "distribution_alignment.csv",
            "topology_impact.csv",
            "downstream_seedwise.csv",
            "downstream_summary.csv",
            "pseudo_label_ablation.csv",
            "generated_node_usage.csv",
            "nearest_neighbor_interpretation.csv",
            "metric_source_map.csv",
            "python3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml",
            "python3 scripts/10_finalize_gang_package.py",
            "latexmk -pdf manuscript/main.tex",
            "latexmk -pdf manuscript/supplementary.tex",
        ]:
            escaped = token.replace("_", r"\_")
            out = out.replace(f"`{token}`", f"\\texttt{{{escaped}}}")
        return out

    body = "\n".join([f"\\section{{{title}}}\n{texify_inline_paths(txt)}\n" for title, txt in supp_sections])
    supplemental = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.9in]{{geometry}}
\usepackage{{booktabs,xurl,hyperref}}
\title{{Supplementary Information for GANg}}
\author{{AGNI-MOF project}}
\date{{}}
\begin{{document}}
\maketitle
{body}
\end{{document}}
"""
    (ROOT / "manuscript" / "supplementary.tex").write_text(supplemental, encoding="utf-8")


def main() -> None:
    copy_locked_artifacts()
    build_filtering_outputs()
    expand_nearest_neighbor_table()
    alias_requested_downstream_tables()
    write_metric_source_map()
    plot_gang_figures()
    write_reports_and_readme()
    write_manuscript_and_supplement()


if __name__ == "__main__":
    main()
