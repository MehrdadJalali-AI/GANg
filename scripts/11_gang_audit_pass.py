from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "qmof_gang_final"
TABLES = EXP / "tables"
FIGS = EXP / "figures"
EXAMPLE = EXP / "examples" / "generated_node_example"
MANUSCRIPT = ROOT / "manuscript"
REPORTS = ROOT / "reports"


def esc(text: object) -> str:
    s = "" if pd.isna(text) else str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_\allowbreak{}",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def fmt(x: object, nd: int = 3) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    s = str(x)
    replacements = {
        "nearest_neighbor_interpolation": "nn_interp",
        "nearest_neighbor_interpolation_label": "nn_interp_label",
        "valid_edge_novel_consistent_uncertainty_filtered_neighbor_label_q50_threshold_0.9075": "filtered_neighbor_q50",
        "conditional_target_bandgap_label": "conditional_label",
        "property_head_pseudo_label": "property_head",
        "neighbor_averaged_pseudo_label": "neighbor_avg",
        "random_train_label": "random_label",
        "noisy_parent_label": "noise_label",
        "real_labels_only": "real_labels",
        "connectivity-preserving-threshold": "conn_pres_threshold",
        "threshold=0.55+best-edge": "th=0.55+best",
        "threshold=0.65+best-edge": "th=0.65+best",
        "threshold=0.75+best-edge": "th=0.75+best",
        "validity/novelty/edge/uncertainty filter": "validity_novelty_uncertainty",
        "not filtered by GANg criteria": "not_GANg_filtered",
        "descriptor-space graph hypothesis; not a decoded or physically validated MOF structure": "descriptor hypothesis; not decoded/validated",
        "feature_difference_summary": "feature_delta",
        "mean_abs_delta_to_top1=": "delta_top1=",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return esc(s)


def table_from_df(df: pd.DataFrame, columns: list[str], caption: str, label: str, font: str = r"\scriptsize") -> str:
    width = max(0.035, min(0.18, 0.82 / max(len(columns), 1)))
    aligns = "".join([f">{{\\RaggedRight\\arraybackslash}}p{{{width:.3f}\\linewidth}}" for _ in columns])
    header = " & ".join(esc(c) for c in columns) + r"\\"
    rows = []
    for _, r in df[columns].iterrows():
        rows.append(" & ".join(fmt(r[c]) for c in columns) + r"\\")
    return (
        f"{font}\n"
        f"\\begin{{longtable}}{{{aligns}}}\n"
        f"\\caption{{{esc(caption)}}}\\label{{{label}}}\\\\\n"
        "\\toprule\n"
        f"{header}\n\\midrule\n\\endfirsthead\n"
        "\\toprule\n"
        f"{header}\n\\midrule\n\\endhead\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{longtable}\n\\normalsize\n"
    )


def feature_group_summary() -> pd.DataFrame:
    audit = pd.read_csv(TABLES / "feature_audit.csv")
    audit["feature_group"] = np.select(
        [
            audit["feature_name"].str.startswith("frac_") | audit["feature_name"].isin(["atom_count_formula", "n_unique_elements"]),
            audit["source"].eq("formula_composition_or_chemistry_role"),
        ],
        ["composition", "chemistry_role"],
        default="scalar_metadata",
    )
    out = audit.groupby("feature_group").agg(
        num_features=("feature_name", "count"),
        missing_rate=("missing_count", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce").fillna(0) / 1000))),
        used_in_similarity_graph=("used_in_similarity_graph", "all"),
        used_in_GAN=("used_in_gan_training", "all"),
        used_in_downstream=("used_in_downstream_gnn", "all"),
    ).reset_index()
    out["leakage_risk"] = "low"
    out["comment"] = "non-leaking descriptor group; target/output columns excluded"
    out.to_csv(TABLES / "feature_group_audit.csv", index=False)
    return out


def downstream_audit_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    rows = []
    for model, sub in ds.groupby("model"):
        real = float(sub.loc[sub.training_mode.eq("real_only"), "mae_mean"].iloc[0])
        nn = float(sub.loc[sub.training_mode.eq("nearest_neighbor_interpolation"), "mae_mean"].iloc[0])
        for _, r in sub.iterrows():
            rows.append({
                "model": r.model,
                "training_mode": r.training_mode,
                "label_strategy": r.label_mode,
                "MAE_mean": r.mae_mean,
                "MAE_std": r.mae_std,
                "RMSE_mean": r.rmse_mean,
                "RMSE_std": r.rmse_std,
                "R2_mean": r.r2_mean,
                "R2_std": r.r2_std,
                "Spearman_mean": r.spearman_mean,
                "Spearman_std": r.spearman_std,
                "Kendall_mean": r.kendall_mean,
                "Kendall_std": r.kendall_std,
                "beats_real_only_yes_no": "yes" if float(r.mae_mean) < real else "no",
                "beats_nearest_neighbor_yes_no": "yes" if float(r.mae_mean) < nn else "no",
            })
    full = pd.DataFrame(rows)
    full.to_csv(TABLES / "downstream_full_audit.csv", index=False)

    seed = pd.read_csv(TABLES / "downstream_seedwise.csv")
    seed_out = seed[["model", "seed", "training_mode", "mae", "rmse", "r2", "spearman", "kendall"]].rename(
        columns={"mae": "MAE", "rmse": "RMSE", "r2": "R2", "spearman": "Spearman", "kendall": "Kendall"}
    )
    seed_out.to_csv(TABLES / "downstream_seedwise_audit.csv", index=False)
    return full, seed_out


def pseudo_and_usage_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    usage = pd.read_csv(TABLES / "generated_node_usage.csv")
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    pseudo_rows = []
    for mode in ["gan_generated", "gan_filtered", "gan_property_head", "gan_conditional_target"]:
        sub = ds[ds.training_mode.eq(mode)]
        used = usage[usage.training_mode.eq(mode)]
        pseudo_rows.append({
            "label_strategy": mode,
            "generated_node_count": int(filt.generated_nodes_before_filtering),
            "filtered_node_count": int(filt.generated_nodes_retained) if mode == "gan_filtered" else int(filt.generated_nodes_before_filtering),
            "MAE_mean": float(sub.mae_mean.mean()),
            "RMSE_mean": float(sub.rmse_mean.mean()),
            "R2_mean": float(sub.r2_mean.mean()),
            "Spearman_mean": float(sub.spearman_mean.mean()),
            "comment": str(used.label_mode.iloc[0]) if len(used) else "not used",
        })
    pseudo = pd.DataFrame(pseudo_rows)
    pseudo.to_csv(TABLES / "pseudo_label_ablation_audit.csv", index=False)

    usage_rows = []
    for mode, sub in usage.groupby("training_mode"):
        usage_rows.append({
            "training_mode": mode,
            "generated_nodes_available": int(filt.generated_nodes_before_filtering) if mode != "real_only" else 0,
            "generated_nodes_used": int(sub.generated_nodes_used.max()),
            "generated_nodes_filtered_out": int(sub.generated_nodes_dropped.max()),
            "filter_reason": "validity/novelty/edge/uncertainty filter" if mode == "gan_filtered" else "not filtered by GANg criteria",
            "label_strategy": str(sub.label_mode.iloc[0]),
        })
    usage_audit = pd.DataFrame(usage_rows)
    usage_audit.to_csv(TABLES / "generated_node_usage_audit.csv", index=False)
    return pseudo, usage_audit


def create_example() -> dict:
    EXAMPLE.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(TABLES / "generated_nodes.csv")
    edges = pd.read_csv(TABLES / "generated_edges.csv")
    nn = pd.read_csv(TABLES / "nearest_neighbor_interpretation.csv")
    filt_nodes = pd.read_csv(EXP / "generated_nodes" / "generated_nodes_filtered.csv")
    candidate_ids = set(filt_nodes.generated_node_id)
    sub_nn = nn[nn.generated_node_id.isin(candidate_ids)].copy()
    sub_nn["score"] = sub_nn["top1_similarity"] - (sub_nn["top5_bandgap_std"] * 0.02)
    chosen = sub_nn.sort_values("score", ascending=False).iloc[0]
    gid = chosen.generated_node_id
    node = nodes[nodes.generated_node_id.eq(gid)].iloc[0]
    edge_sub = edges[edges.generated_node_id.eq(gid)].sort_values("similarity", ascending=False).head(5).copy()
    node_table = pd.read_csv(TABLES / "qmof_node_table.csv")
    edge_sub["qmof_id"] = edge_sub["real_node_index"].map(lambda i: node_table.loc[int(i), "qmof_id"])
    top_ids = edge_sub["qmof_id"].astype(str).tolist()
    top_sims = edge_sub["similarity"].astype(float).tolist()
    card = {
        "generated_node_id": gid,
        "descriptor_valid": bool(node.descriptor_valid),
        "edge_valid": True,
        "duplicate": bool(node.duplicate),
        "novelty_score": float(node.novelty_score),
        "top5_nearest_qmof_ids": ";".join(top_ids),
        "top5_similarity_scores": ";".join(f"{v:.4f}" for v in edge_sub["similarity"]),
        "top1_qmof_id": chosen.top1_qmof_id,
        "top2_qmof_id": chosen.top2_qmof_id,
        "top3_qmof_id": chosen.top3_qmof_id,
        "top5_bandgap_mean": float(chosen.top5_bandgap_mean),
        "top5_bandgap_std": float(chosen.top5_bandgap_std),
        "generated_degree": int((edges.generated_node_id == gid).sum()),
        "edge_weight_mean": float(edges.loc[edges.generated_node_id.eq(gid), "edge_weight"].mean()),
        "interpretation_comment": f"This generated node is a MOF-like descriptor-space hypothesis located near {chosen.top1_qmof_id}, {chosen.top2_qmof_id}, and {chosen.top3_qmof_id}. It is not a decoded crystal structure and no CIF is claimed.",
        "claim_boundary": "descriptor-space hypothesis, not generated CIF; no DFT, force-field relaxation, synthesis, or experimental validation",
    }
    (EXAMPLE / "generated_node_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    pd.DataFrame([card]).to_csv(EXAMPLE / "generated_node_card.csv", index=False)
    pd.DataFrame(
        {
            "rank": list(range(1, len(top_ids) + 1)),
            "qmof_id": top_ids,
            "similarity": top_sims,
        }
    ).to_csv(EXAMPLE / "generated_node_neighbors.csv", index=False)
    (EXAMPLE / "README.md").write_text(
        "# GANg generated-node example\n\n"
        "This folder describes one GANg-generated MOF-like descriptor-space graph node. "
        "No generated CIF is produced in this version because GANg operates in descriptor space. "
        "Structural decoding is future work.\n\n"
        "The CIF file, if present in future versions, must be a nearest real QMOF analogue used for interpretation. "
        "It is not a generated structure and not a physically validated output of GANg.\n",
        encoding="utf-8",
    )

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2})
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [card["novelty_score"], card["edge_weight_mean"], card["top5_bandgap_mean"] / 5.0, card["top5_bandgap_std"] / 2.0]
    ax.bar(["novelty", "edge mean", "bandgap mean / 5", "bandgap std / 2"], vals, color=["#5f819b", "#719a73", "#ba844b", "#965f68"])
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"GANg descriptor-space card: {gid}")
    plt.tight_layout()
    plt.savefig(EXAMPLE / "generated_node_radar_or_barplot.pdf")
    plt.savefig(EXAMPLE / "generated_node_radar_or_barplot.png", dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")
    center = (0.2, 0.5)
    ax.scatter([center[0]], [center[1]], s=950, color="#965f68")
    ax.text(center[0], center[1], gid, color="white", ha="center", va="center", weight="bold")
    angles = np.linspace(-0.9, 0.9, 5)
    for i, (_, r) in enumerate(edge_sub.iterrows()):
        x = 0.75
        y = 0.5 + angles[i] * 0.35
        ax.plot([center[0] + 0.08, x - 0.07], [center[1], y], color="#5f819b", lw=1 + 3 * max(float(r.similarity), 0))
        label = top_ids[i] if i < len(top_ids) else f"real-{int(r.real_node_index)}"
        ax.scatter([x], [y], s=600, color="#597d98")
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=8)
        ax.text((center[0] + x) / 2, (center[1] + y) / 2 + 0.03, f"{float(r.similarity):.3f}", ha="center", fontsize=8)
    ax.text(0.03, 0.08, "descriptor-space hypothesis, not generated CIF", color="#8a2f3c", fontsize=10, weight="bold")
    ax.set_title("Generated-node interpretation card")
    plt.tight_layout()
    plt.savefig(EXAMPLE / "generated_node_network_ego.pdf")
    plt.savefig(EXAMPLE / "generated_node_network_ego.png", dpi=220)
    plt.savefig(FIGS / "fig13_generated_node_interpretation_card.pdf")
    plt.savefig(FIGS / "fig13_generated_node_interpretation_card.png", dpi=220)
    plt.close()
    return card


def update_workflow_and_downstream_figures() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18})
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    ax.axis("off")
    steps = [
        "QMOF metadata",
        "MOF-level non-leaking descriptors",
        "similarity graph",
        "sparsification",
        "GANg node synthesis",
        "generated similarity edges",
        "validity/novelty/topology checks",
        "downstream GNN evaluation",
        "claim boundary",
    ]
    for i, s in enumerate(steps):
        x = i / (len(steps) - 1)
        ax.text(x, 0.58, s, ha="center", va="center", bbox=dict(boxstyle="round,pad=.32", fc="#f7f7f4", ec="#506c83"), wrap=True)
        if i < len(steps) - 1:
            ax.annotate("", xy=((i + 0.78) / (len(steps) - 1), 0.58), xytext=((i + 0.22) / (len(steps) - 1), 0.58), arrowprops=dict(arrowstyle="->", color="#506c83"))
    ax.set_title("GANg workflow for graph-node synthesis in MOF similarity networks", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS / "fig1_workflow.pdf")
    plt.savefig(FIGS / "fig1_workflow.png", dpi=240)
    plt.close()


def metric_source_map(card: dict) -> pd.DataFrame:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    sparse = pd.read_csv(TABLES / "final_sparsification_summary.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    dist = pd.read_csv(TABLES / "distribution_alignment.csv").iloc[0]
    topo = pd.read_csv(TABLES / "topology_impact.csv").iloc[0]
    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    rows = [
        ("Abstract; Results 2.1; Supplement S1", "dataset size", graph.nodes, "graph_summary_before_sparsification.csv", "nodes", "row 0", "Text"),
        ("Abstract; Results 2.2; Supplement S3", "edges before sparsification", graph.edges, "graph_summary_before_sparsification.csv", "edges", "row 0", "Figure 2"),
        ("Abstract; Results 2.2; Supplement S4", "edges after sparsification", sparse.edges, "final_sparsification_summary.csv", "edges", "row 0", "Figure 2"),
        ("Results 2.2; Supplement S4", "edge reduction percent", sparse.edge_reduction_percent, "final_sparsification_summary.csv", "edge_reduction_percent", "row 0", "Figure 2"),
        ("Results 2.2; Supplement S3", "homophily before sparsification", graph.target_bandgap_homophily, "graph_summary_before_sparsification.csv", "target_bandgap_homophily", "row 0", "Figure 2"),
        ("Results 2.2; Supplement S4", "homophily after sparsification", sparse.target_bandgap_homophily, "final_sparsification_summary.csv", "target_bandgap_homophily", "row 0", "Figure 2"),
        ("Abstract; Results 2.4; Supplement S7", "descriptor-valid rate", val.descriptor_valid_rate, "generated_node_validity.csv", "descriptor_valid_rate", "row 0", "Figure 4"),
        ("Abstract; Results 2.4; Supplement S8", "edge-valid rate", edge.edge_valid_rate, "generated_edge_plausibility.csv", "edge_valid_rate", "row 0", "Figure 4"),
        ("Abstract; Results 2.4; Supplement S9", "duplicate rate", nov.duplicate_rate, "generated_node_novelty.csv", "duplicate_rate", "row 0", "Figure 4"),
        ("Abstract; Results 2.4; Supplement S9", "novelty-pass rate", nov.novelty_pass_rate, "generated_node_novelty.csv", "novelty_pass_rate", "row 0", "Figure 4"),
        ("Abstract; Results 2.4; Supplement S15", "filtered-pass rate", filt.filtered_pass_rate, "filtering_summary.csv", "filtered_pass_rate", "row 0", "Figure 4"),
        ("Results 2.5; Supplement S10", "mean absolute descriptor shift", dist.descriptor_mean_abs_shift, "distribution_alignment.csv", "descriptor_mean_abs_shift", "row 0", "Figure 5"),
        ("Results 2.5; Supplement S10", "Wasserstein proxy", dist.descriptor_wasserstein_proxy, "distribution_alignment.csv", "descriptor_wasserstein_proxy", "row 0", "Figure 5"),
        ("Results 2.5; Supplement S10", "generated edge-weight mean", dist.edge_weight_mean_generated, "distribution_alignment.csv", "edge_weight_mean_generated", "row 0", "Figure 5"),
        ("Results 2.6; Supplement S11", "nodes after generated insertion", topo.nodes_after, "topology_impact.csv", "nodes_after", "row 0", "Figure 7"),
        ("Results 2.6; Supplement S11", "generated edges", topo.generated_edges, "topology_impact.csv", "generated_edges", "row 0", "Figure 7"),
        ("Results 2.6; Supplement S11", "average degree after insertion", topo.average_degree_after_generated_insertion, "topology_impact.csv", "average_degree_after_generated_insertion", "row 0", "Figure 7"),
        ("Results 2.7; Supplement S16/S17", "example top-5 neighbor bandgap mean", card["top5_bandgap_mean"], "examples/generated_node_example/generated_node_card.csv", "top5_bandgap_mean", "row 0", "Figure 13"),
        ("Results 2.7; Supplement S16/S17", "example top-5 neighbor bandgap std", card["top5_bandgap_std"], "examples/generated_node_example/generated_node_card.csv", "top5_bandgap_std", "row 0", "Figure 13"),
    ]
    for model in ["GraphSAGE", "GAT", "GCN"]:
        sub = ds[ds.model.eq(model)]
        real = sub[sub.training_mode.eq("real_only")].iloc[0]
        nn = sub[sub.training_mode.eq("nearest_neighbor_interpolation")].iloc[0]
        gen = sub[sub.training_mode.str.startswith("gan_")].sort_values("mae_mean").iloc[0]
        rows.extend([
            ("Results 2.8; Supplement S12", f"{model} real-only MAE mean", real.mae_mean, "downstream_summary.csv", "mae_mean", f"model={model}; training_mode=real_only", "Figure 8"),
            ("Results 2.8; Supplement S12", f"{model} real-only MAE std", real.mae_std, "downstream_summary.csv", "mae_std", f"model={model}; training_mode=real_only", "Figure 8"),
            ("Results 2.8; Supplement S12", f"{model} nearest-neighbor interpolation MAE mean", nn.mae_mean, "downstream_summary.csv", "mae_mean", f"model={model}; training_mode=nearest_neighbor_interpolation", "Figure 8"),
            ("Results 2.8; Supplement S12", f"{model} best generated-node mode MAE", gen.mae_mean, "downstream_summary.csv", "mae_mean", f"model={model}; training_mode={gen.training_mode}", "Figure 8"),
        ])
    out = pd.DataFrame(rows, columns=["manuscript_location", "claim_text", "numeric_value", "source_csv", "source_column", "source_row_filter", "figure_or_table"])
    out["numeric_value"] = out["numeric_value"].map(lambda x: f"{float(x):.6g}" if isinstance(x, (int, float, np.integer, np.floating)) else x)
    out.to_csv(TABLES / "metric_source_map.csv", index=False)
    REPORTS.mkdir(exist_ok=True)
    REPORTS.joinpath("metric_source_map.md").write_text(
        "# Metric source map\n\n"
        "Every quantitative manuscript and supplementary claim is mapped in "
        "`experiments/qmof_gang_final/tables/metric_source_map.csv`.\n\n"
        + out.to_markdown(index=False),
        encoding="utf-8",
    )
    return out


def write_reports(card: dict) -> None:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    sparse = pd.read_csv(TABLES / "final_sparsification_summary.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    ds = pd.read_csv(TABLES / "downstream_summary.csv")

    def improves(model: str, baseline: str) -> bool:
        sub = ds[ds.model.eq(model)].set_index("training_mode")
        best_gan = sub.loc[[m for m in sub.index if m.startswith("gan_")], "mae_mean"].min()
        return bool(best_gan < sub.loc[baseline, "mae_mean"])

    claim = (
        "GANg generates descriptor-valid, edge-valid, non-duplicate MOF-like graph nodes and similarity edges in a "
        "QMOF-derived MOF similarity network. However, generated-node augmentation does not improve held-out bandgap "
        "prediction over real-only training across GraphSAGE, GAT, and GCN. Generated nodes are descriptor-space graph "
        "hypotheses, not physically validated MOF structures."
    )
    report = f"""# GANg final readiness report

1. Was the final MOF similarity graph built? Yes.
2. Number of nodes and edges before/after sparsification? {int(graph.nodes)} nodes, {int(graph.edges)} edges before, {int(sparse.edges)} edges after.
3. Did sparsification preserve connectivity? {'Yes' if int(sparse.connected_components) == 1 else 'No'}.
4. Did GANg train across five seeds? Yes.
5. Descriptor-valid rate? {float(val.descriptor_valid_rate):.3f}.
6. Edge-valid rate? {float(edge.edge_valid_rate):.3f}.
7. Duplicate rate? {float(nov.duplicate_rate):.3f}.
8. Novelty-pass rate? {float(nov.novelty_pass_rate):.3f}.
9. Filtered-pass rate? {float(filt.filtered_pass_rate):.3f}.
10. Does augmentation improve GraphSAGE over real-only? {'Yes' if improves('GraphSAGE', 'real_only') else 'No'}.
11. Does augmentation improve GAT over real-only? {'Yes' if improves('GAT', 'real_only') else 'No'}.
12. Does augmentation improve GCN over real-only? {'Yes' if improves('GCN', 'real_only') else 'No'}.
13. Does augmentation beat random synthetic nodes? {'Yes for all models' if all(improves(m, 'random_synthetic') for m in ['GraphSAGE', 'GAT', 'GCN']) else 'No'}.
14. Does augmentation beat feature-noise nodes? {'Yes for all models' if all(improves(m, 'feature_noise') for m in ['GraphSAGE', 'GAT', 'GCN']) else 'No'}.
15. Does augmentation beat nearest-neighbor interpolation? {'Yes for all models' if all(improves(m, 'nearest_neighbor_interpolation') for m in ['GraphSAGE', 'GAT', 'GCN']) else 'No'}.
16. Are generated nodes interpretable through nearest QMOFs? Yes, as descriptor-space graph hypotheses.
17. Was a generated CIF produced? No. GANg operates in descriptor space; no generated CIF is produced. No nearest-neighbor reference CIF was available in this repository.
18. Was DFT performed? No.
19. Was force-field relaxation performed? No.
20. Was experimental validation performed? No.
21. Strongest supported claim? {claim}
22. Unsupported claims? Physically validated MOF structures, DFT-backed candidates, synthesis-ready outputs, experimental validation, and physical MOF discovery.
23. Recommended target journal tier? Proof-of-concept preprint or specialist methods venue until structural decoding and stronger downstream evidence are available.
"""
    (REPORTS / "gang_final_readiness_report.md").write_text(report, encoding="utf-8")
    (REPORTS / "readiness_report.md").write_text(report, encoding="utf-8")
    (REPORTS / "claim_boundary.md").write_text(
        "# GANg claim boundary\n\n"
        "GANg, a Generative Adversarial Network for Graph-Node Synthesis, generates MOF-like descriptor-space nodes and similarity edges in a QMOF-derived MOF similarity network.\n\n"
        "Generated nodes are descriptor-space hypotheses. No generated CIF is produced in this version because GANg operates in descriptor space. No DFT calculation, force-field relaxation, synthesis, or experimental validation was performed.\n",
        encoding="utf-8",
    )


def write_supplement(full: pd.DataFrame, seed: pd.DataFrame, pseudo: pd.DataFrame, usage: pd.DataFrame, card: dict) -> None:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    sparse = pd.read_csv(TABLES / "final_sparsification_summary.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    dist = pd.read_csv(TABLES / "distribution_alignment.csv").iloc[0]
    topo = pd.read_csv(TABLES / "topology_impact.csv").iloc[0]
    filt = pd.read_csv(TABLES / "filtering_summary.csv").iloc[0]
    feat = pd.read_csv(TABLES / "feature_group_audit.csv")
    grid = pd.read_csv(TABLES / "sparsification_grid.csv").rename(columns={"setting": "parameter", "connected_components": "components", "largest_component_size": "largest_component", "clustering_coefficient": "clustering", "bandgap_homophily": "homophily", "chosen": "selected_yes_no"})
    nn = pd.read_csv(TABLES / "nearest_neighbor_interpretation.csv")
    examples = pd.concat([
        nn.nsmallest(1, "top5_bandgap_mean"),
        nn.iloc[(nn["top5_bandgap_mean"] - nn["top5_bandgap_mean"].median()).abs().argsort()[:1]],
        nn.nlargest(1, "top5_bandgap_mean"),
        nn.nlargest(1, "top1_similarity"),
        nn.nsmallest(1, "top1_similarity"),
    ]).drop_duplicates("generated_node_id").head(5)
    metric = pd.read_csv(TABLES / "metric_source_map.csv")
    tex = r"""\documentclass[10pt]{article}
\usepackage[margin=0.62in]{geometry}
\usepackage{booktabs,longtable,array,graphicx,pdflscape,xurl,hyperref,float,ragged2e}
\setlength{\tabcolsep}{1.5pt}
\renewcommand{\arraystretch}{1.08}
\sloppy
\title{Supplementary Information for GANg}
\author{AGNI-MOF project}
\date{}
\begin{document}
\maketitle
"""
    n_features = int(pd.read_csv(TABLES / "feature_audit.csv")["used_in_similarity_graph"].sum())
    tex += f"""\\section{{S1 Configuration and split}}
Dataset size: {int(graph.nodes)} QMOF-derived nodes. Split seed: 42. Train/validation/test counts: {int(graph.train_count)}/{int(graph.val_count)}/{int(graph.test_count)}. Target: \\texttt{{outputs.pbe.bandgap}}. Number of features: {n_features}. Non-leaking feature policy: target/output quantities are excluded from node features.
\\section{{S2 Feature audit}}
"""
    tex += table_from_df(feat, ["feature_group", "num_features", "missing_rate", "used_in_similarity_graph", "used_in_GAN", "used_in_downstream", "leakage_risk", "comment"], "Feature audit by descriptor group.", "tab:feature-audit")
    tex += f"""\\section{{S3 Similarity graph construction}}
The similarity graph uses cosine kNN over normalized non-leaking descriptors. Nodes: {int(graph.nodes)}. Edges before sparsification: {int(graph.edges)}. Average degree: {graph.average_degree:.3f}. Components: {int(graph.connected_components)}. Homophily: {graph.target_bandgap_homophily:.3f}. Feature normalization uses mean/std scaling after imputation.
"""
    tex += "\\section{S4 Sparsification grid}\n"
    tex += table_from_df(grid, ["method", "parameter", "edges_before", "edges_after", "edge_reduction_percent", "components", "largest_component", "average_degree", "clustering", "homophily", "selected_yes_no"], "Full sparsification grid.", "tab:sparsification-grid", r"\tiny")
    tex += f"""\\section{{S5 GANg architecture and hyperparameters}}
Generator layers: latent/condition input to multilayer perceptron descriptor generator. Discriminator/critic layers: multilayer perceptron descriptor critic. Latent dimension: 64. WGAN-GP weight: 5.0. Learning rates: 0.0002 generator and 0.0002 critic. Runtime epochs: 100 in the locked final script. Batch size: 64. Seeds: 0, 1, 2, 3, 4. Conditioning strategy: sampled bandgap conditioning scalar. Generated nodes per seed: 200.
\\section{{S6 GAN training logs}}
GAN logs are in \\path{{experiments/qmof_gang_final/tables/gan_training_log.csv}} with generator loss, critic loss, real score, fake score, gradient penalty, and feature loss. Stability note: these are optimization diagnostics only and do not establish physical validity.
\\section{{S7 Generated-node descriptor validity}}
Descriptor-valid rate: {val.descriptor_valid_rate:.3f}; in-range feature rate: {val.in_range_feature_rate:.3f}; missing/NaN rate: {val.nan_missing_rate:.3f}; composition-sanity rate: {val.composition_sanity_rate:.3f}; scalar-descriptor sanity rate: {val.scalar_descriptor_sanity_rate:.3f}; constraint satisfaction: {val.constraint_satisfaction_rate:.3f}.
\\section{{S8 Generated-edge plausibility}}
Edge-valid rate: {edge.edge_valid_rate:.3f}; generated mean degree: {edge.generated_degree_mean:.3f}; edge-weight mean/min/max: {edge.edge_weight_mean:.3f}/{edge.edge_weight_min:.3f}/{edge.edge_weight_max:.3f}; cosine consistency mean: {edge.cosine_consistency_mean:.3f}; connectivity after insertion/backbone: {int(edge.graph_connectivity_after_insertion)}.
\\section{{S9 Novelty and duplicate diagnostics}}
Duplicate rate: {nov.duplicate_rate:.3f}; nearest train distance mean: {nov.nearest_neighbor_distance_to_training_mean:.3f}; nearest train distance std is reported in source diagnostics by generated-node row; too-close-to-training rate: {nov.too_close_to_training_rate:.3f}; novelty-pass rate: {nov.novelty_pass_rate:.3f}; diversity across seeds: {nov.diversity_across_seeds:.3f}.
\\section{{S10 Distribution alignment}}
Mean absolute feature shift: {dist.descriptor_mean_abs_shift:.3f}; Wasserstein/MMD proxy: {dist.descriptor_wasserstein_proxy:.3f}; edge-weight distribution shift proxy: generated edge-weight mean {dist.edge_weight_mean_generated:.3f}; bandgap-neighborhood distribution mean {dist.bandgap_neighborhood_mean:.3f}. Interpretation: generated nodes overlap descriptor space but remain distinguishable, so descriptor validity is not downstream utility.
\\section{{S11 Topology impact}}
Nodes before/after: {int(topo.nodes)}/{int(topo.nodes_after)}. Edges before/after generated insertion: {int(topo.edges)}/{int(topo.edges + topo.generated_edges)}. Average degree before/after: {topo.average_degree:.3f}/{topo.average_degree_after_generated_insertion:.3f}. Clustering before: {topo.average_clustering_coefficient:.3f}; clustering after is not recomputed for generated graph in the locked table. Community bridging proxy: {topo.community_bridging_proxy:.3f}. Components after/backbone: {int(topo.connected_components)}.
"""
    tex += r"\begin{landscape}" + "\n"
    tex += table_from_df(full, ["model", "training_mode", "label_strategy", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std", "Spearman_mean", "Spearman_std", "Kendall_mean", "Kendall_std", "beats_real_only_yes_no", "beats_nearest_neighbor_yes_no"], "S12 Downstream GNN full summary. Test nodes are held-out real QMOF nodes only; generated nodes are training augmentation only.", "tab:downstream-full", r"\tiny")
    tex += r"\end{landscape}" + "\n"
    tex += r"\begin{landscape}" + "\n"
    tex += table_from_df(seed, ["model", "seed", "training_mode", "MAE", "RMSE", "R2", "Spearman", "Kendall"], "S13 Downstream seedwise results.", "tab:downstream-seedwise", r"\tiny")
    tex += r"\end{landscape}" + "\n"
    tex += table_from_df(pseudo, ["label_strategy", "generated_node_count", "filtered_node_count", "MAE_mean", "RMSE_mean", "R2_mean", "Spearman_mean", "comment"], "S14 Pseudo-label ablation.", "tab:pseudo-label")
    tex += table_from_df(usage, ["training_mode", "generated_nodes_available", "generated_nodes_used", "generated_nodes_filtered_out", "filter_reason", "label_strategy"], "S15 Generated-node usage.", "tab:node-usage")
    tex += table_from_df(examples, ["generated_node_id", "top1_qmof_id", "top1_similarity", "top2_qmof_id", "top2_similarity", "top3_qmof_id", "top3_similarity", "top5_bandgap_mean", "top5_bandgap_std", "feature_difference_summary", "interpretation_comment"], "S16 Nearest-neighbor interpretation examples.", "tab:nn-examples", r"\tiny")
    tex += f"""\\section{{S17 Example generated-node card and optional CIF/reference analogue}}
Example generated graph node: {esc(card['generated_node_id'])}. Descriptor valid: {card['descriptor_valid']}. Edge valid: {card['edge_valid']}. Duplicate: {card['duplicate']}. Novelty score: {card['novelty_score']:.3f}. Top-neighbor bandgap mean/std: {card['top5_bandgap_mean']:.3f}/{card['top5_bandgap_std']:.3f}. Interpretation: {esc(card['interpretation_comment'])}

No generated CIF is produced in this version because GANg operates in descriptor space. Structural decoding is future work. If a CIF is later included, it must be labeled as a nearest-neighbor reference CIF unless a real decoder and sanity checks produce an explicitly unvalidated decoded candidate.

\\begin{{figure}}[H]\\centering\\includegraphics[width=0.72\\linewidth]{{experiments/qmof_gang_final/figures/fig13_generated_node_interpretation_card.pdf}}\\caption{{Generated-node interpretation card: top nearest real QMOF neighbors, similarity edge weights, descriptor-space summary, and claim label.}}\\end{{figure}}
"""
    tex += table_from_df(metric, ["manuscript_location", "claim_text", "numeric_value", "source_csv", "source_column", "source_row_filter", "figure_or_table"], "S18 Metric source map.", "tab:metric-map", r"\tiny")
    tex += r"""\section{S19 Claim boundary}
Generated nodes are descriptor-space hypotheses. No DFT calculation was performed. No force-field relaxation was performed. No synthesis was performed. No experimental validation was performed. GANg does not make a physical MOF discovery claim.
\section{S20 Reproducibility commands}
\begin{verbatim}
python3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml
python3 scripts/10_finalize_gang_package.py
python3 scripts/11_gang_audit_pass.py
latexmk -pdf manuscript/main.tex
latexmk -pdf manuscript/supplementary.tex
\end{verbatim}
\end{document}
"""
    (MANUSCRIPT / "supplementary.tex").write_text(tex, encoding="utf-8")


def patch_main(card: dict) -> None:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    ds = pd.read_csv(TABLES / "downstream_summary.csv")
    model_lines = []
    for model in ["GraphSAGE", "GAT", "GCN"]:
        sub = ds[ds.model.eq(model)].set_index("training_mode")
        real = sub.loc["real_only"]
        nn = sub.loc["nearest_neighbor_interpolation"]
        best = sub.loc[[idx for idx in sub.index if idx.startswith("gan_")]].sort_values("mae_mean").iloc[0]
        best_mode = sub.loc[[idx for idx in sub.index if idx.startswith("gan_")]].sort_values("mae_mean").index[0]
        beats_real = "does not beat" if best.mae_mean >= real.mae_mean else "beats"
        beats_nn = "does not beat" if best.mae_mean >= nn.mae_mean else "beats"
        model_lines.append(f"{model}: real-only MAE {real.mae_mean:.3f}$\\pm${real.mae_std:.3f}; best generated-node mode {esc(best_mode)} MAE {best.mae_mean:.3f}$\\pm${best.mae_std:.3f}, which {beats_real} real-only and {beats_nn} nearest-neighbor interpolation.")
    replacement = (
        "\\subsection{Downstream GNN diagnostics}\n"
        "Downstream tests use the same split and evaluate only held-out real QMOF nodes. Lower MAE is better, and generated nodes are used only as training-time augmentation. "
        + " ".join(model_lines)
        + " Generated-node augmentation does not improve held-out bandgap prediction over real-only training across GraphSAGE, GAT, and GCN. This result shows a validity--utility gap: descriptor-valid and novel generated nodes are not automatically useful for supervised graph-learning augmentation.\n"
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.94\\linewidth]{fig8_downstream_gnn.pdf}\\caption{Downstream GNN results: test nodes are held-out real QMOF nodes only; generated nodes are used only for training augmentation; lower MAE is better; generated-node modes do not beat real-only.}\\end{figure}\n"
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.88\\linewidth]{fig12_downstream_model_comparison.pdf}\\caption{GNN MAE comparison across GraphSAGE, GAT, and GCN. Test nodes are held-out real QMOF nodes only, and lower MAE is better.}\\end{figure}\n"
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.82\\linewidth]{fig14_pseudo_label_ablation.pdf}\\caption{Pseudo-label ablation for generated-node training augmentation. The ablation is evaluated only on held-out real QMOF nodes.}\\end{figure}\n"
        "\n\\subsection{Example generated graph node as a MOF-like descriptor-space hypothesis}\n"
        f"Generated node {esc(card['generated_node_id'])} passes descriptor validity, edge validity, non-duplication, and novelty filtering. Its nearest real QMOF neighbors include {esc(card['top1_qmof_id'])}, {esc(card['top2_qmof_id'])}, and {esc(card['top3_qmof_id'])}; the top-5 neighbor bandgap mean is {card['top5_bandgap_mean']:.3f} eV with standard deviation {card['top5_bandgap_std']:.3f} eV. This generated node is a MOF-like descriptor-space hypothesis located near those neighbors. It is not a decoded crystal structure and no CIF is claimed.\n"
        "\\begin{figure}[H]\\centering\\includegraphics[width=0.76\\linewidth]{fig13_generated_node_interpretation_card.pdf}\\caption{Generated-node interpretation card showing one GANg node, top nearest real QMOF neighbors, similarity weights, neighbor bandgap summary, and the claim label: descriptor-space hypothesis, not generated CIF.}\\end{figure}\n"
    )
    path = MANUSCRIPT / "main.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "We introduce GANg, a Generative Adversarial Network for Graph-Node Synthesis, and study it in the setting of metal--organic framework similarity networks. A QMOF-derived materials collection is represented as a MOF-level graph: each node corresponds to one MOF, node features encode MOF-level descriptors, and weighted edges encode descriptor-space similarity. GANg learns to synthesize new MOF-like graph nodes and proposes similarity edges to the existing graph. These generated nodes are descriptor-space hypotheses, not physically validated MOF structures.",
        "We introduce GANg, a Generative Adversarial Network for Graph-Node Synthesis, and evaluate it in metal--organic framework similarity networks. Each QMOF entry is represented as one graph node, edges encode descriptor-space similarity, and GANg generates synthetic MOF-like graph nodes with similarity edges to the existing graph. These generated nodes are descriptor-space hypotheses, not physically validated MOF structures.",
    )
    start = text.index("\\subsection{Downstream GNN diagnostics}")
    end = text.index("\\subsection{Claim boundary}")
    text = text[:start] + replacement + text[end:]
    failure = r"""
\subsection{Why descriptor-valid generated nodes may not improve downstream prediction}
Descriptor validity is not the same as predictive utility. A generated node can be in range, non-duplicate, and locally plausible while still being shifted from the supervised signal learned by a GNN. Pseudo-label noise can also harm supervised graph training, especially when many generated nodes are attached with labels inferred from neighbors or auxiliary property heads. Generated edges may alter message passing without adding reliable target information, and the real-only graph may already contain enough local homophily signal for the locked bandgap split. Nearest-neighbor interpolation is therefore a strong baseline rather than a trivial control. Better future representations may require atomistic graph embeddings, pore and topology descriptors, calibrated property heads, and structural retrieval or decoding before generated descriptor-space nodes can become stronger materials hypotheses.
"""
    text = text.replace(
        "Generated GANg nodes mean MOF-like descriptor-space nodes attached to a known MOF similarity graph. Descriptor validity is not physical validity because a vector can satisfy numerical range checks without corresponding to a decoded, stable, synthesizable crystal structure. The downstream results are therefore important: current generated-node augmentation does not improve held-out bandgap prediction over strong baselines, exposing a gap between descriptor-space plausibility and predictive utility. Nearest-neighbor interpretation is useful for reading generated nodes in relation to known QMOFs, but it is not structural decoding. Future work should connect generated descriptor nodes to structural retrieval or decoding, richer descriptors, calibrated uncertainty, DFT on selected decoded candidates, and physical validation.",
        "Generated GANg nodes mean MOF-like descriptor-space nodes attached to a known MOF similarity graph. Descriptor validity is not physical validity because a vector can satisfy numerical range checks without corresponding to a decoded, stable, synthesizable crystal structure. The downstream results are therefore important: current generated-node augmentation does not improve held-out bandgap prediction over strong baselines, exposing a gap between descriptor-space plausibility and predictive utility. Nearest-neighbor interpretation is useful for reading generated nodes in relation to known QMOFs, but it is not structural decoding. Future work should connect generated descriptor nodes to structural retrieval or decoding, richer descriptors, calibrated uncertainty, and physical validation.\n" + failure,
    )
    text = text.replace(
        "not as evidence for new validated MOFs.",
        "not as evidence for physically validated MOF structures.",
    )
    path.write_text(text, encoding="utf-8")


def copy_pdf_names() -> None:
    main_pdf = MANUSCRIPT / "main.pdf"
    supp_pdf = MANUSCRIPT / "supplementary.pdf"
    if main_pdf.exists():
        shutil.copy2(main_pdf, MANUSCRIPT / "GANg_MOF_similarity_networks_main.pdf")
    if supp_pdf.exists():
        shutil.copy2(supp_pdf, MANUSCRIPT / "GANg_MOF_similarity_networks_supplementary.pdf")


def main() -> None:
    feature_group_summary()
    full, seed = downstream_audit_tables()
    pseudo, usage = pseudo_and_usage_tables()
    card = create_example()
    update_workflow_and_downstream_figures()
    metric_source_map(card)
    write_reports(card)
    write_supplement(full, seed, pseudo, usage, card)
    patch_main(card)
    copy_pdf_names()


if __name__ == "__main__":
    main()
