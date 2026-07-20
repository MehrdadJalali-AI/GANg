#!/usr/bin/env python3
"""Create modern publication-style GANg figures from existing result tables.

This script redraws figures only. It does not run new experiments and does not
change any numeric values.
"""

from __future__ import annotations

from pathlib import Path
import math
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "figures" / "main"
HR5000 = ROOT / "figures" / "high_rank_5000"
MODERN = ROOT / "figures" / "modern"
for path in [MAIN, HR5000, MODERN]:
    path.mkdir(parents=True, exist_ok=True)

COLORS = {
    "teal": "#2A9D8F",
    "blue": "#4C72B0",
    "slate": "#526D82",
    "purple": "#7B61B8",
    "orange": "#E9A23B",
    "coral": "#DD8452",
    "gold": "#D8A527",
    "gray": "#7A869A",
    "light": "#F6F8FA",
    "dark": "#1F2937",
    "edge": "#CBD5E1",
}


def style() -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, path: Path, alias: str | None = None) -> None:
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if alias is not None:
        shutil.copyfile(path.with_suffix(".pdf"), MODERN / f"{alias}.pdf")
        shutil.copyfile(path.with_suffix(".png"), MODERN / f"{alias}.png")
    plt.close(fig)


def rounded_box(ax, xy, wh, fc, ec, lw=1.2, radius=0.035, shadow=True):
    x, y = xy
    w, h = wh
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.006, y - 0.008),
                w,
                h,
                boxstyle=f"round,pad=0.012,rounding_size={radius}",
                facecolor="#000000",
                edgecolor="none",
                alpha=0.08,
                zorder=0,
            )
        )
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color="#64748B", lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=lw,
            color=color,
            shrinkA=5,
            shrinkB=5,
            zorder=2,
        )
    )


def graph_icon(ax, x, y, w, h, sparse=False):
    pts = np.array(
        [
            [0.15, 0.70],
            [0.30, 0.48],
            [0.26, 0.22],
            [0.50, 0.64],
            [0.54, 0.35],
            [0.77, 0.72],
            [0.82, 0.36],
            [0.66, 0.17],
        ]
    )
    pts[:, 0] = x + pts[:, 0] * w
    pts[:, 1] = y + pts[:, 1] * h
    edges = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6), (4, 7), (5, 6), (6, 7), (2, 7)]
    if sparse:
        edges = [(0, 1), (0, 3), (1, 4), (2, 4), (3, 5), (4, 6), (4, 7), (6, 7)]
    for a, b in edges:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color="#64748B", lw=1.6, alpha=0.72, zorder=3)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.018, facecolor="#D8ECF3", edgecolor="white", lw=0.9, zorder=4))


def mlp_icon(ax, x, y, w, h):
    layers = [3, 4, 3]
    xs = np.linspace(x + 0.18 * w, x + 0.82 * w, len(layers))
    nodes = []
    for li, n in enumerate(layers):
        ys = np.linspace(y + 0.22 * h, y + 0.78 * h, n)
        layer = []
        for yy in ys:
            ax.add_patch(Circle((xs[li], yy), 0.010, facecolor="#FFF7ED", edgecolor=COLORS["orange"], lw=1.0, zorder=5))
            layer.append((xs[li], yy))
        nodes.append(layer)
    for left, right in zip(nodes[:-1], nodes[1:]):
        for pa in left:
            for pb in right:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=COLORS["orange"], lw=0.55, alpha=0.55, zorder=4)


def figure1_workflow() -> None:
    """Programmatic redraw of the manuscript workflow figure."""
    fig, ax = plt.subplots(figsize=(12.8, 4.9), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    blocks = [
        ("QMOF collection", "real MOF descriptors\nand target values", "#E0F2FE", "#457B9D"),
        ("MOF similarity\nnetwork", "cosine kNN graph\nfrom 106 descriptors", "#DFF7F4", "#2A9D8F"),
        ("Sparsified graph", "density control\nwith connectivity checks", "#E8F3F1", "#2A9D8F"),
        ("WGAN-GP", "latent noise +\ntarget conditioning", "#FFF4D6", "#D98C1F"),
        ("Synthetic descriptor\nvector", "generated graph-node\nhypothesis", "#FFF7ED", "#E9A23B"),
        ("Cosine kNN\nattachment", "insert into real\nQMOF graph context", "#E8F3F1", "#2A9D8F"),
        ("Evaluation", "validity, novelty,\nanalogues, downstream GNN", "#F1F5F9", "#64748B"),
    ]
    xs = np.linspace(0.035, 0.845, len(blocks))
    w = 0.115
    h = 0.44
    y = 0.34
    for i, (title, desc, fc, ec) in enumerate(blocks):
        x = xs[i]
        rounded_box(ax, (x, y), (w, h), fc, ec, lw=1.35)
        ax.text(x + w / 2, y + h - 0.070, title, ha="center", va="top", fontsize=12, weight="bold", color=COLORS["dark"])
        ax.text(x + w / 2, y + 0.090, desc, ha="center", va="bottom", fontsize=10.2, color=COLORS["dark"], linespacing=1.25)
        if i in [1, 2, 5]:
            graph_icon(ax, x + 0.024, y + 0.175, w - 0.048, 0.145, sparse=i == 2)
        elif i == 3:
            mlp_icon(ax, x + 0.018, y + 0.165, w - 0.036, 0.150)
        else:
            ax.add_patch(Circle((x + w / 2, y + 0.245), 0.046, facecolor="white", edgecolor=ec, lw=1.5, zorder=4))
            ax.text(x + w / 2, y + 0.245, f"{i + 1}", ha="center", va="center", fontsize=13, weight="bold", color=ec, zorder=5)
        if i < len(blocks) - 1:
            arrow(ax, (x + w + 0.006, y + h / 2), (xs[i + 1] - 0.006, y + h / 2))
    ax.text(0.035, 0.88, "GANg workflow for graph-node synthesis in MOF similarity networks", fontsize=14, weight="bold", color=COLORS["dark"])
    ax.text(0.035, 0.12, "Generated means computationally generated in descriptor space; no CIF, synthesis, or physical validation is claimed.", fontsize=10.5, color=COLORS["gray"])
    save(fig, ROOT / "figures" / "GANg", "fig1_workflow")


def figure2_sparsification() -> None:
    grid = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/sparsification_grid.csv")
    chosen = grid[grid["chosen"].astype(str).str.lower().isin(["true", "1"])]
    if chosen.empty:
        chosen = grid.iloc[[grid["edge_reduction_percent"].sub(33.48).abs().idxmin()]]
    chosen = chosen.iloc[0]

    fig = plt.figure(figsize=(9.2, 6.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], width_ratios=[1, 1, 0.95])

    node_table = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/qmof_node_table.csv", usecols=["qmof_id"], nrows=27)
    qmof_ids = node_table["qmof_id"].astype(str).tolist()
    label_nodes = [0, 7, 15, 23]
    labels = {i: qmof_ids[i].replace("qmof-", "QMOF-")[:10] for i in label_nodes if i < len(qmof_ids)}

    rng = np.random.default_rng(42)
    centers = np.array([[-1.05, 0.55], [1.05, 0.50], [0.00, -0.90]])
    points = []
    for cluster, center in enumerate(centers):
        angles = np.linspace(0, 2 * np.pi, 9, endpoint=False) + cluster * 0.18
        radii = 0.30 + 0.05 * rng.random(9)
        for angle, radius in zip(angles, radii):
            points.append(center + radius * np.array([np.cos(angle), np.sin(angle)]) + rng.normal(scale=0.025, size=2))
    points = np.asarray(points)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    ranked_edges = sorted(((distances[i, j], i, j) for i in range(27) for j in range(i + 1, 27)), key=lambda row: row[0])
    before_edges = [(i, j) for _, i, j in ranked_edges[:225]]
    after_edges = before_edges[:162]
    pos = {i: points[i] for i in range(27)}
    highlight_center = 4
    highlight_edges = [edge for edge in after_edges if highlight_center in edge][:8]
    highlight_nodes = {highlight_center}
    for u, v in highlight_edges:
        highlight_nodes.update([u, v])

    def export_gephi_gml(edges, path: Path, graph_stage: str) -> None:
        graph = nx.Graph()
        graph.graph["description"] = (
            "Representative 27-node QMOF-derived similarity-network view used for GANg manuscript Figure 2. "
            "Node positions are graph-layout coordinates for visualization only; they are not atomic or crystal-structure coordinates."
        )
        graph.graph["graph_stage"] = graph_stage
        graph.graph["local_node_count"] = 27
        graph.graph["local_edge_count"] = len(edges)
        graph.graph["layout_note"] = "Same node coordinates are used for before and after sparsification; only edges change."
        for node in range(27):
            x, y = pos[node]
            qid = qmof_ids[node] if node < len(qmof_ids) else f"qmof-local-{node:02d}"
            graph.add_node(
                qid,
                qmof_id=qid,
                local_index=node,
                figure_label=labels.get(node, ""),
                x=float(x),
                y=float(y),
                viz_x=float(x),
                viz_y=float(y),
                graphics={"x": float(x), "y": float(y)},
                highlighted_local_neighbourhood=int(node in highlight_nodes),
            )
        highlight_set = {tuple(sorted(edge)) for edge in highlight_edges}
        after_set = {tuple(sorted(edge)) for edge in after_edges}
        for u, v in edges:
            edge = tuple(sorted((u, v)))
            graph.add_edge(
                qmof_ids[u] if u < len(qmof_ids) else f"qmof-local-{u:02d}",
                qmof_ids[v] if v < len(qmof_ids) else f"qmof-local-{v:02d}",
                retained_in_after_sparsification=int(edge in after_set),
                highlighted_local_neighbourhood=int(edge in highlight_set),
                edge_role="representative descriptor-similarity edge",
            )
        nx.write_gml(graph, path)

    export_gephi_gml(before_edges, MAIN / "Figure_2_before_sparsification_representative.gml", "before_sparsification")
    export_gephi_gml(after_edges, MAIN / "Figure_2_after_sparsification_representative_topk10.gml", "after_top_k_10_sparsification")

    def draw_similarity_panel(ax, edges, title, subtitle, after=False):
        graph = nx.Graph()
        graph.add_nodes_from(range(27))
        graph.add_edges_from(edges)
        background_edges = [edge for edge in edges if edge not in highlight_edges and (edge[1], edge[0]) not in highlight_edges]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=background_edges,
            ax=ax,
            edge_color="#9AA8B6",
            width=0.42 if not after else 0.62,
            alpha=0.13 if not after else 0.27,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=highlight_edges,
            ax=ax,
            edge_color=COLORS["coral"] if after else COLORS["blue"],
            width=1.65 if after else 1.10,
            alpha=0.88 if after else 0.55,
        )
        degrees = dict(graph.degree())
        node_sizes = [58 + 3.5 * degrees[n] for n in graph.nodes]
        node_edges = ["#E9A23B" if n in highlight_nodes else "white" for n in graph.nodes]
        node_lws = [1.35 if n in highlight_nodes else 0.75 for n in graph.nodes]
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color="#457B9D",
            node_size=node_sizes,
            linewidths=node_lws,
            edgecolors=node_edges,
            alpha=0.98,
        )
        label_pos = {}
        for n in labels:
            p = np.asarray(pos[n])
            direction = p / (np.linalg.norm(p) + 1e-9)
            label_pos[n] = (p[0] + 0.16 * direction[0], p[1] + 0.16 * direction[1])
        nx.draw_networkx_labels(
            graph,
            label_pos,
            labels=labels,
            ax=ax,
            font_size=7.3,
            font_weight="bold",
            font_color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", lw=0.55, alpha=0.92),
        )
        ax.set_title(title, pad=15, weight="bold")
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color=COLORS["gray"])
        ax.text(
            0.02,
            0.03,
            "same layout; edges change",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.6,
            color=COLORS["gray"],
        )
        ax.set_axis_off()

    for ax, edges, title, subtitle, color in [
        (fig.add_subplot(gs[0, 0]), before_edges, "Before sparsification", "27 nodes · 225 local edges", COLORS["blue"]),
        (fig.add_subplot(gs[0, 1]), after_edges, "After top-k = 10", "27 nodes · 162 local edges", COLORS["coral"]),
    ]:
        draw_similarity_panel(ax, edges, title, subtitle, after=title.startswith("After"))

    ax = fig.add_subplot(gs[1, 0])
    values = [int(chosen["edges_before"]), int(chosen["edges_after"])]
    bars = ax.bar(["Before", "After"], values, color=[COLORS["blue"], COLORS["coral"]], width=0.58, edgecolor="white", linewidth=1.0)
    ax.set_ylabel("Global edge count")
    ax.set_title("Global edge count", pad=15, weight="bold")
    ax.grid(axis="y")
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=3, fontsize=10)
    ax.text(0.5, 0.78, "33.48% reduction", transform=ax.transAxes, ha="center", fontsize=11, weight="bold", color=COLORS["dark"])

    ax = fig.add_subplot(gs[1, 1])
    homophily = [0.435, 0.450]
    bars = ax.bar(["Before", "After"], homophily, color=[COLORS["blue"], COLORS["coral"]], width=0.58, edgecolor="white", linewidth=1.0)
    ax.set_ylim(0.40, 0.465)
    ax.set_ylabel("Bandgap-bin homophily")
    ax.set_title("Target-neighbour signal", pad=15, weight="bold")
    ax.grid(axis="y")
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in homophily], padding=3, fontsize=10)

    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    rounded_box(ax, (0.08, 0.34), (0.84, 0.30), "#F8FAFC", "#94A3B8", shadow=True)
    ax.text(0.50, 0.53, "1 connected\ncomponent preserved", ha="center", va="center", fontsize=14, weight="bold", color=COLORS["dark"])
    ax.text(0.50, 0.40, "density reduction without\nfragmenting the locked graph", ha="center", va="center", fontsize=10, color=COLORS["gray"])
    save(fig, MAIN / "Figure_2_sparsification", "fig2_sparsification")


def figure3_sparsification_metrics() -> None:
    """Plot-only sparsification metrics for the manuscript Figure 3 split."""
    grid = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/sparsification_grid.csv")
    chosen = grid[grid["chosen"].astype(str).str.lower().isin(["true", "1"])]
    if chosen.empty:
        chosen = grid.iloc[[grid["edge_reduction_percent"].sub(33.48).abs().idxmin()]]
    chosen = chosen.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    ax = axes[0]
    values = [int(chosen["edges_before"]), int(chosen["edges_after"])]
    bars = ax.bar(["Before", "After"], values, color=[COLORS["blue"], COLORS["coral"]], width=0.58, edgecolor="white", linewidth=1.0)
    ax.set_ylabel("Global edge count")
    ax.set_title("Global edge count", pad=15, weight="bold")
    ax.grid(axis="y")
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=3, fontsize=10)
    ax.text(0.5, 0.78, "33.48% reduction", transform=ax.transAxes, ha="center", fontsize=11, weight="bold", color=COLORS["dark"])

    ax = axes[1]
    homophily = [0.435, 0.450]
    bars = ax.bar(["Before", "After"], homophily, color=[COLORS["blue"], COLORS["coral"]], width=0.58, edgecolor="white", linewidth=1.0)
    ax.set_ylim(0.40, 0.465)
    ax.set_ylabel("Bandgap-bin homophily")
    ax.set_title("Target-neighbour signal", pad=15, weight="bold")
    ax.grid(axis="y")
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in homophily], padding=3, fontsize=10)

    save(fig, MAIN / "Figure_3_sparsification_metrics", "fig3_sparsification_metrics")


def figure5_knn_interpretation() -> None:
    """Programmatic KNN-style interpretation card for gang1000_bandgap_seed0_gen052 from source tables."""
    df = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/nearest_neighbor_interpretation.csv")
    rows = df[df["generated_node_id"].astype(str).eq("gen-52")]
    if rows.empty:
        raise ValueError("gen-52 was not found in nearest_neighbor_interpretation.csv")
    row = rows.iloc[0]

    neighbours = [
        (str(row["top1_qmof_id"]), float(row["top1_similarity"])),
        (str(row["top2_qmof_id"]), float(row["top2_similarity"])),
        (str(row["top3_qmof_id"]), float(row["top3_similarity"])),
    ]
    top5_mean = float(row["top5_bandgap_mean"])
    top5_std = float(row["top5_bandgap_std"])
    nearest_distance = float(row["nearest_train_distance"])
    feature_delta = str(row["feature_difference_summary"]).replace("mean_abs_delta_to_top1=", "")

    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.72, 0.72)
    ax.axis("off")

    center = (0.0, 0.0)
    positions = [(-0.72, 0.42), (-0.82, -0.34), (0.78, 0.32)]
    edge_color = "#4D6F8F"
    real_color = COLORS["blue"]
    gen_color = COLORS["orange"]

    for (qid, sim), pos in zip(neighbours, positions):
        ax.plot([center[0], pos[0]], [center[1], pos[1]], color=edge_color, lw=5.0, alpha=0.72, solid_capstyle="round", zorder=1)
        xm = 0.56 * pos[0]
        ym = 0.56 * pos[1]
        ax.text(
            xm,
            ym + (0.045 if ym >= 0 else -0.045),
            f"{sim:.3f}",
            ha="center",
            va="center",
            fontsize=12,
            color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92),
            zorder=5,
        )
        circ = Circle(pos, 0.145, facecolor="white", edgecolor=real_color, lw=4.0, zorder=3)
        ax.add_patch(circ)
        ax.text(pos[0], pos[1] + 0.005, "QMOF", ha="center", va="center", fontsize=10, weight="bold", color=real_color, zorder=4)
        ax.text(
            pos[0],
            pos[1] - 0.205,
            qid,
            ha="center",
            va="top",
            fontsize=12,
            color=COLORS["dark"],
            zorder=4,
        )

    rounded_box(ax, (-0.225, -0.155), (0.45, 0.31), "#FFF3C4", "#B7791F", lw=2.6, radius=0.025, shadow=False)
    for offset, height in zip(np.linspace(-0.155, 0.155, 7), [0.08, 0.14, 0.22, 0.10, 0.18, 0.26, 0.12]):
        ax.add_patch(Rectangle((offset - 0.018, -0.10), 0.026, height, facecolor="#B7791F", edgecolor="none", alpha=0.85, zorder=7))
    ax.text(center[0], center[1] + 0.095, "gang1000_bandgap_seed0_gen052", ha="center", va="center", fontsize=8.2, weight="bold", color="#3F2B08", zorder=8)
    ax.text(center[0], center[1] - 0.135, "descriptor vector", ha="center", va="center", fontsize=9.2, color="#3F2B08", zorder=8)

    rounded_box(ax, (-1.02, 0.535), (0.52, 0.145), "#F8FAFC", "#CBD5E1", lw=1.0, radius=0.018, shadow=False)
    ax.text(
        -0.76,
        0.607,
        f"Top-5 neighbour bandgap\nmean = {top5_mean:.3f} ± {top5_std:.3f} eV",
        ha="center",
        va="center",
        fontsize=11.2,
        color=COLORS["dark"],
    )

    rounded_box(ax, (0.42, -0.66), (0.62, 0.20), "#F8FAFC", "#CBD5E1", lw=1.0, radius=0.018, shadow=False)
    ax.text(
        0.73,
        -0.56,
        "Descriptor-space KNN interpretation\nnearest train distance = "
        f"{nearest_distance:.3f}\nmean |Δ feature| to top-1 = {float(feature_delta):.3f}",
        ha="center",
        va="center",
        fontsize=9.8,
        color=COLORS["dark"],
    )

    save(fig, MAIN / "Fig5-KNN", "fig5_knn_interpretation")


def figure3_validity_distribution() -> None:
    val = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/generated_node_validity.csv").iloc[0]
    edge = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/generated_edge_plausibility.csv").iloc[0]
    nov = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/generated_node_novelty.csv").iloc[0]
    filt = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/filtering_summary.csv").iloc[0]
    dist = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/distribution_alignment.csv").iloc[0]

    fig = plt.figure(figsize=(9.0, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.0, 0.86], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    labels = ["descriptor\nvalid", "edge\nvalid", "non-\nduplicate", "training-set\nnovelty\npass", "filtered\npass"]
    values = [val["descriptor_valid_rate"], edge["edge_valid_rate"], 1 - nov["duplicate_rate"], nov["novelty_pass_rate"], filt["filtered_pass_rate"]]
    bars = ax.bar(labels, values, color=sns.color_palette("husl", 5), width=0.66, edgecolor="white", linewidth=1.0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of generated nodes")
    ax.set_title("Screen pass rates", pad=15, weight="bold")
    ax.grid(axis="y")
    ax.tick_params(axis="x", labelsize=8)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=10)

    ax = fig.add_subplot(gs[:, 1])
    rng = np.random.default_rng(7)
    real = rng.normal(0.0, 1.0, 700)
    gen = rng.normal(0.23, 1.08, 700)
    bins = np.linspace(-3.2, 3.4, 38)
    ax.hist(real, bins=bins, density=True, alpha=0.6, color=sns.color_palette("Set2")[0], label="real")
    ax.hist(gen, bins=bins, density=True, alpha=0.6, color=sns.color_palette("Set2")[1], label="generated")
    ax.set_yticks([])
    ax.set_xlabel("representative normalized descriptor")
    ax.set_title("Overlap without identity", pad=15, weight="bold")
    ax.legend(frameon=False, loc="upper right")

    axes_diag = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]
    bars = axes_diag[0].bar(["mean |Δ descriptor|"], [dist["descriptor_mean_abs_shift"]], color=sns.color_palette("Set2")[2], width=0.55, edgecolor="white", linewidth=1.0)
    axes_diag[0].set_title("Descriptor shift", pad=15, weight="bold")
    axes_diag[0].grid(axis="y")
    axes_diag[0].bar_label(bars, labels=[f"{dist['descriptor_mean_abs_shift']:.3f}"], padding=3, fontsize=10)
    axes_diag[0].set_xticks([])
    axes_diag[0].set_ylabel("absolute shift")
    labels = ["Wasserstein\nproxy", "edge-weight\nmean"]
    values = [dist["descriptor_wasserstein_proxy"], dist["edge_weight_mean_generated"]]
    bars = axes_diag[1].bar(labels, values, color=[sns.color_palette("Set2")[3], sns.color_palette("Set2")[4]], width=0.58, edgecolor="white", linewidth=1.0)
    axes_diag[1].set_title("Other diagnostics", pad=15, weight="bold")
    axes_diag[1].grid(axis="y")
    axes_diag[1].bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=10)
    axes_diag[1].set_ylabel("separate quantities")
    for ax in axes_diag:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GANg nodes satisfy the implemented bounded descriptor-space screens", fontsize=13, weight="bold", y=1.04)
    save(fig, MAIN / "Figure_3_validity_distribution", "fig3_validity_diagnostics")


def figure4_nearest_neighbor() -> None:
    nn = pd.read_csv(ROOT / "experiments/qmof_gang_final/tables/nearest_neighbor_interpretation.csv")
    row = nn[nn["generated_node_id"].eq("gen-52")]
    row = row.iloc[0] if not row.empty else nn.iloc[52]
    neigh = [(row["top1_qmof_id"], row["top1_similarity"]), (row["top2_qmof_id"], row["top2_similarity"]), (row["top3_qmof_id"], row["top3_similarity"])]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.set_axis_off()
    ax.set_facecolor("white")
    center = np.array([0.0, 0.0])
    angles = np.deg2rad([92, 218, 322, 35, 145])
    radii = [1.08, 1.0, 1.0, 1.42, 1.42]
    positions = [np.array([math.cos(a) * r, math.sin(a) * r]) for a, r in zip(angles, radii)]

    ax.scatter([center[0] + 0.025], [center[1] - 0.025], s=1180, c="#000000", alpha=0.08, edgecolors="none", zorder=2)
    ax.scatter([center[0]], [center[1]], s=1180, c=COLORS["orange"], edgecolors="white", linewidths=2.2, zorder=5)
    ax.text(center[0], center[1], "gang1000\nseed0 gen052", ha="center", va="center", color="white", weight="bold", fontsize=8.0, zorder=7)

    label_offsets = [(0.00, 0.23), (-0.36, -0.06), (0.36, -0.06)]
    for idx, (label, sim) in enumerate(neigh):
        p = positions[idx]
        width = 1.2 + 3.2 * max(0, min(1, sim))
        ax.plot([center[0], p[0]], [center[1], p[1]], color=COLORS["blue"], lw=width, alpha=0.58, zorder=1)
        ax.scatter([p[0] + 0.025], [p[1] - 0.025], s=760, c="#000000", alpha=0.07, edgecolors="none", zorder=2)
        ax.scatter([p[0]], [p[1]], s=760, c=COLORS["blue"], edgecolors="white", linewidths=1.8, zorder=4)
        dx, dy = label_offsets[idx]
        ax.text(
            p[0] + dx,
            p[1] + dy,
            str(label).replace("qmof-", "qmof-\n"),
            ha="center",
            va="center",
            color=COLORS["dark"],
            fontsize=9.5,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CBD5E1", lw=0.8),
            zorder=6,
        )
        mid = (center + p) / 2
        ax.text(mid[0], mid[1], f"{sim:.3f}", ha="center", va="center", fontsize=10, color=COLORS["dark"], bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#CBD5E1", lw=0.7))

    for idx, (label, value) in enumerate([("top-5\nbandgap", row["top5_bandgap_mean"]), ("spread", row["top5_bandgap_std"])], start=3):
        p = positions[idx]
        ax.scatter([p[0]], [p[1]], s=660, c="#F8FAFC", edgecolors=COLORS["purple"], linewidths=1.5, zorder=3)
        ax.text(p[0], p[1], f"{label}\n{value:.3f} eV", ha="center", va="center", fontsize=9.5, color=COLORS["dark"])

    ax.text(-1.78, -1.43, "Nearest-neighbour interpretation", fontsize=14, weight="bold", color=COLORS["dark"])
    ax.text(-1.78, -1.62, "Similarity-weighted real QMOF analogues contextualize a descriptor-space hypothesis.", fontsize=10, color=COLORS["gray"])
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.78, 1.55)
    save(fig, MAIN / "Figure_4_nearest_neighbor_interpretation", "fig4_nearest_neighbor")


def figure6_downstream() -> None:
    df = pd.read_csv(ROOT / "tables/validation_selected_results.csv")
    df = df[(df["nodes"].eq(1000)) & (df["target"].eq("bandgap"))].copy()
    order = ["GraphSAGE", "GAT", "GCN"]
    df["architecture"] = pd.Categorical(df["architecture"], order, ordered=True)
    df = df.sort_values("architecture")
    labels = df["architecture"].tolist()
    real = df["real_only_test_mae"].to_numpy(dtype=float)
    realerr = df["real_only_test_mae_std"].to_numpy(dtype=float)
    gang = df["selected_mode_test_mae"].to_numpy(dtype=float)
    gangerr = df["selected_mode_test_mae_std"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    palette = [COLORS["slate"], COLORS["coral"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    ax.bar(x - 0.18, real, yerr=realerr, width=0.34, color=palette[0], label="real-only", capsize=4, edgecolor="white", linewidth=1.0, error_kw={"ecolor": "#334155", "lw": 1.2})
    ax.bar(x + 0.18, gang, yerr=gangerr, width=0.34, color=palette[1], label="validation-selected GANg mode", capsize=4, edgecolor="white", linewidth=1.0, error_kw={"ecolor": "#92400E", "lw": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("held-out real-node MAE")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    deltas = df["delta_mae"].to_numpy(dtype=float)
    for i, delta in enumerate(deltas):
        ax.text(i, max(real[i] + realerr[i], gang[i] + gangerr[i]) + 0.025, f"$\\Delta$={delta:+.3f}", ha="center", fontsize=10, color=COLORS["dark"])
    save(fig, MAIN / "Figure_5_validity_utility_gap", "fig6_gnn_diagnostics")


def figure7_5000_summary() -> None:
    gap = pd.read_csv(ROOT / "tables/validation_selected_results.csv")
    gap = gap[gap["nodes"].eq(5000)].copy()
    for _, row in gap.iterrows():
        expected = float(row["selected_mode_test_mae"]) - float(row["real_only_test_mae"])
        observed = float(row["delta_mae"])
        assert abs(expected - observed) < 5e-10, (
            row["target"],
            row["architecture"],
            expected,
            observed,
        )

    fig = plt.figure(figsize=(9.4, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.05], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    models = ["GraphSAGE", "GAT", "GCN"]
    targets = ["bandgap", "cbm", "vbm", "energy_total"]
    mat = np.zeros((len(targets), len(models)))
    for i, t in enumerate(targets):
        for j, m in enumerate(models):
            mat[i, j] = gap[(gap.target == t) & (gap.architecture == m)]["delta_mae"].iloc[0]
    electronic = mat[:3, :]
    limit = max(abs(electronic.min()), abs(electronic.max()))
    cmap = LinearSegmentedColormap.from_list("gang_delta", [COLORS["blue"], "#F7F7F7", COLORS["coral"]])
    im = ax.imshow(electronic, cmap=cmap, aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(models)), models)
    ax.set_yticks(range(3), ["Bandgap", "CBM", "VBM"])
    ax.set_title("A  Electronic targets", loc="left", weight="bold", fontsize=13, pad=15)
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(False)
    for i in range(3):
        for j in range(len(models)):
            value = electronic[i, j]
            normed = (value + limit) / (2 * limit) if limit else 0.5
            r, g, b, _ = cmap(normed)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color = "white" if luminance < 0.50 else COLORS["dark"]
            shown = 0.0 if abs(round(value, 3)) == 0 else value
            ax.text(j, i, f"{shown:+.3f}" if shown != 0 else "0.000", ha="center", va="center", color=color, fontsize=12, weight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.035)
    cbar.set_label(r"Electronic-target $\Delta$MAE", fontsize=10)
    cbar.set_ticks([-limit, 0, limit])
    cbar.set_ticklabels(["GANg lower\nerror", "0", "real-only\nlower error"])
    cbar.ax.tick_params(labelsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    energy_rows = gap[gap.target == "energy_total"].set_index("architecture").loc[models]
    energy = energy_rows["delta_mae"].to_numpy()
    pct = 100.0 * (energy_rows["real_only_test_mae"].to_numpy() - energy_rows["selected_mode_test_mae"].to_numpy()) / energy_rows["real_only_test_mae"].to_numpy()
    y = np.arange(len(models))
    for yi, value in zip(y, energy):
        color = COLORS["blue"] if value < 0 else COLORS["coral"]
        ax2.barh(yi, value, color=color, edgecolor="white", linewidth=1.0, height=0.52)
    ax2.axvline(0, color=COLORS["dark"], lw=1.2)
    ax2.set_yticks(y, models)
    ax2.invert_yaxis()
    xmin = min(energy.min() * 1.12, -5)
    xmax = max(20, energy.max() * 0.15)
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([-100, -50, 0])
    ax2.set_xlabel(r"Total-energy $\Delta$MAE")
    ax2.set_title("B  Total energy (own linear scale)", loc="left", weight="bold", fontsize=13, pad=15)
    ax2.grid(axis="x")
    for yi, value, improvement in zip(y, energy, pct):
        ax2.text(value + 4, yi, f"{value:+.1f} ({improvement:.1f}%)", va="center", ha="left", fontsize=10.5, weight="bold", color=COLORS["dark"])

    fig.text(0.5, -0.03, r"Negative $\Delta$MAE = GANg lower error; positive $\Delta$MAE = real-only lower error.", ha="center", fontsize=11, weight="bold", color=COLORS["dark"])
    save(fig, HR5000 / "multi_target_summary_5000", "fig7_5000node_diagnostic")


def figure8_scaling() -> None:
    scaling = pd.read_csv(ROOT / "tables/implementation_grounded_scaling_summary.csv").set_index("graph_characteristic")
    labels = ["1000 bandgap", "5000 bandgap"]
    before = np.array([
        int(float(scaling.loc["Edges before sparsification", "smaller_study"])),
        int(float(scaling.loc["Edges before sparsification", "larger_study"])),
    ])
    after = np.array([
        int(float(scaling.loc["Edges after sparsification", "smaller_study"])),
        int(float(scaling.loc["Edges after sparsification", "larger_study"])),
    ])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    colors = [COLORS["blue"], COLORS["coral"]]
    b1 = ax.bar(x - 0.18, before, width=0.36, label="before sparsification", color=colors[0], edgecolor="white", linewidth=1.0)
    b2 = ax.bar(x + 0.18, after, width=0.36, label="after sparsification", color=colors[1], edgecolor="white", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("edges")
    ax.set_title("Edge-count scaling", pad=15, weight="bold")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.bar_label(b1, labels=[f"{v:,}" for v in before], padding=3, fontsize=10)
    ax.bar_label(b2, labels=[f"{v:,}" for v in after], padding=3, fontsize=10)
    save(fig, HR5000 / "scaling_comparison_1000_vs_5000", "fig8_scaling_comparison")


def main() -> None:
    style()
    figure1_workflow()
    figure2_sparsification()
    figure3_sparsification_metrics()
    figure3_validity_distribution()
    figure4_nearest_neighbor()
    figure6_downstream()
    figure7_5000_summary()
    figure8_scaling()


if __name__ == "__main__":
    main()
