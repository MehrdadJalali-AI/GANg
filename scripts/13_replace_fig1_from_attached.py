from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "experiments" / "qmof_gang_final" / "figures"
MANUSCRIPT = ROOT / "manuscript"


def rounded_box(ax, xy, wh, fc, ec="#6f8fa5", lw=1.6, radius=0.06, linestyle="-"):
    box = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, color="#6f8fa5", lw=2.0, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw,
            color=color,
            linestyle=style,
            shrinkA=4,
            shrinkB=4,
        )
    )


def graph_icon(ax, x, y, w, h, sparse=False, inserted=False):
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
    edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 6),
        (6, 7),
        (2, 7),
    ]
    if sparse:
        edges = [(0, 1), (0, 3), (1, 4), (2, 4), (3, 5), (4, 6), (4, 7), (6, 7)]
    for a, b in edges:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color="#4d6f7b", lw=2.0, alpha=0.85)
    if sparse:
        for a, b in [(1, 3), (5, 6), (2, 7)]:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color="#9db2b8", lw=1.4, alpha=0.35, linestyle=":")
    colors = ["#c7d4e8", "#c7d4e8", "#c7d4e8", "#d8d0e6", "#c7d4e8", "#ead2cf", "#ead2cf", "#d8d0e6"]
    for (px, py), c in zip(pts, colors):
        ax.add_patch(Circle((px, py), 0.018, facecolor=c, edgecolor="#1d3340", lw=1.1, zorder=5))
    if inserted:
        gen = np.array([[0.10, 0.25], [0.82, 0.78], [0.88, 0.30]])
        gen[:, 0] = x + gen[:, 0] * w
        gen[:, 1] = y + gen[:, 1] * h
        for gx, gy in gen:
            ax.add_patch(Circle((gx, gy), 0.022, facecolor="#ffe3a4", edgecolor="#e67e22", lw=2.4, zorder=6))
        for g, p in [(0, 2), (0, 4), (1, 5), (1, 6), (2, 6), (2, 7)]:
            ax.plot(
                [gen[g, 0], pts[p, 0]],
                [gen[g, 1], pts[p, 1]],
                color="#d48a37",
                lw=1.6,
                alpha=0.9,
                linestyle="--",
            )


def mof_icon(ax, x, y, w, h):
    rng = np.random.default_rng(5)
    pts = rng.uniform([x + 0.10 * w, y + 0.18 * h], [x + 0.90 * w, y + 0.82 * h], size=(22, 2))
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.argsort(d)[1:3]:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#8d5d68", lw=1.5, alpha=0.65)
    for k, (px, py) in enumerate(pts):
        c = "#5fc28a" if k % 4 == 0 else "#55a6c9" if k % 5 == 0 else "#8c4962"
        ax.add_patch(Circle((px, py), 0.011 if k % 4 else 0.014, facecolor=c, edgecolor="white", lw=0.5, zorder=5))


def network_mini(ax, cx, cy, scale, orange=False):
    pts = np.array([[0, 0.15], [-0.06, -0.03], [0.07, -0.02], [0.0, -0.12], [0.12, 0.11]]) * scale
    pts[:, 0] += cx
    pts[:, 1] += cy
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (2, 4)]
    for a, b in edges:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color="#345a68", lw=1.1)
    for i, (px, py) in enumerate(pts):
        color = "#f4a340" if orange and i in (2, 4) else "#a9c7d8"
        ax.add_patch(Circle((px, py), 0.011 * scale, facecolor=color, edgecolor="#1d3340", lw=0.9))


def mlp_icon(ax, x, y, w, h, color="#8e5597"):
    layers = [3, 4, 3]
    xs = np.linspace(x + 0.18 * w, x + 0.82 * w, len(layers))
    nodes = []
    for li, n in enumerate(layers):
        ys = np.linspace(y + 0.22 * h, y + 0.78 * h, n)
        layer = []
        for yy in ys:
            ax.add_patch(Circle((xs[li], yy), 0.010, facecolor="#f3e9f3", edgecolor=color, lw=1.0, zorder=5))
            layer.append((xs[li], yy))
        nodes.append(layer)
    for a, b in zip(nodes[:-1], nodes[1:]):
        for pa in a:
            for pb in b:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=0.6, alpha=0.55)


def draw():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    blue = "#dff1f8"
    teal = "#dff7f4"
    purple = "#f5e9f6"
    amber = "#fff3d7"
    edge = "#6f8fa5"

    stage_y, stage_h = 0.58, 0.34
    xs = [0.02, 0.275, 0.53]
    ws = [0.215, 0.17, 0.17]
    titles = ["Stage 1 — QMOF Database", "Stage 2 — Similarity Graph", "Stage 3 — Sparsified Graph"]
    labels = ["QMOF database", "MOF similarity graph", "sparsified graph"]
    for x, w, title, label in zip(xs, ws, titles, labels):
        ax.text(x, 0.965, title, fontsize=8, ha="left", va="center", color="#111")
        rounded_box(ax, (x, stage_y), (w, stage_h), blue, ec="#7ea6ba", radius=0.008)
        ax.add_patch(plt.Rectangle((x, stage_y), w, 0.07, color="#c6e4f0", ec="none", alpha=0.8))
        ax.text(x + w / 2, stage_y + 0.035, label, fontsize=8, ha="center", va="center")
    mof_icon(ax, xs[0] + 0.02, stage_y + 0.09, ws[0] - 0.04, stage_h - 0.12)
    graph_icon(ax, xs[1] + 0.015, stage_y + 0.08, ws[1] - 0.03, stage_h - 0.11, sparse=False)
    graph_icon(ax, xs[2] + 0.015, stage_y + 0.08, ws[2] - 0.03, stage_h - 0.11, sparse=True)

    arrow(ax, (xs[0] + ws[0] + 0.012, stage_y + stage_h * 0.52), (xs[1] - 0.012, stage_y + stage_h * 0.52))
    arrow(ax, (xs[1] + ws[1] + 0.012, stage_y + stage_h * 0.52), (xs[2] - 0.012, stage_y + stage_h * 0.52))

    # Main Stage 4 panel
    x4, w4 = 0.735, 0.245
    ax.text(x4, 0.965, "Stage 4 — GANg · WGAN-GP", fontsize=8, ha="left", va="center", color="#111")
    rounded_box(ax, (x4, stage_y), (w4, stage_h), "#fbf5fb", ec="#7a4285", lw=2.0, radius=0.02, linestyle=(0, (4, 4)))
    ax.text(x4 + w4 / 2, stage_y + stage_h - 0.035, "GANg", fontsize=13, weight="bold", ha="center", color="#7a2f80")
    gen_box = rounded_box(ax, (x4 + 0.018, stage_y + 0.08), (0.105, 0.205), purple, ec="#b27ab7", radius=0.008)
    crit_box = rounded_box(ax, (x4 + 0.145, stage_y + 0.08), (0.085, 0.205), amber, ec="#e7b46a", radius=0.008)
    ax.text(x4 + 0.071, stage_y + 0.255, "Generator", ha="center", fontsize=7)
    ax.text(x4 + 0.187, stage_y + 0.255, "Critic", ha="center", fontsize=7)
    ax.text(x4 + 0.028, stage_y + 0.175, "noise z\n+ bandgap\nconditioning", fontsize=6.5, va="center")
    mlp_icon(ax, x4 + 0.058, stage_y + 0.105, 0.055, 0.13)
    ax.text(x4 + 0.115, stage_y + 0.170, r"$\hat{x}$", fontsize=7)
    network_mini(ax, x4 + 0.175, stage_y + 0.17, 0.38)
    ax.text(x4 + 0.218, stage_y + 0.17, "Wasserstein\nscore", fontsize=7, va="center")
    arrow(ax, (x4 + 0.123, stage_y + 0.17), (x4 + 0.145, stage_y + 0.17), color="#333", lw=1.4)
    arrow(ax, (x4 + 0.148, stage_y + 0.12), (x4 + 0.125, stage_y + 0.12), color="#333", lw=1.2, style="--")
    ax.text(x4 + 0.135, stage_y + 0.07, "loss signal", fontsize=8, ha="center")
    arrow(ax, (xs[2] + ws[2] + 0.012, stage_y + stage_h * 0.52), (x4 - 0.012, stage_y + stage_h * 0.52))

    # Lower panels
    y2, h2 = 0.08, 0.36
    ax.text(0.02, 0.515, "Zoom-in: GANg · WGAN-GP", fontsize=8, ha="left")
    rounded_box(ax, (0.02, y2), (0.44, h2), "#fbf5fb", ec="#7a4285", lw=2.0, radius=0.02, linestyle=(0, (4, 4)))
    rounded_box(ax, (0.04, y2 + 0.08), (0.18, 0.22), purple, ec="#b27ab7", radius=0.008)
    rounded_box(ax, (0.275, y2 + 0.08), (0.16, 0.22), amber, ec="#e7b46a", radius=0.008)
    ax.text(0.13, y2 + 0.27, "Generator", ha="center", fontsize=8)
    ax.text(0.355, y2 + 0.27, "Critic", ha="center", fontsize=8)
    ax.text(0.055, y2 + 0.17, "noise z\n+ bandgap\nconditioning", fontsize=8, va="center")
    mlp_icon(ax, 0.10, y2 + 0.105, 0.09, 0.15)
    ax.text(0.198, y2 + 0.17, "synthetic\nMOF-like\nnode " + r"$\hat{x}$", fontsize=8, va="center")
    arrow(ax, (0.225, y2 + 0.19), (0.275, y2 + 0.19), color="#333", lw=1.4)
    network_mini(ax, 0.33, y2 + 0.19, 0.54)
    ax.text(0.40, y2 + 0.19, "Wasserstein\nscore", fontsize=8, va="center")
    arrow(ax, (0.275, y2 + 0.12), (0.225, y2 + 0.12), color="#333", lw=1.2, style="--")
    ax.text(0.245, y2 + 0.075, "loss signal", fontsize=8, ha="center")
    ax.text(0.24, y2 + 0.025, "descriptor-space hypothesis", fontsize=7, ha="center", color="#6f4b77")

    ax.text(0.49, 0.515, "Stage 5 — Node Insertion", fontsize=8, ha="left")
    rounded_box(ax, (0.49, y2), (0.20, h2), teal, ec="#4a9b96", radius=0.008)
    graph_icon(ax, 0.505, y2 + 0.08, 0.17, 0.23, sparse=True, inserted=True)
    ax.add_patch(plt.Rectangle((0.49, y2), 0.20, 0.07, color="#bde6e0", ec="none", alpha=0.9))
    ax.text(0.59, y2 + 0.035, "synthetic nodes inserted\nvia cosine kNN", ha="center", va="center", fontsize=7)
    arrow(ax, (0.46, y2 + h2 * 0.50), (0.49, y2 + h2 * 0.50))

    ax.text(0.72, 0.515, "Stage 6 — Downstream GNN Evaluation", fontsize=8, ha="left")
    rounded_box(ax, (0.72, y2), (0.26, h2), teal, ec="#4a9b96", radius=0.008)
    for cx, name, orange in [(0.765, "GraphSAGE", False), (0.84, "GAT", True), (0.92, "GCN", False)]:
        network_mini(ax, cx, y2 + 0.285, 0.34, orange=orange)
        ax.text(cx, y2 + 0.21, name, fontsize=7, ha="center")
    ax.text(0.735, y2 + 0.155, "real-only", fontsize=8, va="center")
    ax.text(0.735, y2 + 0.105, "GANg augmented", fontsize=8, va="center")
    bar_x, bar_w = 0.83, 0.125
    real = 0.600 / 0.70 * bar_w
    gang = 0.630 / 0.70 * bar_w
    ax.add_patch(plt.Rectangle((bar_x, y2 + 0.135), real, 0.028, facecolor="#8db6d9", edgecolor="#333"))
    ax.add_patch(plt.Rectangle((bar_x, y2 + 0.085), gang, 0.028, facecolor="#4fb7b4", edgecolor="#333"))
    ax.text(bar_x + real + 0.006, y2 + 0.149, "0.600", fontsize=7, va="center")
    ax.text(bar_x + gang + 0.006, y2 + 0.099, "0.630", fontsize=7, va="center")
    ax.text(0.72 + 0.13, y2 + 0.052, "lower MAE is better · no improvement over real-only", fontsize=7.5, ha="center")
    ax.add_patch(plt.Rectangle((0.72, y2), 0.26, 0.05, color="#bde6e0", ec="none", alpha=0.9))
    ax.text(0.85, y2 + 0.025, "bandgap prediction · held-out test nodes", ha="center", va="center", fontsize=7)
    arrow(ax, (0.69, y2 + h2 * 0.50), (0.72, y2 + h2 * 0.50))

    for path in [
        FIG_DIR / "GANg_workflow_nature_style.pdf",
        FIG_DIR / "fig1_workflow.pdf",
    ]:
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
    for path in [
        FIG_DIR / "GANg_workflow_nature_style.png",
        FIG_DIR / "fig1_workflow.png",
        FIG_DIR / "GANg_workflow_nature_style.jpg",
    ]:
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.06,
            dpi=220,
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )
    plt.close(fig)


if __name__ == "__main__":
    draw()
