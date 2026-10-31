"""README overview figure for GANg (vector). Units are millimetres.

    python docs/figures/make_overview.py   ->  docs/figures/gang_overview.{svg,png}
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

plt.rcParams.update({"font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "svg.fonttype": "none",
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold"})

W, H = 250.0, 104.0
MM = 1 / 25.4
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

INK, MUTED, LINE = "#1F2933", "#5B6670", "#9AA5B1"
REAL, REAL_L = "#2E4A7D", "#D6E0F0"
SYN, SYN_L = "#E07B39", "#FBE3D3"
GEN, GEN_L = "#3B6FB6", "#E3ECF8"
OK = "#2A9D8F"
PANEL, GREY_L = "#F5F7FA", "#ECEEF1"
METAL, LINKER, PORE = "#4F7A95", "#6B7785", "#F4E3A1"
rng = np.random.default_rng(3)
ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=-5))


def box(x, y, w, h, fc="white", ec=LINE, lw=0.6, r=1.5, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def t(x, y, s, size=7.5, weight="normal", ha="center", va="center", color=INK, style="normal", z=8):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, fontstyle=style, zorder=z,
            linespacing=1.15)


def arrow(x1, y1, x2, y2, color=MUTED, lw=0.9, ms=6, z=5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def edge(p, q, color="#8C9BB4", lw=0.6, ls="-", z=3):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def node(x, y, r=1.5, fc=REAL, ec="white", lw=0.5, z=6):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))


def mof_icon(cx, cy, s, z=5):
    d = np.array([s * 0.36, s * 0.30])
    f = [np.array([cx - s / 2, cy - s / 2]), np.array([cx + s / 2, cy - s / 2]),
         np.array([cx + s / 2, cy + s / 2]), np.array([cx - s / 2, cy + s / 2])]
    b = [p + d for p in f]
    ax.add_patch(Circle((cx + d[0] / 2, cy + d[1] / 2), s * 0.30, fc=PORE, ec="none", zorder=z))
    diamond = lambda p, r: [p + [0, r], p + [r, 0], p + [0, -r], p + [-r, 0]]
    for i in range(4):
        edge(b[i], b[(i + 1) % 4], color="#B3BBC4", lw=0.6, z=z)
        edge(f[i], b[i], color="#97A1AC", lw=0.65, z=z + 1)
    for p in b:
        ax.add_patch(Polygon(diamond(p, s * .11), fc="#9DB4C4", ec="white", lw=0.3, zorder=z + 1))
    for i in range(4):
        p, q = f[i], f[(i + 1) % 4]
        edge(p, q, color=LINKER, lw=0.85, z=z + 2)
        m, r = (p + q) / 2, s * 0.10
        ang = np.pi / 6 + np.arange(6) * np.pi / 3
        ax.add_patch(Polygon(np.c_[m[0] + r * np.cos(ang), m[1] + r * np.sin(ang)], fc="white", ec=LINKER, lw=0.55,
                             zorder=z + 3))
    for p in f:
        ax.add_patch(Polygon(diamond(p, s * .14), fc=METAL, ec="white", lw=0.35, zorder=z + 4))


def vector(x, y, n, w, h, colors, horizontal=True):
    for i in range(n):
        xy = (x + i * w, y) if horizontal else (x, y + i * h)
        ax.add_patch(Rectangle(xy, w, h, fc=colors[i % len(colors)], ec="white", lw=0.3, zorder=5))


def card(x, w, num, title):
    box(x, 19, w, 66, fc=PANEL, ec="none", r=2.5, z=0)
    ax.add_patch(Circle((x + 5.5, 79.5), 3.0, fc=INK, ec="none", zorder=6))
    t(x + 5.5, 79.5, str(num), size=8.5, weight="bold", color="white")
    t(x + 10.5, 79.5, title, size=9, weight="bold", ha="left")


# ------------------------------------------------------------------ title
t(4, 97, "GANg", size=20, weight="bold", ha="left", color=GEN)
t(33, 98.6, "Target-conditioned, constraint-aware synthesis of auditable graph nodes", size=10.5, ha="left")
t(33, 93.4, "for metal–organic framework similarity networks", size=10.5, ha="left", color=MUTED)

# ------------------------------------------------------------------ 1 real MOFs
x1, w1 = 4, 50
card(x1, w1, 1, "Real MOFs")
mof_icon(x1 + 14, 64, 9.5)
mof_icon(x1 + 33, 64, 9.5)
t(x1 + w1 / 2, 52.5, "featurize", size=7.5, color=MUTED, style="italic")
arrow(x1 + w1 / 2, 55.5, x1 + w1 / 2, 47.5)
blues = ["#C9D7EC", "#9DB6DB", "#6F93C8", "#B3C6E3", "#87A5D2"]
vector(x1 + 6, 41, 19, 2.0, 4.2, blues)
t(x1 + w1 / 2, 35.2, "106 target-free descriptors", size=7.8, weight="bold")
t(x1 + w1 / 2, 30.6, "composition, chemistry roles,", size=7.3, color=MUTED)
t(x1 + w1 / 2, 26.8, "cell size, pores, density, symmetry", size=7.3, color=MUTED)
t(x1 + w1 / 2, 22.6, "target (e.g. band gap) = condition", size=7.3, color=SYN)

# ------------------------------------------------------------------ 2 generator
x2, w2 = 58, 80
card(x2, w2, 2, "Constraint-aware generator")
vector(x2 + 4, 60, 5, 3.0, 2.2, ["#C9D1DB", "#AEB8C5", "#D7DDE5"], horizontal=False)
t(x2 + 5.5, 73.0, r"$z$", size=9, weight="bold")
box(x2 + 3, 49.5, 5.2, 5.2, fc=SYN_L, ec=SYN, lw=0.8, r=0.8, z=5)
t(x2 + 5.6, 52.1, r"$c$", size=9, weight="bold", color=SYN)
t(x2 + 5.6, 45.8, "target", size=6.8, color=MUTED)
ax.add_patch(Polygon([[x2 + 12, 54], [x2 + 12, 68], [x2 + 21, 71], [x2 + 21, 51]], closed=True, fc=GEN_L, ec=GEN,
                     lw=0.9, zorder=4))
t(x2 + 16.5, 61, "G", size=11, weight="bold", color=GEN)
arrow(x2 + 7.5, 65, x2 + 11.6, 65)
arrow(x2 + 8.6, 53, x2 + 11.6, 56, color=SYN)
cx = x2 + 25
bars = np.zeros(13)
bars[[1, 4, 5, 9]] = [0.45, 0.2, 0.25, 0.12]
for i, v in enumerate(bars):
    ax.add_patch(Rectangle((cx + i * 1.4, 61.5), 1.0, 0.4 + v * 20 if v else 0.3, fc=GEN if v else LINE, ec="none",
                           zorder=5))
ax.plot([cx - .3, cx + 18.3], [61.3, 61.3], color=LINE, lw=0.5, zorder=4)
t(cx + 9, 58.6, "sparse composition", size=6.8, color=MUTED)
for i in range(6):
    ax.add_patch(Rectangle((cx + 1 + i * 2.8, 51.5), 2.3, 2.3, fc=GEN, ec="white", lw=0.3, zorder=5))
t(cx + 9, 48.6, "6 cell scalars", size=6.8, color=MUTED)
t(cx + 9, 76.2 - 2.5, "independent core", size=7.3, weight="bold")
arrow(x2 + 21.4, 61, cx - 1.0, 61)
arrow(cx + 19.3, 61, x2 + 48.6, 61)
box(x2 + 49, 53.5, 12, 15, fc="white", ec=GEN, lw=0.9, r=1.2, z=4)
t(x2 + 55, 63.3, r"$\phi$", size=12, color=GEN)
t(x2 + 55, 57.0, "exact\nfeaturizer", size=6.4)
arrow(x2 + 61.3, 61, x2 + 64.6, 61)
vector(x2 + 65, 47.5, 13, 3.4, 2.05, ["#F3B48A", "#E89A66", "#F8CDAE", "#EFA779"], horizontal=False)
t(x2 + 72.2, 73.0, r"$\tilde{x}$", size=9.5, weight="bold", color=SYN)
box(x2 + 4, 23, w2 - 8, 17.5, fc="white", ec="none", r=1.5, z=2)
t(x2 + w2 / 2, 36.4, "all 106 descriptors rebuilt  →  consistent by construction", size=7.4, weight="bold", color=OK)
t(x2 + w2 / 2, 31.4, "WGAN-GP critic compares full 106-d descriptors with real MOFs", size=7.0, color=MUTED)
t(x2 + w2 / 2, 26.8, "checkpoint chosen by coverage of validation MOFs", size=7.0, color=MUTED)

# ------------------------------------------------------------------ 3 graph insertion
x3, w3 = 142, 52
card(x3, w3, 3, "Graph insertion")
g = np.array([[149, 68], [158, 72], [168, 68], [186, 70], [152, 58], [163, 57], [176, 60], [187, 56],
              [149, 47], [160, 45], [172, 48], [184, 45]])
E = [(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7), (4, 5), (5, 6), (6, 7),
     (4, 8), (5, 9), (8, 9), (9, 10), (6, 10), (10, 11), (7, 11)]
for i, j in E:
    edge(g[i], g[j])
s1, s2 = np.array([170.5, 63.5]), np.array([166.0, 51.5])
for nb in (2, 5, 6):
    edge(s1, g[nb], color=SYN, ls=(0, (2, 1.2)), lw=0.85)
for nb in (5, 9, 10):
    edge(s2, g[nb], color=SYN, ls=(0, (2, 1.2)), lw=0.85)
for p in g:
    node(*p, r=1.6)
node(*s1, r=1.8, fc=SYN)
node(*s2, r=1.8, fc=SYN)
t(x3 + w3 / 2, 36.2, "cosine kNN links to real MOFs", size=7.6, weight="bold")
t(x3 + w3 / 2, 31.4, "edges are assigned, not generated", size=7.0, color=MUTED, style="italic")
node(x3 + 9, 25.5, r=1.3)
t(x3 + 11.5, 25.5, "real MOF", size=7.0, ha="left")
node(x3 + 29, 25.5, r=1.3, fc=SYN)
t(x3 + 31.5, 25.5, "generated", size=7.0, ha="left")

# ------------------------------------------------------------------ 4 audit
x4, w4 = 198, 48
card(x4, w4, 4, "Audit every node")
rows = ["schema consistency", "coverage & diversity", "novelty & duplicates", "nearest real analogs",
        "pseudo-labels (optional)"]
for k, label in enumerate(rows):
    y = 70 - k * 8.6
    ix = x4 + 7
    if k == 0:
        ax.plot([ix - 2.2, ix - 0.8, ix + 2.2], [y, y - 1.4, y + 1.6], color=OK, lw=1.2, zorder=6)
    elif k == 1:
        pts = rng.normal(0, 1, (14, 2)) * [1.5, 0.8]
        ax.scatter(ix + pts[:, 0], y + pts[:, 1], s=1.4, c=REAL, lw=0, zorder=6)
        pts = rng.normal(0, 1, (7, 2)) * [1.4, 0.7]
        ax.scatter(ix + pts[:, 0], y + pts[:, 1], s=2.0, c=SYN, lw=0, zorder=7)
    elif k == 2:
        ax.add_patch(Circle((ix - 0.4, y + 0.4), 1.6, fc="none", ec=INK, lw=0.8, zorder=6))
        ax.plot([ix + 0.8, ix + 2.2], [y - 0.8, y - 2.2], color=INK, lw=1.0, zorder=6)
    elif k == 3:
        mof_icon(ix - 0.6, y - 0.3, 3.2, z=6)
    else:
        box(ix - 2.3, y - 1.5, 4.6, 3.0, fc=SYN_L, ec=SYN, lw=0.6, r=0.6, z=6)
        t(ix, y, r"$\hat{y}$", size=7, color=SYN)
    t(x4 + 12.5, y, label, size=7.6, ha="left")

# ------------------------------------------------------------------ flow arrows between cards
for xa, xb in [(x1 + w1, x2), (x2 + w2, x3), (x3 + w3, x4)]:
    arrow(xa + 0.4, 52, xb - 0.4, 52, color=INK, lw=1.2, ms=7, z=7)

# ------------------------------------------------------------------ bottom strip
box(4, 3, 156, 12.5, fc=GEN_L, ec="none", r=2, z=1)
t(8, 9.25, "gang-mof", size=8.5, weight="bold", ha="left", color=GEN)
stats = [("100%", "schema-consistent"), ("0.77", "coverage of held-out MOFs"), ("1.06", "spread vs. real MOFs")]
for xs, (num, lab) in zip([42, 82, 130], stats):
    t(xs, 9.25, num, size=10, weight="bold", ha="right", color=INK)
    t(xs + 1.8, 9.25, lab, size=7.4, ha="left", color=MUTED)
box(163, 3, 83, 12.5, fc=SYN_L, ec="none", r=2, z=1)
t(204.5, 11.2, "Descriptor-space hypotheses", size=8, weight="bold", color=INK)
t(204.5, 6.6, "not CIFs, structures or validated materials", size=7.2, color=MUTED)

out = Path(__file__).with_name("gang_overview")
fig.savefig(out.with_suffix(".svg"))
fig.savefig(out.with_suffix(".png"), dpi=300)
print("wrote", out.with_suffix(".png"))
