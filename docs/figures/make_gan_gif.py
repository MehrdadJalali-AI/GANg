"""Make the README's schematic MOF GAN animation (no model/data required).

Run from any directory: python docs/figures/make_gan_gif.py
Requires matplotlib, numpy and Pillow. Writes gang_gan.gif beside this script
and six hold-frame PNGs in gang_gan_frames/ for visual inspection.

The original cube-cage function is loaded from make_overview.py using AST,
without executing that script's figure-generation side effects. Generated
nodes are always circles/fingerprint cards, never crystal structures.
Training examples and the critic score are illustrative, not measured results.
GIF delays use an 80/80/90 ms cadence: exactly 12 frames/s on average.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image

W, H, FPS, SECONDS, SEED = 1200, 500, 12, 13.5, 42
N_FRAMES = int(FPS * SECONDS)
OUT = Path(__file__).resolve().parent
REAL, ORANGE, BLUE = "#2E4A7D", "#E07B39", "#3B6FB6"
GREY, TEAL, METAL, PORE = "#6B7785", "#2A9D8F", "#4F7A95", "#F4E3A1"
INK, MUTED, PALE, LINE = "#263647", "#5B6670", "#F3F5F7", "#C7CED5"
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
font_family = next(name for name in ["Liberation Sans", "Arial", "DejaVu Sans"]
                   if name in available_fonts)
plt.rcParams.update({"font.family": font_family})
fig = plt.figure(figsize=(12, 5), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])

# Reuse the actual overview functions, rather than copying the cage drawing.
overview_ast = ast.parse((OUT / "make_overview.py").read_text())
glyph_ast = ast.Module(body=[n for n in overview_ast.body
                            if isinstance(n, ast.FunctionDef) and n.name in {"edge", "mof_icon"}],
                       type_ignores=[])
glyph_globals = {"ax": ax, "np": np, "Circle": Circle, "Polygon": Polygon,
                 "METAL": METAL, "LINKER": GREY, "PORE": PORE}
exec(compile(glyph_ast, str(OUT / "make_overview.py"), "exec"), glyph_globals)

RNG = np.random.default_rng(SEED)
REAL_PATTERNS = np.array([[.30, .02, .90, .72, .55], [.03, .36, .84, .65, .60],
                          [.25, .01, .82, .63, .67], [.01, .29, .88, .74, .53]])
INITIAL = RNG.uniform(.13, .98, 5)
TARGET = np.array([.27, .03, .88, .69, .59])
GRAPH = np.array([[970, 180], [1040, 170], [1120, 185], [1160, 245],
                  [1140, 340], [1080, 385], [1000, 380], [955, 310],
                  [975, 245], [1065, 330]], dtype=float)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
         (7, 8), (8, 0), (8, 1), (4, 9), (5, 9), (6, 9)]
INSERTS = np.array([[1055, 235], [1105, 280], [1015, 315]], dtype=float)
NEIGHBORS = [(1, 2, 8), (3, 4, 9), (8, 7, 6)]
STARTS = [0, 2.5, 4.5, 6.5, 9.5, 12]
ENDS = [2.5, 4.5, 6.5, 9.5, 12, 13.5]
TITLES = ["1  Real MOFs", "2  Generate a fingerprint", "3  Compare with real MOFs",
          "4  Learn through feedback", "5  Insert by similarity", "GANg"]
CAPTIONS = [
    "Real MOFs, each described by a fingerprint of 106 descriptors",
    "The generator proposes a new MOF fingerprint for the requested band gap",
    "The critic compares it with real MOFs",
    "Trained against each other, the generator learns what real MOFs look like",
    "New hypothetical MOFs join the similarity graph,\nlinked to their most similar real MOFs",
    "github.com/MehrdadJalali-AI/GANg",
]


def ease(v):
    v = np.clip(v, 0, 1)
    return v * v * (3 - 2 * v)


def text(x, y, label, px=20, color=INK, weight="normal", ha="center", alpha=1, z=20):
    # Matplotlib font sizes are in points; all layout and minimums are pixels.
    return ax.text(x, y, label, fontsize=px * .72, color=color, fontweight=weight,
                   ha=ha, va="center", alpha=float(alpha), zorder=z, linespacing=1.15)


def rounded(x, y, w, h, fc="white", ec=LINE, alpha=1, radius=8, lw=1.3, z=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                      fc=fc, ec=ec, lw=lw, alpha=float(alpha), zorder=z)
    ax.add_patch(p)
    return p


def cage(x, y, size, alpha=1):
    before_p, before_l = len(ax.patches), len(ax.lines)
    glyph_globals["mof_icon"](x - .18 * size, y - .15 * size, size)
    for artist in list(ax.patches)[before_p:] + list(ax.lines)[before_l:]:
        artist.set_alpha(float(alpha))
        artist.set_linewidth(artist.get_linewidth() * 1.4)


def fingerprint(x, y, values, color, alpha=1, scale=1, gauges=(.62, .48), labels=True):
    """An expanded labelled card, or a compact transfer/graph glyph."""
    w, h = 180 * scale, 136 * scale
    rounded(x, y, w, h, ec=color, alpha=alpha, radius=7 * scale)
    # Bar labels and both gauge labels are 20 px on expanded cards.
    for i, (name, value) in enumerate(zip(["Zn", "Cu", "C", "H", "O"], values)):
        bx = x + ([22, 61, 96, 128, 159][i] - 9) * scale
        bh = max(2, 43 * value) * scale
        ax.add_patch(Rectangle((bx, y + 52 * scale - bh), 18 * scale, bh,
                               fc=color, ec="none", alpha=float(alpha), zorder=6))
        if labels:
            text(bx + 9 * scale, y + 68 * scale, name, px=20, alpha=alpha)
    for j, (name, value) in enumerate(zip(["pore", "density"], gauges)):
        gy = y + (95 + j * 27) * scale
        if labels:
            text(x + 10 * scale, gy, name, px=20, ha="left", alpha=alpha)
        gx = x + 88 * scale
        ax.add_patch(Rectangle((gx, gy - 3 * scale), 80 * scale, 6 * scale,
                               fc="#E4E8EC", ec="none", alpha=float(alpha), zorder=6))
        ax.add_patch(Rectangle((gx, gy - 3 * scale), 80 * value * scale, 6 * scale,
                               fc=color, ec="none", alpha=float(alpha), zorder=7))


def arrow(points, color=MUTED, alpha=1, curved=False, lw=2.2):
    # One continuous patch per arrow, never overlapping separate arrow segments.
    codes = [MplPath.MOVETO] + ([MplPath.CURVE4] * 3 if curved else [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(path=MplPath(points, codes), arrowstyle="-|>",
                            mutation_scale=13, lw=lw, color=color, alpha=float(alpha), zorder=9)
    ax.add_patch(patch)


def database(t, alpha):
    positions = [(92, 185), (202, 185), (92, 320), (202, 320)]
    text(150, 98, "Real MOF database", px=20, weight="bold", alpha=alpha)
    # Each cage flips to a compact card in turn; a labelled close-up occupies
    # the central slot so descriptors remain readable without crowding the grid.
    active = .30 <= t < 1.70
    k = min(3, int((t - .30) / .35)) if active else -1
    phase = ((t - .30) % .35) / .35 if active else 0
    flip = np.sin(np.pi * phase) ** 2 if active else 0
    for i, (x, y) in enumerate(positions):
        a = alpha * (1 - ease(min(flip * 2, 1))) if i == k else alpha
        rounded(x - 33, y + 52, 74, 12, fc="#E4EAF3", ec="none", alpha=a, radius=3)
        rounded(x - 38, y + 47, 74, 12, fc="#F0F3F8", ec=REAL, alpha=a, radius=3)
        # Tiny descriptor marks remain under each real MOF, with the readable
        # labelled version expanded in the central slot during this scene.
        for j, value in enumerate(REAL_PATTERNS[i]):
            ax.add_patch(Rectangle((x - 30 + j * 13, y + 55 - 6 * value), 7, 6 * value,
                                   fc=REAL, ec="none", alpha=float(a), zorder=7))
        cage(x, y, 45, a)
    if active:
        fa = alpha * ease(min(flip * 2, 1))
        x, y = positions[k]
        fingerprint(x - 38, y - 30, REAL_PATTERNS[k], REAL, fa, .42, labels=False)
    if t < 2.5:
        a = alpha * ease(t / .3)
        text(610, 98, "MOF fingerprint", weight="bold", alpha=a)
        cage(610, 200, 35, a)
        text(610, 253, "real MOF", color=REAL, alpha=a)
        fingerprint(520, 275, REAL_PATTERNS[max(k, 3) if not active else k], REAL, a)
        arrow([(272, 235), (556, 235)], color=REAL, alpha=a)
        text(414, 205, "106 descriptors", color=REAL, alpha=a)


def graph(alpha=1, inserted=0, edge_progress=None):
    text(1055, 98, "Similarity graph", px=20, weight="bold", alpha=alpha)
    for i, j in EDGES:
        # Trim each similarity edge to leave twelve pixels beyond cage bounds.
        p, q = GRAPH[i], GRAPH[j]
        unit = (q - p) / np.linalg.norm(q - p)
        a, b = p + unit * 29, q - unit * 29
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LINE, lw=1.25, alpha=alpha, zorder=1)
    for x, y in GRAPH:
        cage(x, y, 22, alpha)
    for k in range(inserted):
        p = INSERTS[k]
        progresses = edge_progress if k == inserted - 1 and edge_progress is not None else [1, 1, 1]
        for j, progress in zip(NEIGHBORS[k], progresses):
            q = GRAPH[j]
            unit = (q - p) / np.linalg.norm(q - p)
            a, b = p + 22 * unit, q - 29 * unit
            b = a + ease(progress) * (b - a)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=ORANGE, lw=1.8,
                    ls=(0, (3, 2)), alpha=alpha, zorder=2)
        generated(p[0], p[1], alpha=alpha, small=True)


def generated(x=610, y=200, alpha=1, small=False):
    radius = 10 if small else 24
    ax.add_patch(Circle((x, y), radius, fc=ORANGE, ec="white", lw=1.5,
                        alpha=float(alpha), zorder=12))
    if small:
        # The third node's card sits above it, away from its downward link.
        cy = y - 48 if abs(x - INSERTS[2, 0]) < 1 else y + 22
        fingerprint(x - 11, cy, TARGET, ORANGE, alpha, scale=.12, labels=False)


def machines(alpha=1, pulse=0, inputs=True, input_progress=1):
    text(400, 98, "Generator G", weight="bold", color=BLUE, alpha=alpha)
    text(830, 98, "Critic D", weight="bold", color=GREY, alpha=alpha)
    if inputs:
        a = alpha * ease(input_progress * 4)
        shift = (1 - ease(input_progress)) * 18
        text(310 - shift, 133, "z", px=24, alpha=a)
        for j in range(6):
            ax.add_patch(Rectangle((300 - shift, 155 + j * 24), 20, 20,
                                   fc=GREY, ec="white", lw=1, alpha=float(a), zorder=5))
        arrow([(332, 235), (353, 235)], alpha=a)
        rounded(305, 329 + shift, 165, 61, fc="#FFF0E5", ec=ORANGE, alpha=a)
        text(387.5, 347 + shift, "c: band gap", color=ORANGE, alpha=a)
        text(387.5, 373 + shift, "2 eV", color=ORANGE, weight="bold", alpha=a)
        arrow([(387, 317), (387, 303), (415, 303), (415, 292)], color=ORANGE, alpha=a)
    ax.add_patch(Polygon([[365, 185], [458, 202], [458, 268], [365, 285]],
                         fc=BLUE, ec=BLUE, lw=2 + 3 * pulse, alpha=float(alpha), zorder=5))
    text(411, 234, "G", px=36, weight="bold", color="white", alpha=alpha)
    ax.add_patch(Polygon([[760, 202], [900, 185], [900, 285], [760, 268]],
                         fc="#EFF1F4", ec=GREY, lw=2, alpha=float(alpha), zorder=5))
    text(830, 235, "D", px=36, weight="bold", color=GREY, alpha=alpha)


def verdict(marker, alpha=1, final=False, flash=0):
    color = TEAL if final else ORANGE
    text(830, 310, "schematic critic", color=MUTED, alpha=alpha)
    rounded(730, 335, 200, 18, fc="#E6EAEE", ec=color if flash else LINE,
            lw=2 + flash * 2, radius=9, alpha=alpha)
    rounded(730, 335, 200 * np.clip(marker, .03, .97), 18,
            fc="#CDEBE5" if final else "#F6D5BF", ec="none", radius=9, alpha=alpha)
    ax.add_patch(Circle((730 + 200 * marker, 344), 7, fc=color, ec="white", lw=1.5,
                        alpha=float(alpha), zorder=12))
    text(730, 374, "generated", ha="left", color=ORANGE, alpha=alpha)
    text(930, 374, "real", ha="right", color=REAL, alpha=alpha)
    if final:
        text(830, 407, "can't tell", color=TEAL, weight="bold", alpha=alpha)


def transfer_card(x, y, color, alpha=1, values=TARGET, scale=.20):
    fingerprint(x - 90 * scale, y - 68 * scale, values, color, alpha, scale=scale, labels=False)


def interpolate_route(points, p):
    p = ease(p)
    pts = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    distance = p * cumulative[-1]
    idx = min(len(lengths) - 1, np.searchsorted(cumulative[1:], distance, side="right"))
    return pts[idx] + (pts[idx + 1] - pts[idx]) * (distance - cumulative[idx]) / lengths[idx]


def render(frame):
    t = frame / FPS
    scene = min(5, np.searchsorted(STARTS, t, side="right") - 1)
    local = t - STARTS[scene]
    ax.clear()
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=-20))
    ax.add_patch(Rectangle((0, 430), W, 70, fc=PALE, ec="none", zorder=30))
    # A scene tracker also keeps held images distinct in GIF encoders. It is
    # outside the content/caption regions and does not distract from the story.
    ax.plot([40, 1160], [56, 56], color="#E8ECF0", lw=2, zorder=3)
    ax.plot([40, 40 + 1120 * (frame + 1) / N_FRAMES], [56, 56], color=BLUE, lw=2, zorder=4)
    if scene < 5:
        text(40, 28, TITLES[scene], px=28, weight="bold", ha="left")
        database(t if scene == 0 else 3, .25 + .75 * ease(t / .3))
        graph()
        if scene == 0:
            # Graph and database share the same 0.3-second opening fade.
            if t < .3:
                for artist in list(ax.patches)[1:] + list(ax.lines)[2:] + list(ax.texts)[1:]:
                    artist.set_alpha(float(.25 + .75 * ease(t / .3)))
        else:
            ma = ease(local / .3) if scene == 1 else 1
            pulse = np.sin(np.pi * np.clip((local - .3) / .7, 0, 1)) ** 2 if scene == 1 else 0
            machines(ma, pulse, inputs=scene in (1, 4), input_progress=local / .4 if scene == 1 else 1)
            if scene == 1:
                text(610, 98, "New fingerprint", weight="bold", alpha=ma)
                ga = ease((local - .40) / .3)
                arrow([(482, 215), (574, 200)], color=BLUE, alpha=ga)
                generated(alpha=ga)
                text(610, 253, "hypothetical MOF", color=ORANGE, alpha=ga)
                fingerprint(520, 275, INITIAL, ORANGE, ga, gauges=(.92, .12))
            elif scene in (2, 3):
                generated()
                text(610, 253, "hypothetical MOF", color=ORANGE)
                arrow([(482, 215), (574, 200)], color=BLUE)
                arrow([(646, 200), (748, 215)], color=GREY)
                if scene == 2:
                    text(610, 98, "New fingerprint", weight="bold")
                    fingerprint(520, 275, INITIAL, ORANGE, gauges=(.92, .12))
                    move = ease(local / .9)
                    if local < 1.0:
                        a = 1 - ease((local - .75) / .25)
                        p = interpolate_route([(285, 145), (795, 145), (795, 222)], move)
                        transfer_card(*p, REAL, a, REAL_PATTERNS[0])
                        p = interpolate_route([(610, 320), (730, 320), (730, 145), (860, 145), (860, 222)], move)
                        if p[0] >= 730 or p[1] < 250:
                            transfer_card(*p, ORANGE, a, INITIAL)
                    verdict(.52 - .40 * ease(local / 1.0), ease(local / .3), flash=np.sin(np.pi * min(local, 1)) ** 2)
                else:
                    # Three schematic rounds complete within 2.2 seconds,
                    # followed by the requested 0.8-second final hold.
                    phase = np.clip(local / (2.2 / 3), 0, 2.999999)
                    cycle = int(phase)
                    u = ease((phase - cycle) / .65)
                    amounts = [0, .35, .65, .90]
                    blend = amounts[cycle] + (amounts[cycle + 1] - amounts[cycle]) * u
                    values = INITIAL * (1 - blend) + TARGET * blend
                    gauges = (.92 * (1 - blend) + .62 * blend, .12 * (1 - blend) + .48 * blend)
                    fingerprint(520, 275, values, ORANGE, gauges=gauges)
                    rounded(532, 82, 156, 32, fc="#E6F4F0", ec="none")
                    text(610, 98, f"training {cycle + 1}/3", color=TEAL, weight="bold")
                    fa = ease(local / .3)
                    arrow([(820, 173), (820, 115), (415, 115), (415, 173)], TEAL, fa, curved=True)
                    text(610, 154, "feedback", color=TEAL, alpha=fa)
                    # A compact packet follows G -> D once per round, above
                    # the large card. Fade it before reaching either machine.
                    packet = phase - cycle
                    if .1 < packet < .7:
                        pa = np.sin(np.pi * (packet - .1) / .6) ** 2
                        transfer_card(665 + 60 * ease((packet - .1) / .6), 184, ORANGE, pa, values, scale=.10)
                    final = local >= 1.95
                    verdict(.12 + .38 * blend / .90, final=final)
            elif scene == 4:
                text(610, 98, "Hypothetical MOFs", weight="bold")
                verdict(.5, alpha=1 - .7 * ease(local / .3), final=True)
                # The upper lane clears both the critic and the feedback arc.
                # Card labels fade before shrinking, leaving only glyphs while
                # travelling; the large card remains readable below its slot.
                fingerprint(520, 275, TARGET, ORANGE, 1 - .60 * ease(local / .3))
                schedules = [(0, .58, .98), (.69, 1.10, 1.39), (1.10, 1.47, 1.70)]
                for k, (begin, arrive, complete) in enumerate(schedules):
                    if local < begin:
                        continue
                    if local < arrive:
                        u = ease((local - begin) / (arrive - begin))
                        # One smooth cubic path, passing above the critic.
                        p0 = np.array([610., 200.])
                        p1 = np.array([650., 90.])
                        p2 = np.array([1000., 100.])
                        p3 = INSERTS[k]
                        pos = (1-u)**3*p0 + 3*(1-u)**2*u*p1 + 3*(1-u)*u*u*p2 + u**3*p3
                        scale = 1 - .58 * ease(min(u * 3, 1))
                        ax.add_patch(Circle(pos, 24 * scale, fc=ORANGE, ec="white", lw=1.5, zorder=15))
                    else:
                        edge_u = np.clip((local - arrive) / (complete - arrive), 0, 1)
                        progresses = [np.clip(edge_u * 3 - j, 0, 1) for j in range(3)]
                        # Base graph is already present; draw only this node
                        # and its three sequential similarity links.
                        p = INSERTS[k]
                        for j, progress in zip(NEIGHBORS[k], progresses):
                            q = GRAPH[j]
                            unit = (q - p) / np.linalg.norm(q - p)
                            a, b = p + 22 * unit, q - 29 * unit
                            end = a + ease(progress) * (b - a)
                            ax.plot([a[0], end[0]], [a[1], end[1]], color=ORANGE,
                                    lw=1.8, ls=(0, (3, 2)), zorder=2)
                        generated(*p, small=True)
    else:
        graph(alpha=.16, inserted=3)
        fade = ease(local / .3)
        text(600, 176, "GANg", px=72, color=BLUE, weight="bold", alpha=fade)
        text(600, 257, "GAN-based node synthesis for MOF similarity graphs", px=30, alpha=fade)
        text(600, 323, "hypothetical MOFs = descriptor fingerprints, not crystal structures",
             px=22, color=MUTED, alpha=fade)
    text(600, 465, CAPTIONS[scene], px=24, z=40)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in ax.texts:
        if label.get_alpha() == 0:
            continue
        bounds = label.get_window_extent(renderer)
        assert 0 <= bounds.x0 <= bounds.x1 <= W, label.get_text()
        top, bottom = H - bounds.y1, H - bounds.y0
        if label.get_zorder() == 40:
            assert 430 <= top <= bottom <= H, label.get_text()
        elif label.get_position()[1] >= 70:
            assert 70 <= top <= bottom <= 420, label.get_text()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def main():
    snapshots = OUT / "gang_gan_frames"
    snapshots.mkdir(exist_ok=True)
    holds = {int(round((end - .4) * FPS)): i for i, end in enumerate(ENDS)}
    frames = []
    for frame in range(N_FRAMES):
        img = render(frame)
        if frame in holds:
            img.save(snapshots / f"scene_{holds[frame] + 1:02d}.png")
        frames.append(img)
    # One deterministic palette across the whole story prevents flicker. Sample
    # every sixth frame, then quantize without dithering for a compact flat GIF.
    sample = Image.new("RGB", (W, 125 * len(frames[::6])), "white")
    for j, img in enumerate(frames[::6]):
        sample.paste(img.resize((W, 125)), (0, j * 125))
    palette = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    encoded = [img.quantize(palette=palette, dither=Image.Dither.NONE) for img in frames]
    output = OUT / "gang_gan.gif"
    delays = [80, 80, 90] * (N_FRAMES // 3)
    encoded[0].save(output, save_all=True, append_images=encoded[1:], duration=delays,
                    loop=0, optimize=True, disposal=1)
    with Image.open(output) as gif:
        duration = sum(gif.seek(i) or gif.info["duration"] for i in range(gif.n_frames))
        count = gif.n_frames
        assert gif.size == (W, H) and gif.info["loop"] == 0
        assert count == N_FRAMES and duration == int(SECONDS * 1000)
    size = output.stat().st_size
    assert size < 3_000_000, f"GIF exceeds 3 MB: {size:,} bytes"
    print(f"{output.name}: {W} x {H}; {count} frames; {FPS} fps average; "
          f"{duration / 1000:.2f} s; {size:,} bytes; loop forever")
    print("SHA256:", hashlib.sha256(output.read_bytes()).hexdigest())
    plt.close(fig)


if __name__ == "__main__":
    main()
