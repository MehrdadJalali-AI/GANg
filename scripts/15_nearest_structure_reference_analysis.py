from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "qmof_gang_final" / "tables"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
REPORTS = ROOT / "reports"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def main() -> None:
    nn = pd.read_csv(SRC / "nearest_neighbor_interpretation.csv")
    node = pd.read_csv(SRC / "qmof_node_table.csv")
    cols = ["qmof_id", "formula", "info_pld", "info_lcd", "info_density", "info_volume", "info_symmetry_spacegroup_number"]
    meta = node[[c for c in cols if c in node.columns]].copy()
    rows = []
    for _, r in nn.head(25).iterrows():
        entry = {
            "generated_node_id": r.generated_node_id,
            "reference_language": "nearest real QMOF analogue; not a generated CIF",
            "top1_qmof_id": r.top1_qmof_id,
            "top1_similarity": r.top1_similarity,
            "top2_qmof_id": r.top2_qmof_id,
            "top2_similarity": r.top2_similarity,
            "top3_qmof_id": r.top3_qmof_id,
            "top3_similarity": r.top3_similarity,
            "top5_bandgap_mean": r.top5_bandgap_mean,
            "top5_bandgap_std": r.top5_bandgap_std,
        }
        match = meta[meta.qmof_id.eq(r.top1_qmof_id)]
        if len(match):
            for c, v in match.iloc[0].items():
                if c != "qmof_id":
                    entry[f"top1_{c}"] = v
        rows.append(entry)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "generated_node_reference_analogues.csv", index=False)

    sample = out.head(5)
    fig, axes = plt.subplots(len(sample), 1, figsize=(8, 1.7 * len(sample)), dpi=200)
    if len(sample) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, sample.iterrows()):
        ax.axis("off")
        ax.text(0.01, 0.70, f"{r.generated_node_id}", fontsize=12, fontweight="bold")
        ax.text(0.18, 0.70, f"top analogue: {r.top1_qmof_id}  sim={r.top1_similarity:.3f}", fontsize=10)
        ax.text(0.18, 0.38, f"top-5 bandgap {r.top5_bandgap_mean:.3f} ± {r.top5_bandgap_std:.3f} eV", fontsize=10)
        ax.text(0.18, 0.10, "descriptor-space hypothesis; nearest real QMOF reference only", fontsize=9, color="#7E5AA2")
    fig.tight_layout()
    fig.savefig(FIGS / "generated_node_reference_cards.pdf")
    fig.savefig(FIGS / "generated_node_reference_cards.png", dpi=600)
    plt.close(fig)

    report = [
        "# Nearest-structure reference analysis",
        "",
        "GANg does not generate CIFs. This analysis maps generated descriptor-space nodes to nearest real QMOF analogues for interpretation only.",
        f"Rows written: {len(out)}.",
        "No CIF files were found or exported by this script. If raw QMOF CIF paths are later added, they should be labeled as nearest-neighbor reference CIFs, not generated CIFs.",
    ]
    (REPORTS / "nearest_structure_reference_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
