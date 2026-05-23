"""Structure-aware nearest real QMOF reference analogue table (no generated CIFs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "qmof_gang_final" / "tables"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures" / "npj"
REPORTS = ROOT / "reports"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def main() -> None:
    nn = pd.read_csv(SRC / "nearest_neighbor_interpretation.csv")
    node = pd.read_csv(SRC / "qmof_node_table.csv")
    meta_cols = [c for c in ["formula", "info_pld", "info_lcd", "info_density", "info_volume", "info_symmetry_spacegroup_number"] if c in node.columns]
    meta = node[["qmof_id", *meta_cols]].copy()

    rows = []
    for _, r in nn.iterrows():
        row = {
            "generated_node_id": r.generated_node_id,
            "reference_language": "nearest real QMOF reference analogue; not a generated CIF",
            "top1_qmof_id": r.top1_qmof_id,
            "top1_similarity": r.top1_similarity,
            "top2_qmof_id": r.top2_qmof_id,
            "top2_similarity": r.top2_similarity,
            "top3_qmof_id": r.top3_qmof_id,
            "top3_similarity": r.top3_similarity,
            "top5_bandgap_mean": r.top5_bandgap_mean,
            "top5_bandgap_std": r.top5_bandgap_std,
            "cif_path_available": False,
        }
        m = meta[meta.qmof_id == r.top1_qmof_id]
        if len(m):
            for c in meta_cols:
                row[f"top1_{c}"] = m.iloc[0][c]
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "generated_node_reference_analogues.csv", index=False)

    sample = out.head(6)
    fig, axes = plt.subplots(len(sample), 1, figsize=(8, 1.5 * len(sample)), dpi=200)
    if len(sample) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, sample.iterrows()):
        ax.axis("off")
        ax.text(0.01, 0.7, r.generated_node_id, fontweight="bold", fontsize=11)
        ax.text(0.2, 0.7, f"ref analogue: {r.top1_qmof_id} (sim={r.top1_similarity:.3f})", fontsize=10)
        ax.text(0.2, 0.35, f"top-5 bandgap {r.top5_bandgap_mean:.3f} ± {r.top5_bandgap_std:.3f} eV", fontsize=10)
        ax.text(0.2, 0.05, "not generated CIF; not physically validated", fontsize=9, color="#7E5AA2")
    fig.tight_layout()
    fig.savefig(FIGS / "generated_node_reference_analogue_cards.pdf")
    plt.close(fig)

    missing_cif = ROOT / "reports" / "missing_cif_paths_for_analogue_validation.md"
    if not missing_cif.exists():
        missing_cif.write_text(
            "# Missing CIF paths for analogue validation\n\n"
            "Raw QMOF CIF paths are not bundled in this repository. "
            "Analogue analysis is descriptor-level only.\n",
            encoding="utf-8",
        )

    (REPORTS / "structure_aware_analogue_analysis.md").write_text(
        "# Structure-aware analogue analysis\n\n"
        f"Exported {len(out)} generated-node rows with nearest real QMOF reference analogues.\n"
        "No generated structures or CIFs are claimed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
