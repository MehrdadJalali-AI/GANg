"""Export npj-style main figures from locked CSV artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_MAIN = ROOT / "figures" / "main"
OUT = ROOT / "figures" / "npj"
OUT.mkdir(parents=True, exist_ok=True)

MAP = {
    "Figure_1_workflow": "Figure_1_workflow",
    "Figure_2_sparsification": "Figure_2_sparsification",
    "Figure_3_validity_distribution": "Figure_3_validity_distribution",
    "Figure_4_nearest_neighbor_interpretation": "Figure_4_reference_analogues",
    "Figure_5_validity_utility_gap": "Figure_5_validity_utility_gap",
}


def copy_pair(stem_src: str, stem_dst: str) -> None:
    for ext in (".pdf", ".png"):
        s = SRC_MAIN / f"{stem_src}{ext}"
        d = OUT / f"{stem_dst}{ext}"
        if s.exists():
            shutil.copy2(s, d)


def main() -> None:
    for src, dst in MAP.items():
        copy_pair(src, dst)
    # Optional extras from analysis scripts
    for name in [
        "graph_free_vs_gnn_baselines.pdf",
        "statistical_effect_sizes.pdf",
        "generator_baseline_comparison.pdf",
        "generated_node_reference_analogue_cards.pdf",
    ]:
        s = ROOT / "figures" / "npj" / name
        if s.exists():
            continue
        alt = ROOT / "figures" / name
        if alt.exists():
            shutil.copy2(alt, OUT / name)


if __name__ == "__main__":
    main()
