from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmof_galaxy_gan.utils.io import PROJECT, ensure


def _maybe(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def build_report(config: str) -> None:
    reports = ensure(PROJECT / "reports")
    tables = PROJECT / "experiments" / "qmof_galaxy_1000" / "tables"
    graph = _maybe(tables / "graph_summary.csv")
    sparse = _maybe(tables / "sparsification_summary.csv")
    validity = _maybe(tables / "generated_node_validity.csv")
    novelty = _maybe(tables / "generated_node_novelty.csv")
    downstream = _maybe(tables / "downstream_gnn_summary.csv")
    lines = [
        "# GANg readiness report",
        "",
        "This report summarizes the current reproducible project artifacts. Generated nodes are descriptor-space graph hypotheses, not validated MOF structures.",
        "",
        "## Decision questions",
    ]
    if graph is not None:
        r = graph.iloc[0]
        lines += [
            f"1. Was the QMOF similarity graph built successfully? Yes.",
            f"2. How many nodes and edges? {int(r['number_of_nodes'])} nodes and {int(r['number_of_edges'])} undirected similarity edges before sparsification.",
        ]
    else:
        lines += ["1. Was the QMOF similarity graph built successfully? Not yet."]
    if sparse is not None:
        r = sparse.iloc[0]
        lines += [
            f"3. Was sparsification successful? Yes: edges reduced from {int(r['edges_before'])} to {int(r['edges_after'])}.",
        ]
    else:
        lines += ["3. Was sparsification successful? Not yet."]
    if validity is not None and novelty is not None:
        v, n = validity.iloc[0], novelty.iloc[0]
        lines += [
            f"4. Did generated nodes satisfy descriptor-space constraints? Descriptor-valid rate = {v['descriptor_valid_rate']:.3f}.",
            "5. Did generated edges match similarity rules? Edges are attached by cosine kNN in normalized descriptor space; this is a graph-space rule, not physical validation.",
            f"6. Are generated nodes novel relative to real QMOF nodes? Duplicate rate = {n['duplicate_rate']:.3f}; novelty-pass rate = {n['novelty_pass_rate']:.3f}.",
            "7. Do generated nodes collapse into one region or cover meaningful graph space? See PCA coverage and nearest-neighbor figures; this remains a diagnostic, not proof of chemical novelty.",
        ]
    else:
        lines += ["4. Generated-node validity and novelty have not been evaluated yet."]
    if downstream is not None:
        real = downstream.loc[downstream["mode"] == "real_only"].iloc[0]
        gan = downstream.loc[downstream["mode"] == "real_plus_gan"].iloc[0]
        improved = bool(gan["mae"] < real["mae"])
        answer = "Yes on MAE" if improved else "No; the current generated-node augmentation is slightly worse on MAE"
        lines += [f"8. Does adding generated nodes improve downstream GNN prediction? {answer} in the current lightweight MOF-level node-prediction proxy (real-only MAE {real['mae']:.4f}; real+GAN MAE {gan['mae']:.4f}). Treat this as preliminary until full GraphSAGE/GAT training is added."]
    else:
        lines += ["8. Does adding generated nodes improve downstream GNN prediction? Not evaluated yet."]
    lines += [
        "9. Are generated nodes interpretable through nearest real MOFs? Yes if nearest-neighbor interpretation table exists; examples are plotted in Figure 9.",
        "10. Was DFT performed? No.",
        "11. Was force-field relaxation performed? No.",
        "12. Was experimental validation performed? No.",
        "13. Strongest supported claim: GANg generates MOF-like nodes and similarity edges in a QMOF-derived similarity network; generated nodes are descriptor-space graph hypotheses.",
        "14. Not supported: synthesis-ready structures, new MOF discovery, DFT validation, force-field validation, or experimental validation.",
        "15. Recommended target: methods/preprint stage until full graph-neural downstream experiments and stronger feature/decoder validation are complete.",
    ]
    (reports / "readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qmof_galaxy_1000.yaml")
    args = parser.parse_args()
    build_report(args.config)


if __name__ == "__main__":
    main()
