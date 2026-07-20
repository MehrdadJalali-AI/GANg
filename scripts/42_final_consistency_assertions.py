#!/usr/bin/env python3
"""Final implementation-grounded consistency assertions.

These checks do not contain manuscript-facing expected numerical results.
They recompute consistency relationships from generated output tables and fail
if manuscript-facing summaries diverge from authoritative implementation data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GAN_MODES = {"gan_generated", "gan_filtered", "gan_property_head", "gan_conditional_target"}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def assert_allclose(actual, expected, label: str, atol: float = 1e-10) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(label)


def check_validation_selected() -> None:
    results = pd.read_csv(require(ROOT / "tables/validation_selected_results.csv"))
    seedwise = pd.read_csv(require(ROOT / "tables/validation_selected_seedwise_results.csv"))
    stats = pd.read_csv(require(ROOT / "tables/validation_selected_statistics.csv"))

    assert_allclose(
        results["selected_mode_test_mae"] - results["real_only_test_mae"],
        results["delta_mae"],
        "validation-selected delta MAE mismatch",
    )

    for _, r in results.iterrows():
        source = pd.read_csv(require(ROOT / r["source_summary_table"]))
        candidates = source[(source["model"].eq(r["architecture"])) & (source["training_mode"].isin(GAN_MODES))].copy()
        if candidates.empty:
            raise AssertionError(f"No GANg candidates for {r['target']} {r['architecture']}")
        selected = candidates.sort_values(["best_val_mae_mean", "training_mode"], kind="mergesort").iloc[0]
        if selected["training_mode"] != r["validation_selected_mode"]:
            raise AssertionError(f"Validation-selected mode mismatch for {r['target']} {r['architecture']}")
        if abs(float(selected["mae_mean"]) - float(r["selected_mode_test_mae"])) > 1e-10:
            raise AssertionError(f"Selected-mode test MAE mismatch for {r['target']} {r['architecture']}")

    grouped = seedwise.groupby(["target", "nodes", "architecture"])
    for _, r in stats.iterrows():
        key = (r["target"], r["nodes"], r["architecture"])
        sub = grouped.get_group(key)
        delta = sub["selected_mode_mae"].to_numpy() - sub["real_only_mae"].to_numpy()
        if abs(delta.mean() - float(r["delta_mae_mean"])) > 1e-10:
            raise AssertionError(f"Seedwise mean delta mismatch for {key}")
        if int((delta < 0).sum()) != int(r["seeds_gang_lower_mae"]):
            raise AssertionError(f"GANg-lower seed count mismatch for {key}")
        if int((delta > 0).sum()) != int(r["seeds_real_lower_mae"]):
            raise AssertionError(f"Real-lower seed count mismatch for {key}")


def check_scaling() -> None:
    scaling = pd.read_csv(require(ROOT / "tables/implementation_grounded_scaling_summary.csv")).set_index("graph_characteristic")
    small_before = float(scaling.loc["Edges before sparsification", "smaller_study"])
    small_after = float(scaling.loc["Edges after sparsification", "smaller_study"])
    large_before = float(scaling.loc["Edges before sparsification", "larger_study"])
    large_after = float(scaling.loc["Edges after sparsification", "larger_study"])
    small_reduction = float(scaling.loc["Edge reduction (%)", "smaller_study"])
    large_reduction = float(scaling.loc["Edge reduction (%)", "larger_study"])
    if abs(100 * (small_before - small_after) / small_before - small_reduction) > 1e-9:
        raise AssertionError("Small-study edge reduction mismatch")
    if abs(100 * (large_before - large_after) / large_before - large_reduction) > 1e-9:
        raise AssertionError("Large-study edge reduction mismatch")


def check_baselines_and_claim_boundary() -> None:
    baseline = pd.read_csv(require(ROOT / "tables/remaining_issues/generator_baseline_validity_with_gang.csv"))
    expected_generators = {"gaussian_noise", "nearest_neighbor_interpolation", "kde_bootstrap", "gmm", "vae", "GANg"}
    if set(baseline["generator"]) != expected_generators:
        raise AssertionError("Supplementary descriptor-generator baseline does not contain exactly six generators")

    feature_summary = pd.read_csv(require(ROOT / "tables/remaining_issues/feature_category_summary.csv"))
    if int(feature_summary["count"].sum()) != len(pd.read_csv(require(ROOT / "experiments/qmof_gang_final/tables/feature_audit.csv"))):
        raise AssertionError("Feature-category summary does not match generated feature audit row count")

    uniqueness = json.loads(require(ROOT / "tables/remaining_issues/generated_node_global_id_uniqueness.json").read_text())
    if uniqueness["duplicate_global_ids"] != 0:
        raise AssertionError("Duplicate generated-node global IDs detected")

    main_path = ROOT / "manuscript/main.tex"
    supp_path = ROOT / "manuscript/supplementary.tex"
    if main_path.exists() and supp_path.exists():
        main = main_path.read_text()
        supp = supp_path.read_text()
        for phrase in [
            "No generated CIF",
            "No generated CIF, DFT validation, force-field relaxation, synthesis-ready assessment, or experimental validation is claimed",
        ]:
            if phrase not in main:
                raise AssertionError(f"Required claim-boundary phrase missing from main manuscript: {phrase}")
        if "full-reference-pool novelty-pass rate" not in main:
            raise AssertionError("Main manuscript does not distinguish full-reference-pool novelty")
        if "training-set novelty-pass rate" not in main:
            raise AssertionError("Main manuscript does not distinguish training-set novelty")
        if "generator-quality diagnostics only" not in supp:
            raise AssertionError("Supplementary baseline caption does not preserve generator-quality scope")


def main() -> None:
    check_validation_selected()
    check_scaling()
    check_baselines_and_claim_boundary()
    print("PASS final implementation-grounded consistency assertions")


if __name__ == "__main__":
    main()
