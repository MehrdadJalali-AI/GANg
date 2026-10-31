"""Candidate checks, distribution metrics and nearest-real-analog retrieval.

Everything here works on descriptor tables, so it can audit GANg output, the baseline,
any other generator, or real data with the same code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import Standardizer, cosine_normalize, prdc
from .chemistry import AMU_A3_TO_G_CM3, ELEMENTS
from .graph import _topk


# ---------------------------------------------------------------------------- schema consistency
def descriptor_consistency(X: pd.DataFrame, tol: float = 1e-3, density_rtol: float = 0.02) -> pd.DataFrame:
    """Check the QMOF-106 descriptor relations row by row.

    Columns returned (booleans): ``composition_sums_to_one``, ``carbon_fraction_matches``,
    ``integer_formula_units`` (cell atoms / formula atoms is an integer), ``density_matches``
    (density = cell atoms x mean atomic mass / volume), ``pld_le_lcd`` and ``consistent`` (all).
    These are relations of the descriptor schema, not chemical or structural validity.
    """
    frac = X[[f"frac_{e}" for e in ELEMENTS]].to_numpy(float)
    ratio = X["info_natoms"].to_numpy(float) / X["atom_count_formula"].to_numpy(float)
    dens = X["info_natoms"].to_numpy(float) * X["atomic_mass_mean"].to_numpy(float) * AMU_A3_TO_G_CM3 / X["info_volume"].to_numpy(float)
    out = pd.DataFrame(index=X.index)
    out["composition_sums_to_one"] = np.abs(frac.sum(1) - 1) <= tol
    out["carbon_fraction_matches"] = np.abs(X["carbon_fraction"].to_numpy(float) - X["frac_C"].to_numpy(float)) <= tol
    out["integer_formula_units"] = np.abs(ratio - np.round(ratio)) <= tol * np.maximum(ratio, 1)
    out["density_matches"] = np.abs(dens - X["info_density"].to_numpy(float)) <= density_rtol * np.abs(X["info_density"].to_numpy(float))
    out["pld_le_lcd"] = X["info_pld"].to_numpy(float) <= X["info_lcd"].to_numpy(float) + 1e-9
    out["consistent"] = out.all(axis=1)
    return out


# ---------------------------------------------------------------------------- per-candidate checks
def candidate_checks(generated: pd.DataFrame, train_X: pd.DataFrame, scaler: Standardizer | None = None,
                     constrained_schema: bool = False, novelty_threshold: float = 0.01,
                     duplicate_threshold: float = 0.999) -> pd.DataFrame:
    """Validity, duplication and novelty screens for generated descriptors.

    * ``finite`` – no NaN/inf.
    * ``within_training_range`` – every descriptor inside the training min–max range.
    * ``schema_consistent`` – QMOF-106 relations hold (only for the qmof106 schema).
    * ``nearest_training_similarity`` / ``novelty_score`` (= 1 − similarity) in standardized cosine space.
    * ``duplicate_of_training`` – similarity above ``duplicate_threshold``.
    * ``duplicate_in_batch`` – identical to an earlier generated row.
    * ``novel`` – novelty score above ``novelty_threshold``.
    * ``accepted`` – passes all of the above; ``rejection_reasons`` lists failures.
    """
    G = generated.to_numpy(float)
    T = train_X.to_numpy(float)
    scaler = scaler or Standardizer.fit(T)
    out = pd.DataFrame(index=generated.index)
    out["finite"] = np.isfinite(G).all(1)
    lo, hi = T.min(0), T.max(0)
    span = np.maximum(hi - lo, 1e-12)
    out["n_features_out_of_range"] = ((G < lo - 1e-9 * span) | (G > hi + 1e-9 * span)).sum(1)
    out["within_training_range"] = out["n_features_out_of_range"] == 0
    if constrained_schema:
        out["schema_consistent"] = descriptor_consistency(generated)["consistent"].to_numpy()
    sim = cosine_normalize(scaler.transform(np.nan_to_num(G))) @ cosine_normalize(scaler.transform(T)).T
    nearest = sim.max(1)
    out["nearest_training_similarity"] = nearest
    out["novelty_score"] = 1 - nearest
    out["duplicate_of_training"] = nearest > duplicate_threshold
    rounded = pd.DataFrame(np.round(G, 8))
    out["duplicate_in_batch"] = rounded.duplicated().to_numpy()
    out["novel"] = out["novelty_score"] > novelty_threshold
    must = ["finite", "within_training_range"] + (["schema_consistent"] if constrained_schema else [])
    reasons = []
    for _, r in out.iterrows():
        bad = [c for c in must if not r[c]]
        bad += ["duplicate_of_training"] if r["duplicate_of_training"] else []
        bad += ["duplicate_in_batch"] if r["duplicate_in_batch"] else []
        bad += ["not_novel"] if not r["novel"] else []
        reasons.append(";".join(bad))
    out["rejection_reasons"] = reasons
    out["accepted"] = out["rejection_reasons"] == ""
    return out


# ---------------------------------------------------------------------------- distribution metrics
def effective_rank(Z: np.ndarray) -> float:
    """exp(entropy) of the normalized covariance eigenvalue spectrum."""
    ev = np.clip(np.linalg.eigvalsh(np.cov(np.asarray(Z, np.float64), rowvar=False)), 0, None)
    p = ev / max(ev.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def mean_pairwise_distance(Z: np.ndarray, max_rows: int = 2000, seed: int = 0) -> float:
    Z = np.asarray(Z, np.float64)
    if len(Z) > max_rows:
        Z = Z[np.random.default_rng(seed).choice(len(Z), max_rows, replace=False)]
    sq = (Z ** 2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0))
    n = len(Z)
    return float(d.sum() / max(n * (n - 1), 1))


def distribution_metrics(generated: pd.DataFrame, reference: pd.DataFrame, scaler: Standardizer,
                         train_X: pd.DataFrame | None = None, k: int = 5) -> dict:
    """Coverage/precision/recall/density against ``reference`` plus spread diagnostics.

    Use held-out real nodes as ``reference`` to measure how well generation covers unseen MOFs.
    ``top1_analog_share`` is the fraction of samples whose nearest *training* node is the single
    most frequent one (a direct concentration / mode-collapse signal).
    """
    G = scaler.transform(generated.to_numpy(float))
    R = scaler.transform(reference.to_numpy(float))
    m = {f"{key}": val for key, val in prdc(R, G, k).items()}
    m["pairwise_distance_generated"] = mean_pairwise_distance(G)
    m["pairwise_distance_reference"] = mean_pairwise_distance(R)
    m["pairwise_distance_ratio"] = m["pairwise_distance_generated"] / max(m["pairwise_distance_reference"], 1e-12)
    m["effective_rank_generated"] = effective_rank(G)
    m["effective_rank_reference"] = effective_rank(R)
    if train_X is not None:
        T = cosine_normalize(scaler.transform(train_X.to_numpy(float)))
        nn = (cosine_normalize(G) @ T.T).argmax(1)
        counts = np.bincount(nn, minlength=len(T))
        m["top1_analog_share"] = float(counts.max() / len(nn))
        m["distinct_analog_fraction"] = float((counts > 0).sum() / len(nn))
    return m


# ---------------------------------------------------------------------------- analogs
def nearest_analogs(generated: pd.DataFrame, generated_ids, train_X: pd.DataFrame, train_ids, scaler: Standardizer,
                    k: int = 5, train_y=None, train_extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """The ``k`` nearest real training nodes of each generated node (for interpretation only)."""
    G = cosine_normalize(scaler.transform(generated.to_numpy(float)))
    T = cosine_normalize(scaler.transform(train_X.to_numpy(float)))
    sim = G @ T.T
    nn = _topk(sim, k)
    train_ids = np.asarray(train_ids)
    out = pd.DataFrame({"node_id": np.repeat(np.asarray(generated_ids), nn.shape[1]),
                        "rank": np.tile(np.arange(1, nn.shape[1] + 1), len(G)),
                        "analog_id": train_ids[nn.ravel()],
                        "similarity": np.take_along_axis(sim, nn, 1).ravel()})
    if train_y is not None:
        out["analog_target"] = np.asarray(train_y)[nn.ravel()]
    if train_extra is not None and "formula" in train_extra.columns:
        out["analog_formula"] = train_extra["formula"].to_numpy()[nn.ravel()]
    return out


# ---------------------------------------------------------------------------- surrogate fidelity
def surrogate_predictions(train_X: pd.DataFrame, train_y, generated: pd.DataFrame, random_state: int = 0,
                          model=None) -> np.ndarray:
    """Property predicted for generated descriptors by an independent model trained on real training rows.

    This is a *surrogate* diagnostic of conditional fidelity, limited by the surrogate's own accuracy.
    It is not a measured or computed property of any material.
    """
    if model is None:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state)
    model.fit(train_X.to_numpy(float), np.asarray(train_y, float))
    return model.predict(generated.to_numpy(float))
