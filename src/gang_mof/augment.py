"""Pseudo-labelling and augmented training sets.

In the GANg paper, augmentation helped only when real labels were scarce, and the gain came
mainly from distilling a training-only teacher model into the student. These helpers make that
workflow explicit: synthetic nodes get *pseudo-labels* from a teacher fitted on real training rows,
never their requested condition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pseudo_label(model, result, teacher=None, random_state: int | None = None) -> np.ndarray:
    """Label generated nodes with a teacher trained on the model's real training rows.

    The default teacher is ``ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2)`` as in the paper.
    Labels are stored in ``result.nodes["pseudo_label"]`` and returned.
    """
    if model.reference_y is None:
        raise ValueError("pseudo_label needs a model trained with a target.")
    if teacher is None:
        from sklearn.ensemble import ExtraTreesRegressor
        teacher = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                      random_state=model.random_state if random_state is None else random_state)
    teacher.fit(model.reference_X.to_numpy(float), model.reference_y)
    labels = teacher.predict(result.descriptors.to_numpy(float))
    result.nodes["pseudo_label"] = labels
    result.metadata["pseudo_label_teacher"] = type(teacher).__name__
    return labels


def augmented_training_set(model, result, synthetic_weight: float = 0.25, accepted_only: bool = True) -> pd.DataFrame:
    """Real training rows plus pseudo-labelled synthetic rows, with ``is_synthetic`` and ``sample_weight``.

    Pass ``sample_weight`` to your regressor's ``fit`` (or use it as a per-example loss weight in a GNN).
    Choose ``synthetic_weight`` and the number of synthetic rows on *validation* data, never on test data.
    """
    res = result.accepted() if accepted_only else result
    if "pseudo_label" not in res.nodes:
        pseudo_label(model, res)
    target = model.target_name or "target"
    real = model.reference_X.copy()
    real.insert(0, target, model.reference_y)
    real.insert(0, "node_id", model.reference_ids)
    real["is_synthetic"], real["sample_weight"] = False, 1.0
    syn = res.nodes[["node_id"] + model.feature_names].copy()
    syn.insert(1, target, res.nodes["pseudo_label"].to_numpy())
    syn["is_synthetic"], syn["sample_weight"] = True, float(synthetic_weight)
    return pd.concat([real.reset_index(drop=True), syn.reset_index(drop=True)], ignore_index=True)
