"""Node tables: one row per real node, named descriptor columns, optional target."""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import QMOF106_FEATURES, Schema, get_schema


@dataclass
class NodeDataset:
    """Real nodes for GANg.

    Attributes
    ----------
    ids : np.ndarray
        Unique node identifiers.
    X : pd.DataFrame
        Descriptors in original units, columns in schema order, index = ids.
    y : np.ndarray or None
        Target used for conditioning (never used as a descriptor or for similarity).
    schema : Schema
    target_name : str or None
    extra : pd.DataFrame or None
        Any other columns (e.g. formula), kept for reference.
    """

    ids: np.ndarray
    X: pd.DataFrame
    y: np.ndarray | None
    schema: Schema
    target_name: str | None = None
    extra: pd.DataFrame | None = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, node_id: str | None = None, target: str | None = None,
                       schema: str | Schema = "auto", features: list[str] | None = None) -> "NodeDataset":
        """Build a dataset from a DataFrame.

        ``schema="auto"`` uses the QMOF-106 schema when all 106 descriptor columns are present;
        otherwise pass ``schema="generic", features=[...]``.
        """
        sch = get_schema(schema, columns=df.columns, features=features)
        missing = [c for c in sch.features if c not in df.columns]
        if missing:
            raise KeyError(f"{len(missing)} descriptor column(s) missing for schema '{sch.name}': {missing[:8]}...")
        if target is not None and target in sch.features:
            raise ValueError(f"Target '{target}' must not also be a descriptor column.")
        ids = (df[node_id].astype(str).to_numpy() if node_id is not None
               else np.array([f"node-{i}" for i in range(len(df))]))
        if len(set(ids)) != len(ids):
            dup = pd.Series(ids)[pd.Series(ids).duplicated()].unique()[:5]
            raise ValueError(f"Node IDs must be unique; duplicates such as {list(dup)} found.")
        X = df[sch.features].apply(pd.to_numeric, errors="coerce")
        bad = X.isna().any(axis=1) | ~np.isfinite(X.to_numpy(float)).all(axis=1)
        y = None
        if target is not None:
            y = pd.to_numeric(df[target], errors="coerce").to_numpy(float)
            bad |= ~np.isfinite(y)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} row(s) have missing or non-finite descriptor/target values; "
                             f"first IDs: {list(ids[bad.to_numpy()][:5])}. Clean or drop them first.")
        X.index = ids
        rest = [c for c in df.columns if c not in sch.features and c not in {node_id, target}]
        extra = df[rest].copy() if rest else None
        if extra is not None:
            extra.index = ids
        return cls(ids, X, y, sch, target, extra)

    @classmethod
    def from_csv(cls, path, node_id: str | None = None, target: str | None = None,
                 schema: str | Schema = "auto", features: list[str] | None = None, **read_csv_kwargs) -> "NodeDataset":
        return cls.from_dataframe(pd.read_csv(path, **read_csv_kwargs), node_id, target, schema, features)

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        t = f", target='{self.target_name}'" if self.target_name else ""
        return f"NodeDataset(n_nodes={len(self)}, n_features={self.X.shape[1]}, schema='{self.schema.name}'{t})"

    def subset(self, index) -> "NodeDataset":
        index = np.asarray(index)
        return NodeDataset(self.ids[index], self.X.iloc[index], None if self.y is None else self.y[index],
                           self.schema, self.target_name, None if self.extra is None else self.extra.iloc[index])

    def split(self, train: float = 0.70, validation: float = 0.15, test: float = 0.15, random_state: int = 42):
        """Random split into (train, validation, test). Use it before fitting anything."""
        if abs(train + validation + test - 1) > 1e-9:
            raise ValueError("train + validation + test must equal 1.")
        rng = np.random.default_rng(random_state)
        order = rng.permutation(len(self))
        n_tr = int(round(train * len(self)))
        n_va = int(round(validation * len(self)))
        return self.subset(order[:n_tr]), self.subset(order[n_tr:n_tr + n_va]), self.subset(order[n_tr + n_va:])

    def to_dataframe(self) -> pd.DataFrame:
        out = self.X.copy()
        out.insert(0, "node_id", self.ids)
        if self.y is not None:
            out.insert(1, self.target_name or "target", self.y)
        return out.reset_index(drop=True)


def load_qmof_example(as_frame: bool = False):
    """1,000 QMOF MOFs with the 106 GANg descriptors and the PBE band gap (eV).

    Derived from the QMOF database (Rosen et al., *Matter* 2021; *npj Comput. Mater.* 2022),
    CC BY 4.0. Columns: ``qmof_id``, ``formula``, ``bandgap``, then the 106 descriptors.
    Returns a :class:`NodeDataset` (target = band gap) or, with ``as_frame=True``, the raw DataFrame.
    """
    ref = resources.files("gang_mof") / "datasets" / "qmof_example_1000.csv.gz"
    with resources.as_file(ref) as path, gzip.open(Path(path), "rt") as fh:
        df = pd.read_csv(fh)
    if as_frame:
        return df
    return NodeDataset.from_dataframe(df, node_id="qmof_id", target="bandgap", schema="qmof106")


__all__ = ["NodeDataset", "load_qmof_example", "QMOF106_FEATURES"]
