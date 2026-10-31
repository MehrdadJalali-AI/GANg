"""The auditable output of a GANg generation run."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GenerationResult:
    """Generated nodes plus everything needed to audit them.

    nodes : one row per generated node: ``node_id``, ``requested_condition`` (the target value
        the generator was *asked* for; never a label), ``formula`` (qmof106 schema) and the descriptors
        in original units.
    checks : per-node validity, duplication, novelty and acceptance (see :func:`gang_mof.evaluation.candidate_checks`).
    edges : cosine kNN edges ``source`` (generated id) -> ``target`` (real id), filled by ``attach``.
    analogs : nearest real training nodes, for interpretation only, filled by ``attach``.
    metrics : distribution diagnostics, filled by :meth:`gang_mof.GANg.evaluate`.
    metadata : model, schema, seed and settings used.
    """

    nodes: pd.DataFrame
    checks: pd.DataFrame
    feature_names: list[str]
    edges: pd.DataFrame | None = None
    analogs: pd.DataFrame | None = None
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def descriptors(self) -> pd.DataFrame:
        """Generated descriptors only (index = generated node ids)."""
        return self.nodes.set_index("node_id")[self.feature_names]

    def __len__(self) -> int:
        return len(self.nodes)

    def accepted(self) -> "GenerationResult":
        """A copy keeping only nodes that passed every check."""
        keep = self.checks["accepted"].to_numpy()
        ids = set(self.nodes.loc[keep, "node_id"])
        sub = lambda df, col: None if df is None else df[df[col].isin(ids)].reset_index(drop=True)
        return GenerationResult(self.nodes[keep].reset_index(drop=True), self.checks[keep].reset_index(drop=True),
                                self.feature_names, sub(self.edges, "source"), sub(self.analogs, "node_id"),
                                dict(self.metrics), {**self.metadata, "filtered": "accepted_only"})

    def summary(self) -> str:
        c = self.checks
        lines = [f"GANg generation: {len(self)} nodes (schema: {self.metadata.get('schema')}, "
                 f"mode: {self.metadata.get('mode')})",
                 f"  accepted:               {c['accepted'].mean():.3f}",
                 f"  within training range:  {c['within_training_range'].mean():.3f}"]
        if "schema_consistent" in c:
            lines.append(f"  schema consistent:      {c['schema_consistent'].mean():.3f}")
        lines += [f"  novel (vs training):    {c['novel'].mean():.3f}",
                  f"  duplicate of training:  {c['duplicate_of_training'].mean():.3f}"]
        if self.edges is not None:
            lines.append(f"  edges attached:         {len(self.edges)} (k = {self.metadata.get('k')})")
        for key in ["coverage", "precision", "recall", "pairwise_distance_ratio", "top1_analog_share",
                    "surrogate_condition_correlation"]:
            if key in self.metrics:
                lines.append(f"  {key + ':':<24}{self.metrics[key]:.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def export(self, directory) -> Path:
        """Write ``nodes.csv``, ``checks.csv``, ``edges.csv``, ``analogs.csv``, ``metrics.json`` and ``metadata.json``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.nodes.to_csv(d / "nodes.csv", index=False)
        self.checks.to_csv(d / "checks.csv", index=False)
        if self.edges is not None:
            self.edges.to_csv(d / "edges.csv", index=False)
        if self.analogs is not None:
            self.analogs.to_csv(d / "analogs.csv", index=False)
        clean = lambda o: o.item() if isinstance(o, np.generic) else str(o)
        (d / "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=clean))
        (d / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=clean))
        return d
