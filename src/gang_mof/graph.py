"""Similarity graphs and cosine k-nearest-neighbour attachment.

Edges are *assigned* after generation, not generated: each synthetic node is linked to its
``k`` most similar real reference nodes by cosine similarity in training-standardized
descriptor space. Real nodes never receive edges from synthetic nodes unless you ask for it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import Standardizer, cosine_normalize


def _topk(sim: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sim.shape[1])
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(sim, idx, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)


def knn_edges(reference: np.ndarray, query: np.ndarray, reference_ids, query_ids, k: int = 15,
              scaler: Standardizer | None = None, exclude_self: bool = False) -> pd.DataFrame:
    """Cosine kNN edges from each query row to its ``k`` nearest reference rows.

    Inputs are descriptors in original units; ``scaler`` (fitted on training rows) standardizes
    both sides. Returns a DataFrame with ``source`` (query id), ``target`` (reference id),
    ``weight`` (cosine similarity) and ``rank``.
    """
    scaler = scaler or Standardizer.fit(reference)
    R = cosine_normalize(scaler.transform(reference))
    Q = cosine_normalize(scaler.transform(query))
    sim = Q @ R.T
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    nn = _topk(sim, k)
    reference_ids, query_ids = np.asarray(reference_ids), np.asarray(query_ids)
    rows = {"source": np.repeat(query_ids, nn.shape[1]), "target": reference_ids[nn.ravel()],
            "weight": np.take_along_axis(sim, nn, 1).ravel(), "rank": np.tile(np.arange(1, nn.shape[1] + 1), len(query_ids))}
    return pd.DataFrame(rows)


def build_reference_graph(dataset, k: int = 15, scaler: Standardizer | None = None) -> pd.DataFrame:
    """Undirected cosine kNN graph among real nodes (one row per edge, max weight kept)."""
    X = dataset.X.to_numpy(float)
    e = knn_edges(X, X, dataset.ids, dataset.ids, k, scaler, exclude_self=True)
    a = np.where(e.source < e.target, e.source, e.target)
    b = np.where(e.source < e.target, e.target, e.source)
    und = pd.DataFrame({"source": a, "target": b, "weight": e.weight})
    return und.groupby(["source", "target"], as_index=False)["weight"].max()


def to_networkx(result, reference_edges: pd.DataFrame | None = None):
    """Augmented graph (real + synthetic nodes) as a NetworkX graph. Requires ``networkx``."""
    try:
        import networkx as nx
    except ImportError as err:  # pragma: no cover
        raise ImportError("to_networkx needs networkx: pip install 'gang-mof[graph]'") from err
    g = nx.Graph()
    if reference_edges is not None:
        for r in reference_edges.itertuples(index=False):
            g.add_edge(r.source, r.target, weight=float(r.weight), synthetic=False)
    for node_id in result.nodes["node_id"]:
        g.add_node(node_id, synthetic=True)
    if result.edges is not None:
        for r in result.edges.itertuples(index=False):
            g.add_edge(r.source, r.target, weight=float(r.weight), synthetic=True)
    return g
