"""gang-mof: target-conditioned generation of auditable graph nodes for MOF similarity networks.

Reference implementation of GANg (Jalali, Orth and Reischl, *Advanced Intelligent Systems*, 2026).

Quick start::

    from gang_mof import GANg, load_qmof_example

    data = load_qmof_example()                      # 1,000 QMOF MOFs, 106 descriptors, band gap
    train, val, test = data.split(random_state=42)
    model = GANg(random_state=0).fit(train, validation_data=val)
    result = model.attach(model.sample(100, condition=2.0), k=15)
    model.evaluate(result, reference=test)
    print(result.summary())
"""
from ._version import __version__
from .augment import augmented_training_set, pseudo_label
from .chemistry import featurize, featurize_dataframe
from .data import NodeDataset, load_qmof_example
from .evaluation import candidate_checks, descriptor_consistency, distribution_metrics
from .graph import build_reference_graph, knn_edges, to_networkx
from .model import GANg, GANgConfig
from .result import GenerationResult
from .schema import QMOF106_FEATURES, Schema, get_schema

__all__ = [
    "__version__", "GANg", "GANgConfig", "GenerationResult", "NodeDataset", "load_qmof_example",
    "featurize", "featurize_dataframe", "QMOF106_FEATURES", "Schema", "get_schema",
    "candidate_checks", "descriptor_consistency", "distribution_metrics",
    "build_reference_graph", "knn_edges", "to_networkx", "pseudo_label", "augmented_training_set",
]
