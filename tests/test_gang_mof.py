"""Fast tests for the gang-mof package (tiny training budgets; minutes on a laptop CPU)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from gang_mof import (GANg, GANgConfig, NodeDataset, QMOF106_FEATURES, augmented_training_set,
                      build_reference_graph, descriptor_consistency, featurize_dataframe, load_qmof_example,
                      pseudo_label, to_networkx)
from gang_mof.cli import main as cli_main

TINY = dict(iterations=60, eval_start=30, eval_every=30, batch=64)


@pytest.fixture(scope="module")
def data():
    return load_qmof_example()


@pytest.fixture(scope="module")
def splits(data):
    return data.subset(np.arange(300)).split(random_state=1)


@pytest.fixture(scope="module")
def fitted(splits):
    train, val, _ = splits
    return GANg(config=GANgConfig(**TINY), random_state=0, verbose=False).fit(train, val)


def test_example_dataset(data):
    assert len(data) == 1000 and list(data.X.columns) == QMOF106_FEATURES
    assert data.schema.name == "qmof106" and data.target_name == "bandgap"


def test_featurizer_reproduces_published_descriptors():
    df = load_qmof_example(as_frame=True).head(100)
    X = featurize_dataframe(df)
    np.testing.assert_allclose(X.to_numpy(), df[QMOF106_FEATURES].to_numpy(), atol=1e-9)


def test_real_data_mostly_consistent(data):
    assert descriptor_consistency(data.X)["consistent"].mean() > 0.95


def test_dataset_validation(data):
    df = load_qmof_example(as_frame=True).head(20)
    with pytest.raises(ValueError, match="unique"):
        NodeDataset.from_dataframe(pd.concat([df, df.head(1)]), node_id="qmof_id", target="bandgap")
    with pytest.raises(KeyError):
        NodeDataset.from_dataframe(df.drop(columns=["frac_H"]), node_id="qmof_id", schema="qmof106")
    bad = df.copy()
    bad.loc[bad.index[0], "info_volume"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        NodeDataset.from_dataframe(bad, node_id="qmof_id", target="bandgap")


def test_split_is_disjoint(data):
    tr, va, te = data.split(random_state=3)
    assert len(tr) + len(va) + len(te) == len(data)
    assert not (set(tr.ids) & set(va.ids) or set(tr.ids) & set(te.ids) or set(va.ids) & set(te.ids))


def test_constrained_samples_are_schema_consistent(fitted):
    res = fitted.sample(64, condition=2.0, random_state=0)
    assert len(res) == 64
    assert res.checks["schema_consistent"].all()
    assert (res.nodes["requested_condition"] == 2.0).all()
    frac = res.descriptors[[c for c in QMOF106_FEATURES if c.startswith("frac_")]].to_numpy()
    np.testing.assert_allclose(frac.sum(1), 1.0, atol=1e-9)
    assert res.nodes["formula"].str.len().gt(0).all()


def test_attach_and_evaluate(fitted, splits):
    res = fitted.attach(fitted.sample(32, random_state=1), k=7, n_analogs=3)
    assert len(res.edges) == 32 * 7
    assert set(res.edges["target"]) <= set(fitted.reference_ids)
    assert res.edges["weight"].between(-1, 1).all()
    assert len(res.analogs) == 32 * 3
    m = fitted.evaluate(res, reference=splits[2])
    for key in ["coverage", "precision", "recall", "top1_analog_share", "schema_consistent_rate"]:
        assert key in m
    assert "surrogate_prediction" in res.nodes


def test_save_load_roundtrip(fitted, tmp_path):
    fitted.save(tmp_path / "m")
    loaded = GANg.load(tmp_path / "m", verbose=False)
    a = fitted.sample(16, condition=1.5, random_state=7).descriptors.to_numpy()
    b = loaded.sample(16, condition=1.5, random_state=7).descriptors.to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_export(fitted, tmp_path):
    res = fitted.attach(fitted.sample(8))
    out = res.export(tmp_path / "res")
    for f in ["nodes.csv", "checks.csv", "edges.csv", "analogs.csv", "metrics.json", "metadata.json"]:
        assert (out / f).exists()


def test_generic_baseline_on_any_table():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(150, 5)), columns=list("abcde"))
    df["y"] = df["a"] * 2 + rng.normal(size=150) * 0.1
    data = NodeDataset.from_dataframe(df, target="y", schema="generic", features=list("abcde"))
    tr, va, _ = data.split(random_state=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = GANg(config=GANgConfig(**TINY), verbose=False).fit(tr, va)
        assert any("baseline" in str(x.message) for x in w)
    res = model.attach(model.sample(20), k=5)
    assert res.checks["within_training_range"].all()
    assert "schema_consistent" not in res.checks


def test_constrained_requires_mof_schema():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=list("abc"))
    data = NodeDataset.from_dataframe(df, schema="generic", features=list("abc"))
    with pytest.raises(ValueError, match="qmof106"):
        GANg(generator="constrained", config=GANgConfig(**TINY), verbose=False).fit(data)


def test_unconditional_without_target(splits):
    tr, va, _ = splits
    tr = NodeDataset(tr.ids, tr.X, None, tr.schema)
    va = NodeDataset(va.ids, va.X, None, va.schema)
    m = GANg(config=GANgConfig(**TINY), verbose=False).fit(tr, va)
    res = m.sample(10)
    assert res.nodes["requested_condition"].isna().all()


def test_pseudo_labels_and_augmented_set(fitted):
    res = fitted.sample(20)
    labels = pseudo_label(fitted, res)
    assert np.isfinite(labels).all()
    aug = augmented_training_set(fitted, res, synthetic_weight=0.25, accepted_only=False)
    assert aug["is_synthetic"].sum() == 20 and set(aug["sample_weight"]) == {1.0, 0.25}


def test_graph_helpers(fitted, splits):
    pytest.importorskip("networkx")
    ref = build_reference_graph(splits[0], k=5)
    res = fitted.attach(fitted.sample(5), k=3)
    g = to_networkx(res, ref)
    assert g.number_of_nodes() >= len(splits[0]) and all(g.nodes[n]["synthetic"] for n in res.nodes["node_id"])


def test_cli_roundtrip(tmp_path):
    csv = tmp_path / "mofs.csv"
    cli_main(["example", "--out", str(csv)])
    small = pd.read_csv(csv).head(200)
    small.to_csv(csv, index=False)
    cli_main(["fit", "--data", str(csv), "--node-id", "qmof_id", "--target", "bandgap",
              "--iterations", "40", "--out", str(tmp_path / "model")])
    cli_main(["sample", "--model", str(tmp_path / "model"), "--n", "10", "--condition", "2.0",
              "--no-surrogate", "--out", str(tmp_path / "res")])
    assert (tmp_path / "res" / "nodes.csv").exists()
