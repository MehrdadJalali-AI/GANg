"""Graph-free supervised baselines on the same 1000-node split and bandgap target."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "experiments" / "qmof_gang_final" / "tables" / "qmof_node_table.csv"
FEAT_NPY = ROOT / "experiments" / "qmof_galaxy_final" / "generated_nodes" / "qmof_node_features_normalized.npy"
GNN = ROOT / "experiments" / "qmof_gang_final" / "tables" / "downstream_summary.csv"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures" / "npj"
REPORTS = ROOT / "reports"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def mae_on_split(model, x_tr, y_tr, x_te, y_te):
    model.fit(x_tr, y_tr)
    return float(mean_absolute_error(y_te, model.predict(x_te)))


def main() -> None:
    node = pd.read_csv(NODE)
    x = np.load(FEAT_NPY)
    y = node["target_bandgap"].to_numpy()
    splits = node["split"].to_numpy()
    tr = splits == "train"
    va = splits == "validation"
    te = splits == "test"
    x_train, y_train = x[tr], y[tr]
    x_val, y_val = x[va], y[va]
    x_test, y_test = x[te], y[te]
    x_trva = np.vstack([x_train, x_val])
    y_trva = np.concatenate([y_train, y_val])

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42),
        ),
    }
    rows = []
    for name, model in models.items():
        mae = mae_on_split(model, x_trva, y_trva, x_test, y_test)
        rows.append({"model": name, "target": "outputs.pbe.bandgap", "test_mae": mae, "split_policy": "700/150/150 seed 42"})

    gnn = pd.read_csv(GNN)
    for model_name in ["GraphSAGE", "GAT", "GCN"]:
        hit = gnn[(gnn.model == model_name) & (gnn.training_mode == "real_only")]
        if len(hit) == 0:
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "model": f"{model_name}_real_only_GNN",
                "target": "outputs.pbe.bandgap",
                "test_mae": float(r.mae_mean),
                "split_policy": "held-out real QMOF test nodes",
            }
        )

    out = pd.DataFrame(rows).sort_values("test_mae")
    out.to_csv(TABLES / "graph_free_baselines.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ax.barh(out.model, out.test_mae, color="#4C9FAD")
    ax.set_xlabel("Test MAE (eV)")
    ax.set_title("Graph-free vs GNN real-only baselines (bandgap)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGS / "graph_free_vs_gnn_baselines.pdf")
    fig.savefig(FIGS.parent / "graph_free_vs_gnn_baselines.pdf")
    plt.close(fig)

    (REPORTS / "graph_free_baselines.md").write_text(
        "# Graph-free baselines\n\n"
        + out.to_markdown(index=False)
        + "\n\nGraph-free models use the same normalized 106-D descriptors and held-out test MOFs. "
        "Lower MAE is better. GNN real-only values are from locked downstream summary.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
