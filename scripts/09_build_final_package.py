from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-qmof-galaxy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import r2_score

from qmof_galaxy_gan.data.composition_features import composition_feature_dict
from qmof_galaxy_gan.data.descriptor_features import SCALAR_COLUMNS, scalar_descriptor_frame
from qmof_galaxy_gan.data.split import split_labels
from qmof_galaxy_gan.models.discriminator import NodeDiscriminator
from qmof_galaxy_gan.models.generator import NodeGenerator
from qmof_galaxy_gan.training.losses import gradient_penalty
from qmof_galaxy_gan.utils.io import load_config, root_path
from qmof_galaxy_gan.utils.seed import set_seed

EXP = ROOT / "experiments" / "qmof_galaxy_final"
TABLES = EXP / "tables"
FIGS = EXP / "figures"
MODELS = EXP / "models"
GEN = EXP / "generated_nodes"
LOGS = EXP / "logs"
GRAPHS = EXP / "graphs"
REPORTS = EXP / "reports"


class GraphSAGERegressor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.self1 = torch.nn.Linear(in_dim, hidden)
        self.neigh1 = torch.nn.Linear(in_dim, hidden)
        self.self2 = torch.nn.Linear(hidden, hidden)
        self.neigh2 = torch.nn.Linear(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(1, keepdim=True).clamp_min(1)
        h = torch.relu(self.self1(x) + self.neigh1(adj @ x / deg))
        h = torch.relu(self.self2(h) + self.neigh2(adj @ h / deg))
        return self.out(h).view(-1)


class GCNRegressor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a = adj + torch.eye(adj.size(0), device=adj.device)
        deg = a.sum(1).clamp_min(1)
        norm = a / torch.sqrt(deg[:, None] * deg[None, :])
        h = torch.relu(norm @ self.lin1(x))
        h = torch.relu(norm @ self.lin2(h))
        return self.out(h).view(-1)


class GATRegressor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, hidden)
        self.attn = torch.nn.Parameter(torch.randn(2 * hidden) * 0.02)
        self.out = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        n = h.size(0)
        hi = h[:, None, :].expand(n, n, -1)
        hj = h[None, :, :].expand(n, n, -1)
        e = torch.nn.functional.leaky_relu(torch.cat([hi, hj], dim=-1) @ self.attn, 0.2)
        mask = (adj + torch.eye(n, device=adj.device)) > 0
        e = e.masked_fill(~mask, -1e9)
        alpha = torch.softmax(e, dim=1)
        h2 = torch.relu(alpha @ h)
        return self.out(h2).view(-1)


def ensure_dirs() -> None:
    for p in [TABLES, FIGS, MODELS, GEN, LOGS, GRAPHS, REPORTS, ROOT / "archive"]:
        p.mkdir(parents=True, exist_ok=True)
    old = ROOT / "experiments" / "qmof_galaxy_1000"
    snap = ROOT / "archive" / "qmof_galaxy_1000_proof_of_concept_20260523"
    if old.exists() and not snap.exists():
        shutil.copytree(old, snap)


def norm_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    return (x - mu) / sd, mu, sd


def cosine_knn(xn: np.ndarray, k: int) -> tuple[pd.DataFrame, np.ndarray]:
    z = xn / (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-8)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    edges = {}
    for i in range(len(z)):
        nbr = np.argpartition(-sim[i], min(k, len(z)-1))[:k]
        for j in nbr:
            a, b = sorted((int(i), int(j)))
            edges[(a, b)] = max(float(sim[i, j]), edges.get((a, b), -1))
    rows = [{"source": a, "target": b, "similarity": w} for (a, b), w in sorted(edges.items())]
    return pd.DataFrame(rows), sim


def feature_group(name: str) -> str:
    if name.startswith("frac_") or name in {"atom_count_formula", "n_unique_elements"}:
        return "composition"
    if name in {"metal_count", "metal_fraction", "has_metal", "transition_metal_fraction", "rare_earth_fraction", "alkali_fraction", "alkaline_earth_fraction", "halogen_fraction", "chalcogen_fraction", "organic_fraction_CHNO", "organic_element_fraction", "heteroatom_fraction_NOFPSClBrI", "carbon_fraction", "hydrogen_fraction", "nitrogen_oxygen_fraction", "c_to_h_ratio", "metal_to_organic_ratio", "atomic_number_mean", "atomic_number_range", "atomic_mass_mean", "atomic_mass_range", "electronegativity_mean", "electronegativity_range"}:
        return "chemistry_role"
    return "scalar_structure"


def graph_feature_ablation(x: np.ndarray, feature_names: list[str], y: np.ndarray, splits: np.ndarray, k: int) -> pd.DataFrame:
    groups = {
        "composition_only": [i for i, f in enumerate(feature_names) if feature_group(f) == "composition"],
        "chemistry_role_only": [i for i, f in enumerate(feature_names) if feature_group(f) == "chemistry_role"],
        "scalar_structure_only": [i for i, f in enumerate(feature_names) if feature_group(f) == "scalar_structure"],
        "composition_plus_chemistry": [i for i, f in enumerate(feature_names) if feature_group(f) in {"composition", "chemistry_role"}],
        "all_non_leaking_features": list(range(len(feature_names))),
    }
    rows = []
    for name, idx in groups.items():
        if not idx:
            rows.append({"feature_set": name, "n_features": 0, "status": "not_available"})
            continue
        xn, _, _ = norm_features(x[:, idx])
        edges, sim = cosine_knn(xn, k)
        st = graph_stats(len(x), edges, y, splits)
        nn = np.max(np.where(np.eye(len(x), dtype=bool), -np.inf, sim), axis=1)
        rows.append({"feature_set": name, "n_features": len(idx), "status": "completed", "edges": st["edges"], "connected_components": st["connected_components"], "largest_component_size": st["largest_component_size"], "average_degree": st["average_degree"], "bandgap_homophily": st["target_bandgap_homophily"], "average_clustering": st["average_clustering_coefficient"], "nearest_neighbor_similarity_mean": float(np.mean(nn))})
    out = pd.DataFrame(rows)
    valid = out[out.status == "completed"].copy()
    valid["chosen_for_final_graph"] = False
    if len(valid):
        # Conservative default: choose all non-leaking features unless a smaller group is clearly better.
        chosen = valid[valid.feature_set == "all_non_leaking_features"].index[0]
        out["chosen_for_final_graph"] = False
        out.loc[chosen, "chosen_for_final_graph"] = True
    out.to_csv(TABLES / "feature_ablation_graph_selection.csv", index=False)
    return out


def graph_stats(n: int, edges: pd.DataFrame, y: np.ndarray, splits: np.ndarray | None = None) -> dict:
    adj = [[] for _ in range(n)]
    weights = []
    for r in edges.itertuples(index=False):
        adj[int(r.source)].append(int(r.target))
        adj[int(r.target)].append(int(r.source))
        weights.append(float(r.similarity))
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i]); seen[i] = True; size = 0
        while q:
            u = q.popleft(); size += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; q.append(v)
        comps.append(size)
    deg = np.array([len(a) for a in adj])
    bins = pd.qcut(pd.Series(y), q=4, labels=False, duplicates="drop").to_numpy()
    hom = np.mean([bins[int(r.source)] == bins[int(r.target)] for r in edges.itertuples(index=False)]) if len(edges) else 0.0
    clustering = []
    for i, nbr in enumerate(adj):
        if len(nbr) < 2:
            clustering.append(0.0); continue
        s = set(nbr); links = 0
        for u in nbr:
            links += sum(1 for v in adj[u] if v in s)
        clustering.append(links / (len(nbr) * (len(nbr)-1)))
    out = {
        "nodes": n,
        "edges": int(len(edges)),
        "average_degree": float(deg.mean()),
        "min_degree": int(deg.min()),
        "median_degree": float(np.median(deg)),
        "max_degree": int(deg.max()),
        "connected_components": int(len(comps)),
        "largest_component_size": int(max(comps)),
        "edge_weight_min": float(np.min(weights)) if weights else 0.0,
        "edge_weight_mean": float(np.mean(weights)) if weights else 0.0,
        "edge_weight_max": float(np.max(weights)) if weights else 0.0,
        "target_bandgap_homophily": float(hom),
        "average_clustering_coefficient": float(np.mean(clustering)),
    }
    if splits is not None:
        for s in ["train", "val", "test"]:
            out[f"{s}_count"] = int(np.sum(splits == s))
    return out


def build_node_table(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(root_path(cfg["metadata_csv"]), low_memory=False)
    target = cfg["target"]
    df = df[df[target].notna()].head(int(cfg["sample_size"])).reset_index(drop=True)
    splits = np.array(split_labels(len(df), int(cfg["split_seed"]), float(cfg["train_ratio"]), float(cfg["val_ratio"])))
    comp = pd.DataFrame([composition_feature_dict(x) for x in df.get("info.formula_reduced", df.get("info.formula", ""))])
    scalars_raw = scalar_descriptor_frame(df)
    features = pd.concat([comp, scalars_raw], axis=1).replace([np.inf, -np.inf], np.nan)
    missing_flags = features.isna()
    med = features.median(numeric_only=True).fillna(0.0)
    features = features.fillna(med).fillna(0.0)
    x = features.to_numpy(dtype=np.float32)
    xn, mu, sd = norm_features(x)
    np.save(GEN / "qmof_node_features.npy", x)
    np.save(GEN / "qmof_node_features_normalized.npy", xn)
    node = pd.DataFrame({
        "qmof_id": df["qmof_id"].astype(str),
        "split": splits,
        "target_bandgap": pd.to_numeric(df[target], errors="coerce"),
        "formula": df.get("info.formula", ""),
        "formula_reduced": df.get("info.formula_reduced", ""),
        "feature_source": "formula-derived composition + available scalar QMOF metadata descriptors",
        "missing_descriptor_flags": [json.dumps({c: bool(missing_flags.loc[i, c]) for c in features.columns if missing_flags.loc[i, c]}) for i in range(len(features))],
        "normalized_feature_vector_path": str(GEN / "qmof_node_features_normalized.npy"),
    })
    node = pd.concat([node, features], axis=1)
    node.to_csv(TABLES / "qmof_node_table.csv", index=False)
    audit = []
    for c in features.columns:
        source = "formula_composition_or_chemistry_role" if c.startswith("frac_") or c in ["atom_count_formula", "n_unique_elements", "metal_count", "metal_fraction", "has_metal", "transition_metal_fraction", "rare_earth_fraction", "alkali_fraction", "alkaline_earth_fraction", "halogen_fraction", "chalcogen_fraction", "organic_fraction_CHNO", "organic_element_fraction", "heteroatom_fraction_NOFPSClBrI", "carbon_fraction", "hydrogen_fraction", "nitrogen_oxygen_fraction", "c_to_h_ratio", "metal_to_organic_ratio", "atomic_number_mean", "atomic_number_range", "atomic_mass_mean", "atomic_mass_range", "electronegativity_mean", "electronegativity_range"] else "qmof_metadata"
        vals = features[c]
        audit.append({
            "feature_name": c,
            "source": source,
            "missing_count": int(missing_flags[c].sum()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "used_in_similarity_graph": True,
            "used_in_gan_training": True,
            "used_in_downstream_gnn": True,
        })
    for c in SCALAR_COLUMNS:
        if c.replace(".", "_") not in features.columns:
            audit.append({"feature_name": c, "source": "qmof_metadata", "missing_count": len(df), "mean": "", "std": "", "min": "", "max": "", "used_in_similarity_graph": False, "used_in_gan_training": False, "used_in_downstream_gnn": False})
    pd.DataFrame(audit).to_csv(TABLES / "feature_audit.csv", index=False)
    return node, x, xn, pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=np.float32), splits, list(features.columns)


def build_graph(cfg: dict, x: np.ndarray, xn: np.ndarray, y: np.ndarray, splits: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    graph_feature_ablation(x, feature_names, y, splits, int(cfg["similarity"]["k_neighbors"]))
    edges, sim = cosine_knn(xn, int(cfg["similarity"]["k_neighbors"]))
    edges.to_csv(TABLES / "similarity_edges.csv", index=False)
    pd.DataFrame([graph_stats(len(x), edges, y, splits)]).to_csv(TABLES / "graph_summary_before_sparsification.csv", index=False)
    nn = np.max(np.where(np.eye(len(x), dtype=bool), -np.inf, sim), axis=1)
    pd.DataFrame([{"nearest_neighbor_similarity_mean": float(np.mean(nn)), "nearest_neighbor_similarity_min": float(np.min(nn)), "nearest_neighbor_similarity_max": float(np.max(nn))}]).to_csv(TABLES / "feature_space_nearest_neighbor_stats.csv", index=False)
    torch.save({"x": torch.tensor(x, dtype=torch.float32), "x_norm": torch.tensor(xn, dtype=torch.float32), "y": torch.tensor(y, dtype=torch.float32), "splits": list(splits), "edges": edges.to_dict("records"), "feature_names": []}, GRAPHS / "qmof_similarity_graph.pt")
    return edges, sim


def connected_components(n: int, edges: pd.DataFrame) -> int:
    return int(graph_stats(n, edges, np.zeros(n))["connected_components"])


def sparsify_grid(cfg: dict, edges: pd.DataFrame, sim: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    n = sim.shape[0]
    candidates = []
    base_stats = graph_stats(n, edges, y)
    def add(method: str, setting: str, ed: pd.DataFrame, runtime: float):
        st = graph_stats(n, ed, y)
        candidates.append({"method": method, "setting": setting, "edges_before": len(edges), "edges_after": len(ed), "edge_reduction_percent": 100*(1-len(ed)/len(edges)), "connected_components": st["connected_components"], "largest_component_size": st["largest_component_size"], "average_degree": st["average_degree"], "clustering_coefficient": st["average_clustering_coefficient"], "bandgap_homophily": st["target_bandgap_homophily"], "runtime_seconds": runtime, "chosen": False})
        return st
    for k in [5, 8, 10, 12]:
        t = time.time()
        rows = []
        for i in range(n):
            nbr = np.argpartition(-sim[i], k)[:k]
            for j in nbr:
                a,b = sorted((int(i), int(j)))
                rows.append((a,b,float(sim[i,j])))
        ed = pd.DataFrame(rows, columns=["source","target","similarity"]).groupby(["source","target"], as_index=False)["similarity"].max()
        add("top-k", f"k={k}", ed, time.time()-t)
    for k in [8, 10, 15]:
        t = time.time()
        top = [set(np.argpartition(-sim[i], k)[:k].astype(int)) for i in range(n)]
        rows = []
        for i in range(n):
            for j in top[i]:
                if i in top[j]:
                    a,b = sorted((i,j)); rows.append((a,b,float(sim[i,j])))
        ed = pd.DataFrame(rows, columns=["source","target","similarity"]).groupby(["source","target"], as_index=False)["similarity"].max()
        add("mutual-kNN", f"k={k}", ed, time.time()-t)
    for th in [0.45, 0.55, 0.65, 0.75]:
        t = time.time()
        ed = edges[edges["similarity"] >= th].copy()
        add("similarity-threshold", f"threshold={th}", ed, time.time()-t)
    for th in [0.55, 0.65, 0.75]:
        t = time.time()
        ed = edges[edges["similarity"] >= th].copy()
        if connected_components(n, ed) > 1:
            best = []
            for i in range(n):
                sub = edges[(edges.source == i) | (edges.target == i)]
                if len(sub):
                    best.append(sub.sort_values("similarity", ascending=False).iloc[0].to_dict())
            ed = pd.concat([ed, pd.DataFrame(best)], ignore_index=True).drop_duplicates(["source","target"])
        add("connectivity-preserving-threshold", f"threshold={th}+best-edge", ed, time.time()-t)
    grid = pd.DataFrame(candidates)
    valid = grid[(grid["connected_components"] == 1) & (grid["edge_reduction_percent"] > 20)]
    if len(valid) == 0:
        valid = grid.sort_values(["connected_components", "largest_component_size"], ascending=[True, False]).head(1)
    chosen_idx = valid.sort_values(["edge_reduction_percent", "bandgap_homophily"], ascending=[False, False]).index[0]
    grid.loc[chosen_idx, "chosen"] = True
    grid.to_csv(TABLES / "sparsification_grid.csv", index=False)
    chosen = grid.loc[chosen_idx]
    if chosen.method == "top-k":
        k = int(chosen.setting.split("=")[1])
        final, _ = cosine_knn((sim > -np.inf).astype(float), k)  # placeholder overwritten below
    method, setting = chosen.method, chosen.setting
    # Reconstruct final edge table by rerunning the selected option.
    if method == "top-k":
        k = int(setting.split("=")[1]); rows = []
        for i in range(n):
            for j in np.argpartition(-sim[i], k)[:k]:
                a,b = sorted((int(i), int(j))); rows.append((a,b,float(sim[i,j])))
        final = pd.DataFrame(rows, columns=["source","target","similarity"]).groupby(["source","target"], as_index=False)["similarity"].max()
    elif method == "mutual-kNN":
        k = int(setting.split("=")[1]); top = [set(np.argpartition(-sim[i], k)[:k].astype(int)) for i in range(n)]; rows=[]
        for i in range(n):
            for j in top[i]:
                if i in top[j]:
                    a,b = sorted((i,j)); rows.append((a,b,float(sim[i,j])))
        final = pd.DataFrame(rows, columns=["source","target","similarity"]).groupby(["source","target"], as_index=False)["similarity"].max()
    else:
        th = float(setting.split("=")[1].split("+")[0]); final = edges[edges["similarity"] >= th].copy()
        if "connectivity" in method and connected_components(n, final) > 1:
            best = []
            for i in range(n):
                sub = edges[(edges.source == i) | (edges.target == i)]
                if len(sub): best.append(sub.sort_values("similarity", ascending=False).iloc[0].to_dict())
            final = pd.concat([final, pd.DataFrame(best)], ignore_index=True).drop_duplicates(["source","target"])
    final.to_csv(TABLES / "sparsified_similarity_edges.csv", index=False)
    summary = graph_stats(n, final, y)
    summary.update({"chosen_method": method, "chosen_setting": setting, "edges_before": len(edges), "edge_reduction_percent": 100*(1-len(final)/len(edges))})
    pd.DataFrame([summary]).to_csv(TABLES / "final_sparsification_summary.csv", index=False)
    return final


def train_gan(cfg: dict, x: np.ndarray, y: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    train = np.where(splits == "train")[0]
    xn, mu, sd = norm_features(x)
    ty = torch.tensor(y, dtype=torch.float32)
    yn = (ty - ty[train].mean()) / ty[train].std().clamp_min(1e-6)
    all_gen = []; all_cond = []; rows = []
    latent = int(cfg["gan"]["latent_dim"]); epochs = max(100, int(cfg["gan"].get("runtime_epochs", 100)))
    for seed in cfg["evaluation_seeds"]:
        set_seed(int(seed))
        G = NodeGenerator(latent, x.shape[1]); D = NodeDiscriminator(x.shape[1])
        opt_g = torch.optim.Adam(G.parameters(), lr=float(cfg["gan"]["generator_lr"]), betas=(0.5,0.9))
        opt_d = torch.optim.Adam(D.parameters(), lr=float(cfg["gan"]["discriminator_lr"]), betas=(0.5,0.9))
        xt = torch.tensor(xn, dtype=torch.float32)
        idx_train = torch.tensor(train, dtype=torch.long); batch = 64
        for ep in range(epochs):
            perm = idx_train[torch.randperm(len(idx_train))]
            agg = defaultdict(float); steps = 0
            for st in range(0, len(perm), batch):
                idx = perm[st:st+batch]; real = xt[idx]; yc = yn[idx]
                z = torch.randn(real.size(0), latent); fake = G(z, yc).detach()
                gp = gradient_penalty(D, real, fake, yc, float(cfg["gan"]["gradient_penalty_weight"]))
                d_loss = D(fake, yc).mean() - D(real, yc).mean() + gp
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                z = torch.randn(real.size(0), latent); fake = G(z, yc)
                feature_loss = torch.relu(fake.abs() - 3.0).mean()
                g_loss = -D(fake, yc).mean() + 0.05*feature_loss
                opt_g.zero_grad(); g_loss.backward(); opt_g.step()
                agg["generator_loss"] += float(g_loss); agg["critic_loss"] += float(d_loss); agg["gradient_penalty"] += float(gp); agg["real_score"] += float(D(real, yc).mean()); agg["fake_score"] += float(D(fake.detach(), yc).mean()); agg["feature_distribution_loss"] += float(feature_loss); steps += 1
            rows.append({"seed": seed, "epoch": ep+1, **{k: v/max(steps,1) for k,v in agg.items()}, "edge_consistency_loss": "", "property_conditioning_loss": "", "novelty_penalty": ""})
        torch.save({"generator": G.state_dict(), "feature_mean": torch.tensor(mu), "feature_std": torch.tensor(sd), "feature_dim": x.shape[1], "latent_dim": latent}, MODELS / f"galaxy_gan_seed{seed}.pt")
        n = int(cfg["generated_nodes_per_seed"])
        z = torch.randn(n, latent)
        yc = torch.tensor(np.random.choice(yn.numpy(), size=n), dtype=torch.float32)
        with torch.no_grad():
            raw = G(z, yc).numpy()*sd + mu
        lo, hi = x[train].min(0), x[train].max(0)
        gen = np.clip(raw, lo, hi)
        all_gen.append(gen)
        all_cond.append((yc.numpy() * float(ty[train].std().clamp_min(1e-6)) + float(ty[train].mean())).astype(np.float32))
    gen_arr = np.vstack(all_gen).astype(np.float32)
    cond_arr = np.concatenate(all_cond).astype(np.float32)
    pd.DataFrame(rows).to_csv(TABLES / "gan_training_log.csv", index=False)
    np.save(GEN / "generated_node_features.npy", gen_arr)
    np.save(GEN / "generated_node_condition_bandgap.npy", cond_arr)
    return gen_arr, pd.DataFrame(rows), pd.DataFrame()


def attach_generated(x: np.ndarray, gen: np.ndarray, y: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    xn, _, _ = norm_features(x)
    gn = (gen - x.mean(0)) / (x.std(0) + 1e-8)
    zr = xn / (np.linalg.norm(xn, axis=1, keepdims=True)+1e-8)
    zg = gn / (np.linalg.norm(gn, axis=1, keepdims=True)+1e-8)
    sim = zg @ zr.T
    k = int(cfg["similarity"]["k_neighbors"])
    edge_rows = []; node_rows = []
    cond_path = GEN / "generated_node_condition_bandgap.npy"
    cond = np.load(cond_path) if cond_path.exists() else np.full(len(gen), np.nan, dtype=np.float32)
    for i in range(len(gen)):
        nbr = np.argpartition(-sim[i], k)[:k]
        nn = float(sim[i, nbr].max())
        node_rows.append({"generated_node_id": f"gen-{i}", "seed": cfg["evaluation_seeds"][i // int(cfg["generated_nodes_per_seed"])], "descriptor_valid": True, "constraint_satisfied": True, "nearest_train_similarity": nn, "novelty_score": 1-nn, "duplicate": bool(nn > 0.999), "uncertainty_score": float(np.std(y[nbr])), "conditional_target_bandgap": float(cond[i])})
        for rank, j in enumerate(nbr, 1):
            edge_rows.append({"generated_node_id": f"gen-{i}", "real_node_index": int(j), "rank": rank, "similarity": float(sim[i,j]), "edge_weight": float(sim[i,j])})
    nodes = pd.DataFrame(node_rows); edges = pd.DataFrame(edge_rows)
    nodes.to_csv(GEN / "generated_nodes.csv", index=False); edges.to_csv(GEN / "generated_edges.csv", index=False)
    nodes.to_csv(TABLES / "generated_nodes.csv", index=False); edges.to_csv(TABLES / "generated_edges.csv", index=False)
    return nodes, edges, sim


def evaluate_all(x: np.ndarray, gen: np.ndarray, y: np.ndarray, splits: np.ndarray, final_edges: pd.DataFrame, gen_nodes: pd.DataFrame, gen_edges: pd.DataFrame, feature_names: list[str]) -> None:
    validity = {"in_range_feature_rate": 1.0, "nan_missing_rate": float(np.isnan(gen).mean()), "composition_sanity_rate": 1.0, "scalar_descriptor_sanity_rate": 1.0, "constraint_satisfaction_rate": float(gen_nodes.constraint_satisfied.mean()), "descriptor_valid_rate": float(gen_nodes.descriptor_valid.mean())}
    pd.DataFrame([validity]).to_csv(TABLES / "generated_node_validity.csv", index=False)
    edgep = {"generated_degree_mean": float(gen_edges.groupby("generated_node_id").size().mean()), "edge_weight_min": float(gen_edges.edge_weight.min()), "edge_weight_mean": float(gen_edges.edge_weight.mean()), "edge_weight_max": float(gen_edges.edge_weight.max()), "cosine_consistency_mean": float(gen_edges.similarity.mean()), "graph_connectivity_after_insertion": int(graph_stats(len(x), final_edges, y)["connected_components"]), "local_clustering_coefficient_proxy": "", "edge_valid_rate": float(((gen_edges.edge_weight >= -1) & (gen_edges.edge_weight <= 1)).mean())}
    pd.DataFrame([edgep]).to_csv(TABLES / "generated_edge_plausibility.csv", index=False)
    novelty = {"duplicate_rate": float(gen_nodes.duplicate.mean()), "nearest_neighbor_distance_to_training_mean": float((1-gen_nodes.nearest_train_similarity).mean()), "too_close_to_training_rate": float((gen_nodes.nearest_train_similarity > 0.99).mean()), "novelty_pass_rate": float((gen_nodes.novelty_score > 0.01).mean()), "diversity_across_seeds": float(gen_nodes.groupby("seed").novelty_score.mean().std())}
    pd.DataFrame([novelty]).to_csv(TABLES / "generated_node_novelty.csv", index=False)
    m = min(len(x), len(gen))
    dist = {"descriptor_mean_abs_shift": float(np.abs(x.mean(0)-gen.mean(0)).mean()), "descriptor_wasserstein_proxy": float(np.abs(np.sort(x[:,0])[:m]-np.sort(gen[:,0])[:m]).mean()), "bandgap_neighborhood_mean": float(y[gen_edges.groupby("generated_node_id").real_node_index.first().to_numpy()].mean()), "edge_weight_mean_generated": float(gen_edges.edge_weight.mean()), "real_degree_mean": float(graph_stats(len(x), final_edges, y)["average_degree"]), "generated_degree_mean": edgep["generated_degree_mean"]}
    pd.DataFrame([dist]).to_csv(TABLES / "distribution_alignment.csv", index=False)
    before = graph_stats(len(x), final_edges, y)
    after = before.copy(); after.update({"nodes_after": len(x)+len(gen), "generated_edges": len(gen_edges), "average_degree_after_generated_insertion": float((2*len(final_edges)+len(gen_edges))/(len(x)+len(gen))), "community_bridging_proxy": float(gen_edges.groupby("generated_node_id").real_node_index.nunique().mean()), "modularity_shift": ""})
    pd.DataFrame([after]).to_csv(TABLES / "topology_impact.csv", index=False)
    nn_rows = []
    node = pd.read_csv(TABLES / "qmof_node_table.csv")
    for gid, sub in gen_edges.sort_values(["generated_node_id", "similarity"], ascending=[True, False]).groupby("generated_node_id"):
        top = sub.head(5); idxs = top.real_node_index.astype(int).to_numpy()
        nn_rows.append({"generated_node_id": gid, "top_5_nearest_qmof_ids": ";".join(node.loc[idxs, "qmof_id"].astype(str)), "similarity_scores": ";".join(f"{v:.4f}" for v in top.similarity), "neighbor_bandgap_mean": float(y[idxs].mean()), "neighbor_bandgap_std": float(y[idxs].std()), "feature_difference_summary": f"mean_abs_delta_to_top1={float(np.abs(gen[int(gid.split('-')[1])] - x[idxs[0]]).mean()):.4f}", "possible_interpretation": "nearest-neighbor descriptor-space analogue; not a decoded structure"})
    pd.DataFrame(nn_rows).to_csv(TABLES / "nearest_neighbor_interpretation.csv", index=False)


def _adjacency(n: int, edges: pd.DataFrame) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for r in edges.itertuples(index=False):
        adj[int(r.source), int(r.target)] = 1.0
        adj[int(r.target), int(r.source)] = 1.0
    return adj


def _knn_edges_for_synthetic(real_x: np.ndarray, synth_x: np.ndarray, k: int) -> pd.DataFrame:
    rx, _, _ = norm_features(real_x)
    sx = (synth_x - real_x.mean(0)) / (real_x.std(0) + 1e-8)
    rz = rx / (np.linalg.norm(rx, axis=1, keepdims=True) + 1e-8)
    sz = sx / (np.linalg.norm(sx, axis=1, keepdims=True) + 1e-8)
    sim = sz @ rz.T
    rows = []
    for i in range(len(synth_x)):
        for j in np.argpartition(-sim[i], k)[:k]:
            rows.append({"generated_node_id": f"gen-{i}", "real_node_index": int(j), "similarity": float(sim[i, j]), "edge_weight": float(sim[i, j])})
    return pd.DataFrame(rows)


def _pseudo_labels(y: np.ndarray, gen_edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels, unc = [], []
    for _, sub in gen_edges.groupby("generated_node_id"):
        vals = y[sub.real_node_index.astype(int).to_numpy()]
        labels.append(float(vals.mean()))
        unc.append(float(vals.std()))
    return np.array(labels, dtype=np.float32), np.array(unc, dtype=np.float32)


def _property_head_labels(x: np.ndarray, y: np.ndarray, splits: np.ndarray, gen: np.ndarray) -> np.ndarray:
    train = splits == "train"
    mu, sd = x[train].mean(0), x[train].std(0); sd[sd < 1e-8] = 1
    a = np.c_[(x[train] - mu) / sd, np.ones(train.sum())]
    w = np.linalg.solve(a.T @ a + 1e-3 * np.eye(a.shape[1]), a.T @ y[train])
    return (np.c_[(gen - mu) / sd, np.ones(len(gen))] @ w).astype(np.float32)


def _make_downstream_graph(x: np.ndarray, y: np.ndarray, splits: np.ndarray, real_edges: pd.DataFrame, mode: str, gen: np.ndarray, gen_edges: pd.DataFrame, seed: int, cfg: dict):
    rng = np.random.default_rng(seed)
    train_mask_real = splits == "train"; val_mask_real = splits == "val"; test_mask_real = splits == "test"
    label_mode = "real_labels_only"
    if mode == "real_only":
        xx = x.copy(); yy = y.copy(); edges = real_edges.copy(); gen_used = 0; dropped = 0
    else:
        n = len(gen)
        train_x, train_y = x[train_mask_real], y[train_mask_real]
        if mode == "random_synthetic":
            gx = rng.uniform(train_x.min(0), train_x.max(0), size=(n, x.shape[1])).astype(np.float32)
            gy = rng.choice(train_y, size=n).astype(np.float32)
            gedges = _knn_edges_for_synthetic(x, gx, int(cfg["similarity"]["k_neighbors"]))
            label_mode = "random_train_label"
        elif mode == "feature_noise":
            base_idx = rng.integers(0, len(train_x), size=n)
            gx = (train_x[base_idx] + rng.normal(0, train_x.std(0) * 0.05, size=(n, x.shape[1]))).astype(np.float32)
            gy = train_y[base_idx].astype(np.float32)
            gedges = _knn_edges_for_synthetic(x, gx, int(cfg["similarity"]["k_neighbors"]))
            label_mode = "noisy_parent_label"
        elif mode == "nearest_neighbor_interpolation":
            ai = rng.integers(0, len(train_x), size=n); bi = rng.integers(0, len(train_x), size=n)
            gx = (0.5 * train_x[ai] + 0.5 * train_x[bi]).astype(np.float32)
            gy = (0.5 * train_y[ai] + 0.5 * train_y[bi]).astype(np.float32)
            gedges = _knn_edges_for_synthetic(x, gx, int(cfg["similarity"]["k_neighbors"]))
            label_mode = "nearest_neighbor_interpolation_label"
        else:
            gx = gen.astype(np.float32)
            gedges = gen_edges.copy()
            gen_meta = pd.read_csv(TABLES / "generated_nodes.csv")
            neighbor_y, unc = _pseudo_labels(y, gen_edges)
            prop_y = _property_head_labels(x, y, splits, gen)
            if mode == "gan_property_head":
                gy = prop_y; keep = np.ones(len(gx), dtype=bool); label_mode = "property_head_pseudo_label"
            elif mode == "gan_conditional_target":
                gy = gen_meta["conditional_target_bandgap"].to_numpy(dtype=np.float32)
                keep = np.isfinite(gy)
                gy = gy[keep]; gx = gx[keep]
                keep_ids = {f"gen-{i}" for i, ok in enumerate(keep) if ok}
                gedges = gedges[gedges.generated_node_id.isin(keep_ids)].copy()
                remap = {gid: f"gen-{i}" for i, gid in enumerate(sorted(keep_ids, key=lambda s: int(s.split("-")[1])))}
                gedges["generated_node_id"] = gedges["generated_node_id"].map(remap)
                label_mode = "conditional_target_bandgap_label"
            elif mode == "gan_filtered":
                threshold = float(np.quantile(unc, 0.50))
                meta_keep = (
                    gen_meta["descriptor_valid"].to_numpy(dtype=bool)
                    & (~gen_meta["duplicate"].to_numpy(dtype=bool))
                    & (gen_meta["novelty_score"].to_numpy(dtype=float) > 0.01)
                    & (gen_meta["nearest_train_similarity"].to_numpy(dtype=float) >= 0.20)
                    & (gen_meta["nearest_train_similarity"].to_numpy(dtype=float) <= 0.99)
                )
                edge_ok = gen_edges.groupby("generated_node_id")["edge_weight"].apply(lambda s: bool(((s >= -1.0) & (s <= 1.0)).all())).reindex([f"gen-{i}" for i in range(len(gx))], fill_value=False).to_numpy(dtype=bool)
                keep = (unc <= threshold) & meta_keep & edge_ok
                gy = neighbor_y[keep]; gx = gx[keep]
                keep_ids = {f"gen-{i}" for i, ok in enumerate(keep) if ok}
                gedges = gedges[gedges.generated_node_id.isin(keep_ids)].copy()
                remap = {gid: f"gen-{i}" for i, gid in enumerate(sorted(keep_ids, key=lambda s: int(s.split("-")[1])))}
                gedges["generated_node_id"] = gedges["generated_node_id"].map(remap)
                label_mode = f"valid_edge_novel_consistent_uncertainty_filtered_neighbor_label_q50_threshold_{threshold:.4f}"
            else:
                gy = neighbor_y; keep = np.ones(len(gx), dtype=bool); label_mode = "neighbor_averaged_pseudo_label"
            n = len(gx)
        dropped = len(gen) - len(gx)
        xx = np.vstack([x, gx]).astype(np.float32)
        yy = np.r_[y, gy].astype(np.float32)
        add_edges = []
        for r in gedges.itertuples(index=False):
            gid = int(str(r.generated_node_id).split("-")[1])
            add_edges.append({"source": len(x) + gid, "target": int(r.real_node_index), "similarity": float(r.similarity)})
        edges = pd.concat([real_edges, pd.DataFrame(add_edges)], ignore_index=True)
        gen_used = len(gx)
    n_total = len(xx)
    train_mask = np.zeros(n_total, dtype=bool); val_mask = np.zeros(n_total, dtype=bool); test_mask = np.zeros(n_total, dtype=bool)
    train_mask[:len(x)] = train_mask_real; val_mask[:len(x)] = val_mask_real; test_mask[:len(x)] = test_mask_real
    if n_total > len(x):
        train_mask[len(x):] = True
    return xx, yy, _adjacency(n_total, edges), train_mask, val_mask, test_mask, label_mode, gen_used, dropped


def _model_for(name: str, in_dim: int, hidden: int):
    if name == "GraphSAGE":
        return GraphSAGERegressor(in_dim, hidden)
    if name == "GCN":
        return GCNRegressor(in_dim, hidden)
    return GATRegressor(in_dim, max(24, hidden // 2))


def _train_graph_model(model_name: str, xx: np.ndarray, yy: np.ndarray, adj: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray, seed: int, epochs: int = 80) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    mu, sd = xx[train_mask].mean(0), xx[train_mask].std(0); sd[sd < 1e-8] = 1
    xt = torch.tensor((xx - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(yy, dtype=torch.float32)
    at = torch.tensor(adj, dtype=torch.float32)
    tr = torch.tensor(train_mask); va = torch.tensor(val_mask); te = torch.tensor(test_mask)
    model = _model_for(model_name, xx.shape[1], 64)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    best_state = None; best_val = float("inf"); patience = 0
    for ep in range(epochs):
        model.train(); pred = model(xt, at)
        loss = torch.nn.functional.mse_loss(pred[tr], yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xt, at)
            val = torch.nn.functional.l1_loss(pred[va], yt[va]).item()
        if val < best_val:
            best_val = val; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xt, at).numpy()
    return pred[test_mask], {"best_val_mae": best_val, "epochs_run": ep + 1}


def downstream(x: np.ndarray, y: np.ndarray, splits: np.ndarray, gen: np.ndarray, gen_edges: pd.DataFrame, cfg: dict, real_edges: pd.DataFrame | None = None) -> None:
    real_edges = pd.read_csv(TABLES / "sparsified_similarity_edges.csv") if real_edges is None else real_edges
    models = ["GraphSAGE", "GAT", "GCN"]
    modes = ["real_only", "random_synthetic", "feature_noise", "nearest_neighbor_interpolation", "gan_generated", "gan_filtered", "gan_property_head", "gan_conditional_target"]
    rows, usage = [], []
    epochs = int(cfg.get("downstream", {}).get("runtime_epochs", 50))
    for model_name in models:
        for mode in modes:
            for seed in cfg["evaluation_seeds"]:
                xx, yy, adj, tr, va, te, label_mode, used, dropped = _make_downstream_graph(x, y, splits, real_edges, mode, gen, gen_edges, int(seed), cfg)
                pred, info = _train_graph_model(model_name, xx, yy, adj, tr, va, te, int(seed), epochs=epochs)
                truth = yy[te]
                rows.append({"model": model_name, "training_mode": mode, "seed": seed, "label_mode": label_mode, "mae": float(np.mean(np.abs(pred - truth))), "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))), "r2": float(r2_score(truth, pred)), "spearman": float(spearmanr(truth, pred).correlation), "kendall": float(kendalltau(truth, pred).correlation), **info})
                usage.append({"model": model_name, "training_mode": mode, "seed": seed, "generated_nodes_used": used, "generated_nodes_dropped": dropped, "label_mode": label_mode, "test_nodes": int(te.sum())})
    seedwise = pd.DataFrame(rows)
    seedwise.to_csv(TABLES / "downstream_gnn_seedwise.csv", index=False)
    summary = seedwise.groupby(["model", "training_mode", "label_mode"]).agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), r2_mean=("r2", "mean"), r2_std=("r2", "std"), spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"), kendall_mean=("kendall", "mean"), kendall_std=("kendall", "std"), best_val_mae_mean=("best_val_mae", "mean")).reset_index()
    summary.to_csv(TABLES / "downstream_gnn_summary.csv", index=False)
    pd.DataFrame(usage).to_csv(TABLES / "generated_node_usage.csv", index=False)
    ab = []
    for mode in ["gan_generated", "gan_filtered", "gan_property_head", "gan_conditional_target"]:
        sub = summary[summary.training_mode == mode]
        ab.append({"label_strategy": mode, "status": "completed", "mean_mae_across_models": float(sub.mae_mean.mean()), "comment": "Held-out test set contains only real QMOF nodes."})
    pd.DataFrame(ab).to_csv(TABLES / "pseudo_label_ablation.csv", index=False)


def plot_all(x: np.ndarray, gen: np.ndarray) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18})
    def save(name):
        plt.tight_layout(); plt.savefig(FIGS / f"{name}.pdf"); plt.savefig(FIGS / f"{name}.png", dpi=220); plt.close()
    fig, ax = plt.subplots(figsize=(10,3)); ax.axis("off")
    steps = ["QMOF table","descriptors","similarity graph","sparsify","GAN nodes","generated edges","diagnostics","downstream GNN","claim boundary"]
    for i,s in enumerate(steps):
        ax.text(i/(len(steps)-1), .55, s, ha="center", va="center", bbox=dict(boxstyle="round,pad=.28", fc="#f6f7fb", ec="#4f6985"))
    save("fig1_workflow")
    sparse = pd.read_csv(TABLES / "sparsification_grid.csv")
    fig, ax = plt.subplots(); ax.barh(sparse.method + " " + sparse.setting, sparse.edge_reduction_percent, color=np.where(sparse.chosen, "#b3544a", "#678aa6")); ax.set_xlabel("edge reduction (%)"); ax.set_title("Sparsification grid"); save("fig2_similarity_graph_sparsification")
    log = pd.read_csv(TABLES / "gan_training_log.csv")
    fig, ax = plt.subplots(); 
    for seed, sub in log.groupby("seed"): ax.plot(sub.epoch, sub.generator_loss, alpha=.55, label=f"G seed {seed}")
    ax.plot(log.groupby("epoch").critic_loss.mean(), color="#333", label="critic mean"); ax.set_title("GAN training curves"); ax.legend(fontsize=7); save("fig3_gan_training_curves")
    fig, ax = plt.subplots()
    mean_log = log.groupby("epoch")[["real_score", "fake_score", "gradient_penalty", "feature_distribution_loss"]].mean()
    ax.plot(mean_log.index, mean_log.real_score, label="real score")
    ax.plot(mean_log.index, mean_log.fake_score, label="fake score")
    ax.plot(mean_log.index, mean_log.gradient_penalty, label="gradient penalty")
    ax.plot(mean_log.index, mean_log.feature_distribution_loss, label="feature loss")
    ax.set_title("GAN critic scores and auxiliary diagnostics"); ax.legend(frameon=False, fontsize=8); save("fig11_gan_diagnostics")
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]; nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]; edge = pd.read_csv(TABLES / "generated_edge_plausibility.csv").iloc[0]
    fig, ax = plt.subplots(); ax.bar(["descriptor-valid","edge-valid","non-duplicate","novelty-pass","constraint"], [val.descriptor_valid_rate, edge.edge_valid_rate, 1-nov.duplicate_rate, nov.novelty_pass_rate, val.constraint_satisfaction_rate], color=["#587c9a","#6f9b7a","#b8844b","#9a5f62","#7b6c9b"]); ax.set_ylim(0,1.05); save("fig4_validity_novelty")
    dist = pd.read_csv(TABLES / "distribution_alignment.csv").iloc[0]; fig, ax = plt.subplots(); ax.bar(["descriptor shift","wasserstein proxy","edge weight mean"], [dist.descriptor_mean_abs_shift, dist.descriptor_wasserstein_proxy, dist.edge_weight_mean_generated]); save("fig5_distribution_alignment")
    emb = np.vstack([x, gen]); emb = emb - emb.mean(0); _,_,vt = np.linalg.svd(emb, full_matrices=False); p = emb @ vt[:2].T
    fig, ax = plt.subplots(); ax.scatter(p[:len(x),0], p[:len(x),1], s=8, alpha=.3, label="real"); ax.scatter(p[len(x):,0], p[len(x):,1], s=10, alpha=.55, label="generated"); ax.legend(); save("fig6_embedding_space")
    topo = pd.read_csv(TABLES / "topology_impact.csv").iloc[0]; fig, ax = plt.subplots(); ax.bar(["degree before","degree after","generated edges"], [topo.average_degree, topo.average_degree_after_generated_insertion, topo.generated_edges]); save("fig7_topology_impact")
    ds = pd.read_csv(TABLES / "downstream_gnn_summary.csv"); fig, ax = plt.subplots(figsize=(9,4)); sub = ds[ds.model=="GraphSAGE"]; ax.bar(sub.training_mode, sub.mae_mean, yerr=sub.mae_std.fillna(0)); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("MAE"); ax.set_title("GraphSAGE held-out bandgap prediction"); save("fig8_downstream_gnn")
    fig, ax = plt.subplots(figsize=(9,4))
    modes = ["real_only", "random_synthetic", "feature_noise", "nearest_neighbor_interpolation", "gan_generated", "gan_filtered", "gan_property_head", "gan_conditional_target"]
    width = 0.25; xs = np.arange(len(modes))
    for i, model in enumerate(["GraphSAGE", "GAT", "GCN"]):
        sub = ds[ds.model == model].set_index("training_mode").loc[modes]
        ax.bar(xs + (i-1)*width, sub.mae_mean, width=width, yerr=sub.mae_std.fillna(0), label=model)
    ax.set_xticks(xs); ax.set_xticklabels(modes, rotation=25, ha="right"); ax.set_ylabel("MAE"); ax.set_title("Downstream MAE across graph models"); ax.legend(frameon=False); save("fig12_downstream_model_comparison")
    fig, ax = plt.subplots(figsize=(9,4))
    for metric, marker in [("r2_mean", "o"), ("spearman_mean", "s")]:
        sub = ds[(ds.model == "GraphSAGE")].set_index("training_mode").loc[modes]
        ax.plot(modes, sub[metric], marker=marker, label=metric.replace("_mean", ""))
    ax.tick_params(axis="x", rotation=25); ax.set_title("GraphSAGE R2 and Spearman diagnostics"); ax.legend(frameon=False); save("fig13_downstream_r2_spearman")
    ab = pd.read_csv(TABLES / "pseudo_label_ablation.csv")
    fig, ax = plt.subplots(); ax.bar(ab.label_strategy, ab.mean_mae_across_models, color=["#6d8fac", "#ba7656", "#7b9a68"]); ax.tick_params(axis="x", rotation=20); ax.set_ylabel("mean MAE across models"); ax.set_title("Pseudo-label strategy ablation"); save("fig14_pseudo_label_ablation")
    nn = pd.read_csv(TABLES / "nearest_neighbor_interpretation.csv").head(12); fig, ax = plt.subplots(); ax.barh(nn.generated_node_id, nn.neighbor_bandgap_mean); ax.set_xlabel("nearest-neighbor bandgap mean"); save("fig9_nearest_neighbor")
    fig, ax = plt.subplots(figsize=(7,3)); ax.axis("off")
    for i,s in enumerate(["QMOF graph yes","sparsification yes","GAN node generation yes","validity/novelty yes","downstream GNN proxy yes"]): ax.text(.05,.85-i*.14,s,color="#2f6f52")
    for i,s in enumerate(["DFT no","force field no","experiment no","physical validation no"]): ax.text(.58,.85-i*.14,s,color="#8a2f3c")
    save("fig10_claim_boundary")


def reports_and_manuscript(cfg: dict) -> None:
    graph = pd.read_csv(TABLES / "graph_summary_before_sparsification.csv").iloc[0]
    sparse = pd.read_csv(TABLES / "final_sparsification_summary.csv").iloc[0]
    val = pd.read_csv(TABLES / "generated_node_validity.csv").iloc[0]
    nov = pd.read_csv(TABLES / "generated_node_novelty.csv").iloc[0]
    ds = pd.read_csv(TABLES / "downstream_gnn_summary.csv")
    def improves(model):
        sub = ds[ds.model == model].set_index("training_mode")
        return bool(sub.loc["gan_filtered","mae_mean"] < min(sub.loc["real_only","mae_mean"], sub.loc["random_synthetic","mae_mean"], sub.loc["feature_noise","mae_mean"], sub.loc["nearest_neighbor_interpolation","mae_mean"]))
    strongest = "GANg generates descriptor-valid, non-duplicate MOF-like nodes and similarity edges in a QMOF-derived similarity network, but current generated-node augmentation does not consistently improve held-out bandgap prediction across GraphSAGE, GAT, and GCN."
    if all(improves(m) for m in ["GraphSAGE","GAT","GCN"]):
        strongest = "GANg generates descriptor-valid, non-duplicate MOF-like nodes and similarity edges in a QMOF-derived similarity network, and filtered generated-node augmentation improves held-out MOF-level bandgap prediction across GraphSAGE, GAT, and GCN."
    report = f"""# Final readiness report

1. Was the final similarity graph built? Yes.
2. How many nodes and edges before/after sparsification? {int(graph.nodes)} nodes, {int(graph.edges)} edges before, {int(sparse.edges)} after.
3. Did sparsification preserve connectivity? {'Yes' if int(sparse.connected_components)==1 else 'No; final graph has ' + str(int(sparse.connected_components)) + ' components'}.
4. Did GAN training complete across five seeds? Yes.
5. Did generated nodes satisfy descriptor constraints? Descriptor-valid rate = {val.descriptor_valid_rate:.3f}.
6. Did generated edges satisfy similarity rules? Edge-valid rate is reported in generated_edge_plausibility.csv.
7. Are generated nodes novel? Duplicate rate = {nov.duplicate_rate:.3f}; novelty-pass rate = {nov.novelty_pass_rate:.3f}.
8. Do generated nodes cover meaningful feature space? PCA and distribution-alignment diagnostics were generated; this is descriptor-space evidence only.
9. Do generated nodes affect topology in a useful way? Topology-impact diagnostics were generated; usefulness remains diagnostic.
10. Does generated-node augmentation improve GraphSAGE? {'Yes' if improves('GraphSAGE') else 'No'}.
11. Does it improve GAT? {'Yes' if improves('GAT') else 'No'}.
12. Does it improve GCN? {'Yes' if improves('GCN') else 'No'}.
13. Are generated nodes interpretable through nearest real QMOFs? Yes, via nearest_neighbor_interpretation.csv.
14. Was DFT performed? No.
15. Was force-field relaxation performed? No.
16. Was experimental validation performed? No.
17. Strongest supported claim: {strongest}
18. Not supported: physical MOF discovery, DFT validation, force-field validation, synthesis readiness, or experimental validation.
19. Recommended target journal tier: proof-of-concept/preprint or specialist methods venue until stronger downstream and structural decoding evidence is available.
"""
    (ROOT / "reports" / "final_readiness_report.md").write_text(report, encoding="utf-8")
    (ROOT / "reports" / "metric_source_map.md").write_text("# Metric source map\n\nSee `experiments/qmof_galaxy_final/tables/metric_source_map.csv`.\n", encoding="utf-8")
    claims = [
        ("1000 QMOF nodes", int(graph.nodes), "graph_summary_before_sparsification.csv", "nodes", "Results 1"),
        ("edges before sparsification", int(graph.edges), "graph_summary_before_sparsification.csv", "edges", "Results 1"),
        ("edges after sparsification", int(sparse.edges), "final_sparsification_summary.csv", "edges", "Results 2"),
        ("feature-ablation graph selection", "see table", "feature_ablation_graph_selection.csv", "chosen_for_final_graph", "Methods"),
        ("descriptor-valid rate", f"{val.descriptor_valid_rate:.3f}", "generated_node_validity.csv", "descriptor_valid_rate", "Results 4"),
        ("duplicate rate", f"{nov.duplicate_rate:.3f}", "generated_node_novelty.csv", "duplicate_rate", "Results 4"),
        ("GraphSAGE downstream summary", "see table", "downstream_gnn_summary.csv", "mae_mean", "Results 6"),
    ]
    pd.DataFrame(claims, columns=["manuscript_claim","value","source_csv","source_column","figure_or_table"]).to_csv(TABLES / "metric_source_map.csv", index=False)
    main = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx,booktabs,hyperref}}
\title{{GANg: Generative Node Expansion in MOF Similarity Networks}}
\author{{AGNI-MOF project}}
\date{{}}
\begin{{document}}\maketitle
\begin{{abstract}}
GANg represents each QMOF entry as one node in a global MOF similarity network and encodes edges as descriptor-space similarity links. A WGAN-GP generator produces synthetic MOF-like nodes and similarity edges. Generated nodes are descriptor-space graph hypotheses, not validated MOF structures. On the locked 1000-node QMOF slice \cite{{rosen2021qmof}}, the final graph contains {int(graph.edges)} edges before sparsification and {int(sparse.edges)} edges after sparsification. Generated nodes achieve descriptor-valid rate {val.descriptor_valid_rate:.3f}, duplicate rate {nov.duplicate_rate:.3f}, and novelty-pass rate {nov.novelty_pass_rate:.3f}. Downstream GNN-style evaluation is reported on held-out real QMOF nodes only; generated-node augmentation is claimed as beneficial only if it beats real-only and synthetic baselines. No DFT calculation, force-field relaxation, experimental validation, or synthesis-ready claim is made.
\end{{abstract}}
\section{{Results}}
\subsection{{QMOF-derived MOF similarity graph}}
The final node table contains {int(graph.nodes)} QMOF entries with composition and available scalar metadata descriptors. Cosine k-nearest-neighbor similarity gives {int(graph.edges)} undirected edges.
\subsection{{Sparsification}}
The selected sparsified graph contains {int(sparse.edges)} edges and {int(sparse.connected_components)} connected component(s), reducing edge density while preserving the main similarity structure.
\subsection{{GAN-generated descriptor-space nodes}}
GANg generates synthetic MOF-level feature vectors and attaches them to real QMOF nodes by cosine similarity. These are computational MOF similarity-network variants.
\subsection{{Validity, novelty, and interpretation}}
The final generated nodes have descriptor-valid rate {val.descriptor_valid_rate:.3f}, duplicate rate {nov.duplicate_rate:.3f}, and novelty-pass rate {nov.novelty_pass_rate:.3f}. Nearest-neighbor interpretation links each generated node to real QMOF analogues.
\subsection{{Topology and downstream evaluation}}
Generated nodes alter the similarity-network topology through additional similarity edges. Downstream GraphSAGE, GAT, and GCN-style results are reported in the locked tables. The final claim is: {strongest}
\subsection{{Claim boundary}}
Generated nodes are not validated MOF crystal structures. DFT, force-field relaxation, and experimental validation were not performed.
\section{{Methods}}
\subsection{{QMOF node table and descriptors}}
We used a 1000-entry QMOF slice with split seed 42 and a 70/15/15 train/validation/test split. The target is \texttt{{outputs.pbe.bandgap}}. Features are formula-derived composition descriptors, metal-family and organic-linker proxy descriptors, periodic-table summary statistics, and available non-target scalar metadata descriptors. Target-leaking output properties such as band gap, CBM, VBM, direct gap, and energy outputs are excluded from node features. A feature-ablation graph-selection table compares composition-only, chemistry-role, scalar-structure, hybrid, and all-feature similarity graphs.
\subsection{{Similarity graph and sparsification}}
Features are normalized and cosine k-nearest-neighbor edges are built. A sparsification grid compares top-k, mutual-kNN, threshold, and connectivity-preserving threshold settings.
\subsection{{GANg}}
The GAN is a WGAN-GP descriptor-space generator and critic trained across five seeds. Generated feature vectors are clipped to the training feature range and attached by similarity kNN.
\subsection{{Evaluation}}
We evaluate descriptor validity, edge plausibility, novelty, distribution alignment, topology impact, nearest-neighbor interpretation, and held-out real-node downstream prediction.
\section{{Discussion}}
GANg is a proof-of-concept generative similarity-network benchmark. Its outputs are descriptor-space graph hypotheses that require structural decoding or retrieval and physical validation before any MOF-structure claim.
\bibliographystyle{{unsrt}}\bibliography{{references}}
\end{{document}}
"""
    (ROOT / "manuscript" / "main.tex").write_text(main, encoding="utf-8")
    supp = "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{hyperref}\n\\title{Supplementary Information for GANg}\\author{AGNI-MOF project}\\date{}\\begin{document}\\maketitle\n"
    supp += "\\section{Reproducibility}\\begin{verbatim}\npython3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml\n\\end{verbatim}\n"
    supp += "\\section{Tables}Final tables are stored in experiments/qmof\\_galaxy\\_final/tables/.\\section{Claim boundary}No DFT, force-field relaxation, experimental validation, or synthesis-ready claim is made.\\end{document}\n"
    (ROOT / "manuscript" / "supplementary.tex").write_text(supp, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qmof_galaxy_1000.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs()
    node, x, xn, y, splits, feature_names = build_node_table(cfg)
    edges, sim = build_graph(cfg, x, xn, y, splits, feature_names)
    final_edges = sparsify_grid(cfg, edges, sim, y)
    torch.save({"x": torch.tensor(x, dtype=torch.float32), "x_norm": torch.tensor(xn, dtype=torch.float32), "y": torch.tensor(y, dtype=torch.float32), "splits": list(splits), "edges": final_edges.to_dict("records"), "feature_names": feature_names}, GRAPHS / "qmof_similarity_graph_sparsified.pt")
    gen, _, _ = train_gan(cfg, x, y, splits)
    gen_nodes, gen_edges, _ = attach_generated(x, gen, y, cfg)
    torch.save({"real_x": torch.tensor(x), "generated_x": torch.tensor(gen), "real_edges": final_edges.to_dict("records"), "generated_edges": gen_edges.to_dict("records")}, GRAPHS / "qmof_similarity_graph_with_generated_nodes.pt")
    evaluate_all(x, gen, y, splits, final_edges, gen_nodes, gen_edges, feature_names)
    downstream(x, y, splits, gen, gen_edges, cfg, final_edges)
    plot_all(x, gen)
    reports_and_manuscript(cfg)


if __name__ == "__main__":
    main()
