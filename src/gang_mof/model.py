"""The GANg model: train, sample, attach, evaluate, save and load."""
from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ._version import __version__
from ._core import ConstrainedGenerator, CoreSpace, Critic, FullGenerator, Standardizer, gradient_penalty, prdc
from .data import NodeDataset
from .evaluation import candidate_checks, distribution_metrics, nearest_analogs, surrogate_predictions
from .graph import knn_edges
from .result import GenerationResult
from .schema import Schema, get_schema


@dataclass
class GANgConfig:
    """Training settings. Defaults follow the GANg paper except ``iterations`` (paper: 6000)."""

    iterations: int = 3000          #: generator updates
    n_critic: int = 5               #: critic updates per generator update
    gp: float = 10.0                #: gradient-penalty weight
    lr: float = 1e-4
    betas: tuple = (0.5, 0.9)
    batch: int = 128
    latent: int = 64
    hidden: int = 256
    ema: float = 0.999              #: exponential moving average of generator weights
    eval_every: int = 250           #: checkpoint-selection interval (generator updates)
    eval_start: int = 1000          #: first checkpoint considered
    range_penalty: float = 0.05     #: full-vector baseline only
    minibatch_std: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def paper(cls) -> "GANgConfig":
        """The configuration used for the constraint-aware results in the paper (6000 updates)."""
        return cls(iterations=6000)

    @classmethod
    def fast(cls) -> "GANgConfig":
        """A quick configuration for tests and demos (results are not paper quality)."""
        return cls(iterations=400, eval_start=200, eval_every=100)


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class GANg:
    """Target-conditioned generative adversarial network for graph-node synthesis.

    Parameters
    ----------
    schema : {"auto", "qmof106", "generic"} or Schema
        ``"auto"`` uses the constraint-aware QMOF-106 schema when the data has all 106 MOF
        descriptors; any other table uses the generic full-vector baseline.
    generator : {"auto", "constrained", "full"}
        ``"constrained"`` (GANg; needs the qmof106 schema) emits only the independent descriptor
        core and rebuilds all derived descriptors. ``"full"`` is the conditional WGAN-GP baseline that
        emits every descriptor and clips to training bounds.
    conditional : bool
        If False, the condition is fixed to zero (parameter-matched unconditional control).
    config : GANgConfig, optional
    random_state : int
    device : {"cpu", "cuda", "auto"}
    verbose : bool

    Examples
    --------
    >>> from gang_mof import GANg, load_qmof_example
    >>> data = load_qmof_example()
    >>> train, val, test = data.split(random_state=42)
    >>> model = GANg(random_state=0).fit(train, validation_data=val)
    >>> result = model.attach(model.sample(100, condition=2.0), k=15)
    """

    def __init__(self, schema: str | Schema = "auto", generator: str = "auto", conditional: bool = True,
                 config: GANgConfig | None = None, random_state: int = 0, device: str = "cpu", verbose: bool = True,
                 **config_overrides):
        self.schema_arg = schema
        self.generator_arg = generator
        self.conditional = conditional
        self.config = config or GANgConfig()
        for key, value in config_overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"Unknown GANgConfig option {key!r}")
            setattr(self.config, key, value)
        self.random_state = random_state
        self.device = _device(device)
        self.verbose = verbose
        self.is_fitted = False

    # ------------------------------------------------------------------ helpers
    def _log(self, msg):
        if self.verbose:
            print(f"[GANg] {msg}", flush=True)

    @property
    def mode(self) -> str:
        return "constrained" if self.constrained else "full-vector baseline"

    def _resolve(self, data: NodeDataset):
        if isinstance(self.schema_arg, Schema):
            self.schema = self.schema_arg
        elif self.schema_arg == "auto":
            self.schema = data.schema
        else:
            self.schema = get_schema(self.schema_arg, columns=data.X.columns, features=list(data.X.columns))
        gen = self.generator_arg
        if gen == "auto":
            gen = "constrained" if self.schema.constrained else "full"
        if gen == "constrained" and not self.schema.constrained:
            raise ValueError("The constrained generator needs the qmof106 schema (all 106 MOF descriptors).")
        if gen not in {"constrained", "full"}:
            raise ValueError("generator must be 'auto', 'constrained' or 'full'.")
        self.constrained = gen == "constrained"
        if not self.constrained:
            warnings.warn("Using the full-vector conditional WGAN-GP baseline: descriptor relations are not enforced "
                          "and outputs are clipped to training bounds.", stacklevel=3)

    def _cond(self, y) -> torch.Tensor:
        y = np.asarray(y, float)
        c = (y - self.y_mean) / self.y_sd * float(self.conditional and self.has_target)
        return torch.tensor(c, dtype=torch.float32, device=self.device)

    def _critic_view(self, g, z, c):
        out = g(z, c)
        return self.space.critic_input(*out) if self.constrained else out

    def _build_networks(self, n_features: int):
        cfg = self.config
        if self.constrained:
            self.g = ConstrainedGenerator(cfg.latent, self.space.n_elements, self.space.n_cont, cfg.hidden,
                                          self.space.element_mask)
            dim = self.space.dim
        else:
            self.g = FullGenerator(cfg.latent, n_features, cfg.hidden)
            dim = n_features
        self.d = Critic(dim, cfg.hidden, cfg.minibatch_std)
        self.g.to(self.device), self.d.to(self.device)

    # ------------------------------------------------------------------ training
    def fit(self, train: NodeDataset, validation_data: NodeDataset | None = None) -> "GANg":
        """Train on real nodes. Only training rows fit scalers, networks and graph references.

        ``validation_data`` is used for checkpoint selection only (coverage of validation
        descriptors, with validation targets as conditions). If omitted, 15% of ``train`` is held out.
        """
        cfg = self.config
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if validation_data is None:
            n_val = max(10, int(round(0.15 * len(train))))
            order = np.random.default_rng(self.random_state).permutation(len(train))
            validation_data, train = train.subset(order[:n_val]), train.subset(order[n_val:])
            self._log(f"no validation_data given: holding out {n_val} training nodes for checkpoint selection")
        if list(validation_data.X.columns) != list(train.X.columns):
            raise ValueError("train and validation_data must have the same descriptor columns.")
        self._resolve(train)
        self.feature_names = list(train.X.columns)
        self.has_target = train.y is not None
        self.target_name = train.target_name
        y_tr = train.y if self.has_target else np.zeros(len(train))
        y_va = validation_data.y if (self.has_target and validation_data.y is not None) else np.zeros(len(validation_data))
        self.y_mean, self.y_sd = float(np.mean(y_tr)), max(float(np.std(y_tr)), 1e-8)
        if not self.has_target and self.conditional:
            self._log("no target given: training an unconditional generator")

        raw_tr = train.X.to_numpy(float)
        raw_va = validation_data.X.to_numpy(float)
        self.scaler = Standardizer.fit(raw_tr)
        if self.constrained:
            self.space = CoreSpace.fit(raw_tr, self.feature_names, self.scaler)
            real = self.space.encode(raw_tr).to(self.device)
            val = self.space.encode(raw_va).numpy()
        else:
            real = torch.tensor(self.scaler.transform(raw_tr), device=self.device)
            val = self.scaler.transform(raw_va)
        self._build_networks(len(self.feature_names))
        cond = self._cond(y_tr)
        val_cond = self._cond(y_va)

        og = torch.optim.Adam(self.g.parameters(), lr=cfg.lr, betas=cfg.betas)
        od = torch.optim.Adam(self.d.parameters(), lr=cfg.lr, betas=cfg.betas)
        ema = copy.deepcopy(self.g).eval()
        eval_start = cfg.eval_start if cfg.eval_start < cfg.iterations else max(cfg.iterations // 2, 1)
        n, best = len(real), (-np.inf, None, 0)
        self.history, self.selection = [], []
        t0 = time.time()
        self._log(f"training {self.mode} on {n} nodes x {len(self.feature_names)} descriptors "
                  f"({cfg.iterations} generator updates, device={self.device})")
        for it in range(1, cfg.iterations + 1):
            for _ in range(cfg.n_critic):
                ix = torch.randint(0, n, (cfg.batch,), device=self.device)
                r, c = real[ix], cond[ix]
                with torch.no_grad():
                    fake = self._critic_view(self.g, torch.randn(cfg.batch, cfg.latent, device=self.device), c)
                ld = self.d(fake, c).mean() - self.d(r, c).mean() + gradient_penalty(self.d, r, fake, c, cfg.gp)
                od.zero_grad()
                ld.backward()
                od.step()
            c = cond[torch.randint(0, n, (cfg.batch,), device=self.device)]
            fake = self._critic_view(self.g, torch.randn(cfg.batch, cfg.latent, device=self.device), c)
            lg = -self.d(fake, c).mean()
            if not self.constrained:
                lg = lg + cfg.range_penalty * torch.relu(fake.abs() - 3).mean()
            og.zero_grad()
            lg.backward()
            og.step()
            with torch.no_grad():
                for pe, pg in zip(ema.parameters(), self.g.parameters()):
                    pe.mul_(cfg.ema).add_(pg, alpha=1 - cfg.ema)
            if it % 50 == 0:
                self.history.append({"iteration": it, "critic_loss": ld.item(), "generator_loss": lg.item()})
            if it >= eval_start and it % cfg.eval_every == 0 or it == cfg.iterations:
                with torch.no_grad():
                    scores = [prdc(val, self._critic_view(
                        ema, torch.randn(len(val_cond), cfg.latent, device=self.device), val_cond).cpu().numpy())
                        for _ in range(3)]
                m = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
                score = m["coverage"] + 0.1 * min(m["density"], 1.0)
                self.selection.append({"iteration": it, **m, "selection_score": score})
                if score > best[0]:
                    best = (score, copy.deepcopy(ema.state_dict()), it)
                self._log(f"update {it:>5}/{cfg.iterations}  validation coverage {m['coverage']:.3f}  "
                          f"({time.time() - t0:.0f}s)")
        self.g.load_state_dict(best[1])
        self.g.eval()
        self.selected_iteration = best[2]
        self._log(f"selected checkpoint at update {self.selected_iteration} (validation coverage "
                  f"{[s for s in self.selection if s['iteration'] == best[2]][0]['coverage']:.3f})")
        # Reference data for attachment, novelty and analogs (training rows only).
        self.reference_ids = train.ids.copy()
        self.reference_X = train.X.copy()
        self.reference_y = None if train.y is None else train.y.copy()
        self.reference_extra = None if train.extra is None else train.extra.copy()
        self.validation_X = validation_data.X.copy()
        self.is_fitted = True
        return self

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() (or GANg.load()) first.")

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, n_samples: int = 100, condition=None, random_state: int | None = None,
               id_prefix: str = "gen") -> GenerationResult:
        """Generate descriptor-space nodes.

        condition : None, float or array of length ``n_samples``
            Requested target value(s), in target units. ``None`` draws requests from the training
            target distribution. The request is stored as ``requested_condition``; it is not a label.
        """
        self._check_fitted()
        seed = self.random_state if random_state is None else random_state
        rng = np.random.default_rng(seed)
        if condition is None:
            req = (rng.choice(self.reference_y, n_samples) if self.reference_y is not None else np.zeros(n_samples))
        else:
            req = np.broadcast_to(np.asarray(condition, float), (n_samples,)).copy()
        z = torch.tensor(rng.normal(size=(n_samples, self.config.latent)), dtype=torch.float32, device=self.device)
        c = self._cond(req)
        extra_cols, repair = {}, {}
        if self.constrained:
            frac, cont = self.g(z, c)
            raw, formulas, repair = self.space.to_descriptors(frac.cpu().numpy(), cont.cpu().numpy())
            extra_cols["formula"] = formulas
        else:
            unclipped = self.scaler.inverse(self.g(z, c).cpu().numpy())
            lo, hi = self.reference_X.min().to_numpy(), self.reference_X.max().to_numpy()
            raw = np.clip(unclipped, lo, hi)
            extra_cols["fraction_features_clipped"] = (raw != unclipped).mean(1)
            repair["mean_fraction_features_clipped"] = float(extra_cols["fraction_features_clipped"].mean())
        ids = [f"{id_prefix}-{i:06d}" for i in range(1, n_samples + 1)]
        nodes = pd.DataFrame(raw, columns=self.feature_names)
        front = pd.DataFrame({"node_id": ids, "requested_condition": req if self.conditional and self.has_target else np.nan,
                              **extra_cols})
        nodes = pd.concat([front, nodes], axis=1)
        X = nodes.set_index("node_id")[self.feature_names]
        checks = candidate_checks(X, self.reference_X, self.scaler, constrained_schema=self.schema.constrained)
        checks.insert(0, "node_id", ids)
        checks = checks.reset_index(drop=True)
        meta = {"gang_mof_version": __version__, "schema": self.schema.name, "mode": self.mode,
                "conditional": bool(self.conditional and self.has_target), "target": self.target_name,
                "n_samples": n_samples, "random_state": seed, "selected_iteration": self.selected_iteration,
                "config": asdict(self.config), "repair_statistics": repair}
        return GenerationResult(nodes, checks, list(self.feature_names), metadata=meta)

    # ------------------------------------------------------------------ graph attachment
    def attach(self, result: GenerationResult, k: int = 15, n_analogs: int = 5) -> GenerationResult:
        """Attach generated nodes to the real training graph by cosine kNN and retrieve analogs."""
        self._check_fitted()
        X = result.descriptors
        result.edges = knn_edges(self.reference_X.to_numpy(float), X.to_numpy(float), self.reference_ids,
                                 X.index.to_numpy(), k=k, scaler=self.scaler)
        result.analogs = nearest_analogs(X, X.index.to_numpy(), self.reference_X, self.reference_ids, self.scaler,
                                         k=n_analogs, train_y=self.reference_y, train_extra=self.reference_extra)
        result.metadata.update({"k": k, "attachment": "cosine kNN to real training nodes (assigned, not generated)"})
        return result

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, result: GenerationResult, reference: NodeDataset | None = None,
                 surrogate: bool = True) -> dict:
        """Distribution and fidelity diagnostics; results are stored in ``result.metrics``.

        reference : held-out real nodes (e.g. the test split). Defaults to the validation nodes.
        surrogate : if True and a target exists, predict the property of generated nodes with an
            independent random forest trained on real training rows (column ``surrogate_prediction``).
        """
        self._check_fitted()
        ref = self.validation_X if reference is None else reference.X[self.feature_names]
        X = result.descriptors
        m = {"reference": "validation" if reference is None else "user-supplied",
             "n_generated": len(X), "acceptance_rate": float(result.checks["accepted"].mean()),
             "within_training_range_rate": float(result.checks["within_training_range"].mean()),
             "novel_rate": float(result.checks["novel"].mean())}
        if "schema_consistent" in result.checks:
            m["schema_consistent_rate"] = float(result.checks["schema_consistent"].mean())
        m.update(distribution_metrics(X, ref, self.scaler, self.reference_X))
        if surrogate and self.reference_y is not None:
            pred = surrogate_predictions(self.reference_X, self.reference_y, X, self.random_state)
            result.nodes["surrogate_prediction"] = pred
            req = result.nodes["requested_condition"].to_numpy(float)
            if np.isfinite(req).all() and np.std(req) > 0:
                m["surrogate_condition_correlation"] = float(np.corrcoef(req, pred)[0, 1])
            if np.isfinite(req).all():
                m["surrogate_condition_mae"] = float(np.mean(np.abs(req - pred)))
        result.metrics.update(m)
        return m

    # ------------------------------------------------------------------ persistence
    def save(self, directory) -> Path:
        """Save networks, scaler, schema, reference data and settings to a directory."""
        self._check_fitted()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"generator": self.g.state_dict(), "critic": self.d.state_dict()}, d / "weights.pt")
        meta = {"gang_mof_version": __version__, "schema": {"name": self.schema.name, "features": self.schema.features,
                                                              "constrained": self.schema.constrained},
                "constrained": self.constrained, "conditional": self.conditional, "has_target": self.has_target,
                "target_name": self.target_name, "y_mean": self.y_mean, "y_sd": self.y_sd,
                "random_state": self.random_state, "selected_iteration": self.selected_iteration,
                "config": asdict(self.config), "feature_names": self.feature_names,
                "space_state": self.space.state() if self.constrained else None,
                "selection": self.selection}
        (d / "model.json").write_text(json.dumps(meta, indent=2))
        np.savez_compressed(d / "reference.npz", scaler_mean=self.scaler.mean, scaler_scale=self.scaler.scale,
                            reference_X=self.reference_X.to_numpy(float), reference_ids=self.reference_ids.astype(str),
                            reference_y=np.array([]) if self.reference_y is None else self.reference_y,
                            validation_X=self.validation_X.to_numpy(float))
        if self.reference_extra is not None and "formula" in self.reference_extra.columns:
            self.reference_extra[["formula"]].to_csv(d / "reference_formula.csv")
        return d

    @classmethod
    def load(cls, directory, device: str = "cpu", verbose: bool = True) -> "GANg":
        """Load a model saved with :meth:`save`."""
        d = Path(directory)
        meta = json.loads((d / "model.json").read_text())
        cfg = meta["config"]
        cfg["betas"] = tuple(cfg["betas"])
        model = cls(schema=Schema(meta["schema"]["name"], meta["schema"]["features"], meta["schema"]["constrained"]),
                    generator="constrained" if meta["constrained"] else "full", conditional=meta["conditional"],
                    config=GANgConfig(**cfg), random_state=meta["random_state"], device=device, verbose=verbose)
        model.schema, model.constrained = model.schema_arg, meta["constrained"]
        model.feature_names, model.has_target = meta["feature_names"], meta["has_target"]
        model.target_name, model.y_mean, model.y_sd = meta["target_name"], meta["y_mean"], meta["y_sd"]
        model.selected_iteration, model.selection = meta["selected_iteration"], meta["selection"]
        arr = np.load(d / "reference.npz", allow_pickle=False)
        model.scaler = Standardizer(arr["scaler_mean"], arr["scaler_scale"])
        ids = arr["reference_ids"].astype(str)
        model.reference_ids = ids
        model.reference_X = pd.DataFrame(arr["reference_X"], columns=model.feature_names, index=ids)
        model.reference_y = arr["reference_y"] if arr["reference_y"].size else None
        model.validation_X = pd.DataFrame(arr["validation_X"], columns=model.feature_names)
        fpath = d / "reference_formula.csv"
        model.reference_extra = pd.read_csv(fpath, index_col=0) if fpath.exists() else None
        if model.constrained:
            model.space = CoreSpace(model.feature_names, model.scaler, meta["space_state"])
        model._build_networks(len(model.feature_names))
        state = torch.load(d / "weights.pt", map_location=model.device, weights_only=True)
        model.g.load_state_dict(state["generator"])
        model.d.load_state_dict(state["critic"])
        model.g.eval()
        model.is_fitted = True
        return model

    def __repr__(self) -> str:
        status = f"fitted, {self.mode}" if self.is_fitted else "not fitted"
        return f"GANg({status}, conditional={self.conditional}, iterations={self.config.iterations})"
