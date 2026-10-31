# GANg

**Target-conditioned generation of auditable graph nodes for metal–organic framework (MOF) similarity networks**

[![tests](https://github.com/MehrdadJalali-AI/GANg/actions/workflows/tests.yml/badge.svg)](https://github.com/MehrdadJalali-AI/GANg/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

<p align="center">
  <img src="GAN2.jpg" alt="GANg: descriptor-space graph-node synthesis in MOF similarity networks" width="80%">
</p>

`gang-mof` is the Python package for **GANg** (Jalali, Orth & Reischl, *Advanced Intelligent Systems*, 2026).
Give it a table of real MOFs; it trains a constraint-aware conditional WGAN-GP and returns synthetic
MOF descriptor vectors, their similarity edges to real MOFs, and a per-node audit.

- **Constraint-aware generation.** GANg generates only the independent descriptor core (a sparse
  elemental composition and six cell scalars) and rebuilds all 106 descriptors with the same
  featurizer used for real MOFs, so every generated node is consistent with the descriptor schema
  by construction.
- **Graph insertion.** Each generated node is attached to its *k* most similar real MOFs by cosine
  similarity. Edges are assigned, not generated.
- **Built-in audit.** Range, consistency, duplication and novelty checks per node; coverage,
  diversity and mode-collapse diagnostics against held-out MOFs; nearest real analogs for interpretation.

> **Claim boundary.** Outputs are *descriptor-space hypotheses*: not CIFs, crystal structures,
> DFT-validated or synthesizable materials. Nearest real QMOF analogs are retrieved for
> interpretation only. A requested target value is never treated as a property of the generated node.

## Installation

```bash
pip install "gang-mof @ git+https://github.com/MehrdadJalali-AI/GANg"
```

Requires Python ≥ 3.9 and PyTorch (CPU is enough; a GPU is optional). For a CPU-only PyTorch, install it first with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. Add `[graph]` for NetworkX export.

## Quick start

```python
from gang_mof import GANg, load_qmof_example

data = load_qmof_example()                          # 1,000 real QMOF MOFs, 106 descriptors, PBE band gap
train, validation, test = data.split(random_state=42)

model = GANg(random_state=0).fit(train, validation_data=validation)

result = model.sample(n_samples=100, condition=2.0)  # request descriptor-space MOFs near 2.0 eV
model.attach(result, k=15)                           # cosine kNN edges to real training MOFs
model.evaluate(result, reference=test)               # audit against held-out MOFs

print(result.summary())
result.export("gang_results")                        # nodes, edges, checks, analogs, metrics, metadata
model.save("gang_model")                             # GANg.load("gang_model") later
```

Training takes about three minutes on a CPU with the default 3,000 generator updates
(`GANgConfig.paper()` uses the paper's 6,000; `GANgConfig.fast()` is for quick tests only).
A typical summary (256 nodes, requested band gaps drawn from the training distribution, audited against the test split):

```text
GANg generation: 256 nodes (schema: qmof106, mode: constrained)
  accepted:               0.816
  within training range:  0.816
  schema consistent:      1.000
  novel (vs training):    1.000
  duplicate of training:  0.000
  edges attached:         3840 (k = 15)
  coverage:               0.773
  precision:              0.793
  recall:                 0.660
  pairwise_distance_ratio:1.063
  top1_analog_share:      0.055
```

Nodes that are not accepted are mostly new compositions whose *derived* descriptors (for example an
element-count ratio) fall outside the training range; use `result.accepted()` to keep only nodes that pass every screen.

## Using your own MOFs

Per MOF you need an ID, the reduced formula, the cell atom count, pore-limiting and largest cavity
diameters, cell volume and space-group number, plus the property you want to condition on.

```python
import pandas as pd
from gang_mof import GANg, NodeDataset, featurize_dataframe

raw = pd.read_csv("my_mofs.csv")
table = featurize_dataframe(
    raw,
    columns={"formula": "formula", "natoms": "n_atoms", "pld": "pld", "lcd": "lcd",
             "volume": "volume", "spacegroup": "spacegroup"},   # map to your column names
    keep=["mof_id", "bandgap"],
)
data = NodeDataset.from_dataframe(table, node_id="mof_id", target="bandgap")
train, validation, test = data.split(random_state=0)
model = GANg().fit(train, validation_data=validation)
```

If your table already has the 106 GANg descriptor columns, skip `featurize_dataframe` and load it
with `NodeDataset.from_csv("table.csv", node_id=..., target=...)`.

## Inputs and outputs

| Input | Required | Notes |
|---|---|---|
| Descriptors | yes | The 106 MOF descriptors (`gang_mof.QMOF106_FEATURES`), or any numeric table with `schema="generic"` |
| Node IDs | recommended | Must be unique |
| Target | for conditional generation | Kept separate from descriptors; never used for similarity |
| Validation nodes | yes (or 15% of train is held out) | Used only for checkpoint selection |

`model.sample()` returns a `GenerationResult`:

| Attribute | Contents |
|---|---|
| `nodes` | `node_id`, `requested_condition`, `formula`, 106 descriptors (original units); after `evaluate`, `surrogate_prediction` |
| `checks` | `within_training_range`, `schema_consistent`, `novelty_score`, `duplicate_of_training`, `duplicate_in_batch`, `accepted`, `rejection_reasons` |
| `edges` | `source` (generated) → `target` (real), cosine `weight`, `rank` |
| `analogs` | nearest real training MOFs with similarity, target value and formula |
| `metrics` | coverage, precision, recall, density, pairwise-distance ratio, effective rank, top-1 analog share, surrogate correlation |
| `metadata` | package version, schema, generator mode, seed, settings, checkpoint |

Three quantities are kept apart on purpose: the **requested condition** (what you asked for), the
**surrogate prediction** (what an independent model trained on real data predicts) and any
**observed property** (only obtainable by external calculation or experiment).

## Generator modes

| Mode | How to select | Use |
|---|---|---|
| Constraint-aware GANg | default for the 106 MOF descriptors | Recommended |
| Full-vector WGAN-GP baseline | `GANg(generator="full")`, or any non-MOF table | Reference; outputs are clipped to training bounds and relations are not enforced |
| Unconditional control | `GANg(conditional=False)`, or no target | Tests whether conditioning matters |

Same data, split and training schedule (3,000 updates), scored against the held-out test MOFs:

| Diagnostic | Constraint-aware GANg | Full-vector baseline |
|---|---|---|
| Schema-consistent nodes | 100% | 0% |
| Coverage of held-out MOFs | 0.77 | 0.83 |
| Recall | 0.66 | 0.22 |
| Spread relative to real MOFs (pairwise-distance ratio) | 1.06 | 0.54 |

The baseline covers held-out MOFs about as well but samples a much narrower region (low recall and
spread) and never satisfies the descriptor relations. In the paper, the baseline with its original,
shorter training schedule reached a coverage of only 0.14.

## Augmentation

In the paper, augmentation lowered error only when real labels were scarce, matched simple
resampling, and worked mainly through **pseudo-label distillation**. The package makes this explicit:

```python
from gang_mof import pseudo_label, augmented_training_set

pseudo_label(model, result)                          # labels from a training-only ExtraTrees teacher
aug = augmented_training_set(model, result, synthetic_weight=0.25)
# fit your regressor on aug with sample_weight=aug["sample_weight"]
```

Choose the augmentation ratio and weight on validation data, compare against a non-GAN baseline,
and evaluate once on the test set. See `examples/03_low_label_augmentation.py`.

## Command line

```bash
gang-mof example --out mofs.csv
gang-mof fit --data mofs.csv --node-id qmof_id --target bandgap --out gang_model
gang-mof sample --model gang_model --n 100 --condition 2.0 --out gang_results
```

## Examples

| File | What it shows |
|---|---|
| [`examples/quickstart.ipynb`](examples/quickstart.ipynb) | The quick start as a notebook, with outputs |
| [`examples/01_quickstart.py`](examples/01_quickstart.py) | Train, sample, attach, evaluate, export |
| [`examples/02_your_own_mofs.py`](examples/02_your_own_mofs.py) | Featurize your own MOF table and train on it |
| [`examples/03_low_label_augmentation.py`](examples/03_low_label_augmentation.py) | Pseudo-label augmentation with honest held-out evaluation |

## Reproducing the paper

The experiment outputs, evidence tables and pipeline scripts behind the article are kept in this
repository: see [`README_REPRODUCIBILITY.md`](README_REPRODUCIBILITY.md) and
[`docs/PAPER_ARTIFACTS.md`](docs/PAPER_ARTIFACTS.md). The package's constraint-aware generator is the
implementation used for the paper's GANg results.

## Citation

If you use GANg, please cite:

> M. Jalali, A. Orth, M. Reischl. GANg: A Target-Conditioned Generative Adversarial Network for Node
> Synthesis in Metal–Organic Framework Similarity Graphs. *Advanced Intelligent Systems* (2026).

See [`CITATION.cff`](CITATION.cff). The bundled example data are derived from the QMOF database
(Rosen et al., *Matter* 4, 1578 (2021); *npj Comput. Mater.* 8, 112 (2022)), released under CC BY 4.0;
please cite QMOF as well when you use them.

## License

MIT, see [`LICENSE`](LICENSE).
