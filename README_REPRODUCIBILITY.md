# GANg reproducibility guide

Repository: https://github.com/MehrdadJalali-AI/GANg

## Environment

- Python 3.10+ recommended (3.9+ tested for analysis scripts)
- Packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `torch`, `pyyaml`
- LaTeX: `latexmk` (TeX Live)

```bash
export MPLBACKEND=Agg
```

## Raw data

Obtain QMOF structures/metadata from the original QMOF source and place:

`data/raw/qmof.csv`

The locked derived 1000-node package is under `experiments/qmof_gang_final/`.

## Minimal rebuild (1000-node package)

```bash
python3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml
python3 scripts/10_finalize_gang_package.py
python3 scripts/11_gang_audit_pass.py
python3 scripts/12_submission_polish.py
```

## npj-readiness analyses (no fabricated large-scale results)

```bash
python3 scripts/18_detect_targets_and_leakage.py
python3 scripts/19_run_multitarget_pipeline.py
python3 scripts/20_graph_free_baselines.py
python3 scripts/21_descriptor_generator_baselines.py
python3 scripts/22_statistical_testing.py
python3 scripts/23_validity_utility_gap_analysis.py
python3 scripts/24_structure_aware_reference_analogues.py
python3 scripts/25_make_npj_figures.py
python3 scripts/26_claim_safety_audit.py
```

## Figures

- Main: `figures/main/Figure_1..5`
- npj export: `figures/npj/`
- Regenerate main panels: `python3 scripts/16_make_high_rank_figures.py`

## Manuscript

```bash
latexmk -pdf manuscript/main.tex
latexmk -pdf manuscript/supplementary.tex
```

## Expected outputs

- `tables/statistical_testing_summary.csv`
- `tables/graph_free_baselines.csv`
- `tables/validity_utility_gap_summary.csv`
- `tables/generated_node_reference_analogues.csv`
- `reports/npj_submission_readiness_decision.md`

## Hardware / runtime

1000-node derived artifacts are lightweight. Full rebuild (GANg + downstream GNN, five seeds) is heavier; CPU is supported, GPU optional for PyTorch.

## Claim boundary

Generated nodes are descriptor-space hypotheses—not generated CIFs, not physically validated MOF structures, not DFT-validated, not synthesis-ready, not experimentally validated.
