# GANg reproducibility guide

This guide describes the public code/data snapshot. It focuses on scripts, locked experiment outputs, and generated evidence tables. Manuscript sources and internal reports are not part of this public repository snapshot.

## 1. Environment

Recommended:

- Python 3.10+
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `networkx`
- `torch`
- `pyyaml`

For non-interactive or server environments:

```bash
export MPLBACKEND=Agg
```

## 2. Fast integrity check

From the repository root:

```bash
python3 scripts/42_final_consistency_assertions.py
```

Expected output:

```text
PASS final implementation-grounded consistency assertions
```

This validates the public evidence tables against locked source outputs, including validation-selected downstream summaries, graph-scaling calculations, six-generator baseline coverage, feature-category consistency, and generated-node ID uniqueness.

## 3. Core 1000-node workflow

The historical locked 1000-node QMOF-derived workflow can be rebuilt with:

```bash
python3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml
python3 scripts/10_finalize_gang_package.py
python3 scripts/11_gang_audit_pass.py
python3 scripts/12_submission_polish.py
```

The main 1000-node derived tables are stored under:

```text
experiments/qmof_gang_final/tables/
experiments/qmof_galaxy_final/tables/
```

## 4. Validation-selected downstream summaries

Validation-selected summaries are generated from locked downstream GNN source tables:

```bash
python3 scripts/46_validation_selected_downstream_audit.py
```

Primary outputs:

```text
tables/validation_selected_results.csv
tables/validation_selected_seedwise_results.csv
tables/validation_selected_statistics.csv
tables/validation_selected_downstream_summary.csv
tables/validation_selected_downstream_seedwise_5000.csv
```

The script selects GANg augmentation mode by validation MAE within each target-architecture pair, then reports held-out real-node test metrics. It does not select by test performance.

## 5. 5000-node source tables

The public repository includes minimal locked source CSVs required to reproduce the validation-selected and graph-scaling audits:

```text
experiments/qmof_gang_5000_outputs_pbe_bandgap/tables/
experiments/qmof_gang_5000_outputs_pbe_cbm/tables/
experiments/qmof_gang_5000_outputs_pbe_vbm/tables/
experiments/qmof_gang_5000_outputs_pbe_energy_total/tables/
```

Each target folder contains the downstream summary and seedwise source tables. The bandgap folder also contains graph-construction and sparsification summaries used for graph-scaling checks.

## 6. Implementation-grounded graph scaling

Run:

```bash
python3 scripts/47_implementation_grounded_audit.py
```

Primary outputs:

```text
tables/implementation_grounded_scaling_summary.csv
tables/implementation_grounded_scaling_summary.tex
tables/implementation_grounded_metric_source_map.csv
tables/implementation_grounded_consistency_checks.csv
tables/implementation_grounded_final_audit.md
figures/high_rank_5000/scaling_comparison_1000_vs_5000.pdf
figures/high_rank_5000/scaling_comparison_1000_vs_5000.png
figures/high_rank_5000/scaling_comparison_1000_vs_5000.svg
```

The script computes graph-scaling values from source graph-construction and sparsification tables. It does not manually insert figure or table values.

## 7. Generator-baseline diagnostics

The harmonised descriptor-generator baseline evidence is under:

```text
tables/remaining_issues/generator_baseline_validity_with_gang.csv
tables/remaining_issues/generator_baseline_diversity_with_gang.csv
```

Regenerate the baseline figure with:

```bash
python3 scripts/29_make_generative_upgrade_figures.py
```

These are generator-quality diagnostics only. They do not establish downstream superiority.

## 8. Figure regeneration

The final public plotting script is:

```bash
python3 scripts/37_journal_style_polish_figures.py
```

It redraws manuscript-facing figure assets from existing tables. It does not run new experiments and does not change numeric values.

## 9. Claim boundary

GANg generates descriptor-space graph-node hypotheses and similarity edges. It does not generate CIFs or validated periodic structures. Nearest real QMOF analogue retrieval provides interpretation only and is not physical validation.

Raw total-energy diagnostics are extensive-target stress tests and should not be interpreted as transferable energetic learning, physical inverse design, or experimental validation.
