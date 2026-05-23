# GANg: Descriptor-Space Graph-Node Synthesis Reveals a Validity–Utility Gap in MOF Similarity Networks
<p align="center">
  <img 
    src="GANg.jpg" 
    alt="GANg conceptual artwork: graph-node synthesis in MOF similarity networks" 
    width="90%">
</p>
GANg is a transparent descriptor-space graph-node synthesis framework for MOF similarity networks. The case study uses a QMOF-derived MOF similarity network: one MOF is one node, edges are descriptor-space similarities, and generated nodes are MOF-like descriptor-space hypotheses.

Generated nodes are not automatically new MOFs, not DFT-validated, not force-field-relaxed, not synthesis-ready, and not experimentally validated.

Repository: https://github.com/MehrdadJalali-AI/GANg

## Reproducibility

```bash
# Minimal rebuild
python3 scripts/09_build_final_package.py --config configs/qmof_galaxy_1000.yaml
python3 scripts/10_finalize_gang_package.py
python3 scripts/11_gang_audit_pass.py
python3 scripts/12_submission_polish.py

# Extended audit and figure rebuild
python3 scripts/13_statistical_significance.py
python3 scripts/14_validity_utility_gap_analysis.py
python3 scripts/15_nearest_structure_reference_analysis.py
python3 scripts/16_make_high_rank_figures.py
python3 scripts/17_claim_safety_audit.py

# Compile
latexmk -pdf manuscript/main.tex
latexmk -pdf manuscript/supplementary.tex
```

## Outputs

- Tables: `experiments/qmof_gang_final/tables/`
- Figures: `experiments/qmof_gang_final/figures/`
- Generated nodes: `experiments/qmof_gang_final/generated_nodes/`
- Reports: `reports/gang_final_readiness_report.md`, `reports/claim_boundary.md`, `reports/metric_source_map.md`
