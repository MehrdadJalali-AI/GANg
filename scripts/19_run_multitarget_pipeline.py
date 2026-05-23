"""Prepare multi-target job manifest; run only when targets exist in derived tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
AVAIL = ROOT / "tables" / "target_availability.csv"
JOBS = ROOT / "configs" / "generated_jobs"
REPORTS = ROOT / "reports"
EXP = ROOT / "experiments" / "qmof_gang_multitarget"
JOBS.mkdir(parents=True, exist_ok=True)
EXP.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SIZES = ["qmof_galaxy_1000.yaml", "qmof_galaxy_5000.yaml", "qmof_galaxy_all_available.yaml"]


def main() -> None:
    if not AVAIL.exists():
        raise SystemExit("Run scripts/18_detect_targets_and_leakage.py first.")

    targets = pd.read_csv(AVAIL)
    runnable = targets[targets.usable_for_multitarget_run]
    manifest = []

    for cfg_name in SIZES:
        cfg_path = ROOT / "configs" / cfg_name
        base = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        for _, t in runnable.iterrows():
            job = dict(base)
            job["target"] = t.target_key
            job["output_directory"] = f"experiments/qmof_gang_multitarget/{cfg_name.replace('.yaml','')}/{t.target_key.replace('.','_')}"
            job_name = f"{cfg_name.replace('.yaml','')}__{t.target_key.replace('.','_')}.yaml"
            out = JOBS / job_name
            out.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
            manifest.append(
                {
                    "config_template": cfg_name,
                    "target": t.target_key,
                    "job_yaml": str(out.relative_to(ROOT)),
                    "status": "ready_to_run" if cfg_name == "qmof_galaxy_1000.yaml" else "blocked_missing_raw_qmof",
                }
            )

    man = pd.DataFrame(manifest)
    man.to_csv(EXP / "tables" / "multitarget_job_manifest.csv", index=False) if (EXP / "tables").exists() else None
    (EXP / "tables").mkdir(parents=True, exist_ok=True)
    man.to_csv(EXP / "tables" / "multitarget_job_manifest.csv", index=False)

    lines = [
        "# Multi-target job manifest",
        "",
        "This script does **not** fabricate multi-target results.",
        "",
        f"Runnable targets detected: {len(runnable)}",
        f"Job YAML files written to `{JOBS.relative_to(ROOT)}/`.",
        "",
        "## Execution",
        "```bash",
        "python3 scripts/09_build_final_package.py --config configs/generated_jobs/<job>.yaml",
        "```",
        "",
        "## Current blocker",
        "- 5000/all-available jobs require `data/raw/qmof.csv`.",
        "- Only bandgap is present in the locked derived 1000-node table unless additional target columns are exported.",
    ]
    (REPORTS / "multitarget_job_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "note": "No multitarget downstream summary generated",
                "reason": "Only single-target bandgap artifacts exist in qmof_gang_final",
            }
        ]
    )
    summary.to_csv(EXP / "tables" / "multitarget_downstream_summary.csv", index=False)
    (REPORTS / "multitarget_experiment_report.md").write_text(
        "# Multi-target experiment report\n\nNo multi-target downstream results were executed in this pass. "
        "See `reports/multitarget_job_manifest.md` and `reports/missing_large_scale_data.md`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
