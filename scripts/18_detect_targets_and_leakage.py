"""Detect candidate regression targets and leakage risk from derived QMOF tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "experiments" / "qmof_gang_final" / "tables" / "qmof_node_table.csv"
FEATURE_AUDIT = ROOT / "experiments" / "qmof_gang_final" / "tables" / "feature_audit.csv"
RAW = ROOT / "data" / "raw" / "qmof.csv"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
TABLES.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

CANDIDATE_TARGETS = [
    "outputs.pbe.bandgap",
    "outputs.pbe.cbm",
    "outputs.pbe.vbm",
    "target_bandgap",
    "info_pld",
    "info_lcd",
    "info_density",
    "info_volume",
    "info_natoms",
]


def main() -> None:
    node = pd.read_csv(NODE)
    audit = pd.read_csv(FEATURE_AUDIT)
    feature_cols = set(audit.feature_name)

    rows = []
    for col in CANDIDATE_TARGETS:
        mapped = col
        if col == "outputs.pbe.bandgap" and "target_bandgap" in node.columns:
            mapped = "target_bandgap"
        present = mapped in node.columns
        n = len(node) if present else 0
        miss = float(node[mapped].isna().mean()) if present else 1.0
        rows.append(
            {
                "target_key": col,
                "column_in_derived_table": mapped if present else "",
                "present_in_derived_1000_table": present,
                "n_rows": n,
                "missing_fraction": miss,
                "usable_for_multitarget_run": bool(present and miss < 0.2),
            }
        )
    avail = pd.DataFrame(rows)
    avail.to_csv(TABLES / "target_availability.csv", index=False)

    leak_rows = []
    for _, t in avail[avail.usable_for_multitarget_run].iterrows():
        col = t.column_in_derived_table
        for feat in sorted(feature_cols):
            leak_rows.append(
                {
                    "target_key": t.target_key,
                    "feature_name": feat,
                    "feature_equals_target": feat == col,
                    "leakage_risk": "high" if feat == col else "low",
                    "excluded_from_node_features": feat != col,
                }
            )
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(TABLES / "feature_leakage_audit_by_target.csv", index=False)

    lines = [
        "# Target and leakage audit",
        "",
        f"Derived node table: `{NODE.relative_to(ROOT)}` ({len(node)} rows).",
        f"Raw QMOF metadata present: `{RAW.exists()}` (`{RAW}`).",
        "",
        "## Usable targets in current derived table",
    ]
    usable = avail[avail.usable_for_multitarget_run]
    if len(usable):
        for _, r in usable.iterrows():
            lines.append(f"- `{r.target_key}` → column `{r.column_in_derived_table}` (missing {r.missing_fraction:.1%})")
    else:
        lines.append("- Only `outputs.pbe.bandgap` / `target_bandgap` is available in the locked 1000-node export.")

    lines += [
        "",
        "## Multi-target expansion status",
        "- Additional QMOF output columns (CBM/VBM/etc.) require raw `data/raw/qmof.csv` or an expanded derived table.",
        "- Do not run multi-target manuscript claims until those columns are exported with leakage checks.",
    ]
    if not RAW.exists():
        lines.append("- **Blocker:** `data/raw/qmof.csv` is missing.")
    (REPORTS / "target_and_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
