from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "manuscript" / "main.tex", ROOT / "manuscript" / "supplementary.tex"]
REPORT = ROOT / "reports" / "claim_safety_audit.md"

UNSAFE = [
    "new MOF",
    "generated MOF structure",
    "synthesis-ready",
    "DFT-validated",
    "physically valid",
    "crystal generation",
    "discovery",
    "optimized material",
    "experimental validation",
    "generated CIF",
]

NEGATORS = [
    "not ",
    "No ",
    "no ",
    "not supported",
    "does not",
    "without",
    "not a",
    "not generated",
    "not physically",
]


def safe_context(line: str) -> bool:
    low = line.lower()
    return any(n.lower() in low for n in NEGATORS)


def main() -> None:
    rows = []
    unsafe_rows = []
    for path in FILES:
        text = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(text, 1):
            for phrase in UNSAFE:
                if phrase.lower() in line.lower():
                    ok = safe_context(line)
                    row = (path.relative_to(ROOT).as_posix(), i, phrase, ok, line.strip())
                    rows.append(row)
                    if not ok:
                        unsafe_rows.append(row)
    lines = ["# Claim safety audit", ""]
    if unsafe_rows:
        lines.append("## Potentially unsafe, non-negated occurrences")
        for f, i, p, _, line in unsafe_rows:
            lines.append(f"- `{f}:{i}` phrase `{p}`: {line}")
    else:
        lines.append("No non-negated unsafe claim phrases were detected.")
    lines.append("")
    lines.append("## All monitored occurrences")
    if not rows:
        lines.append("No monitored phrases found.")
    for f, i, p, ok, line in rows:
        status = "allowed-negative-context" if ok else "review"
        lines.append(f"- `{f}:{i}` `{p}` [{status}]: {line}")
    lines.append("")
    lines.append("Preferred manuscript language: descriptor-space hypothesis; MOF-like graph node; reference analogue; nearest real QMOF neighbor; not a generated CIF; not physically validated.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if unsafe_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
