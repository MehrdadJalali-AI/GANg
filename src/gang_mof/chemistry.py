"""Element tables and the QMOF-style composition featurizer used by GANg.

The featurizer turns a chemical formula (plus six cell-level scalars) into the
106 MOF-level descriptors used in the GANg paper. It is the *same* featurizer
that GANg applies to its generated descriptor core, so real and generated MOFs
live in exactly the same descriptor space.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

import numpy as np
import pandas as pd

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Pb", "Bi",
}
#: The 75-element composition basis (order defines the ``frac_<El>`` columns).
ELEMENTS = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Pb", "Bi",
]
TRANSITION_METALS = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
                     "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}
RARE_EARTHS = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE = {"Be", "Mg", "Ca", "Sr", "Ba"}
HALOGENS = {"F", "Cl", "Br", "I"}
CHALCOGENS = {"O", "S", "Se", "Te"}
ORGANIC = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}

ATOMIC_NUMBER = {el: i + 1 for i, el in enumerate([
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
])}
ATOMIC_MASS = {
    "H": 1.008, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63,
    "As": 74.922, "Se": 78.971, "Br": 79.904, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Ag": 107.868, "Cd": 112.414,
    "In": 114.818, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.904, "Cs": 132.905, "Ba": 137.327,
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25,
    "Tb": 158.925, "Dy": 162.5, "Ho": 164.93, "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967,
    "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.967, "Hg": 200.592, "Pb": 207.2, "Bi": 208.98,
}
ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
    "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55,
    "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16, "Ru": 2.2,
    "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.1,
    "I": 2.66, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17,
    "Eu": 1.2, "Gd": 1.2, "Tb": 1.2, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.1,
    "Lu": 1.27, "Hf": 1.3, "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.2, "Pt": 2.28,
    "Au": 2.54, "Hg": 2.00, "Pb": 2.33, "Bi": 2.02,
}
#: Conversion from amu / Å^3 to g / cm^3.
AMU_A3_TO_G_CM3 = 1.66054

#: Cell-level scalar descriptors, in the order used by the QMOF-106 schema.
CELL_SCALARS = ["info_natoms", "info_pld", "info_lcd", "info_density", "info_volume", "info_symmetry_spacegroup_number"]


def parse_formula(formula: str) -> Counter:
    """Parse a flat formula such as ``"Zn4C24H12O13"`` into element counts."""
    counts: Counter = Counter()
    if not isinstance(formula, str):
        return counts
    for el, num in re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula):
        counts[el] += float(num) if num else 1.0
    return counts


def composition_features(formula: str) -> dict[str, float]:
    """The 100 formula-derived descriptors (75 fractions + 25 composition summaries)."""
    c = parse_formula(formula)
    total = float(sum(c.values())) or 1.0
    out = {f"frac_{el}": float(c.get(el, 0.0) / total) for el in ELEMENTS}
    metal_count = float(sum(v for k, v in c.items() if k in METALS))
    organic_count = float(sum(v for k, v in c.items() if k in ORGANIC))
    share = lambda group: float(sum(v for k, v in c.items() if k in group) / total)
    out.update({
        "atom_count_formula": total,
        "n_unique_elements": float(len(c)),
        "metal_count": metal_count,
        "metal_fraction": metal_count / total,
        "has_metal": float(metal_count > 0),
        "transition_metal_fraction": share(TRANSITION_METALS),
        "rare_earth_fraction": share(RARE_EARTHS),
        "alkali_fraction": share(ALKALI),
        "alkaline_earth_fraction": share(ALKALINE),
        "halogen_fraction": share(HALOGENS),
        "chalcogen_fraction": share(CHALCOGENS),
        "organic_fraction_CHNO": float(sum(c.get(k, 0.0) for k in "CHNO") / total),
        "organic_element_fraction": share(ORGANIC),
        "heteroatom_fraction_NOFPSClBrI": float(sum(c.get(k, 0.0) for k in ["N", "O", "F", "P", "S", "Cl", "Br", "I"]) / total),
        "carbon_fraction": float(c.get("C", 0.0) / total),
        "hydrogen_fraction": float(c.get("H", 0.0) / total),
        "nitrogen_oxygen_fraction": float((c.get("N", 0.0) + c.get("O", 0.0)) / total),
        "c_to_h_ratio": float(c.get("C", 0.0) / max(c.get("H", 0.0), 1.0)),
        "metal_to_organic_ratio": float(metal_count / max(organic_count, 1.0)),
    })

    def weighted(table):
        vals = [(table[k], v) for k, v in c.items() if k in table]
        den = sum(v for _, v in vals) or 1.0
        mean = float(sum(a * v for a, v in vals) / den)
        rng = float(max(a for a, _ in vals) - min(a for a, _ in vals)) if vals else 0.0
        return mean, rng

    for name, table in [("atomic_number", ATOMIC_NUMBER), ("atomic_mass", ATOMIC_MASS),
                        ("electronegativity", ELECTRONEGATIVITY)]:
        out[f"{name}_mean"], out[f"{name}_range"] = weighted(table)
    return out


def featurize(formula: str, natoms: float, pld: float, lcd: float, volume: float, spacegroup: int,
              density: float | None = None) -> dict[str, float]:
    """Build the full 106-descriptor record for one MOF.

    Parameters
    ----------
    formula : str
        Reduced formula (formula unit), e.g. ``"Zn4C24H12O13"``; ``natoms`` must be an integer multiple of its atom count.
    natoms : float
        Number of atoms in the unit cell.
    pld, lcd : float
        Pore-limiting and largest cavity diameter (Å).
    volume : float
        Unit-cell volume (Å^3).
    spacegroup : int
        International space-group number.
    density : float, optional
        Density in g/cm^3. If omitted it is computed from the composition, ``natoms`` and ``volume``.
    """
    f = composition_features(formula)
    if density is None:
        density = natoms * f["atomic_mass_mean"] * AMU_A3_TO_G_CM3 / volume
    f.update({"info_natoms": float(natoms), "info_pld": float(pld), "info_lcd": float(lcd),
              "info_density": float(density), "info_volume": float(volume),
              "info_symmetry_spacegroup_number": float(spacegroup)})
    return f


#: Default column names (QMOF metadata convention) understood by :func:`featurize_dataframe`.
QMOF_COLUMNS: Mapping[str, str] = {
    "formula": "formula", "natoms": "info_natoms", "pld": "info_pld", "lcd": "info_lcd",
    "volume": "info_volume", "spacegroup": "info_symmetry_spacegroup_number", "density": "info_density",
}


def featurize_dataframe(df: pd.DataFrame, columns: Mapping[str, str] | None = None, keep: list[str] | None = None) -> pd.DataFrame:
    """Featurize a table of MOFs into the 106 GANg descriptors.

    ``columns`` maps the keys ``formula, natoms, pld, lcd, volume, spacegroup`` (and optionally
    ``density``) to column names in ``df``; the defaults follow the QMOF metadata names.
    Columns listed in ``keep`` (for example an ID or a target) are copied to the output.
    """
    from .schema import QMOF106_FEATURES

    cols = dict(QMOF_COLUMNS)
    cols.update(columns or {})
    missing = [k for k in ["formula", "natoms", "pld", "lcd", "volume", "spacegroup"] if cols[k] not in df.columns]
    if missing:
        raise KeyError(f"featurize_dataframe: columns not found for {missing}; pass columns={{...}} to map them.")
    has_density = cols.get("density") in df.columns
    rows = []
    for _, r in df.iterrows():
        rows.append(featurize(r[cols["formula"]], r[cols["natoms"]], r[cols["pld"]], r[cols["lcd"]],
                              r[cols["volume"]], r[cols["spacegroup"]],
                              r[cols["density"]] if has_density else None))
    out = pd.DataFrame(rows, index=df.index)[QMOF106_FEATURES]
    if keep:
        out = pd.concat([df[keep], out], axis=1)
    return out


def descriptors_to_formula(row: Mapping[str, float], decimals: int = 0) -> str:
    """Reconstruct a formula string from a descriptor record (``frac_*`` x ``atom_count_formula``)."""
    total = float(row["atom_count_formula"])
    parts = []
    for el in ELEMENTS:
        n = float(row[f"frac_{el}"]) * total
        n = np.round(n, decimals)
        if n > 0:
            parts.append(f"{el}{int(n) if decimals == 0 else n:g}")
    return "".join(parts)
