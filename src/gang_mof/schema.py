"""Descriptor schemas.

A schema tells GANg which columns are node descriptors and whether the descriptors obey
known relations that the generator can enforce by construction.

* ``"qmof106"`` – the 106 MOF descriptors of the GANg paper. GANg generates only the
  independent core (75 elemental fractions + 6 cell scalars) and rebuilds every derived
  descriptor exactly, so all generated samples are schema-consistent.
* ``"generic"`` – any numeric descriptor table. GANg falls back to the full-vector
  conditional WGAN-GP baseline with clipping to training bounds; no relations are enforced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .chemistry import CELL_SCALARS, ELEMENTS

COMPOSITION_SUMMARIES = [
    "atom_count_formula", "n_unique_elements", "metal_count", "metal_fraction", "has_metal",
    "transition_metal_fraction", "rare_earth_fraction", "alkali_fraction", "alkaline_earth_fraction",
    "halogen_fraction", "chalcogen_fraction", "organic_fraction_CHNO", "organic_element_fraction",
    "heteroatom_fraction_NOFPSClBrI", "carbon_fraction", "hydrogen_fraction", "nitrogen_oxygen_fraction",
    "c_to_h_ratio", "metal_to_organic_ratio", "atomic_number_mean", "atomic_number_range", "atomic_mass_mean",
    "atomic_mass_range", "electronegativity_mean", "electronegativity_range",
]
#: The 106 target-free MOF descriptors used in the GANg paper, in canonical order.
QMOF106_FEATURES = [f"frac_{e}" for e in ELEMENTS] + COMPOSITION_SUMMARIES + CELL_SCALARS


@dataclass(frozen=True)
class Schema:
    name: str
    features: list[str] = field(default_factory=list)
    constrained: bool = False

    @property
    def n_features(self) -> int:
        return len(self.features)


QMOF106 = Schema("qmof106", list(QMOF106_FEATURES), constrained=True)


def get_schema(schema: str | Schema = "auto", columns=None, features: list[str] | None = None) -> Schema:
    """Resolve a schema name. ``"auto"`` picks ``qmof106`` when all 106 columns are present."""
    if isinstance(schema, Schema):
        return schema
    if schema == "auto":
        if columns is not None and set(QMOF106_FEATURES) <= set(columns):
            return QMOF106
        schema = "generic"
    if schema == "qmof106":
        return QMOF106
    if schema == "generic":
        if not features:
            raise ValueError("The generic schema needs an explicit list of feature columns (features=[...]).")
        return Schema("generic", list(features), constrained=False)
    raise ValueError(f"Unknown schema {schema!r}; use 'qmof106', 'generic' or 'auto'.")
