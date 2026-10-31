"""Use GANg with your own MOFs.

You need, per MOF: an ID, a reduced formula, the cell atom count, PLD, LCD, cell volume and
space-group number (density is optional; it is recomputed if missing), plus the target you want
to condition on. Column names below follow the QMOF metadata; map yours with `columns=`.
"""
import pandas as pd

from gang_mof import GANg, NodeDataset, featurize_dataframe

# --- your table (here built from the bundled example so the script runs as-is) -----------------
from gang_mof import load_qmof_example
raw = load_qmof_example(as_frame=True)[["qmof_id", "formula", "bandgap", "info_natoms", "info_pld",
                                        "info_lcd", "info_volume", "info_symmetry_spacegroup_number"]]
raw = raw.rename(columns={"qmof_id": "mof", "info_natoms": "n_atoms", "info_pld": "pld", "info_lcd": "lcd",
                          "info_volume": "volume", "info_symmetry_spacegroup_number": "spacegroup"})
# raw = pd.read_csv("my_mofs.csv")  # <- your own file

# --- 1. compute the 106 GANg descriptors ------------------------------------------------------
table = featurize_dataframe(raw, columns={"formula": "formula", "natoms": "n_atoms", "pld": "pld",
                                          "lcd": "lcd", "volume": "volume", "spacegroup": "spacegroup"},
                            keep=["mof", "bandgap"])

# --- 2. wrap as a dataset and split before fitting anything ------------------------------------
data = NodeDataset.from_dataframe(table, node_id="mof", target="bandgap")
train, validation, test = data.split(random_state=0)

# --- 3. train, generate, attach, audit -------------------------------------------------------
model = GANg(random_state=0).fit(train, validation_data=validation)
result = model.attach(model.sample(200), k=15)    # requests drawn from the training band-gap distribution
model.evaluate(result, reference=test)
print(result.summary())
accepted = result.accepted()                       # keep nodes that pass every screen
accepted.export("outputs/my_mofs")
