"""Quick start: train GANg on the bundled QMOF example and generate 100 MOF graph nodes.

Runtime: about 2-3 minutes on a laptop CPU.   python examples/01_quickstart.py
"""
from gang_mof import GANg, load_qmof_example

data = load_qmof_example()                         # 1,000 real MOFs, 106 descriptors, PBE band gap (eV)
train, validation, test = data.split(random_state=42)

model = GANg(random_state=0)                       # constraint-aware GANg (auto-selected for MOF data)
model.fit(train, validation_data=validation)       # validation is used only to pick the checkpoint

result = model.sample(n_samples=100, condition=2.0)  # ask for descriptor-space MOFs near 2.0 eV
model.attach(result, k=15)                         # cosine kNN edges to real training MOFs
model.evaluate(result, reference=test)             # coverage, diversity, surrogate fidelity vs held-out MOFs

print(result.summary())
print(result.nodes[["node_id", "formula", "requested_condition", "surrogate_prediction"]].head())
print(result.analogs.head())

result.export("outputs/quickstart")                # nodes.csv, edges.csv, checks.csv, analogs.csv, metrics.json
model.save("outputs/quickstart_model")             # reload later with GANg.load(...)
