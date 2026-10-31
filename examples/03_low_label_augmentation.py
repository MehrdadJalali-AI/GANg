"""Pseudo-label augmentation in a low-label setting, evaluated honestly on held-out MOFs.

GANg paper finding: augmentation helps modestly only when real labels are scarce, and the gain
comes mainly from distilling a training-only teacher. This script lets you test that on your data.
All choices (ratio, weight) should be made on validation data; the test set is touched once.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gang_mof import GANg, augmented_training_set, load_qmof_example

data = load_qmof_example()
pool, validation, test = data.split(train=0.70, validation=0.15, test=0.15, random_state=1)
labelled = pool.subset(np.arange(100))             # pretend only 100 labelled MOFs exist

model = GANg(random_state=0).fit(labelled, validation_data=validation)
result = model.sample(100)                         # 100% augmentation ratio

def student():
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0))

def mae(m, ds):
    return float(np.mean(np.abs(m.predict(ds.X.to_numpy(float)) - ds.y)))

real_only = student().fit(labelled.X.to_numpy(float), labelled.y)

aug = augmented_training_set(model, result, synthetic_weight=1.0)   # pseudo-labels from a training-only teacher
feats = model.feature_names
X_aug, y_aug = aug[feats].to_numpy(float), aug["bandgap"].to_numpy(float)
augmented = student().fit(X_aug, y_aug)   # MLPRegressor has no sample_weight; weight=1.0 here

print(f"validation MAE  real-only {mae(real_only, validation):.3f} eV   augmented {mae(augmented, validation):.3f} eV")
print(f"test MAE        real-only {mae(real_only, test):.3f} eV   augmented {mae(augmented, test):.3f} eV")
print("Compare against a non-GAN baseline (e.g. bootstrap resampling) before claiming a GAN-specific gain.")
