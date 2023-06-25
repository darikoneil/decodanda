from sklearnex.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from mfr.mouse import Mouse
from mfr.sanitizing import downsample, scale_responses
from pathlib import Path
from binner import bin_data
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns


bin_length = 1000

# Debug
np.random.seed(1123)

# Load Mouse
mouse = Mouse.load(Path("D:\\DEM2"))

# Load Experiment
experiment = mouse.preexposure

# Load / Format Features
features = np.load(experiment.file_tree.get("results")("features"))
features = np.vstack([bin_data(features[trial, :, :].T, bin_length=bin_length, fun="sum")
                      for trial in range(features.shape[0])])

# Load / Format Labels
labels = pd.read_parquet(experiment.file_tree.get("results")("labels"))
label_idx = labels.columns.tolist().index("Trial #")
time_idx = labels.columns.tolist().index("Time (ms)")
labels = labels.to_numpy(copy=True)
trials = np.unique(labels[:, label_idx]).shape[0]
samples, conditions = labels.shape
samples_per_trial = samples // trials
labels_ = np.zeros((trials, conditions, samples_per_trial))
for trial in range(trials):
    labels_[trial, :, :] = labels[np.where(labels[:, label_idx] == trial)[0], :].T
labels_ = np.vstack([bin_data(labels_[trial, :, :].T, bin_length=bin_length, fun="median")
                     for trial in range(labels_.shape[0])])


idx = np.where((labels_[:, 0] == 1) | (labels_[:, 1] == 1) | (labels_[:, 2] == 1) | (labels_[:, 3] == 1))[0]

raster = features[idx, :]
cs = labels_[idx, 0]
cs[np.where(labels_[idx, 3] == 1)[0]] = 2

testing_index = np.where((labels_[idx, -2] == 2) | (labels_[idx, -2] == 3))[0]
training_index = np.where((labels_[idx, -2] == 0) | (labels_[idx, -2] == 1) | (labels_[idx, -2] == 4) | (labels_[idx, -2] == 5) | (labels_[idx, -2] == 6) | (labels_[idx, -2] == 7) | (labels_[idx, -2] == 8) | (labels_[idx, -2] == 9))[0]

training_labels = cs[training_index].copy()
training_raster = raster[training_index, :].copy()
testing_labels = cs[testing_index].copy()
testing_raster = raster[testing_index, :].copy()

scaler = MinMaxScaler().fit(training_raster)
training_raster = scaler.transform(training_raster)
testing_raster = scaler.transform(testing_raster)

mdl = SVC(kernel="rbf")
mdl.fit(training_raster, training_labels)
mdl.score(testing_raster, testing_labels)

ConfusionMatrixDisplay.from_estimator(mdl, testing_raster, testing_labels, normalize="true")
