import numpy as np
import pandas as pd
from mfr.mouse import Mouse
from mfr.sanitizing import downsample, scale_responses
from pathlib import Path


from sklearnex.svm import SVC as classifier_
from sklearn.preprocessing import StandardScaler as scaler_


from dd2 import Decodanda
from dd2._defaults import classifier_parameters_intel as classifier_params_


from binner import bin_data

from scipy.stats import median_abs_deviation


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
experiment = mouse.retrieval

# Load / Format Features
features = np.load(experiment.file_tree.get("results")("features"))
features = [bin_data(features[trial, :, :].T, bin_length=bin_length, fun="mean")
                      for trial in range(features.shape[0])]

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

idx = np.where((labels_[:, 0] == 1) | (labels_[:, 3] == 1))[0]

raster = features[idx, :]


cs = labels_[idx, :]

data = {
    "raster": raster,
    "cs": cs[:, 0],
    "trial": cs[:, -2],
}

conditions = {"cs": [0, 1]}

decoder_params = {
    "cross_validations": 10,
    "n_shuffles": 0,
    "parallel": False,
    "scale": scaler_(copy=False),
    "training_fraction": 0.8,
    "verbose": True
}

decoder = Decodanda(data=data,
                    conditions=conditions,
                    classifier=classifier_,
                    classifier_params=classifier_params_,
                    decodanda_params=decoder_params)

real, null = decoder.decode()

