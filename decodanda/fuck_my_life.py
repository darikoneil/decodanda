import numpy as np
import pandas as pd
from mfr.mouse import Mouse
from pathlib import Path


# load data
mouse = Mouse.load()
experiment = mouse.retrieval

# Load / Format Features
features = np.load(experiment.file_tree.get("results")("features"))
labels = pd.read_parquet(experiment.file_tree.get("results")("labels"))

cs_plus_trials = labels[labels["CS+"] == 1]
cs_plus_trials = np.unique(cs_plus_trials["Trial #"].to_numpy())

cs_minus_trials = labels[labels["CS-"] == 1]
cs_minus_trials = np.unique(cs_minus_trials["Trial #"].to_numpy())

cs_plus_data = features[cs_plus_trials.astype(int), :, 15000:30000]
cs_minus_data = features[cs_minus_trials.astype(int), :, 15000:30000]
