from __future__ import annotations
from typing import Callable, Iterable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import product


def scale_responses(traces: np.ndarray, scaler: Callable = StandardScaler, inplace=True):
    if not inplace:
        traces = traces.copy()

    for neuron in range(traces.shape[0]):
        traces[neuron, :] = np.ravel(scaler().fit_transform(traces[neuron, :].reshape(-1, 1)))

    return traces


def generate_tensor_indices(aligned_data, included_stages: Iterable[str] = ("Trial", "Pretrial"),
                            trial_groups: Optional[Iterable] = None):

    # if trial groups aren't provided we'll automatically use all of them instead
    if trial_groups:
        trials = len(trial_groups)
    else:
        trials = np.where(np.diff(aligned_data["Trial"].to_numpy(copy=True)) == 1)[0].shape[0]
        trial_groups = list(range(trials))

    # calculate number of samples per trial - the median is such a failsafe to prevent gotcha's
    samples_per_trial = int(np.median(
        [np.where((
            np.sum([(aligned_data[label] == 1) for label in included_stages], axis=0))
                  & (aligned_data["Trial Group"] == trial))[0].shape[0]
         for trial in range(trials)]
    ))

    samples_index = np.empty((samples_per_trial, trials))
    samples_index[:] = np.nan

    for trial in range(trials):
        trial = trial_groups[trial]
        trial_data = aligned_data[aligned_data["Trial Group"] == trial]
        trial_data = trial_data.iloc[
            np.where(np.sum([(trial_data[label] == 1) for label in included_stages], axis=0))[0]]
        try:
            samples_index[:, trial] = trial_data.index.to_numpy(copy=True)
        except ValueError:
            samples = trial_data.index.to_numpy(copy=True)
            true_samples = samples[~np.isnan(samples)]
            sample_positions = np.where(~np.isnan(samples))[0]
            samples_index[sample_positions, trial] = true_samples

    return samples_index


def build_trace_tensor(tensor_indices: np.ndarray, traces: pd.DataFrame) -> np.ndarray:
    samples, trials = tensor_indices.shape
    neurons = traces.shape[1]
    trace_tensor = np.empty((trials, neurons, samples))
    trace_tensor[:] = np.nan

    for trial in range(trials):
        trace_tensor[trial, :, :] = traces.loc[tensor_indices[:, trial], :].to_numpy(copy=True).T

    return trace_tensor


def build_trial_tensor(tensor_indices, included_labels, aligned_data):
    samples, trials = tensor_indices.shape
    features = len(included_labels)
    indicator_matrix = np.empty((trials, features, samples))
    indicator_matrix[:] = np.nan

    for trial, (feature_idx, feature) in product(range(trials), zip(range(features), included_labels)):
        try:
            indicator_matrix[trial, feature_idx, :] = \
                aligned_data[feature].loc[tensor_indices[:, trial]].to_numpy(copy=True)
        except KeyError:
            true_indices = tensor_indices[~np.isnan(tensor_indices[:, trial]), trial]
            indices_positions = np.where(~np.isnan(tensor_indices[:, trial]))[0]
            indicator_matrix[trial, feature_idx, indices_positions] = \
                aligned_data[feature].loc[true_indices].to_numpy(copy=True)

    return indicator_matrix


def interpolate_traces(traces, aligned_data) -> pd.DataFrame:
    frame_times = aligned_data["Imaging Frame"].dropna().index
    traces = pd.DataFrame(data=traces.T, index=frame_times, columns=[
        "".join(["Neuron ", str(x)]) for x in range(traces.shape[0])
    ])
    traces = traces.reindex(aligned_data.index)
    traces.interpolate(method="pchip", inplace=True)
    return traces


def downsample(indicator_matrix, target_frame_rate: int = 30):
    current_samples = indicator_matrix.shape[-1]
    target_idx = np.arange(0, current_samples, target_frame_rate)
    return indicator_matrix[:, :, target_idx]
