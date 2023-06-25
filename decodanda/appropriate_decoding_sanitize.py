import numpy as np
import pandas as pd
from mfr.mouse import Mouse
from pathlib import Path
from mfr.sanitizing import generate_tensor_indices, build_trial_tensor, interpolate_traces, scale_responses, build_trace_tensor
from deconvolution import constrained_foopsi
from CalSciPy.event_processing import convert_tau
from scipy.stats import median_abs_deviation


def find_spikes(matrix):
    matrix = matrix.copy()
    for neuron in range(matrix.shape[0]):
        matrix[neuron, np.where(matrix[neuron, :] <= (2 * median_abs_deviation(matrix[neuron, :])))[0]] = 0
    return matrix


mouse = Mouse.load(Path("D:\\DEM2"))

experiment = mouse.retrieval

aligned = pd.read_csv(experiment.file_tree.get("results")("aligned"), index_col=0)

burrow = np.zeros_like(aligned["Position (mm)"])
burrow[np.where(aligned["Position (mm)"] >= 5)[0]] = 1
aligned["Burrow"] = burrow

tensor_indices = generate_tensor_indices(aligned)

included_labels = ["CS+", "Trace+", "Response+", "CS-", "Trace-", "Response-", "Valence", "Burrow"]

trial_tensor = build_trial_tensor(tensor_indices, included_labels, aligned)

dfof = np.load(experiment.file_tree.get("results")("dfof"))


# preallocate
taus = []
baselines = []
init_vals = []
noises = []
spikes = []
regs = []
non_neg = dfof.copy()

frame_rate = 30

# actually calculate using constrained foopsi
for neuron in range(non_neg.shape[0]):
    trace, baseline, initial_value, tau, noise, spike_matrix, reg_param = constrained_foopsi(non_neg[neuron, :], p=1)
    non_neg[neuron, :] = trace
    baselines.append(baseline)
    init_vals.append(initial_value)
    taus.append(convert_tau(tau, 1/frame_rate))
    noises.append(noise)
    spikes.append(spike_matrix)
    regs.append(reg_param)

non_neg = non_neg

# spike_matrix = find_spikes(np.vstack(spikes))

# spike_matrix *= 30

neurons = non_neg.shape[0]
frames, trials = tensor_indices.shape

traces = np.empty((1020, 73882))
traces[:, :] = np.nan
traces[:, 3882:] = non_neg.copy()

traces = interpolate_traces(traces, aligned)

trace_tensor = build_trace_tensor(tensor_indices, traces)

time = np.arange(-15000 / 1000, 40000 / 1000, 1 / 1000)
time = np.hstack([time for _ in range(trials)]).T

trial_id = np.vstack([np.ones((1, 1, 55000,)) * x for x in range(trials)])

trial_stack = np.append(trial_tensor, trial_id, axis=1)

trial_stack = np.hstack(trial_stack)

trial_stack = np.vstack([trial_stack, time])

included_labels.append("Trial #")

included_labels.append("Time (ms)")

samples = np.arange(trial_stack.shape[1])

trial_stack = pd.DataFrame(data=trial_stack.T, index=samples, columns=included_labels)

np.save(experiment.file_tree.get("results")().joinpath("decoding\\features.npy"), trace_tensor)
trial_stack.to_parquet(experiment.file_tree.get("results")().joinpath("decoding\\labels.parquet"))
