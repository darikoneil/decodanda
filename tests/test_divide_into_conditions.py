import numpy as np
import pytest

from decodanda import Decodanda

# Test the core functionality of the Decodanda class,
#  i.e., dividing the data into conditioned arrays

@pytest.mark.parametrize("data", ["base", "undersampled", "tangled"], indirect=["data"])
def test_division_into_conditioned_data(data) -> None:
    d = Decodanda(data=data.data,
                  conditions=dict(data.conditions),
                  verbose=False,
                  min_data_per_condition=0,
                  min_trials_per_condition=0)
    max_features_expected = data.num_neurons
    max_samples_expected = data.num_samples
    num_features = {raster[0].shape[1] for raster in d.conditioned_rasters.values()}
    if len(num_features) > 1:
        raise AssertionError("The number of features should be the same for all conditioned data")
    num_features = num_features.pop()
    num_samples = sum(raster[0].shape[0] for raster in d.conditioned_rasters.values())
    try:
        assert max_features_expected >= num_features
        assert max_samples_expected >= num_samples
    except AssertionError as exc:
        raise AssertionError("The concatenated shape of conditioned data should be"
                             "less than or equal to the number of features x samples in the dataset") from exc


@pytest.mark.parametrize("data", ["base", "undersampled", "tangled"], indirect=["data"])
def test_division_into_conditioned_trials(data) -> None:
    d = Decodanda(data=data.data,
                  conditions=dict(data.conditions),
                  verbose=False,
                  min_data_per_condition=0,
                  min_trials_per_condition=0)
    max_trials_expected = data.num_trials
    num_trials = len(np.concatenate([np.unique(trial) for trials in d.conditioned_trial_index.values()
                                     for trial in trials]))
    assert (max_trials_expected >= num_trials), \
        "The total number of trial index should be less than or equal to the number of trials in the dataset"
