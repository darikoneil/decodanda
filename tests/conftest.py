from dataclasses import dataclass
from functools import singledispatchmethod
from types import MappingProxyType
from typing import Any, Callable, Iterator, Optional

import numpy as np
import pytest
from sklearn.svm import SVC, LinearSVC

from decodanda import Decodanda, generate_synthetic_data, z_pval

"""
This module provides encapsulated containers and fixtures for testing the decodanda package. 
The fixtures are designed to easily provide synthetic data, trained decoders, and  results for testing.
Once generated, the data, decoders, and results are stored in a registry for efficient access by downstream tests.
"""


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Encapsulation of Test Data, Test Results, and the associated Trained Decoders
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


@dataclass(frozen=True)
class Data:
    """
    Encapsulation of synthetic datasets for testing.
    """
    conditions: MappingProxyType[str, list[int | str]]
    data: dict[str, list | np.ndarray]
    key: str
    num_neurons: int
    num_trials: int
    num_samples: int
    num_conditions: int


@dataclass(frozen=True)
class Result:
    """
    Encapsulation of decoding results for providing to test functions.

    """
    dichotomy: str
    performance: float
    null: np.ndarray
    pval: float
    zval: float


class Results:
    """
    Encapsulation of decoding results for providing to test functions. Each dichotomy is separately stored in the
    internal dictionary
    """

    def __init__(self, results: Optional[Result | list[Result]] = None):
        self.results = {}
        if results:
            self.add(results)

    @singledispatchmethod
    def add(self, results: Result | list[Result]) -> None:
        raise TypeError("Invalid type for results")

    @add.register(dict)
    def _(self, result: Result) -> None:
        self.results[result.dichotomy] = result

    @add.register(Result)
    def _(self, result: Result) -> None:
        self.results[result.dichotomy] = result

    @add.register(list)
    def _(self, results: list[Result]) -> None:
        for result in results:
            self.add(result)

    @property
    def dichotomies(self) -> list[str]:
        return list(self.results.keys())

    @property
    def performance(self) -> list[float]:
        return [result.performance for result in self.results.values()]

    @property
    def pvalues(self) -> list[float]:
        return [result.pval for result in self.results.values()]

    @property
    def zvalues(self) -> list[float]:
        return [result.zval for result in self.results.values()]

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except (AttributeError, TypeError):
            return self.results.get(item)

    def __iter__(self) -> Iterator[tuple[str, Result]]:
        return iter(self.results.items())


@dataclass
class TrainedDecodanda:
    """
    Encapsulation of a trained decoder for providing to test functions.
    """
    data: Data
    decoder: Decodanda
    dichotomies: dict[str, list]
    key: tuple[str, str]  # data_key, decoder_key
    parameters: dict[str, Any]
    results: Results
    trained: bool = False


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Registries for provided efficient access to synthetic data, trained decoders, and results
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


class DataRegistry:
    """
    Persistent registry of synthetic data to be accessed by test functions without
    the need to regenerate data for each test.
    """
    # persistent registry of synthetic data
    __registry = {}

    @classmethod
    def get(cls, key: str) -> Data:
        if key in cls.__registry:
            if not isinstance(cls.__registry.get(key), Data):
               cls.__registry[key] = cls.__registry.get(key)()
            return cls.__registry.get(key)
        else:
            raise KeyError(f"Data key '{key}' not found in registry.")

    @classmethod
    def register(cls, alias: Optional[str] = None):  # noqa: ANN206
        """
        Register a synthetic dataset for testing.
        """
        def register_data(constructor: Callable):  # noqa: ANN206, ANN001, ANN201
            nonlocal alias
            alias = alias if alias else constructor.__name__.split("_")[0]
            cls.__registry[alias] = constructor

        return register_data


class DecodandaRegistry:
    """
    Registry of decoder constructors to be accessed by the trained decoder registry (not the test functions).
    """
    # persistent registry of decoder constructors
    __registry = {}

    @classmethod
    def get(cls, key: str, data: Data) -> TrainedDecodanda:
        if key in cls.__registry:
            return cls.__registry.get(key)(data)
        else:
            raise KeyError(f"Decoder key '{key}' not found in registry.")

    @classmethod
    def register(cls, alias: Optional[str] = None):  # noqa: ANN206
        """
        Register a synthetic dataset for testing.
        """
        def register_data(constructor: Callable):  # noqa: ANN206, ANN001, ANN201
            nonlocal alias
            alias = alias if alias else constructor.__name__.split("_")[0]
            cls.__registry[alias] = constructor

        return register_data


class TrainedDecodandaRegistry:
    """
    Persistent registry of trained decoders to be accessed by test functions without
    the need to retrain decoders for each test.
    """
    # persistent registry of trained decoders
    __registry = {}

    @classmethod
    def get(cls, keys: tuple[str, str]) -> None:
        if keys not in cls.__registry:
            data_key, decoder_key = keys
            data = DataRegistry.get(data_key)
            decoder = DecodandaRegistry.get(decoder_key, data)
            cls._train(decoder)
            cls.__registry[keys] = decoder

        return cls.__registry.get(keys)


    @staticmethod
    def _train(decoder: TrainedDecodanda) -> None:

        if decoder.trained:  # safeguard double-training
            return

        decoder_ = decoder.decoder

        # noinspection PyProtectedMember
        for key_, values_ in decoder.dichotomies.items():
            result_, null_ = decoder_.decode_with_nullmodel(values_, **decoder.parameters)
            decoder.results.add(Result(**{"dichotomy": key_,
                         "performance": result_,
                         "null": null_,
                         "pval": z_pval(result_, null_)[1],
                         "zval": z_pval(result_, null_)[0]
                        }))

        # make super sure the decoder is updated because I don't know if there's a reference guarantee
        decoder.decoder = decoder_
        decoder.trained = True


""" 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test Cases (Data)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


# noinspection DuplicatedCode
@DataRegistry.register()
def base_data() -> Data:
    """
    Base dataset for testing.
    """
    data = generate_synthetic_data(n_neurons=80,
                                   n_trials=200,
                                   keyA="stimulus",
                                   rateA=0.3,
                                   keyB="action",
                                   rateB=0.3,
                                   timebins_per_trial=50,
                                   corrAB=0.0,
                                   scale=1.0,
                                   meanfr=0.1,
                                   mixed_term=0.0,
                                   mixing_factor=0.0
                                   )
    return Data(
        conditions=MappingProxyType({"stimulus": [-1, 1], "action": [-1, 1]}),
        data=data,
        key="base",
        num_neurons=data.get("raster").shape[-1],
        num_trials=len(set(data.get("trial").tolist())),
        num_samples=data.get("raster").shape[0],
        num_conditions=2,
    )


# noinspection DuplicatedCode
@DataRegistry.register()
def undersampled_data() -> Data:
    """
    An undersampled dataset for testing.
    """
    data = generate_synthetic_data(n_neurons=80,
                                   n_trials=8,
                                   keyA="stimulus",
                                   rateA=0.3,
                                   keyB="action",
                                   rateB=0.3,
                                   timebins_per_trial=1,
                                   corrAB=0.0,
                                   scale=1.0,
                                   meanfr=0.1,
                                   mixed_term=0.0,
                                   mixing_factor=0.0
                                   )
    return Data(
        conditions=MappingProxyType({"stimulus": [-1, 1], "action": [-1, 1]}),
        data=data,
        key="undersampled",
        num_neurons=data.get("raster").shape[-1],
        num_trials=len(set(data.get("trial").tolist())),
        num_samples=data.get("raster").shape[0],
        num_conditions=2,
    )


# noinspection DuplicatedCode
@DataRegistry.register()
def tangled_data() -> Data:
    """
    A dataset with tangled conditions for testing.
    """
    data = generate_synthetic_data(n_neurons=80,
                                   n_trials=100,
                                   keyA="stimulus",
                                   rateA=0.3,
                                   keyB="action",
                                   rateB=0.0,
                                   timebins_per_trial=5,
                                   corrAB=0.8,
                                   scale=1.0,
                                   meanfr=0.1,
                                   mixed_term=0.0,
                                   mixing_factor=0.0
                                   )
    return Data(
        conditions=MappingProxyType({"stimulus": [-1, 1], "action": [-1, 1]}),
        data=data,
        key="tangled",
        num_neurons=data.get("raster").shape[-1],
        num_trials=len(set(data.get("trial").tolist())),
        num_samples=data.get("raster").shape[0],
        num_conditions=2,
    )


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test Cases (Decoders)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


# noinspection DuplicatedCode
@DecodandaRegistry.register()
def base_decoder(data: Data) -> TrainedDecodanda:
    """
    Standard decodanda decoder for testing
    LinearSVC--liblinear implementation, squared hinge loss, primal optimization, regularized intercept term
    """
    init_parameters = {
        "classifier": LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=5000),
        "conditions": data.conditions,
        "verbose": False
    }
    parameters = {
        "ndata": 100,
        "nshuffles": 10,
        "cross_validations": 10,
        "training_fraction": 0.75,
    }
    decoder = Decodanda(data=data.data, **init_parameters)
    return TrainedDecodanda(data=data,
                            decoder=decoder,
                            dichotomies=decoder.all_dichotomies(balanced=True, semantic_names=True),
                            key=(data.key, "base"),
                            results=Results(results=None),
                            trained=False,
                            parameters=parameters)


# noinspection DuplicatedCode
@DecodandaRegistry.register()
def svc_decoder(data: Data) -> TrainedDecodanda:
    """
    Standard linear support vector classifier. SVC--libsvm implementation, hinge loss
    """
    init_parameters = {
        "classifier":SVC(C=1.0, kernel="linear", class_weight="balanced", max_iter=5000),
        "conditions": data.conditions,
        "verbose": False
    }
    parameters = {
        "ndata": 100,
        "nshuffles": 10,
        "cross_validations": 10,
        "training_fraction": 0.75,
    }
    decoder = Decodanda(data=data.data, **init_parameters)
    return TrainedDecodanda(data=data,
                            decoder=decoder,
                            dichotomies=decoder.all_dichotomies(balanced=True, semantic_names=True),
                            key=(data.key, "base"),
                            results=Results(results=None),
                            trained=False,
                            parameters=parameters)


# noinspection DuplicatedCode
@DecodandaRegistry.register()
def nonlinear_decoder(data: Data) -> TrainedDecodanda:
    init_parameters = {
        "classifier": SVC(C=1.0, kernel='poly', degree=3, gamma=2, max_iter=5000),
        "conditions": data.conditions,
        "verbose": False
    }
    parameters = {
        "ndata": 100,
        "nshuffles": 10,
        "cross_validations": 10,
        "training_fraction": 0.75,
    }
    decoder = Decodanda(data=data.data, **init_parameters)
    return TrainedDecodanda(data=data,
                            decoder=decoder,
                            dichotomies=decoder.all_dichotomies(balanced=True, semantic_names=True),
                            key=(data.key, "base"),
                            results=Results(results=None),
                            trained=False,
                            parameters=parameters)


# noinspection DuplicatedCode
@DecodandaRegistry.register()
def unbalancedbase_decoder(data: Data) -> TrainedDecodanda:
    """
    Standard decodanda decoder for testing
    LinearSVC--liblinear implementation, squared hinge loss, primal optimization, regularized intercept term
    """
    unbalanced_conditions = dict(data.conditions)
    unbalanced_conditions.pop("stimulus")

    init_parameters = {
        "classifier": LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=5000),
        "conditions": unbalanced_conditions,
        "verbose": False
    }
    parameters = {
        "ndata": 100,
        "nshuffles": 10,
        "cross_validations": 10,
        "training_fraction": 0.75,
    }
    decoder = Decodanda(data=data.data, **init_parameters)
    return TrainedDecodanda(data=data,
                            decoder=decoder,
                            dichotomies=decoder.all_dichotomies(balanced=False, semantic_names=True),
                            key=(data.key, "base"),
                            results=Results(results=None),
                            trained=False,
                            parameters=parameters)


# noinspection DuplicatedCode
@DecodandaRegistry.register()
def unbalancedsvc_decoder(data: Data) -> TrainedDecodanda:
    """
    Standard linear support vector classifier. SVC--libsvm implementation, hinge loss
    """
    unbalanced_conditions = dict(data.conditions)
    unbalanced_conditions.pop("stimulus")
    init_parameters = {
        "classifier":SVC(C=1.0, kernel="linear", class_weight="balanced", max_iter=5000),
        "conditions": unbalanced_conditions,
        "verbose": False
    }
    parameters = {
        "ndata": 100,
        "nshuffles": 10,
        "cross_validations": 10,
        "training_fraction": 0.75,
    }
    decoder = Decodanda(data=data.data, **init_parameters)
    return TrainedDecodanda(data=data,
                            decoder=decoder,
                            dichotomies=decoder.all_dichotomies(balanced=False, semantic_names=True),
                            key=(data.key, "base"),
                            results=Results(results=None),
                            trained=False,
                            parameters=parameters)


# noinspection DuplicatedCode
@DecodandaRegistry.register()
def unbalancednonlinear_decoder(data: Data) -> TrainedDecodanda:
    unbalanced_conditions = dict(data.conditions)
    unbalanced_conditions.pop("stimulus")
    init_parameters = {
        "classifier": SVC(C=1.0, kernel='poly', degree=3, gamma=2, max_iter=5000),
        "conditions": unbalanced_conditions,
        "verbose": False
    }
    parameters = {
        "ndata": 100,
        "nshuffles": 10,
        "cross_validations": 10,
        "training_fraction": 0.75,
    }
    decoder = Decodanda(data=data.data, **init_parameters)
    return TrainedDecodanda(data=data,
                            decoder=decoder,
                            dichotomies=decoder.all_dichotomies(balanced=False, semantic_names=True),
                            key=(data.key, "base"),
                            results=Results(results=None),
                            trained=False,
                            parameters=parameters)


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fixtures for providing the data, decoders, and results upon requests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


@pytest.fixture(scope="function")
def data(request) -> Data:
    return DataRegistry.get(request.param)


@pytest.fixture(scope="function")
def trained_decoder(request):
    return TrainedDecodandaRegistry.get(request.param)
