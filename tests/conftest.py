from dataclasses import dataclass
from functools import singledispatchmethod, partial
from types import MappingProxyType
from typing import Any, Callable, Iterator, Optional
import numpy as np
from decodanda import generate_synthetic_data, Decodanda, z_pval
import pytest
from typing import Protocol, runtime_checkable
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from itertools import product


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Structural Subtype (Classifier)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


# noinspection PyPep8Naming
@runtime_checkable
class Classifier(Protocol):
    """
    Structural subtype for classifier implementations. All classifier implementations must implement the following
    methods. These methods must adhere to the designated return type, and (at minimum) accept parameters in the form of
    the specified types.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional = None) -> None:
        """
        Method that fits the classifier to the provided data, where X is the data and y is the target labels.
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Method that predicts the class labels for the provided data (X)
        """
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Method that returns the balanced accuracy of the classifier on the provided data (X) and target labels (y).
        """
        ...


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Encapsulations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


@dataclass(frozen=True)
class Dataset:
    """
    Encapsulation of a test dataset with specific characteristics for providing to a Decodanda constructor
    """
    #: the conditions dictionary containing the condition names and their values (read only)
    conditions: MappingProxyType[str, list[int | str]]
    #: the data dictionary containing the raster, trial, and condition information (read only)
    data: MappingProxyType[str, list | np.ndarray]
    #: linearly separable conditions
    linearly_separable_dichotomies: list[str | None]
    #: non-linearly separable conditions
    non_linearly_separable_dichotomies: list[str | None]
    #: identifier for this dataset
    key: str

    @property
    def num_conditions(self) -> int:
        """
        The number of conditions in the dataset
        """
        return len(self.conditions)

    @property
    def num_neurons(self) -> int:
        """
        The number of neurons in the dataset
        """
        return self.data.get("raster").shape[-1]

    @property
    def num_samples(self) -> int:
        """
        The number of samples in the dataset
        """
        return self.data.get("raster").shape[0]

    @property
    def num_trials(self) -> int:
        """
        The number of trials in the dataset
        """
        return len(set(self.data.get("trial").tolist()))


@dataclass(frozen=True)
class Result:
    """
    Encapsulation of the results for one dichotomy
    """
    #: the specific dichotomy tested
    dichotomy: str
    #: the balanced accuracy for the cross-validated classifier
    performance: float | list | np.ndarray
    #: the balanced accuracy for each null model
    null: np.ndarray
    #: the p-value of the classifier performance
    pval: float
    #: the z-value of the classifier performance
    zval: float


@dataclass(frozen=False)
class Results:
    """
    Encapsulation of the results for multiple dichotomies
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

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except (AttributeError, TypeError):
            return self.results.get(item)

    def __iter__(self) -> Iterator[tuple[str, Result]]:
        return iter(self.results.items())


@dataclass(frozen=False)
class DecodandaTestCase:
    """
    Encapsulation of one test case. Contains a test dataset with specific characteristics, a decodanda object, the
    dichotomies to be tested, the results of the test, and the parameters used in constructing the decodanda object.
    """
    #: the classifier implementation for the test case
    classifier: partial
    #: the test dataset associated with this test case
    dataset: Dataset
    #: the decodanda object associated with this test case
    decodanda: Decodanda | None
    #: any parameters used in constructing the decodanda object (read only)
    parameters: MappingProxyType[str, Any]
    #: the results associated with each dichotomy
    results: Results


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Registries
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


class ClassifierRegistry:
    """
    Registry of classifier implementations with preset parameters. This implementation ensures that classifiers are
    always called with the exact same parameters during testing while simultaneously providing readable access to the
    classifier implementations (e.g., calling "liblinear" instead of LinearSVC(...)). We specifically retrieve the
    classifier's constructor and its desired parameters by binding them together as a partial function on a "get" call.
    By doing so, we can easily access a classifier that is both uninitialized and guaranteed to have the same
    parameters across multiple test cases.
    """
    #: registry of classifier implementations
    __registry = {}

    @classmethod
    def get(cls, key: str) -> partial:
        if key in cls.__registry:
            constructor, parameters = cls.__registry.get(key).__call__()
            # return the classifier's constructor
            return partial(constructor, **parameters)
        else:
            raise KeyError(f"Classifier with key '{key}' not found in registry.")

    @classmethod
    def register(cls, alias: str  = None):  # noqa: ANN206
        """
        Register a specific classifier implementation for subsequent access by test functions.
        """
        def register_classifier(fixture_function):  # noqa: ANN206, ANN001, ANN201
            """
            Registers the classifier constructor in the registry.
            """
            nonlocal alias
            alias = alias if alias else fixture_function.__name__.split("_")[0]
            cls.__registry[alias] = fixture_function

        return register_classifier


class DatasetRegistry:
    """
    Registry of test datasets with specific characteristics. Initially, the registry maps the dataset key to the
    dataset constructor. The dataset constructor is called ONLY the first time the dataset is requested by a test
    function. Thereafter, the dataset is stored in the registry for subsequent access by other test functions.
    """
    #: registry of dataset constructors
    __registry = {}

    @classmethod
    def get(cls, key: str) -> Dataset:
        """
        Get the test datatest with specific characteristics using its key.
        """
        if key in cls.__registry:
            # if the dataset is not already constructed, construct it
            if not isinstance(cls.__registry.get(key), Dataset):
               cls.__registry[key] = cls.__registry.get(key).__call__()
            # return the constructed dataset
            return cls.__registry.get(key)
        else:
            raise KeyError(f"Dataset key '{key}' not found in registry.")

    @classmethod
    def register(cls, alias: Optional[str] = None):  # noqa: ANN206
        """
        Register a specific test dataset for subsequent access by test functions.
        """
        def register_data(constructor: Callable):  # noqa: ANN206, ANN001, ANN201
            """
            Registers the synthetic data constructor in the registry.
            """
            nonlocal alias
            alias = alias if alias else constructor.__name__
            cls.__registry[alias] = constructor

        return register_data


""" 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test Datasets
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


@DatasetRegistry.register()
def base_dataset() -> Dataset:
    """
    Standard, 'base' dataset for testing containing two conditions (a_condition & b_condition).
    Its identifier is 'base_data'.
    """
    data = generate_synthetic_data(n_neurons=80,
                                   n_trials=200,
                                   keyA="a_condition",
                                   rateA=0.3,
                                   keyB="b_condition",
                                   rateB=0.3,
                                   timebins_per_trial=50,
                                   corrAB=0.0,
                                   scale=1.0,
                                   meanfr=0.1,
                                   mixed_term=0.0,
                                   mixing_factor=0.0
                                   )
    return Dataset(
        conditions=MappingProxyType({"a_condition": [-1, 1], "b_condition": [-1, 1]}),
        linearly_separable_dichotomies=["a_condition", "b_condition"],
        non_linearly_separable_dichotomies=["XOR", ],
        data=MappingProxyType(data),
        key="base_dataset",
    )


@DatasetRegistry.register()
def undersampled_dataset() -> Dataset:
    """
    Undersampled dataset for testing. In contains many more neuronal features than samples. It contains two conditions
    (a_condition & b_condition). Its identifier is 'undersampled_data'.
    """
    data = generate_synthetic_data(n_neurons=80,
                                   n_trials=8,
                                   keyA="a_condition",
                                   rateA=0.3,
                                   keyB="b_condition",
                                   rateB=0.3,
                                   timebins_per_trial=1,
                                   corrAB=0.0,
                                   scale=1.0,
                                   meanfr=0.1,
                                   mixed_term=0.0,
                                   mixing_factor=0.0
                                   )
    return Dataset(
        conditions=MappingProxyType({"a_condition": [-1, 1], "b_condition": [-1, 1]}),
        linearly_separable_dichotomies=["a_condition", "b_condition"],
        non_linearly_separable_dichotomies=["XOR", ],
        data=MappingProxyType(data),
        key="undersampled_dataset",
    )


@DatasetRegistry.register()
def correlated_dataset() -> Dataset:
    """
    Dataset containing a high correlation between two conditions (a_condition & b_condition). The b_condition is a
    confounding condition. Its identifier is 'correlated_data'.
    """
    data = generate_synthetic_data(n_neurons=80,
                                   n_trials=100,
                                   keyA="a_condition",
                                   rateA=0.3,
                                   keyB="b_condition",
                                   rateB=0.0,
                                   timebins_per_trial=5,
                                   corrAB=0.8,
                                   scale=1.0,
                                   meanfr=0.1,
                                   mixed_term=0.0,
                                   mixing_factor=0.0
                                   )
    return Dataset(
        conditions=MappingProxyType({"a_condition": [-1, 1], "b_condition": [-1, 1]}),
        data=MappingProxyType(data),
        linearly_separable_dichotomies=["a_condition"],
        non_linearly_separable_dichotomies=[None, ],
        key="correlated_dataset",
    )


@DatasetRegistry.register()
def random_dataset() -> Dataset:
    """
    Dataset containing random activity. It contains two conditions (a_condition & b_condition).
    Its identifier is 'random_data'.
    """
    data = generate_synthetic_data(n_neurons=80,
                                   n_trials=100,
                                   keyA="a_condition",
                                   rateA=0.0,
                                   keyB="b_condition",
                                   rateB=0.0,
                                   timebins_per_trial=5,
                                   corrAB=0.0,
                                   scale=1.0,
                                   meanfr=0.25,
                                   mixed_term=0.0,
                                   mixing_factor=0.0
                                   )
    return Dataset(
        conditions=MappingProxyType({"a_condition": [-1, 1], "b_condition": [-1, 1]}),
        data=MappingProxyType(data),
        linearly_separable_dichotomies=[None, ],
        non_linearly_separable_dichotomies=[None, ],
        key="random_dataset",
    )


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Classifier Implementations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


@ClassifierRegistry.register()
def liblinear() -> tuple[Classifier, dict[str, Any]]:
    """
    Partial binding the constructor for a liblinear implementation of a support vector classifier. The
    classifier uses a squared hinge loss function, primal optimization, a regularized intercept term, L2 regularization
    term, and balanced class weight.
    """
    # noinspection PyTypeChecker
    return LinearSVC, {"dual": False, "C": 1.0, "class_weight": "balanced", "max_iter": 5000}


@ClassifierRegistry.register()
def libsvm() -> tuple[Classifier, dict[str, Any]]:
    """
    Partial binding the constructor for a libsvm implementation of a support vector classifier with a linear kernel.
    The classifier uses the hinge loss and balanced class weight. This implementation is slightly different than the
    liblinear implementation. Most notably, it uses a hinge loss function and does not regularize the intercept.
    """
    # noinspection PyTypeChecker
    return SVC, {"C": 1.0, "kernel": "linear", "class_weight": "balanced", "max_iter": 5000}


@ClassifierRegistry.register()
def largeliblinear() -> tuple[Classifier, dict[str, Any]]:
    """
    Partial binding the constructor for a 'near-identical' implementation of the liblinear classifier that avoids
    making a memory copy of the data. This implementation is recommended by sklearn as an alternative to the liblinear
    when in a memory-constrained environment.
    """
    # noinspection PyTypeChecker
    return SGDClassifier, {"alpha": 1.0,
                           "class_weight": "balanced",
                           "max_iter": 5000,
                           "loss": "squared_hinge",
                           "penalty": "l2",
                            "n_jobs": 1,
                           }


@ClassifierRegistry.register()
def rbf() -> tuple[Classifier, dict[str, Any]]:
    """
    Partial binding the constructor for a support vector classifier with an RBF kernel. The classifier uses a squared
    hinge loss function, primal optimization, squared-L2 regularization term, and balanced class
    weight.
    """
    # noinspection PyTypeChecker
    return SVC, {"C": 1.0, "kernel": "rbf", "class_weight": "balanced", "max_iter": 5000}


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fixtures
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


@pytest.fixture(scope="session")
def initialization_parameters() -> MappingProxyType[str, Any]:
    """
    Standard parameters for initializing a Decodanda object EXCLUDING the data, conditions, and classifier.
    """
    return MappingProxyType({
        "neural_attr": "raster",
        "trial_attr": "trial",
        "squeeze_trials": False,
        "min_data_per_condition": 2,
        "min_trials_per_condition": 1,
        "min_activations_per_cell": 1,
        "trial_chunk": None,
        "exclude_contiguous_chunks": False,
        "exclude_silent": False,
        "verbose": False,
        "zscore": False,
        "fault_tolerance": False,
        "debug": False,
    })


@pytest.fixture(scope="session")
def decoding_parameters() -> MappingProxyType[str, Any]:
    return MappingProxyType({
        "training_fraction": 0.75,
        "cross_validations": 10,
        "nshuffles": 10,
        "ndata": 100,
    })


@pytest.fixture(scope="session")
def ccgp_parameters() -> MappingProxyType[str, Any]:
    ...


@pytest.fixture(scope="session")
def ps_parameters() -> MappingProxyType[str, Any]:
    ...


@pytest.fixture(scope="class")
def decoding_test_case(request, initialization_parameters, decoding_parameters) -> DecodandaTestCase:
    """
    Fixture for generating test cases for decoding with a specific dataset and classifier.
    """
    dataset_key = request.param[0]
    classifier_key = request.param[1]
    test_case = DecodandaTestCase(dataset=DatasetRegistry.get(dataset_key),
                                  decodanda=None,
                                  classifier=ClassifierRegistry.get(classifier_key),
                                  parameters=initialization_parameters,
                                  results=Results())
    decodanda = Decodanda(data=dict(test_case.dataset.data),
                          conditions=dict(test_case.dataset.conditions),
                          **initialization_parameters)
    # TRAIN SEMANTIC DICHOTOMIES
    for key, value in decodanda.all_dichotomies(balanced=True, semantic_names=True).items():
        result, null = decodanda.decode_with_nullmodel(value, **decoding_parameters)
        zval, pval = z_pval(result, null)
        test_case.results.add(Result(
            dichotomy=key,
            performance=result,
            null=null,
            pval=pval,
            zval=zval,
        ))
    # TRAIN NON-SEMANTIC DICHOTOMIES
    for key, value in decodanda.all_dichotomies(balanced=True, semantic_names=False).items():
        result, null = decodanda.decode_with_nullmodel(value, **decoding_parameters)
        zval, pval = z_pval(result, null)
        test_case.results.add(Result(
            dichotomy=key,
            performance=result,
            null=null,
            pval=pval,
            zval=zval,
        ))
    # ADD THE DECODANDA OBJECT TO THE TEST CASE
    test_case.decodanda = decodanda
    return test_case


@pytest.fixture(scope="class")
def ccgp_test_case(request, initialization_parameters, ccgp_parameters):
    ...


@pytest.fixture(scope="class")
def ps_test_case(request, initialization_parameters, ps_parameters):
    ...


"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dynamic Test Generation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


def _parse_id(value: Any) -> str:
    """
    Parse the id for the dynamically generated test cases.
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, tuple):
        return f"{value[0]}_{value[1]}"
    else:
        return f"{value}"


def pytest_generate_tests(metafunc):
    """
    Dynamically generate test cases for various test suites.
    """
    if "decoding_test_case" in metafunc.fixturenames:
        datasets = ["base_dataset", "correlated_dataset", "random_dataset"] #"undersampled_dataset"]
        classifiers = ["liblinear", "libsvm", "largeliblinear", "rbf"]
        metafunc.parametrize("decoding_test_case", list(product(datasets, classifiers)), indirect=True, ids=_parse_id)
