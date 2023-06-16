from __future__ import annotations
from types import MappingProxyType
from typing import Mapping, Iterable, Union, Any, List, Tuple
from dataclasses import dataclass, Field, field  # use for complex property checks (e.g., range)
from collections import ChainMap
from math import inf
import re

from sklearn.base import BaseEstimator
import numpy as np


# SET VRAM LIMIT FOR THUNDER
from . import NVIDIA_SMI
from ._dev import determine_gpu_memory, MultiExceptionLogger, ParameterException, TerminalStyle
if NVIDIA_SMI:
    GPU_MEMORY = determine_gpu_memory(0)
else:
    GPU_MEMORY = 0


# BACKPORT INSPECTIONS IF NECESSARY
try:
    from inspect import get_annotations
except ImportError:
    from get_annotations import get_annotations


classifier_parameters_thunder = MappingProxyType({
    "C": 1.0,
    "class_weight": "balanced",
    "gpu_id": 0,
    "kernel": "linear",
    "max_iter": 5000,
    "max_mem_size": GPU_MEMORY,
    "n_jobs": -1,
    "probability": False,
    "shrinking": False,
    "tol": 1e-4,
})


classifier_parameters_intel = MappingProxyType({
    "C": 1.0,
    "class_weight": "balanced",
    "kernel": "linear",
    "max_iter": 5000,
    "probability": False,
    "shrinking": False,
    "tol": 1e-4,
})


classifier_parameters = MappingProxyType({
    "C": 1.0,
    "class_weight": "balanced",
    "dual": False,
    "max_iter": 5000,
    "tol": 1e-4
})


"""
Self-validating dataclass replicated as-needed with permission from DAO's unreleased package
"""


@dataclass
class DecodandaParameters:
    cross_validations: int = field(default=10, metadata={"range": (0, inf)})
    debug: bool = False
    exclude_contiguous_trials: bool = False
    exclude_silent: bool = False
    fault_tolerance: bool = False
    # label_shuffling: str = field(default="bin", metadata={"permitted": ("bin", "condition", "shift")})
    max_conditioned_data: np.int32 = field(default=np.int32(0), metadata={"range": (0, inf)})
    min_conditioned_data: np.int32 = field(default=np.int32(10 ** 6), metadata={"range": (0, inf)})
    min_data_per_condition: int = field(default=2, metadata={"range": (1, inf)})
    min_trials_per_condition: int = field(default=2, metadata={"range": (1, inf)})
    min_activations_per_cell: int = field(default=1, metadata={"range": (0, inf)})
    neural_attr: str = "raster"
    non_semantic: bool = False
    n_shuffles: int = field(default=25, metadata={"range": (0, inf)})
    parallel: bool = False
    scale: BaseEstimator = None
    trial_average: bool = False
    trial_attr: str = "trial"
    trial_chunk: int = None
    testing_trials: list = None
    training_fraction: float = field(default=0.7, metadata={"range": (0, 1)})
    verbose: bool = False

    def __post_init__(self):
        """
        Post initialization type-checking for current and child classes
        """
        self._validate_fields()

    def __str__(self):
        """
        Verbose printing of the decodanda parameters
        """
        parameters = f"\n{TerminalStyle.BOLD}{TerminalStyle.UNDERLINE}{TerminalStyle.YELLOW}" \
                     f"{self.__name__()}{TerminalStyle.RESET}\n"
        annotations_ = self._collect_annotations()

        for key, type_ in annotations_.items():
            parameters += f"{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{key}: " \
                          f"{TerminalStyle.RESET}{self.__dict__.get(key)}" \
                          f"{TerminalStyle.BLUE} ({type_}){TerminalStyle.RESET}\n"
        return parameters

    @staticmethod
    def _type_check_nested_types(var: Any, expected: str) -> bool:
        """
        Checks type of nested types. WORKS FOR ONLY ONE NEST in this implementation.
        Doesn't work for union/optional in some edge cases.

        :param var: variable to check
        :param expected: expected type
        :returns: boolean type comparison
        """
        try:
            if isinstance(var, (MappingProxyType, dict)):
                pass
            _ = iter(var)
        except TypeError:
            return isinstance(var, eval(expected))
        else:
            # separate out the nested types
            expected = re.split(r"[\[\],]", expected)[:-1]
            outer_type = isinstance(var, eval(expected[0]))
            expected.pop(0)

            var_list = list(var)

            # if the provided types aren't equal to number of items in var, then fill out with last type
            if len(var_list) != len(expected):
                while len(var_list) != len(expected):
                    expected.append(expected[-1])
            try:
                assert (len(var.keys()) >= 1)
                checks = [isinstance(nested_var, eval(nested_type))
                          for nested_var, nested_type in zip(var.values(), expected)]
                checks.append(outer_type)
            except AttributeError:
                checks = [isinstance(nested_var, eval(nested_type))
                          for nested_var, nested_type in zip(var, expected)]
                checks.append(outer_type)
            return all(checks)

    @staticmethod
    def _validate_permitted(var: Any, values: Tuple[Any]) -> List[Exception]:
        logger = MultiExceptionLogger()
        try:
            _ = iter(var)
        except TypeError:
            if var not in values:
                return [AssertionError(f"{var} not permitted value")]
        else:
            for val in var:
                try:
                    assert (val in values), f"{val} not permitted value"
                except AssertionError as e:
                    logger += e
            return logger.exceptions

    @staticmethod
    def _validate_range(var: Any, val_range: Tuple[Any, Any]) -> List[Exception]:
        logger = MultiExceptionLogger()
        val_min, val_max = val_range
        try:
            _ = iter(var)
        except TypeError:
            if not val_min <= var <= val_max:
                return [AssertionError("Range")]
        else:
            for val in var:
                try:
                    assert (val_min <= val <= val_max), "Range"
                except AssertionError as e:
                    logger += e
            return logger.exceptions

    @classmethod
    def build(cls: DecodandaParameters, mappings: Union[Mapping, Iterable[Mapping]]) -> dict:
        """
        Builds validated dictionary

        :param mappings: mapping/s to update dataclass values with
        :returns: dictionary with validated types & properties
        """
        mappings = [mapping for mapping in mappings if isinstance(mapping, Mapping)]
        if len(mappings) > 1:
            return cls(**ChainMap(*mappings))
        elif len(mappings) == 1:
            return cls(**mappings[0])
        else:
            return cls()

    @classmethod
    def _collect_annotations(cls: DecodandaParameters) -> dict:
        """

        Collects annotations from all parent classes for type-checking
        :returns: a dictionary containing key-type pairs
        """
        return dict(ChainMap(*(get_annotations(cls_) for cls_ in cls.__mro__)))
    # It feels dumb to wrap with dict but for whatever reason it seems to be ordering it which is nice

    @classmethod
    def _field_validator(cls: DecodandaParameters, key: str, value: Any, var: Field) -> List[Exception]:
        """
        Static method for validating one field

        """
        if value is None:
            return
        logger = MultiExceptionLogger()
        type_ = var.type
        # first check type always
        try:
            # Type Check
            if not isinstance(value, eval(type_)):
                raise AttributeError(f"Field {key} must be type {type_} not {type(value).__name__}")
        except TypeError:
            # Type Check
            if not cls._type_check_nested_types(value, str(type_)):
                raise AttributeError(f"Field {key} must be type {type_} not {type(value).__name__}")
        except AttributeError as e:
            logger += e
            return logger.exceptions
            # short-circuit to prevent shenanigans on the field validators which are type specific

        # now use validators on metadata
        meta = var.metadata

        for key in meta.keys():
            if key in META_VALIDATORS.keys():
                e = META_VALIDATORS.get(key)(value, meta.get(key))
                if e:
                    logger += e

        return logger.exceptions

    def hint_types(self) -> None:
        """
        Verbose printing of type information to assist users in setting decodanda parameters
        """
        type_hints = f"\n{TerminalStyle.BOLD}{TerminalStyle.YELLOW}{TerminalStyle.UNDERLINE}" \
                     f"{self.__name__()}{TerminalStyle.RESET}\n"

        annotations_ = self._collect_annotations()

        for key, type_ in annotations_.items():
            type_hints += f"{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{key}: {TerminalStyle.RESET}" \
                          f"{TerminalStyle.BLUE}{type_}{TerminalStyle.RESET}\n"

        print(type_hints)

    def _format_fields(self) -> tuple:
        """
        Format fields for validation

        :returns: fields organized in tuple for validation
        """
        return tuple([(key, vars(self).get(key), self.__dataclass_fields__.get(key))
                      for key in sorted(self.__dataclass_fields__)])

    def _validate_fields(self) -> bool:
        """
        Primary method for field validation

        :returns: True if valid
        """
        logger = MultiExceptionLogger(exception_class=ParameterException)
        fields_tuple = self._format_fields()
        for key_, value_, field_ in fields_tuple:
            logger.add_exception(self._field_validator(key=key_, value=value_, var=field_))

        if logger.exceptions:
            logger.raise_exceptions()

    def __setattr__(self, key: str, value: Any):
        """
        Override of magic method to provide type-checking whenever the dataclass is edited

        :param key: key of the configuration parameter to be changed
        :param value: value of the configuration parameter to be changed

        """
        vars(self)[key] = value
        self.__post_init__()

    def __name__(self):
        return "Decodanda Parameters"


# noinspection PyProtectedMember
META_VALIDATORS = MappingProxyType({
    "range": DecodandaParameters._validate_range,
    "permitted": DecodandaParameters._validate_permitted,
})
