from __future__ import annotations
from types import MappingProxyType
from typing import Optional, Mapping, Iterable, Union, Any, List, Tuple
from dataclasses import dataclass, Field, field  # use for complex property checks (e.g., range)
from collections import ChainMap
from math import inf
import re

try:
    from inspect import get_annotations
except ImportError:
    from get_annotations import get_annotations  # backport


classifier_parameters = MappingProxyType({
    "dual": False,
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 5000,
})


"""
Self-validating dataclass replicated as-needed with permission from DAO until package publication
"""


@dataclass
class DecodandaParameters:
    cross_validations: int = field(default=10, metadata={"range": (0, inf)})
    debug: bool = False
    exclude_contiguous_trials: bool = False
    exclude_silent: bool = False
    fault_tolerance: bool = False
    max_conditioned_data: int = field(default=0, metadata={"range": (0, inf)})
    min_conditioned_data: int = field(default=10 ** 6, metadata={"range": (0, inf)})
    min_data_per_condition: int = field(default=2, metadata={"range": (1, inf)})
    min_trials_per_condition: int = field(default=2, metadata={"range": (1, inf)})
    min_activations_per_cell: int = field(default=1, metadata={"range": (0, inf)})
    neural_attr: str = "raster"
    non_semantic: bool = False
    n_shuffles: int = field(default=25, metadata={"range": (0, inf)})
    parallel: bool = False
    testing_trials: Optional[list] = None
    training_fraction: float = field(default=0.7, metadata={"range": (0, 1)})
    trial_average: bool = False
    trial_attr: str = "trial"
    trial_chunk: Optional[int] = None
    verbose: bool = False
    zscore: bool = False

    def __post_init__(self):
        """
        Post initialization type-checking for current and child classes
        """
        self._validate_fields()

    @staticmethod
    def _type_check_nested_types(var: Any, expected: str) -> bool:
        """
        Checks type of nested types. WORKS FOR ONLY ONE NEST.

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
    def _validate_range(var: Any, val_range: Tuple[Any, Any]) -> List[Exception]:
        logger = _ParameterLogger()
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
            return vars(cls(**ChainMap(*mappings)))
        elif len(mappings) == 1:
            return vars(cls(**mappings[0]))
        else:
            return vars(cls())

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
        logger = _ParameterLogger()
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
        logger = _ParameterLogger()
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
})


class _ParameterLogger:
    def __init__(self, exceptions: List[Exception] = None):
        self.exceptions = []
        if exceptions:
            self.add_exception(exceptions)

    def __str__(self):
        exceptions = "".join([f"\n{message}" for message in self.exceptions])
        return "".join([f"{_TerminalStyle.YELLOW}", *exceptions, f"{_TerminalStyle.RESET}"])

    def add_exception(self, other: Exception or List[Exception]) -> _ParameterLogger:
        try:
            _ = iter(other)
        except TypeError:
            self.exceptions.append(other)
        else:
            for exception in other:
                self.add_exception(exception)
        self.exceptions = [exception for exception in self.exceptions if exception is not None]

    def raise_exceptions(self) -> _ParameterLogger:
        raise _ParameterException(self.__str__())

    def __add__(self, other: Exception or List[Exception]):
        try:
            _ = iter(other)
        except TypeError:
            self.add_exception(other)
        else:
            for exception in other:
                self.add_exception(exception)
        return _ParameterLogger(self.exceptions)

    def __call__(self, *args, **kwargs):
        self.raise_exceptions()


class _ParameterException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        return


class _TerminalStyle:
    """
    Font styles for printing to terminal
    """
    RED = "\u001b[31m"
    GREEN = "\u001b[32m"
    BLUE = "\u001b[38;5;39m"
    YELLOW = "\u001b[38;5;11m"
    ORANGE = "\u001b[38;5;208m"
    BOLD = "\u001b[1m"
    UNDERLINE = "\u001b[7m"
    RESET = "\033[0m"
