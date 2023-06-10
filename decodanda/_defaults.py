from __future__ import annotations
from types import MappingProxyType
from typing import Optional, Mapping
from dataclasses import dataclass, field
from collections import ChainMap


classifier_parameters = MappingProxyType({
    "dual": False,
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 5000,
})


@dataclass
class DecodandaParameters:
    debug: bool = False
    exclude_contiguous_trials: bool = False
    exclude_silent: bool = False
    fault_tolerance: bool = False
    min_data_per_condition: int = 2
    min_trials_per_condition: int = 2
    min_activations_per_cell: int = 1
    neural_attr: str = "raster"
    trial_average: bool = False
    trial_attr: str = "trial"
    trial_chunk: Optional[int] = None
    verbose: bool = False
    zscore: bool = False

    @classmethod
    def build(cls, mappings):
        mappings = [mapping for mapping in mappings if isinstance(mapping, Mapping)]
        if len(mappings) > 1:
            return vars(cls(**ChainMap(mappings)))
        else:
            return vars(cls(**mappings[0]))

    def __post_init__(self):
        """
        Post initialization type-checking for child classes
        """
        # validate_fields(self)

    def __setattr__(self, key: str, value: Any) -> ConfigTemplate:
        """
        Override of magic method to provide type-checking whenever a configuration is edited

        :param key: key of the configuration parameter to be changed
        :param value: value of the configuration parameter to be changed

        """
        self.__dict__[key] = value
        self.__post_init__()

    def _format_fields(self) -> tuple:
        return tuple([(key, self.__dict__.get(key), self.__dataclass_fields__.get(key))
                      for key in sorted(self.__dataclass_fields__)])

    def __name__(self):
        return "Decodanda Parameters"
