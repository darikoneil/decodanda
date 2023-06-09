from __future__ import annotations
from types import MappingProxyType
from dataclasses import dataclass, field


classifier_parameters = MappingProxyType({
    "dual": False,
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 5000,
})


@dataclass
class Parameters:
    min_data_per_condition: int = field(default=2)

    def __post_init__(self):
        """
        Post initialization type-checking for child classes
        """
        validate_fields(self)

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
        return "Parameters"
