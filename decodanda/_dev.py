from __future__ import annotations
from inspect import stack


def identify_calling_function() -> str:
    """
    Function returns the name of the function that called it

    """
    return f"[{stack()[1][3]}]"
