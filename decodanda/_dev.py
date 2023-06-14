from __future__ import annotations
from typing import List
from inspect import stack


try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
    NVIDIA_SMI = True
except ImportError:
    NVIDIA_SMI = False


try:
    from sklearnex import patch_sklearn
    INTELX = True
except ImportError:
    INTELX = False


def determine_gpu_memory(gpu_index: int = 0) -> int:
    """
    Function returns the amount of VRAM the current GPU
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    vram = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return vram


def identify_calling_function() -> str:
    """
    Function returns the name of the function that called it

    """
    return f"[{stack()[1][3]}]"


