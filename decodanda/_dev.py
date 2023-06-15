from __future__ import annotations
from inspect import stack


from . import NVIDIA_SMI


if NVIDIA_SMI:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def determine_gpu_memory(gpu_index: int = 0) -> int:
    """
    Function returns the amount of VRAM the current GPU
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    vram = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return vram


def identify_calling_function() -> str:
    """
    Function returns the name of the function that called it

    """
    return f"[{stack()[1][3]}]"
