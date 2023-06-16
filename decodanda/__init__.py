__all__ = [
    "Decodanda",
    "DecodandaParameters",
    "visualize",
]

# FLAG SCIKIT-LEARN-INTELX
try:
    import sklearnex as _sklearnex  # noqa: F401
    INTELX = True
except ImportError:
    INTELX = False


# FLAG NVIDIA-SMI
try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    assert (_pynvml.nvmlDeviceGetCount() > 0)
    NVIDIA_SMI = True
    _pynvml.nvmlShutdown()
except (ImportError, _pynvml.NVMLError, AssertionError):
    NVIDIA_SMI = False


from .classes import Decodanda  # noqa: F401
from .defaults import DecodandaParameters  # noqa: F401
