from __future__ import annotations
from inspect import stack


from . import NVIDIA_SMI


if NVIDIA_SMI:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


class MultiExceptionLogger:
    def __init__(self, exception_class: Exception = Exception, exceptions: List[Exception] = None):
        self.exceptions = []
        self.exception_class = exception_class
        if exceptions:
            self.add_exception(exceptions)

    def __str__(self):
        exceptions = "".join([f"\n{message}" for message in self.exceptions])
        return "".join([f"{_TerminalStyle.YELLOW}", *exceptions, f"{_TerminalStyle.RESET}"])

    def add_exception(self, other: Exception or List[Exception]) -> _MultiExceptionLogger:
        try:
            _ = iter(other)
        except TypeError:
            self.exceptions.append(other)
        else:
            for exception in other:
                self.add_exception(exception)
        self.exceptions = [exception for exception in self.exceptions if exception is not None]

    def raise_exceptions(self) -> _MultiExceptionLogger:
        # noinspection PyCallingNonCallable
        raise self.exception_class(self.__str__())

    def __add__(self, other: Exception or List[Exception]):
        try:
            _ = iter(other)
        except TypeError:
            self.add_exception(other)
        else:
            for exception in other:
                self.add_exception(exception)
        return _MultiExceptionLogger(self.exceptions)

    def __call__(self, *args, **kwargs):
        self.raise_exceptions()


class ParameterException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        return


class ClassifierException(Exception):
    def __init__(self, errors: List[Exception]):
        self.errors = errors
        super().__init__(self.errors)
        return


class TerminalStyle:
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


def validate_classifier(classifier: Any) -> None:
    logger = MultiExceptionLogger(exception_class=ClassifierException)
    for method in ["fit", "score"]:
        try:
            assert (hasattr(classifier, method))
        except AssertionError as error:
            logger.add_exception(error)

    if logger.exceptions:
        logger.raise_exceptions()
