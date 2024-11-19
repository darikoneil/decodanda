import pytest

from tests.conftest import trained_decoder


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("trained_decoder", [("base", "base"),
                                              ("base", "svc"),
                                             ("base", "nonlinear")
                                              ], indirect=["trained_decoder"])
def test_variable_dichotomies(trained_decoder) -> None:
    pvals = [result.pval for key, result in iter(trained_decoder.results) if key != "XOR"]
    assert all(pval < 0.05 for pval in pvals), \
        (f"Variable dichotomies should able to be decoded in a low-dimensional geometry,"
         f"{trained_decoder.key=}")


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("trained_decoder", [("base", "base"),
                                              ("base", "svc")
                                              ], indirect=["trained_decoder"])
def test_xor_dichotomies(trained_decoder) -> None:
    pval = trained_decoder.results.XOR.pval
    assert pval > 0.05, \
       (f"XOR dichotomy should not be linearly-decodable in a low-dimensional geometry, "
        f"{trained_decoder.key=}")
