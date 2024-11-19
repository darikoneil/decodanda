import pytest

from tests.conftest import trained_decoder


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("trained_decoder", [("base", "base"),
                                              ("base", "svc"),
                                             ("base", "nonlinear")
                                              ], indirect=["trained_decoder"])
def test_inclusive_or_dichotomies(trained_decoder) -> None:
    pvals = [result.pval for key, result in iter(trained_decoder.results) if key != "XOR"]
    assert all(pval < 0.05 for pval in pvals), \
        (f"All inclusive-or dichotomies should able to be decoded in a low-dimensional geometry,"
         f"{trained_decoder.key=}")


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("trained_decoder", [("base", "base"),
                                              ("base", "svc")
                                              ], indirect=["trained_decoder"])
def test_exclusive_or_dichotomies(trained_decoder) -> None:
    pval = trained_decoder.results.XOR.pval
    assert pval > 0.05, \
       (f"An exclusive-or dichotomy should not be linearly-decodable in a low-dimensional geometry, "
        f"{trained_decoder.key=}")
