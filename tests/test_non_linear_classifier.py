import pytest


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("trained_decoder", [("base", "nonlinear")] , indirect=["trained_decoder"])
def test_non_linear_decoder(trained_decoder) -> None:
    pval = trained_decoder.results.XOR.pval
    assert pval < 0.05, "XOR should able to be decoded with a non-linear classifier in a low-dimensional geometry."

