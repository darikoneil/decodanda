import pytest


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("trained_decoder", [("tangled", "unbalancedbase"),
                                              ("tangled", "unbalancedsvc"),
                                              ("tangled", "unbalancednonlinear")
                                              ], indirect=["trained_decoder"])
def test_without_balancing(trained_decoder):
    pval = trained_decoder.results.action.pval
    zval = trained_decoder.results.action.zval
    assert "stimulus" not in trained_decoder.results, (f"Unbalanced decoders should not have stimulus in the conditions, "
                                                       f"{trained_decoder.key=}")

    try:
        assert zval > 2.0
        #assert pval < 0.05
    except AssertionError as exc:
        raise AssertionError(f"Unbalanced decoding should be significant for the condition with rate 0.0 & corr 0.8,"
                             f"{trained_decoder.key=}, {pval=}, {zval=}") from exc


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("trained_decoder", [("tangled", "base"),
                                              ("tangled", "svc"),
                                              ("tangled", "nonlinear")
                                              ], indirect=["trained_decoder"])
def test_with_balancing(trained_decoder):
    pval = trained_decoder.results.action.pval
    zval = trained_decoder.results.action.zval
    try:
        assert zval < 2.0
        #assert pval > 0.05
    except AssertionError as exc:
        raise AssertionError(f"Balanced decoding should not be significant for the condition with rate 0.0 & corr 0.8,"
                             f"{trained_decoder.key=}, {pval=}, {zval=}") from exc
