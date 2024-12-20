import pytest


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
class TestDecodingCase:
    """
    This class is used to test specific decoding test cases
    """
    def test_linearly_separable_dichotomies(self, decoding_test_case):
        linearly_separable_dichotomies = [key for key, result in iter(decoding_test_case.results)
                                         if result.pval < 0.05
                                         and key in decoding_test_case.dataset.linearly_separable_dichotomies]
        try:
            assert set(linearly_separable_dichotomies) == set(decoding_test_case.dataset.linearly_separable_dichotomies)
        except AssertionError as exc:
            missing = set(decoding_test_case.dataset.linearly_separable_dichotomies).difference(
                set(linearly_separable_dichotomies)
            )
            raise AssertionError(
                "Linearly separable dichotomies should be linearly separable with any of the test classifiers, "
                f"Dataset: {decoding_test_case.dataset.key}, "
                f"Classifier: {decoding_test_case.classifier}",
                f"Failures: {missing}") from exc

    def test_non_linearly_separable_dichotomies(self, decoding_test_case):
        non_linearly_separable_dichotomies = [key for key, result in iter(decoding_test_case.results)
                                             if result.pval < 0.05
                                             and key in decoding_test_case.dataset.non_linearly_separable_dichotomies]
        if (len(decoding_test_case.dataset.non_linearly_separable_dichotomies) == 0
                and len(non_linearly_separable_dichotomies) > 0):
            raise AssertionError(
                "No dichotomies should be separable only when using a nonlinear classifier",
                f"Dataset: {decoding_test_case.dataset.key}, "
                f"Classifier: {decoding_test_case.classifier}",
                f"Failures: {non_linearly_separable_dichotomies}")

        if decoding_test_case.classifier.func.__name__ == "rbf":
            try:
                assert (set(non_linearly_separable_dichotomies) ==
                        set(decoding_test_case.dataset.non_linearly_separable_dichotomies))
            except AssertionError as exc:
                missing = set(decoding_test_case.dataset.non_linearly_separable_dichotomies).difference(
                    set(non_linearly_separable_dichotomies)
                )
                raise AssertionError(
                    "Non-linearly separable dichotomies should be non-linearly separable"
                    f"Dataset: {decoding_test_case.dataset.key}, "
                    f"Classifier: {decoding_test_case.classifier}",
                    f"Failures: {missing}") from exc
        else:
            assert len(non_linearly_separable_dichotomies) == 0, (
                "Non-linearly separable dichotomies should not be linearly separable with a linear classifier",
                f"Dataset: {decoding_test_case.dataset.key}, "
                f"Classifier: {decoding_test_case.classifier}",
                f"Failures: {non_linearly_separable_dichotomies}")
