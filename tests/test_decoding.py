import pytest


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
class TestDecodingCase:
    """
    This class is used to test specific decoding test cases. It checks that the decoding results are consistent with the
    expected results for the dataset and classifier used in the test case. That is, it checks that the dichotomies that
    are expected to be linearly separable are separable by a linear classifier and that the dichotomies that are
    expected to be non-linearly separable are separably only by a non-linear classifier. We filter out sklearn's
    ConvergenceWarning because meeting the convergence criteria is not necessary for this test.

    Methods
    -------
    test_linearly_separable_semantic_dichotomies(decoding_test_case)
        Test that the linearly separable semantic dichotomies are separable by the classifier used in the test case.
    test_non_linearly_separable_semantic_dichotomies(decoding_test_case)
        Test that the non-linearly separable semantic dichotomies are separable only by a non-linear classifier.
    """
    def test_linearly_separable_semantic_dichotomies(self, decoding_test_case):
        linearly_separable_semantic_dichotomies = [key for key, result in iter(decoding_test_case.results)
                                          if result.pval < 0.05
                                          and key in decoding_test_case.dataset.linearly_separable_semantic_dichotomies]
        try:
            assert set(linearly_separable_semantic_dichotomies) == set(decoding_test_case.dataset.linearly_separable_semantic_dichotomies)
        except AssertionError as exc:
            missing = set(decoding_test_case.dataset.linearly_separable_semantic_dichotomies).difference(
                set(linearly_separable_semantic_dichotomies)
            )
            raise AssertionError(
                "Linearly separable semantic dichotomies should be linearly separable with all test classifiers:\n"
                f"\tDataset: {decoding_test_case.dataset.key}\n"
                f"\tClassifier: {decoding_test_case.classifier}\n"
                f"\tFailures: {missing}") from exc

    def test_non_linearly_separable_semantic_dichotomies(self, decoding_test_case):
        non_linearly_separable_semantic_dichotomies = \
            [key for key, result in iter(decoding_test_case.results)
             if result.pval < 0.05
             and key in decoding_test_case.dataset.non_linearly_separable_semantic_dichotomies]
        if (len(decoding_test_case.dataset.non_linearly_separable_semantic_dichotomies) == 0
                and len(non_linearly_separable_semantic_dichotomies) > 0):
            raise AssertionError(
                "No semantic dichotomies should be separable ONLY when using a nonlinear classifier:\n"
                f"\tDataset: {decoding_test_case.dataset.key}\n"
                f"\tClassifier: {decoding_test_case.classifier}\n"
                f"\tFailures: {non_linearly_separable_semantic_dichotomies}")

        if decoding_test_case.classifier.func.__name__ == "rbf":
            try:
                assert (set(non_linearly_separable_semantic_dichotomies) ==
                        set(decoding_test_case.dataset.non_linearly_separable_semantic_dichotomies))
            except AssertionError as exc:
                missing = set(decoding_test_case.dataset.non_linearly_separable_semantic_dichotomies).difference(
                    set(non_linearly_separable_semantic_dichotomies)
                )
                raise AssertionError(
                    "Non-linearly separable semantic dichotomies should be non-linearly separable:\n"
                    f"\tDataset: {decoding_test_case.dataset.key}\n"
                    f"\tClassifier: {decoding_test_case.classifier}\n"
                    f"\tFailures: {missing}") from exc
        else:
            assert len(non_linearly_separable_semantic_dichotomies) == 0, (
                "Non-linearly separable semantic dichotomies should not be separable with a linear classifier:\n",
                f"\tDataset: {decoding_test_case.dataset.key}\n"
                f"\tClassifier: {decoding_test_case.classifier}\n"
                f"\tFailures: {non_linearly_separable_semantic_dichotomies}")
