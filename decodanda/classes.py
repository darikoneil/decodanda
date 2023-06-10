from __future__ import annotations
from typing import Tuple, Union, Callable, Iterable, List, Mapping, Optional
from types import MappingProxyType
from itertools import chain, combinations
from multiprocessing import Pool
import copy


import numpy as np
from sklearn.svm import LinearSVC
from sklearn.base import clone
from tqdm import tqdm

from ._defaults import classifier_parameters, DecodandaParameters
from ._dev import identify_calling_function
from .utilities import generate_binary_words, string_bool, sample_training_testing_from_rasters, CrossValidator, \
    log_dichotomy, hamming, generate_dichotomies, contiguous_chunking, non_contiguous_mask, generate_binary_conditions


# Main class

class Decodanda:
    def __init__(self,
                 data: Union[Iterable, dict],
                 conditions: dict,
                 classifier: Callable = LinearSVC,
                 classifier_params: Mapping = classifier_parameters,
                 decodanda_params: Optional[Mapping] = None,
                 **kwargs
                 ):

        """
        Main class that implements the decoding pipelines with built-in best practices.

        It works by separating the input data into all possible conditions - defined as specific
        combinations of variable values - and sampling data points from these conditions
        according to the specific decoding problem.

        Parameters
        ----------
        data
            A dictionary or a list of dictionaries each containing
            (1) the neural data (2) a set of variables that we want to decode from the neural data
            (3) a trial number. See the ``Data Structure`` section for more details.
            If a list is passed, the analyses will be performed on the pseudo-population built by pooling
            all the data sets in the list.

        conditions
            A dictionary that specifies which values for which variables of `data` we want to decode.
            See the ``Data Structure`` section for more details.

        classifier
            The classifier used for all decoding analyses. Default: ``sklearn.svm.LinearSVC``.

        neural_attr
            The key under which the neural features are stored in the ``data`` dictionary.

        trial_attr
            The key under which the trial numbers are stored in the ``data`` dictionary.
            Each different trial is considered as an independent sample to be used in
            during cross validation.
            If ``None``, trials are defined as consecutive bouts of data in time
            where all the variables have a constant value.

        squeeze_trials
            If True, all population vectors corresponding to the same trial number for the same
            condition will be squeezed into a single average activity vector.

        min_data_per_condition
            The minimum number of data points per each condition, defined as a specific
            combination of values of all variables in the ``conditions`` dictionary,
            that a data set needs to have to be included in the analysis.

        min_trials_per_condition
            The minimum number of unique trial numbers per each condition, defined as a specific
            combination of values of all variables in the ``conditions`` dictionary,
            that a data set needs to have to be included in the analysis.

        min_activations_per_cell
            The minimum number of non-zero bins that single neurons / features need to have to be
            included into the analysis.

        trial_chunk
            Only used when ``trial_attr=None``. The maximum number of consecutive data points
            within the same bout. Bouts longer than ``trial_chunk`` data points are split into
            different trials.

        exclude_contiguous_chunks
            Only used when ``trial_attr=None`` and ``trial_chunks != None``. Discards every second trial
            that has the same value of all variables as the previous one. It can be useful to avoid
            decoding temporal artifacts when there are long auto-correlation times in the neural
            activations.

        exclude_silent
            If ``True``, all silent population vectors (only zeros) are excluded from the analysis.

        verbose
            If ``True``, most operations and analysis results are logged in standard output.

        zscore
            If ``True``, neural features are z-scored before being separated into conditions.

        fault_tolerance
            If ``True``, the constructor raises a warning instead of an error if no data set
            passes the inclusion criteria specified by ``min_data_per_condition`` and ``min_trials_per_condition``.

        debug
            If ``True``, operations are super verbose. Do not use unless you are developing.


        Data structure
        --------------
        Decodanda works with datasets organized into Python dictionaries.
        For ``N`` recorded neurons and ``T`` trials (or time bins), the data dictionary must contain:

        1. a ``TxN`` array, under the ``raster`` key
            This is the set of features we use to decode. Can be continuous (e.g., calcium fluorescence) or discrete (e.g., spikes) values.

        2. a ``Tx1`` array specifying a ``trial`` number
            This array will define the subdivisions for cross validation: trials (or time bins) that share the
            same ```trial``` value will always go together in either training or testing samples.

        3. a ``Tx1`` array for each variable we want to decode
            Each value will be used as a label for the ``raster`` feature. Make sure these arrays are
            synchronized with the ``raster`` array.


        Say we have a data set with N=50 neurons, T=800 time bins divided into 80 trials, where two experimental
        variables are specified ``stimulus`` and ``action``.
        A properly-formatted data set would look like this:

        The ``conditions`` dictionary is used to specify which variables - out of
        all the keywords in the ``data`` dictionary, and which and values - out of
        all possible values of each specified variable - we want to decode.

        It has to be in the form ``{key: [value1, value2]}``:

        If more than one variable is specified, `Decodanda` will balance all
        conditions during each decoding analysis to disentangle
        the variables and avoid confounding correlations.


        Examples
        --------

        Using the data set defined above:

        [Decodanda]	building conditioned rasters for dataset 0
                    (stimulus = A, action = left):	Selected 150 time bin out of 800, divided into 15 trials
                    (stimulus = A, action = right):	Selected 210 time bin out of 800, divided into 21 trials
                    (stimulus = B, action = left):	Selected 210 time bin out of 800, divided into 21 trials
                    (stimulus = B, action = right):	Selected 230 time bin out of 800, divided into 23 trials


        The constructor divides the data into conditions using the ``stimulus`` and ``action`` values
        and stores them in the ``self.conditioned_rasters`` object.
        This condition structure is the basis for all the balanced decoding analyses.

        """

        # abstracted for readability - DAO 06/08/2023
        self.data = self._sanitize_data(data)

        # abstracted to directly passing a callable with sklearn syntax - DAO 06/08/2023
        self.classifier = classifier(**classifier_params)

        # handling discrete dict conditions
        if type(list(conditions.values())[0]) == list:
            conditions = generate_binary_conditions(conditions)

        # setting input parameters
        self.conditions = conditions

        # decodanda parameters
        self.parameters = DecodandaParameters.build([decodanda_params, kwargs])
        # integrate parameters
        for key, value in self.parameters.items():
            vars(self)["".join(["_", key])] = value

        # deriving dataset(s) attributes
        self.n_datasets = len(self.data)
        self.n_conditions = len(self.conditions)
        self._max_conditioned_data = 0
        self._min_conditioned_data = 10 ** 6
        self.n_neurons = 0
        self.n_brains = 0
        self.which_brain = []

        # keys and stuff
        self._condition_vectors = generate_binary_words(self.n_conditions)
        self._semantic_keys = list(self.conditions.keys())
        self._semantic_vectors = {string_bool(w): [] for w in generate_binary_words(self.n_conditions)}
        self._generate_semantic_vectors()

        # decoding weights
        self.decoding_weights = {}
        self.decoding_weights_null = {}

        # creating conditioned array with the following structure:
        #   define a condition_vector with boolean values for each semantic condition, es. 100
        #   use this vector as the key for a dictionary
        #   as a value, create a list of neural data for each dataset conditioned as per key

        #   >>> main object: neural rasters conditioned to semantic vector <<<
        self.conditioned_rasters = {string_bool(w): [] for w in self._condition_vectors}

        # conditioned null model index is the chunk division used for null model shuffles
        self.conditioned_trial_index = {string_bool(w): [] for w in self._condition_vectors}

        #   >>> main part: create conditioned arrays <<< ---------------------------------
        self._divide_data_into_conditions(self.data)
        #  \ >>> main part: create conditioned arrays <<< --------------------------------

        # raising exceptions
        if self.n_brains == 0:
            if not fault_tolerance:
                raise RuntimeError(
                    f"\n{identify_calling_function()}\tNo dataset passed the minimum data threshold for conditioned "
                    f"arrays.\n\t\t Check for mutually-exclusive conditions or try using less restrictive thresholds."
                )
        else:
            # derived attributes
            self._compute_centroids()

            # null model variables
            self.random_translations = {string_bool(w): [] for w in self._condition_vectors}
            self.subset = np.arange(self.n_neurons)

            self.ordered_conditioned_rasters = {}
            self.ordered_conditioned_trial_index = {}

            for w in self.conditioned_rasters.keys():
                self.ordered_conditioned_rasters[w] = self.conditioned_rasters[w].copy()
                self.ordered_conditioned_trial_index[w] = self.conditioned_trial_index[w].copy()

    @staticmethod
    def _sanitize_data(data: Union[Iterable, dict]) -> List[dict]:

        # casting single dataset to a list so that it is compatible with all loops below
        if isinstance(data, dict):  # faster and considers inheritance - DAO -6/08/2023
            data = [data]

        # ensure each dataset in data is a dictionary - DAO -6/08/2023
        for idx, dataset in zip(range(len(data)), data):
            assert (isinstance(dataset, dict)), f"Dataset {idx} is a {type(dataset)} not a dictionary"

        # make sure all dataset data is numpy array
        if isinstance(data[0], dict):
            data = [{key: np.asarray(value) for key, value in dataset.items()} for dataset in data]
            # List Comprehension, pythonic / faster
            # Ensuring all numpy array using dict comprehension
            # This ensures all dictionary methods / optimizations (including getattr)
            # Preferable to mutable mapping abstract base class with regards to performance
            # DAO 06/08/2023

        return data

    # basic decoding functions

    def _train(self, training_raster_A, training_raster_B, label_A, label_B):

        training_labels_A = np.repeat(label_A, training_raster_A.shape[0]).astype(object)
        training_labels_B = np.repeat(label_B, training_raster_B.shape[0]).astype(object)

        training_raster = np.vstack([training_raster_A, training_raster_B])
        training_labels = np.hstack([training_labels_A, training_labels_B])

        self.classifier = clone(self.classifier)

        training_raster = training_raster[:, self.subset]

        self.classifier.fit(training_raster, training_labels)

    def _test(self, testing_raster_A, testing_raster_B, label_A, label_B):

        testing_labels_A = np.repeat(label_A, testing_raster_A.shape[0]).astype(object)
        testing_labels_B = np.repeat(label_B, testing_raster_B.shape[0]).astype(object)

        testing_raster = np.vstack([testing_raster_A, testing_raster_B])
        testing_labels = np.hstack([testing_labels_A, testing_labels_B])

        testing_raster = testing_raster[:, self.subset]

        if self._debug:
            print("Real labels")
            print(testing_labels)
            print("Predicted labels")
            print(self.classifier.predict(testing_raster))
        performance = self.classifier.score(testing_raster, testing_labels)
        return performance

    def _one_cv_step(self, dic, training_fraction, ndata, shuffled=False, testing_trials=None, dic_key=None):
        if dic_key is None:
            dic_key = self._dic_key(dic)

        set_A = dic[0]
        label_A = ''
        for d in set_A:
            label_A += (self._semantic_vectors[d] + ' ')
        label_A = label_A[:-1]

        set_B = dic[1]
        label_B = ''
        for d in set_B:
            label_B += (self._semantic_vectors[d] + ' ')
        label_B = label_B[:-1]

        training_array_A = []
        training_array_B = []
        testing_array_A = []
        testing_array_B = []

        # allow for unbalanced dichotomies
        n_conditions_A = float(len(dic[0]))
        n_conditions_B = float(len(dic[1]))
        fraction = n_conditions_A / n_conditions_B

        for d in set_A:
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     int(ndata / fraction),
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     debug=self._debug,
                                                                     testing_trials=testing_trials)
            if self._debug:
                plt.title('Condition A')
                print("Sampling for condition A, d=%s" % d)
                print("Conditioned raster mean:")
                print(np.nanmean(self.conditioned_rasters[d][0], 0))

            training_array_A.append(training)
            testing_array_A.append(testing)

        for d in set_B:
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     int(ndata),
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     debug=self._debug,
                                                                     testing_trials=testing_trials)
            training_array_B.append(training)
            testing_array_B.append(testing)
            if self._debug:
                plt.title('Condition B')
                print("Sampling for condition B, d=%s" % d)
                print("Conditioned raster mean:")
                print(np.nanmean(self.conditioned_rasters[d][0], 0))

        training_array_A = np.vstack(training_array_A)
        training_array_B = np.vstack(training_array_B)
        testing_array_A = np.vstack(testing_array_A)
        testing_array_B = np.vstack(testing_array_B)

        if self._debug:
            selectivity_training = np.nanmean(training_array_A, 0) - np.nanmean(training_array_B, 0)
            selectivity_testing = np.nanmean(testing_array_A, 0) - np.nanmean(testing_array_B, 0)
            # corr_scatter(selectivity_training, selectivity_testing, 'Selectivity (training)', 'Selectivity (testing)')

        if self._zscore:
            big_raster = np.vstack([training_array_A, training_array_B])  # z-scoring using the training data
            big_mean = np.nanmean(big_raster, 0)
            big_std = np.nanstd(big_raster, 0)
            big_std[big_std == 0] = np.inf
            training_array_A = (training_array_A - big_mean) / big_std
            training_array_B = (training_array_B - big_mean) / big_std
            testing_array_A = (testing_array_A - big_mean) / big_std
            testing_array_B = (testing_array_B - big_mean) / big_std

        self._train(training_array_A, training_array_B, label_A, label_B)

        if hasattr(self.classifier, 'coef_'):
            if dic_key and not shuffled:
                if dic_key not in self.decoding_weights.keys():
                    self.decoding_weights[dic_key] = []
                self.decoding_weights[dic_key].append(self.classifier.coef_)
            if dic_key and shuffled:
                if dic_key not in self.decoding_weights_null.keys():
                    self.decoding_weights_null[dic_key] = []
                self.decoding_weights_null[dic_key].append(self.classifier.coef_)

        performance = self._test(testing_array_A, testing_array_B, label_A, label_B)

        return performance

    # Dichotomy analysis functions

    def decode_dichotomy(self, dichotomy: Union[str, list], training_fraction: float,
                         cross_validations: int = 10, ndata: Optional[int] = None,
                         shuffled: bool = False, parallel: bool = False,
                         testing_trials: Optional[list] = None,
                         dic_key: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Function that performs cross-validated decoding of a specific dichotomy.
        Decoding is performed by sampling a balanced amount of data points from each condition in each class of the
        dichotomy, so to ensure that only the desired variable is analyzed by balancing confounds.
        Before sampling, each condition is individually divided into training and testing bins
        by using the ``self.trial`` array specified in the data structure when constructing the ``Decodanda`` object.



        Parameters
        ----------
            dichotomy : str || list
                The dichotomy to be decoded, expressed in a double-list binary format, e.g. [['10', '11'], ['01', '00']], or as a variable name.
            training_fraction:
                the fraction of trials used for training in each cross-validation fold.
            cross_validations:
                the number of cross-validations.
            ndata:
                the number of data points (population vectors) sampled for training and for testing for each condition.
            shuffled:
                if True, population vectors for each condition are sampled in a random way compatibly with a null model for decoding performance.
            parallel:
                if True, each cross-validation is performed by a dedicated thread (experimental, use with caution).
            testing_trials:
                if specified, data sampled from the specified trial numbers will be used for testing, and the remaining ones for training.
            dic_key:
                if specified, weights of the decoding analysis will be saved in self.decoding_weights using dic_key as the dictionary key.

        Returns
        -------
            performances: list of decoding performance values for each cross-validation.

        Note
        ----
        ``dichotomy`` can be passed as a string or as a list.
        If a string is passed, it has to be a name of one of the variables specified in the conditions dictionary.

        If a list is passed, it needs to contain two lists in the shape [[...], [...]].
        Each sub list contains the conditions used to define one of the two decoded classes
        in binary notation.

        For example, if the data set has two variables
        ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, the condition
        ``stimulus=-1`` & ``action=-1`` will correspond to the binary notation ``'00'``,
        the condition ``stimulus=+1`` & ``action=-1`` will correspond to ``10`` and so on.
        Therefore, the notation:


        >>> dic = 'stimulus'

        is equivalent to

        >>> dic = [['00', '01'], ['10', '11']]

        and

        >>> dic = 'action'

        is equivalent to

        >>> dic = [['00', '10'], ['01', '11']]

        However, not all dichotomies have names (are semantic). For example, the dichotomy

        >>> [['01','10'], ['00', '11']]

        can only be defined using the binary notation.

        Note that this function gives you the flexibility to use sub-sets of conditions, for example

        >>> dic = [['10'], ['01']]

        will decode stimulus=1 & action=-1  vs.  stimulus=-1 & action=1


        Example
        -------
        >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
        >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
        >>> perfs = dec.decode_dichotomy('stimulus', training_fraction=0.75, cross_validations=10)
        >>> perfs
        [0.82, 0.87, 0.75, ..., 0.77] # 10 values

        """

        if isinstance(dichotomy, str):  # DAO 06/08/2023
            dic = self._dichotomy_from_key(dichotomy)
        else:
            dic = dichotomy

        if ndata is None and self.n_brains == 1:
            ndata = self._max_conditioned_data
        if ndata is None and self.n_brains > 1:
            ndata = max(self._max_conditioned_data, 2 * self.n_neurons)

        if shuffled:
            self._shuffle_conditioned_arrays(dic)

        if self._verbose and not shuffled:
            print(dic, ndata)
            log_dichotomy(self, dic, ndata, 'Decoding')
            count = tqdm(range(cross_validations))
        else:
            count = range(cross_validations)

        if parallel:
            pool = Pool()
            performances = pool.map(CrossValidator(classifier=self.classifier,
                                                   conditioned_rasters=self.conditioned_rasters,
                                                   conditioned_trial_index=self.conditioned_trial_index,
                                                   dic=dic,
                                                   training_fraction=training_fraction,
                                                   ndata=ndata,
                                                   subset=self.subset,
                                                   semantic_vectors=self._semantic_vectors),
                                    range(cross_validations))

        else:
            performances = np.zeros(cross_validations)
            if self._verbose and not shuffled:
                print('\nLooping over decoding cross validation folds:')
            for i in count:
                performances[i] = self._one_cv_step(dic=dic, training_fraction=training_fraction, ndata=ndata,
                                                    shuffled=shuffled, testing_trials=testing_trials, dic_key=dic_key)

        if shuffled:
            self._order_conditioned_rasters()
        return np.asarray(performances)

    # Dichotomy analysis functions with null model

    def decode_with_nullmodel(self, dichotomy: Union[str, list],
                              training_fraction: float,
                              cross_validations: int = 10,
                              nshuffles: int = 10,
                              ndata: Optional[int] = None,
                              parallel: bool = False,
                              return_CV: bool = False,
                              testing_trials: Optional[list] = None,
                              plot: bool = False,
                              dic_key: Optional[str] = None) -> Tuple[Union[list, np.ndarray], np.ndarray]:
        """
        Function that performs cross-validated decoding of a specific dichotomy and compares the resulting values with
        a null model where the relationship between the neural data and the two sides of the dichotomy is
        shuffled.

        Decoding is performed by sampling a balanced amount of data points from each condition in each class of the
        dichotomy, so to ensure that only the desired variable is analyzed by balancing confounds.

        Before sampling, each condition is individually divided into training and testing bins
        by using the ``self.trial`` array specified in the data structure when constructing the ``Decodanda`` object.


        Parameters
        ----------
            dichotomy : str || list
                The dichotomy to be decoded, expressed in a double-list binary format, e.g. [['10', '11'], ['01', '00']], or as a variable name.
            training_fraction:
                the fraction of trials used for training in each cross-validation fold.
            cross_validations:
                the number of cross-validations.
            nshuffles:
                the number of null-model iterations of the decoding procedure.
            ndata:
                the number of data points (population vectors) sampled for training and for testing for each condition.
            parallel:
                if True, each cross-validation is performed by a dedicated thread (experimental, use with caution).
            return_CV:
                if True, invidual cross-validation values are returned in a list. Otherwise, the average performance over the cross-validation folds is returned.
            testing_trials:
                if specified, data sampled from the specified trial numbers will be used for testing, and the remaining ones for training.
            plot:
                if True, a visualization of the decoding results is shown.
            dic_key:
                if specified, weights of the decoding analysis will be saved in self.decoding_weights using dic_key as the dictionary key.


        Returns
        -------
            performances, null_performances: list of decoding performance values for each cross-validation.


        See Also
        --------
        Decodanda.decode_dichotomy : The method used for each decoding iteration.


        Note
        ----
        ``dichotomy`` can be passed as a string or as a list.
        If a string is passed, it has to be a name of one of the variables specified in the conditions dictionary.

        If a list is passed, it needs to contain two lists in the shape [[...], [...]].
        Each sub list contains the conditions used to define one of the two decoded classes
        in binary notation.

        For example, if the data set has two variables
        ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, the condition
        ``stimulus=-1`` & ``action=-1`` will correspond to the binary notation ``'00'``,
        the condition ``stimulus=+1`` & ``action=-1`` will correspond to ``10`` and so on.
        Therefore, the notation:


        >>> dic = 'stimulus'

        is equivalent to

        >>> dic = [['00', '01'], ['10', '11']]

        and

        >>> dic = 'action'

        is equivalent to

        >>> dic = [['00', '10'], ['01', '11']]

        However, not all dichotomies have names (are semantic). For example, the dichotomy

        >>> [['01','10'], ['00', '11']]

        can only be defined using the binary notation.

        Note that this function gives you the flexibility to use sub-sets of conditions, for example

        >>> dic = [['10'], ['01']]

        will decode stimulus=1 & action=-1  vs.  stimulus=-1 & action=1


        Example
        -------
        >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
        >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
        >>> perf, null = dec.decode_with_nullmodel('stimulus', training_fraction=0.75, cross_validations=10, nshuffles=20)
        >>> perf
        0.88
        >>> null
        [0.51, 0.54, 0.48, ..., 0.46] # 25 values
        """

        # am i necessary
        if isinstance(dichotomy, str):  # DAO 06/08/2023
            dic = self._dichotomy_from_key(dichotomy)
        else:
            dic = dichotomy

        # Decode Real Data
        real_performance = self.decode_dichotomy(dichotomy=dic,
                                               training_fraction=training_fraction,
                                               cross_validations=cross_validations,
                                               ndata=ndata,
                                               parallel=parallel,
                                               testing_trials=testing_trials,
                                               dic_key=dic_key)
        # Report Progress
        if self._verbose:
            print(f"\n{identify_calling_function()}\t Fraction Correct: {np.nanmean(real_performance):.2f}")

        if self._verbose:
            print(f"\n{identify_calling_function()}\t Looping over null model shuffles")
            count = tqdm(range(nshuffles))
        else:
            count = range(nshuffles)

        null_model_performances = [np.nanmean(self.decode_dichotomy(dichotomy=dic,
                                                 training_fraction=training_fraction,
                                                 cross_validations=cross_validations,
                                                 ndata=ndata,
                                                 parallel=parallel,
                                                 testing_trials=testing_trials,
                                                 shuffled=True))
                                   for _ in count]

        return real_performance, null_model_performances

    # Decoding analysis for semantic dichotomies

    def decode(self, training_fraction: float,
               cross_validations: int = 10,
               nshuffles: int = 10,
               ndata: Optional[int] = None,
               parallel: bool = False,
               non_semantic: bool = False,
               return_CV: bool = False,
               testing_trials: Optional[list] = None,
               **kwargs):

        """
        Main function to decode the variables specified in the ``conditions`` dictionary.

        It returns a single decoding value per variable which represents the average over
        the cross-validation folds.

        It also returns an array of null-model values for each variable to test the significance of
        the corresponding decoding result.

        Notes
        -----

        Each decoding analysis is performed by first re-sampling an equal number of data points
        from each condition (combination of variable values), so to ensure that possible confounds
        due to correlated conditions are balanced out.


        Before sampling, each condition is individually divided into training and testing bins
        by using the ``self.trial`` array specified in the data structure when constructing the ``Decodanda`` object.


        To generate the null model values, the relationship between the neural data and
        the decoded variable is randomly shuffled. Eeach null model value corresponds to the
        average across ``cross_validations``` iterations after a single data shuffle.


        If ``non_semantic=True``, dichotomies that do not correspond to variables will also be decoded.
        Note that, in the case of 2 variables, there is only one non-semantic dichotomy
        (corresponding to grouping together conditions that have the same XOR value in the
        binary notation: ``[['10', '01'], ['11', '00']]``). However, the number of non-semantic dichotomies
        grows exponentially with the number of conditions, so use with caution if more than two variables
        are specified in the conditions dictionary.


        Parameters
        ----------
        training_fraction:
            the fraction of trials used for training in each cross-validation fold.
        cross_validations:
            the number of cross-validations.
        nshuffles:
            the number of null-model iterations of the decoding procedure.
        ndata:
            the number of data points (population vectors) sampled for training and for testing for each condition.
        parallel:
            if True, each cross-validation is performed by a dedicated thread (experimental, use with caution).
        return_CV:
            if True, invidual cross-validation values are returned in a list. Otherwise, the average performance over the cross-validation folds is returned.
        testing_trials:
            if specified, data sampled from the specified trial numbers will be used for testing, and the remaining ones for training.
        non_semantic:
            if True, non-semantic dichotomies (i.e., dichotomies that do not correspond to a variable) will also be decoded.
        plot:
            if True, a visualization of the decoding results is shown.
        ax:
            if specified and ``plot=True``, the results will be displayed in the specified axis instead of a new figure.
        plot_all:
            if True, a more in-depth visualization of the decoding results and of the decoded data is shown.


        Returns
        -------
            perfs:
                a dictionary containing the decoding performances for all variables in the form of ``{var_name_1: performance1, var_name_2: performance2, ...}``
            null:
                a dictionary containing an array of null model decoding performance for each variable in the form ``{var_name_1: [...], var_name_2: [...], ...}``.

        See Also
        --------
            Decodanda.decode_with_nullmodel: The method used for each decoding analysis.


        Example
        -------
        >>> from decodanda import Decodanda, generate_synthetic_data
        >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
        >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
        >>> perfs, null = dec.decode(training_fraction=0.75, cross_validations=10, nshuffles=20)
        >>> perfs
        {'stimulus': 0.88, 'action': 0.85}  # mean over 10 cross-validation folds
        >>> null
        {'stimulus': [0.51, ..., 0.46], 'action': [0.48, ..., 0.55]}  # null model means, 20 values each
        """

        semantic_dics, semantic_keys = self._find_semantic_dichotomies()

        perfs = {}
        perfs_nullmodel = {}

        for key, dic in zip(semantic_keys, semantic_dics):

            if self._verbose:
                print(f"\n{identify_calling_function()}\tTesting decoding performance for semantic dichotomy: {key}")

            performance, null_model_performances = self.decode_with_nullmodel(
                dic,
                training_fraction,
                cross_validations=cross_validations,
                ndata=ndata,
                nshuffles=nshuffles,
                parallel=parallel,
                testing_trials=testing_trials,
                plot=plot_all)

            perfs[key] = performance
            perfs_nullmodel[key] = null_model_performances

        return perfs, perfs_nullmodel

    # Geometrical analysis for semantic dichotomies

    def _divide_data_into_conditions(self, datasets):
        # TODO: make sure conditions don't overlap somehow

        for si, dataset in enumerate(datasets):

            if self._verbose:
                if hasattr(dataset, 'name'):
                    print(f"\n{identify_calling_function()}\tBuilding conditioned rasters for dataset {dataset.name}")
                else:
                    print(f"\n{identify_calling_function()}\tBuilding conditioned rasters for dataset {si}")

            dataset_conditioned_rasters = {}
            dataset_conditioned_trial_index = {}

            # exclude inactive neurons across the specified conditions
            array = dataset.get(self._neural_attr)
            total_mask = np.zeros(len(array)) > 0

            for condition_vec in self._condition_vectors:
                mask = np.ones(len(array)) > 0
                for i, sk in enumerate(self._semantic_keys):
                    semantic_values = list(self.conditions[sk])
                    mask_i = self.conditions[sk][semantic_values[condition_vec[i]]](dataset)
                    mask = mask & mask_i
                total_mask = total_mask | mask

            min_activity_mask = np.sum(array[total_mask] > 0, 0) >= self._min_activations_per_cell

            for condition_vec in self._condition_vectors:
                # get the array from the dataset object
                array = dataset.get(self._neural_attr)
                array = array[:, min_activity_mask]

                # create a mask that becomes more and more restrictive by iterating on semanting conditions
                mask = np.ones(len(array)) > 0
                for i, sk in enumerate(self._semantic_keys):
                    semantic_values = list(self.conditions[sk])
                    mask_i = self.conditions[sk][semantic_values[condition_vec[i]]](dataset)
                    mask = mask & mask_i

                # select bins conditioned on the semantic behavioural vector
                conditioned_raster = array[mask, :]

                # Define trial logic
                def condition_no(cond):
                    no = 0
                    for i in range(len(cond)):
                        no += cond[i] * 10 ** (i + 2)
                    return no

                if self._trial_attr is not None:
                    conditioned_trial = dataset.get(self._trial_attr)[mask]
                elif self._trial_chunk is None:
                    if self._verbose:
                        print('[Decodanda]\tUsing contiguous chunks of the same labels as trials.')
                    conditioned_trial = contiguous_chunking(mask)[mask]
                    conditioned_trial += condition_no(condition_vec)
                else:
                    conditioned_trial = contiguous_chunking(mask, self._trial_chunk)[mask]
                    conditioned_trial += condition_no(condition_vec)

                if self._exclude_contiguous_trials:
                    contiguous_chunks = contiguous_chunking(mask)[mask]
                    nc_mask = non_contiguous_mask(contiguous_chunks, conditioned_trial)
                    conditioned_raster = conditioned_raster[nc_mask, :]
                    conditioned_trial = conditioned_trial[nc_mask]

                # exclude empty time bins (only for binary discrete decoding)
                if self._exclude_silent:
                    active_mask = np.sum(conditioned_raster, 1) > 0
                    conditioned_raster = conditioned_raster[active_mask, :]
                    conditioned_trial = conditioned_trial[active_mask]

                # squeeze into trials
                if self._trial_average:
                    unique_trials = np.unique(conditioned_trial)
                    squeezed_raster = []
                    squeezed_trial_index = []
                    for t in unique_trials:
                        trial_raster = conditioned_raster[conditioned_trial == t]
                        squeezed_raster.append(np.nanmean(trial_raster, 0))
                        squeezed_trial_index.append(t)
                    # set the new arrays
                    conditioned_raster = np.asarray(squeezed_raster)
                    conditioned_trial = np.asarray(squeezed_trial_index)

                # set the conditioned neural data in the conditioned_rasters dictionary
                dataset_conditioned_rasters[string_bool(condition_vec)] = conditioned_raster
                dataset_conditioned_trial_index[string_bool(condition_vec)] = conditioned_trial

                if self._verbose:
                    semantic_vector_string = []
                    for i, sk in enumerate(self._semantic_keys):
                        semantic_values = list(self.conditions[sk])
                        semantic_vector_string.append("%s = %s" % (sk, semantic_values[condition_vec[i]]))
                    semantic_vector_string = ', '.join(semantic_vector_string)
                    print("\t\t\t(%s):\tSelected %u time bin out of %u, divided into %u trials "
                          % (semantic_vector_string, conditioned_raster.shape[0], len(array),
                             len(np.unique(conditioned_trial))))

            dataset_conditioned_data = [r.shape[0] for r in list(dataset_conditioned_rasters.values())]
            dataset_conditioned_trials = [len(np.unique(c)) for c in list(dataset_conditioned_trial_index.values())]

            self._max_conditioned_data = max([self._max_conditioned_data, np.max(dataset_conditioned_data)])
            self._min_conditioned_data = min([self._min_conditioned_data, np.min(dataset_conditioned_data)])

            # if the dataset has enough data for each condition, append it to the main data dictionary

            if np.min(dataset_conditioned_data) >= self._min_data_per_condition and \
                    np.min(dataset_conditioned_trials) >= self._min_trials_per_condition:
                for cv in self._condition_vectors:
                    self.conditioned_rasters[string_bool(cv)].append(dataset_conditioned_rasters[string_bool(cv)])
                    self.conditioned_trial_index[string_bool(cv)].append(
                        dataset_conditioned_trial_index[string_bool(cv)])
                if self._verbose:
                    print('\n')
                self.n_brains += 1
                self.n_neurons += list(dataset_conditioned_rasters.values())[0].shape[1]
                self.which_brain.append(np.ones(list(dataset_conditioned_rasters.values())[0].shape[1]) * self.n_brains)
            else:
                if self._verbose:
                    print('\t\t\t===> dataset discarded for insufficient data.\n')
        if len(self.which_brain):
            self.which_brain = np.hstack(self.which_brain)

    def _find_semantic_dichotomies(self):
        # I think this exports possible vals x conditions
        d_keys, dics = generate_dichotomies(self.n_conditions)
        semantic_dics = []
        semantic_keys = []

        for i, dic in enumerate(dics):
            d = [string_bool(x) for x in dic[0]]
            col_sum = np.sum(d, 0)
            if (0 in col_sum) or (len(dic[0]) in col_sum):
                semantic_dics.append(dic)
                semantic_keys.append(self._semantic_keys[np.where(col_sum == len(dic[0]))[0][0]])
        return semantic_dics, semantic_keys

    def _find_nonsemantic_dichotomies(self):
        d_keys, dics = generate_dichotomies(self.n_conditions)
        nonsemantic_dics = []

        for i, dic in enumerate(dics):
            d = [string_bool(x) for x in dic[0]]
            col_sum = np.sum(d, 0)
            if not ((0 in col_sum) or (len(dic[0]) in col_sum)):
                nonsemantic_dics.append(dic)
        return nonsemantic_dics

    def all_dichotomies(self, balanced=True, semantic_names=False):
        if balanced:
            dichotomies = {}
            sem, keys = self._find_semantic_dichotomies()
            nsem = self._find_nonsemantic_dichotomies()
            if (self.n_conditions == 2) and semantic_names:
                dichotomies[keys[0]] = sem[0]
                dichotomies[keys[1]] = sem[1]
                dichotomies['XOR'] = nsem[0]
            else:
                for i in range(len(sem)):
                    dichotomies[keys[i]] = sem[i]
                for dic in nsem:
                    dichotomies[_powerchotomy_to_key(dic)] = dic
        else:
            powerchotomies = self._powerchotomies()
            dichotomies = {}
            for dk in powerchotomies:
                k = self._dic_key(powerchotomies[dk])
                if k and semantic_names:
                    dichotomies[k] = powerchotomies[dk]
                else:
                    dichotomies[dk] = powerchotomies[dk]
            if self.n_conditions == 2:
                dichotomies['XOR'] = dichotomies['00_11_v_01_10']
                del dichotomies['00_11_v_01_10']
        return dichotomies

    def _powerchotomies(self):
        conditions = list(self._semantic_vectors.keys())
        powerset = list(chain.from_iterable(combinations(conditions, r) for r in range(1, len(conditions))))
        dichotomies = {}
        for i in range(len(powerset)):
            for j in range(i + 1, len(powerset)):
                if len(np.unique(powerset[i] + powerset[j])) == len(conditions):
                    if len(powerset[i] + powerset[j]) == len(conditions):
                        dic = [list(powerset[i]), list(powerset[j])]
                        dichotomies[_powerchotomy_to_key(dic)] = dic
        return dichotomies

    def _dic_key(self, dic):
        if len(dic[0]) == 2 ** (self.n_conditions - 1) and len(dic[1]) == 2 ** (self.n_conditions - 1):
            for i in range(len(dic)):
                d = [string_bool(x) for x in dic[i]]
                col_sum = np.sum(d, 0)
                if len(dic[0]) in col_sum:
                    return self._semantic_keys[np.where(col_sum == len(dic[0]))[0][0]]
        return 0

    def _dichotomy_from_key(self, key):
        dics, keys = self._find_semantic_dichotomies()
        if key in keys:
            dic = dics[np.where(np.asarray(keys) == key)[0][0]]
        else:
            raise RuntimeError(
                "\n[dichotomy_from_key] The specified key does not correspond to a semantic dichotomy. Check the key value.")

        return dic

    def _generate_semantic_vectors(self):
        for condition_vec in self._condition_vectors:
            semantic_vector = '('
            for i, sk in enumerate(self._semantic_keys):
                semantic_values = list(self.conditions[sk])
                semantic_vector += semantic_values[condition_vec[i]]
            semantic_vector = semantic_vector + ')'
            self._semantic_vectors[string_bool(condition_vec)] = semantic_vector

    def _compute_centroids(self):
        self.centroids = {w: np.hstack([np.nanmean(r, 0) for r in self.conditioned_rasters[w]])
                          for w in self.conditioned_rasters.keys()}

    def _zscore_activity(self):
        keys = [string_bool(w) for w in self._condition_vectors]
        for n in range(self.n_brains):
            n_neurons = self.conditioned_rasters[keys[0]][n].shape[1]
            for i in range(n_neurons):
                r = np.hstack([self.conditioned_rasters[key][n][:, i] for key in keys])
                m = np.nanmean(r)
                std = np.nanstd(r)
                if std:
                    for key in keys:
                        self.conditioned_rasters[key][n][:, i] = (self.conditioned_rasters[key][n][:, i] - m) / std

    def _print(self, string):
        if self._verbose:
            print(string)

    # null model utilities

    def _generate_random_subset(self, n):
        if n < self.n_neurons:
            self.subset = np.random.choice(self.n_neurons, n, replace=False)
        else:
            self.subset = np.arange(self.n_neurons)

    def _reset_random_subset(self):
        self.subset = np.arange(self.n_neurons)

    def _shuffle_conditioned_arrays(self, dic):
        """
        the null model is built by interchanging trials between conditioned arrays that are in different
        dichotomies but have only hamming distance = 1. This ensures that even in the null model the other
        conditions (i.e., the one that do not define the dichotomy), are balanced during sampling.
        So if my dichotomy is [1A, 1B] vs [0A, 0B], I will change trials between 1A and 0A, so that,
        with oversampling, I will then ensure balance between A and B.
        If the dichotomy is not semantic, then I'll probably have to interchange between conditions regardless
        (to be implemented).

        :param dic: The dichotomy to be decoded
        """
        # if the dichotomy is semantic, shuffle between rasters at semantic distance=1
        if self._dic_key(dic):
            set_A = dic[0]
            set_B = dic[1]

            for i in range(len(set_A)):
                for j in range(len(set_B)):
                    test_condition_A = set_A[i]
                    test_condition_B = set_B[j]
                    if hamming(string_bool(test_condition_A), string_bool(test_condition_B)) == 1:
                        for n in range(self.n_brains):
                            # select conditioned rasters
                            arrayA = np.copy(self.conditioned_rasters[test_condition_A][n])
                            arrayB = np.copy(self.conditioned_rasters[test_condition_B][n])

                            # select conditioned trial index
                            trialA = np.copy(self.conditioned_trial_index[test_condition_A][n])
                            trialB = np.copy(self.conditioned_trial_index[test_condition_B][n])

                            n_trials_A = len(np.unique(trialA))
                            n_trials_B = len(np.unique(trialB))

                            # assign randomly trials between the two conditioned rasters, keeping the same
                            # number of trials between the two conditions

                            all_rasters = []
                            all_trials = []

                            for index in np.unique(trialA):
                                all_rasters.append(arrayA[trialA == index, :])
                                all_trials.append(trialA[trialA == index])

                            for index in np.unique(trialB):
                                all_rasters.append(arrayB[trialB == index, :])
                                all_trials.append(trialB[trialB == index])

                            all_trial_index = np.arange(n_trials_A + n_trials_B).astype(int)
                            np.random.shuffle(all_trial_index)

                            new_rasters_A = [all_rasters[iA] for iA in all_trial_index[:n_trials_A]]
                            new_rasters_B = [all_rasters[iB] for iB in all_trial_index[n_trials_A:]]

                            new_trials_A = [all_trials[iA] for iA in all_trial_index[:n_trials_A]]
                            new_trials_B = [all_trials[iB] for iB in all_trial_index[n_trials_A:]]

                            self.conditioned_rasters[test_condition_A][n] = np.vstack(new_rasters_A)
                            self.conditioned_rasters[test_condition_B][n] = np.vstack(new_rasters_B)

                            self.conditioned_trial_index[test_condition_A][n] = np.hstack(new_trials_A)
                            self.conditioned_trial_index[test_condition_B][n] = np.hstack(new_trials_B)

        else:
            for n in range(self.n_brains):
                # select conditioned rasters
                for iteration in range(10):
                    all_conditions = list(self._semantic_vectors.keys())
                    all_data = np.vstack([self.conditioned_rasters[cond][n] for cond in all_conditions])
                    all_trials = np.hstack([self.conditioned_trial_index[cond][n] for cond in all_conditions])
                    all_n_trials = {cond: len(np.unique(self.conditioned_trial_index[cond][n])) for cond in
                                    all_conditions}

                    unique_trials = np.unique(all_trials)
                    np.random.shuffle(unique_trials)

                    i = 0
                    for cond in all_conditions:
                        cond_trials = unique_trials[i:i + all_n_trials[cond]]
                        new_cond_array = []
                        new_cond_trial = []
                        for trial in cond_trials:
                            new_cond_array.append(all_data[all_trials == trial])
                            new_cond_trial.append(all_trials[all_trials == trial])
                        self.conditioned_rasters[cond][n] = np.vstack(new_cond_array)
                        self.conditioned_trial_index[cond][n] = np.hstack(new_cond_trial)
                        i += all_n_trials[cond]

        if not self._check_trial_availability():  # if the trial distribution is not cross validatable, redo the shuffling
            print("Note: re-shuffling arrays")
            self._order_conditioned_rasters()
            self._shuffle_conditioned_arrays(dic)

    def _rototraslate_conditioned_rasters(self):
        # DEPCRECATED

        for i in range(self.n_brains):
            # brain_means = np.vstack([np.nanmean(self.conditioned_rasters[key][i], 0) for key in self.conditioned_rasters.keys()])
            # mean_centroid = np.nanmean(brain_means, axis=0)
            for w in self.conditioned_rasters.keys():
                raster = self.conditioned_rasters[w][i]
                rotation = np.arange(raster.shape[1]).astype(int)
                np.random.shuffle(rotation)
                raster = raster[:, rotation]
                # mean = np.nanmean(raster, 0)
                # randomdir = np.random.rand()-0.5
                # randomdir = randomdir/np.sqrt(np.dot(randomdir, randomdir))
                # vector_from_mean_centroid = mean - mean_centroid
                # distance_from_mean_centroid = np.sqrt(np.dot(vector_from_mean_centroid, vector_from_mean_centroid))
                # raster = raster - vector_from_mean_centroid + randomdir*distance_from_mean_centroid
                self.conditioned_rasters[w][i] = raster

    def _order_conditioned_rasters(self):
        for w in self.conditioned_rasters.keys():
            self.conditioned_rasters[w] = self.ordered_conditioned_rasters[w].copy()
            self.conditioned_trial_index[w] = self.ordered_conditioned_trial_index[w].copy()

    def _check_trial_availability(self):
        if self._debug:
            print('\nCheck trial availability')
        for k in self.conditioned_trial_index:
            for i, ti in enumerate(self.conditioned_trial_index[k]):
                if self._debug:
                    print(k, 'raster %u:' % i, np.unique(ti).shape[0])
                    print(ti)
                if np.unique(ti).shape[0] < 2:
                    return False
        return True
