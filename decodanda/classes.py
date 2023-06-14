from __future__ import annotations
from typing import Tuple, Union, Callable, Iterable, List, Mapping, Optional, Any
from multiprocessing import Pool
import copy


# support for scikit-learn-intelex
from ._dev import INTELX
if INTELX:
    from sklearnex import patch_sklearn
    patch_sklearn(verbose=False)  # Patch scikit-learn with intel extension, must be done before importing sklearn


from sklearn.svm import LinearSVC
import numpy as np
from sklearn.base import clone
from tqdm import tqdm


from ._dev import identify_calling_function
from ._defaults import classifier_parameters, DecodandaParameters
from .utilities import generate_binary_words, string_bool, sample_training_testing_from_rasters, \
    log_dichotomy, hamming, generate_dichotomies, contiguous_chunking, non_contiguous_mask, \
    generate_binary_conditions, compute_label
from .geometry import compute_centroids


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

        # decodanda parameters; making call to hashmap each time but more pythonic & flexible--also faster fwiw
        # kwargs take precedence over passed params dictionary -- DAO 06/10/2023
        if isinstance(decodanda_params, DecodandaParameters):
            decodanda_params = vars(decodanda_params)  # in case a user passes the actual dataclass object
        self._parameters = DecodandaParameters.build([kwargs, decodanda_params])
        # it's still protected as before,
        # but we have a dedicated getter/setter now that allows the user to view & change the values
        # DAO 06/11/2023

        # deriving dataset(s) attributes
        self.n_datasets = len(self.data)
        self.n_conditions = len(self.conditions)
        self.n_neurons = 0
        self.n_brains = 0
        self.which_brain = []

        # keys and stuff
        self._condition_vectors = generate_binary_words(self.n_conditions)
        self._semantic_keys = list(self.conditions.keys())
        self._semantic_vectors = {string_bool(w): [] for w in generate_binary_words(self.n_conditions)}
        self._generate_semantic_vectors()

        # results 6/11/2023; memory costs are small, save it all DAO 06/11/2023
        self.real_performance = {}
        self.real_performance_folds = {}
        self.real_performance_weights = {}
        self.null_performance = {}
        self.null_performance_folds = {}
        self.null_performance_weights = {}

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
            if not self._parameters.get("fault_tolerance"):
                raise RuntimeError(
                    f"\n{identify_calling_function()}\tNo dataset passed the minimum data threshold for conditioned "
                    f"arrays.\n\t\t Check for mutually-exclusive conditions or try using less restrictive thresholds."
                )
        else:
            # derived attributes
            self.centroids = compute_centroids(self.conditioned_rasters)

            # null model variables
            self.random_translations = {string_bool(w): [] for w in self._condition_vectors}
            self.subset = np.arange(self.n_neurons)

            self.ordered_conditioned_rasters = {}
            self.ordered_conditioned_trial_index = {}

            for w in self.conditioned_rasters.keys():
                self.ordered_conditioned_rasters[w] = self.conditioned_rasters[w].copy()
                self.ordered_conditioned_trial_index[w] = self.conditioned_trial_index[w].copy()

    @property
    def parameters(self) -> dict:
        """
        Decodanda parameters
        """
        return self._parameters

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
    @staticmethod
    def parameter_info() -> None:
        DecodandaParameters().hint_types()

    @classmethod
    def _train(cls: Decodanda, classifier: Callable, subset: np.ndarray, training_raster_a: np.ndarray,
               training_raster_b: np.ndarray, label_a: np.ndarray, label_b: np.ndarray) -> Callable:

        training_labels_a = np.repeat(label_a, training_raster_a.shape[0]).astype(object)
        training_labels_b = np.repeat(label_b, training_raster_b.shape[0]).astype(object)

        training_raster = np.vstack([training_raster_a, training_raster_b])
        training_labels = np.hstack([training_labels_a, training_labels_b])

        classifier = clone(classifier)

        training_raster = training_raster[:, subset]

        return classifier.fit(training_raster, training_labels)

    @classmethod
    def _test(cls: Decodanda, classifier: Callable, subset: np.ndarray, testing_raster_a: np.ndarray,
              testing_raster_b: np.ndarray, label_a: np.ndarray, label_b: np.ndarray) -> Tuple[float, np.ndarray]:

        testing_labels_a = np.repeat(label_a, testing_raster_a.shape[0]).astype(object)
        testing_labels_b = np.repeat(label_b, testing_raster_b.shape[0]).astype(object)

        testing_raster = np.vstack([testing_raster_a, testing_raster_b])
        testing_labels = np.hstack([testing_labels_a, testing_labels_b])

        testing_raster = testing_raster[:, subset]

        return classifier.score(testing_raster, testing_labels)

    @classmethod
    def _one_cv_step(cls: Decodanda,
                     classifier: Callable,
                     dichotomy: Union[str, List[str]],
                     training_fraction: float,
                     semantic_vectors: dict,
                     conditioned_rasters: dict,
                     conditioned_trial_index: dict,
                     ndata: int,
                     subset: np.ndarray,
                     scale: Optional[Callable] = None,
                     testing_trials: Optional[list] = None,
                     ) -> Tuple[float, np.ndarray, np.ndarray]:

        set_a = dichotomy[0]
        # abstracted DAO 06/11/2023
        label_a = compute_label(semantic_vectors, set_a)

        set_b = dichotomy[1]
        # abstracted DAO 06/11/2023
        label_b = compute_label(semantic_vectors, set_b)

        training_array_a = []
        training_array_b = []
        testing_array_a = []
        testing_array_b = []

        # allow for unbalanced dichotomies
        n_conditions_a = float(len(dichotomy[0]))
        n_conditions_b = float(len(dichotomy[1]))
        fraction = n_conditions_a / n_conditions_b

        for d in set_a:
            training, testing = \
                sample_training_testing_from_rasters(conditioned_rasters[d],
                                                     int(ndata / fraction),
                                                     training_fraction,
                                                     conditioned_trial_index[d],
                                                     testing_trials=testing_trials)

            training_array_a.append(training)
            testing_array_a.append(testing)

        for d in set_b:
            training, testing = sample_training_testing_from_rasters(conditioned_rasters[d],
                                                                     int(ndata),
                                                                     training_fraction,
                                                                     conditioned_trial_index[d],
                                                                     testing_trials=testing_trials)
            training_array_b.append(training)
            testing_array_b.append(testing)

        training_array_a = np.vstack(training_array_a)
        training_array_b = np.vstack(training_array_b)
        testing_array_a = np.vstack(testing_array_a)
        testing_array_b = np.vstack(testing_array_b)

        if scale:
            scaler = scale().fit(np.vstack([training_array_a, training_array_b]))
            training_array_a = scaler.transform(training_array_a, copy=False)
            training_array_b = scaler.transform(training_array_b, copy=False)
            testing_array_a = scaler.transform(testing_array_a, copy=False)
            testing_array_b = scaler.transform(testing_array_b, copy=False)

        classifier = cls._train(classifier=classifier,
                                subset=subset,
                                training_raster_a=training_array_a,
                                training_raster_b=training_array_b,
                                label_a=label_a,
                                label_b=label_b)

        if hasattr(classifier, 'coef_'):
            weights = classifier.coef_
        else:
            weights = None

        performance = cls._test(classifier=classifier,
                                subset=subset,
                                testing_raster_a=testing_array_a,
                                testing_raster_b=testing_array_b,
                                label_a=label_a,
                                label_b=label_b)

        return performance, weights

    @parameters.setter
    def parameters(self, value: Union[Mapping, Tuple[str, Any]]) -> Decodanda:
        """
        Set a parameter by passing a key-value tuple or a mapping

        """
        if isinstance(value, Mapping):
            self._parameters = DecodandaParameters().build([value, self._parameters])
        elif isinstance(value, tuple) and isinstance(value[0], str):
            key, value = value
            self._parameters = DecodandaParameters().build([{key: value}, self._parameters])
        else:
            raise TypeError(f"Argument must be a key-value tuple or mapping not {type(value)}")

    # Dichotomy analysis functions

    def decode_dichotomy(self, dichotomy: Union[str, list], shuffled: bool = False) -> np.ndarray:
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
        # INGEST PARAMETERS - Actually saves a few hundred ms / model, relevant when 100's of small models
        # DAO 06/11/2023
        ndata = self._parameters.get("ndata")
        max_conditioned_data = self._parameters.get("max_conditioned_data")
        verbose = self._parameters.get("verbose")
        cross_validations = self._parameters.get("cross_validations")
        parallel = self._parameters.get("parallel")
        subset = self.subset
        training_fraction = self._parameters.get("training_fraction")
        testing_trials = self._parameters.get("testing_trials")
        scale = self._parameters.get("scale")

        # Estimate n_data if needed if needed
        if ndata is None and self.n_brains == 1:
            ndata = max_conditioned_data
            self._parameters["ndata"] = ndata
        if ndata is None and self.n_brains > 1:
            ndata = max(max_conditioned_data, 2 * self.n_neurons)
            self._parameters["ndata"] = ndata
        if shuffled:
            self._shuffle_conditioned_arrays(dichotomy)

        if verbose and not shuffled:
            print(f"{dichotomy=}, {ndata=}")
            log_dichotomy(self, dichotomy, ndata, 'Decoding')
            count = tqdm(range(cross_validations), delay=1)
        else:
            count = range(cross_validations)

        if parallel:
            pool = Pool()
            performances = pool.map(_CrossValidator(classifier=self.classifier,
                                                    dichotomy=dichotomy,
                                                    training_fraction=training_fraction,
                                                    semantic_vectors=self._semantic_vectors,
                                                    conditioned_rasters=self.conditioned_rasters,
                                                    conditioned_trial_index=self.conditioned_trial_index,
                                                    ndata=ndata,
                                                    subset=self.subset,
                                                    scale=scale,
                                                    testing_trials=testing_trials),
                                    range(cross_validations))
        else:
            if verbose and not shuffled:
                print('\nLooping over decoding cross validation folds:')
            performances = [self._one_cv_step(classifier=self.classifier,
                                              dichotomy=dichotomy,
                                              training_fraction=training_fraction,
                                              semantic_vectors=self._semantic_vectors,
                                              conditioned_rasters=self.conditioned_rasters,
                                              conditioned_trial_index=self.conditioned_trial_index,
                                              ndata=ndata,
                                              subset=subset,
                                              scale=scale,
                                              testing_trials=testing_trials) for _ in count]

        # noinspection PyUnboundLocalVariable
        scores, weights = map(list, zip(*performances))
        scores = np.asarray(scores)
        weights = np.asarray(weights)

        if shuffled:
            self._order_conditioned_rasters()

        return np.nanmean(scores), scores, weights

    # Dichotomy analysis functions with null model

    def decode_with_nullmodel(self, dichotomy: Union[str, list]) -> Tuple[Union[list, np.ndarray], np.ndarray]:
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
            testing_trials:
                if specified, data sampled from the specified trial numbers will be used for testing, and the remaining ones for training.

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
        if isinstance(dichotomy, str):  # DAO 06/08/2023
            dichotomy = self._dichotomy_from_key(dichotomy)

        # Decode Real Data
        real_model = \
            self.decode_dichotomy(dichotomy=dichotomy, shuffled=False)

        # Report Progress
        if self._parameters.get("verbose"):
            print(f"\n{identify_calling_function()}\t Fraction Correct: {np.nanmean(real_model[0]):.2f}")
            print(f"\n{identify_calling_function()}\t Looping over null model shuffles")
            count = tqdm(range(self._parameters.get("n_shuffles")), delay=1)
            # delay prevents progress bar duplication glitch
        else:
            count = range(self._parameters.get("n_shuffles"))

        # Decode Null data
        null_models = [self.decode_dichotomy(dichotomy=dichotomy, shuffled=True) for _ in count]
        null_model_performance = [performance[0] for performance in null_models]

        # Match formatting
        null_model_folds = [folds[1] for folds in null_models]
        null_model_weights = [weights[2] for weights in null_models]
        null_models = (null_model_performance, null_model_folds, null_model_weights)

        return real_model, null_models

    # Decoding analysis for semantic dichotomies

    def decode(self, **kwargs) -> Decodanda:

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
        testing_trials:
            if specified, data sampled from the specified trial numbers will be used for testing, and the remaining ones for training.
        non_semantic:
            if True, non-semantic dichotomies (i.e., dichotomies that do not correspond to a variable) will also be decoded.

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

        # Streamlined parameter ingestion. All parameters automatically passed from main structure unless overriden
        # by kwargs DAO - 06/11/2023
        if kwargs:
            # noinspection PyArgumentList
            self._parameters = DecodandaParameters().build([kwargs, self._parameters])

        semantic_dics, semantic_keys = self._find_semantic_dichotomies()

        # General DAO 06/11/2023
        for semantic_dic, semantic_key in zip(semantic_dics, semantic_keys):

            if self._parameters.get("verbose"):
                print(f"\n{identify_calling_function()}\tTesting decoding performance for semantic dichotomy: "
                      f"{semantic_key}")

            # Abstraction
            real_model, null_models = self.decode_with_nullmodel(semantic_dic)

            self.real_performance[semantic_key] = real_model[0]
            self.real_performance_folds[semantic_key] = real_model[1]
            self.real_performance_weights[semantic_key] = real_model[2]

            self.null_performance[semantic_key] = null_models[0]
            self.null_performance_folds = null_models[1]
            self.null_performance_weights = null_models[2]

        return self.real_performance, self.null_performance

    def _divide_data_into_conditions(self, datasets: dict) -> Decodanda:
        # TODO: make sure conditions don't overlap somehow
        # TODO: This is too complex and needs broken down for maintainability

        for si, dataset in enumerate(datasets):

            if self._parameters.get("verbose"):
                if hasattr(dataset, 'name'):
                    print(f"\n{identify_calling_function()}\tBuilding conditioned rasters for dataset {dataset.name}")
                else:
                    print(f"\n{identify_calling_function()}\tBuilding conditioned rasters for dataset {si}")

            dataset_conditioned_rasters = {}
            dataset_conditioned_trial_index = {}

            # exclude inactive neurons across the specified conditions
            array = dataset.get(self._parameters.get("neural_attr"))
            total_mask = np.zeros(len(array)) > 0

            for condition_vec in self._condition_vectors:
                mask = np.ones(len(array)) > 0
                for i, sk in enumerate(self._semantic_keys):
                    semantic_values = list(self.conditions[sk])
                    mask_i = self.conditions[sk][semantic_values[condition_vec[i]]](dataset)
                    mask = mask & mask_i
                total_mask = total_mask | mask

            min_activity_mask = np.sum(array[total_mask] > 0, 0) >= self._parameters.get("min_activations_per_cell")

            for condition_vec in self._condition_vectors:
                # get the array from the dataset object
                array = dataset.get(self._parameters.get("neural_attr"))
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
                def condition_no(cond):  # noqa
                    no = 0
                    for i in range(len(cond)):
                        no += cond[i] * 10 ** (i + 2)
                    return no

                if self._parameters.get("trial_attr") is not None:
                    conditioned_trial = dataset.get(self._parameters.get("trial_attr"))[mask]
                elif self._trial_chunk is None:
                    if self._parameters.get("verbose"):
                        print('[Decodanda]\tUsing contiguous chunks of the same labels as trials.')
                    conditioned_trial = contiguous_chunking(mask)[mask]
                    conditioned_trial += condition_no(condition_vec)
                else:
                    conditioned_trial = contiguous_chunking(mask, self._trial_chunk)[mask]
                    conditioned_trial += condition_no(condition_vec)

                if self._parameters.get("exclude_contiguous_trials"):
                    contiguous_chunks = contiguous_chunking(mask)[mask]
                    nc_mask = non_contiguous_mask(contiguous_chunks, conditioned_trial)
                    conditioned_raster = conditioned_raster[nc_mask, :]
                    conditioned_trial = conditioned_trial[nc_mask]

                # exclude empty time bins (only for binary discrete decoding)
                if self._parameters.get("exclude_silent"):
                    active_mask = np.sum(conditioned_raster, 1) > 0
                    conditioned_raster = conditioned_raster[active_mask, :]
                    conditioned_trial = conditioned_trial[active_mask]

                # squeeze into trials
                if self._parameters.get("trial_average"):
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

                if self._parameters.get("verbose"):
                    for idx, semantic_key in enumerate(self._semantic_keys):
                        print(f"\t\t\t({semantic_key} = {list(self.conditions[semantic_key])[condition_vec[idx]]}):\t "
                              f"Selected {conditioned_raster.shape[0]} time bins out of {len(array)}, "
                              f"divided into {len(np.unique(conditioned_trial))} trials")

            dataset_conditioned_data = [r.shape[0] for r in list(dataset_conditioned_rasters.values())]
            dataset_conditioned_trials = [len(np.unique(c)) for c in list(dataset_conditioned_trial_index.values())]

            self._parameters["max_conditioned_data"] = \
                max([self._parameters.get("max_conditioned_data"), np.max(dataset_conditioned_data)])
            self._parameters["min_conditioned_data"] =\
                min([self._parameters.get("min_conditioned_data"), np.min(dataset_conditioned_data)])

            # if the dataset has enough data for each condition, append it to the main data dictionary

            if np.min(dataset_conditioned_data) >= self._parameters.get("min_data_per_condition") and \
                    np.min(dataset_conditioned_trials) >= self._parameters.get("min_trials_per_condition"):
                for cv in self._condition_vectors:
                    self.conditioned_rasters[string_bool(cv)].append(dataset_conditioned_rasters[string_bool(cv)])
                    self.conditioned_trial_index[string_bool(cv)].append(
                        dataset_conditioned_trial_index[string_bool(cv)])
                self.n_brains += 1
                self.n_neurons += list(dataset_conditioned_rasters.values())[0].shape[1]
                self.which_brain.append(np.ones(list(dataset_conditioned_rasters.values())[0].shape[1]) * self.n_brains)
            else:
                if self._parameters.get("verbose"):
                    print('\n\t\t\t===> dataset discarded for insufficient data.\n')
        if len(self.which_brain):
            self.which_brain = np.hstack(self.which_brain)

    def _find_semantic_dichotomies(self) -> Tuple[list, list]:
        # I think this exports possible vals x conditions
        d_keys, dics = generate_dichotomies(self.n_conditions)
        semantic_dics = []
        semantic_keys = []

        for _, dic in enumerate(dics):
            d = [string_bool(x) for x in dic[0]]
            col_sum = np.sum(d, 0)
            if (0 in col_sum) or (len(dic[0]) in col_sum):
                semantic_dics.append(dic)
                semantic_keys.append(self._semantic_keys[np.where(col_sum == len(dic[0]))[0][0]])
        return semantic_dics, semantic_keys

    def _dichotomy_from_key(self, key: str) -> Union[str, List[str]]:
        dics, keys = self._find_semantic_dichotomies()
        if key in keys:
            dic = dics[np.where(np.asarray(keys) == key)[0][0]]
        else:
            raise RuntimeError(
                f"\n{identify_calling_function()} "
                f"The specified key does not correspond to a semantic dichotomy. Check the key value.")

        return dic

    def _shuffle_conditioned_arrays(self, dic) -> Decodanda:
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
                for _ in range(10):
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

    def _check_trial_availability(self) -> bool:
        if self._parameters.get("debug"):
            print('\nCheck trial availability')
        for k in self.conditioned_trial_index:
            for i, ti in enumerate(self.conditioned_trial_index[k]):
                if self._parameters.get("debug"):
                    print(f"{k} raster {i} {np.unique(ti).shape[0]} {ti}")
                if np.unique(ti).shape[0] < 2:
                    return False
        return True

    def _order_conditioned_rasters(self) -> Decodanda:
        for w in self.conditioned_rasters.keys():
            self.conditioned_rasters[w] = self.ordered_conditioned_rasters[w].copy()
            self.conditioned_trial_index[w] = self.ordered_conditioned_trial_index[w].copy()

    def _dic_key(self, dic):
        if len(dic[0]) == 2 ** (self.n_conditions - 1) and len(dic[1]) == 2 ** (self.n_conditions - 1):
            for i in range(len(dic)):
                d = [string_bool(x) for x in dic[i]]
                col_sum = np.sum(d, 0)
                if len(dic[0]) in col_sum:
                    return self._semantic_keys[np.where(col_sum == len(dic[0]))[0][0]]
        return 0

    def _generate_semantic_vectors(self) -> Decodanda:
        for condition_vec in self._condition_vectors:
            semantic_vector = '('
            for i, sk in enumerate(self._semantic_keys):
                semantic_values = list(self.conditions[sk])
                semantic_vector += semantic_values[condition_vec[i]]
            semantic_vector = semantic_vector + ')'
            self._semantic_vectors[string_bool(condition_vec)] = semantic_vector

    def _zscore_activity(self) -> Decodanda:
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

    def _generate_random_subset(self, n: int) -> Decodanda:
        if n < self.n_neurons:
            self.subset = np.random.choice(self.n_neurons, n, replace=False)
        else:
            self.subset = np.arange(self.n_neurons)

    def _reset_random_subset(self) -> Decodanda:
        self.subset = np.arange(self.n_neurons)


class _CrossValidator(Decodanda):
    # noinspection PyMissingConstructor
    def __init__(self,
                 classifier: Callable,
                 dichotomy: Union[str, List[str]],
                 training_fraction: float,
                 semantic_vectors: dict,
                 conditioned_rasters: dict,
                 conditioned_trial_index: dict,
                 ndata: int,
                 subset: np.ndarray,
                 scale: Optional[Callable] = None,
                 testing_trials: Optional[list] = None):
        self.classifier = classifier
        self.dichotomy = dichotomy
        self.training_fraction = training_fraction
        self.semantic_vectors = semantic_vectors
        self.conditioned_rasters = copy.deepcopy(conditioned_rasters)
        self.conditioned_trial_index = copy.deepcopy(conditioned_trial_index)
        self.ndata = ndata
        self.subset = subset
        self.testing_trials = testing_trials
        self.scale = scale

    def __call__(self, cur_iter: int) -> Tuple[float, np.ndarray, np.ndarray]:
        self.cur_iter = cur_iter
        self.random_state = np.random.RandomState(cur_iter)
        return self._one_cv_step(classifier=self.classifier,
                                 dichotomy=self.dichotomy,
                                 training_fraction=self.training_fraction,
                                 semantic_vectors=self.semantic_vectors,
                                 conditioned_rasters=self.conditioned_rasters,
                                 conditioned_trial_index=self.conditioned_trial_index,
                                 ndata=self.ndata,
                                 subset=self.subset,
                                 scale=self.scale,
                                 testing_trials=self.testing_trials)
