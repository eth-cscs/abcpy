from abc import ABCMeta, abstractmethod

import numpy as np
from glmnet import LogitNet
from sklearn import linear_model

from abcpy.utils import wass_dist


class Distance(metaclass=ABCMeta):
    """This abstract base class defines how the distance between the observed and
    simulated data should be implemented.
    """

    def __init__(self, statistics_calc):
        """The constructor of a sub-class must accept a non-optional statistics
        calculator as a parameter; then, it must call the __init__ method of the parent class. This ensures that the
        object is initialized correctly so that the _calculate_summary_stat private method can be called when computing
        the distances.

        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        """

        self.statistics_calc = statistics_calc

        # Since the observations do always stay the same, we can save the
        #  summary statistics of them and not recalculate it each time
        self.s1 = None
        self.data_set = None
        self.dataSame = False

    @abstractmethod
    def distance(self, d1, d2):
        """To be overwritten by any sub-class: should calculate the distance between two
        sets of data d1 and d2 using their respective statistics.
        
        Usually, calling the _calculate_summary_stat private method to obtain statistics from the datasets is handy;
        that also keeps track of the first provided dataset (which is the observation in ABCpy inference schemes) and
        avoids computing the statistics for that multiple times.

        Notes
        -----
        The data sets d1 and d2 are array-like structures that contain n1 and n2 data
        points each.  An implementation of the distance function should work along
        the following steps:

        1. Transform both input sets dX = [ dX1, dX2, ..., dXn ] to sX = [sX1, sX2,
        ..., sXn] using the statistics object. See _calculate_summary_stat method.

        2. Calculate the mutual desired distance, here denoted by - between the
        statistics; for instance, dist = [s11 - s21, s12 - s22, ..., s1n - s2n] (in some cases however you
        may want to compute all pairwise distances between statistics elements.

        Important: any sub-class must not calculate the distance between data sets
        d1 and d2 directly. This is the reason why any sub-class must be
        initialized with a statistics object.

        Parameters
        ----------
        d1: Python list
            Contains n1 data points.
        d2: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.float
            The distance between the two input data sets.
        """

        raise NotImplementedError

    @abstractmethod
    def dist_max(self):
        """To be overwritten by sub-class: should return maximum possible value of the
        desired distance function.

        Examples
        --------
        If the desired distance maps to :math:`\mathbb{R}`, this method should return numpy.inf.

        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        raise NotImplementedError

    def _calculate_summary_stat(self, d1, d2):
        """Helper function that extracts the summary statistics s1 and s2 from d1 and
        d2 using the statistics object stored in self.statistics_calc. This stores s1 for the purpose of checking
        whether that is repeated in next calls to the function, and avoiding computing the statitistics for the same
        dataset several times.

        Parameters
        ----------
        d1 : array-like
            d1 contains n data sets.
        d2 : array-like
            d2 contains m data sets.

        Returns
        -------
        tuple
            Tuple containing numpy.ndarray's with the summary statistics extracted from d1 and d2.
        """
        if not isinstance(d1, list):
            raise TypeError('Data is not of allowed types')
        if not isinstance(d2, list):
            raise TypeError('Data is not of allowed types')

        # Check whether d1 is same as self.data_set
        if self.data_set is not None:
            # check that the the observations have the same length; if not, they can't be the same:
            if len(d1) != len(self.data_set):
                self.dataSame = False
            elif len(np.array(d1[0]).reshape(-1, )) == 1:
                self.dataSame = self.data_set == d1
            else:
                self.dataSame = all([(np.array(self.data_set[i]) == np.array(d1[i])).all() for i in range(len(d1))])

        # Extract summary statistics from the dataset
        if self.s1 is None or self.dataSame is False:
            self.s1 = self.statistics_calc.statistics(d1)
            self.data_set = d1

        s2 = self.statistics_calc.statistics(d2)

        return self.s1, s2


class Euclidean(Distance):
    """
    This class implements the Euclidean distance between two vectors.

    The maximum value of the distance is np.inf.
    """

    def __init__(self, statistics):
        """
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        """
        super(Euclidean, self).__init__(statistics)

    def distance(self, d1, d2):
        """Calculates the distance between two datasets, by computing Euclidean distance between each element of d1 and
        d2 and taking their average.

        Parameters
        ----------
        d1: Python list
            Contains n1 data points.
        d2: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.float
            The distance between the two input data sets.
        """
        s1, s2 = self._calculate_summary_stat(d1, d2)

        # compute distance between the statistics
        dist = np.zeros(shape=(s1.shape[0], s2.shape[0]))
        for ind1 in range(0, s1.shape[0]):
            for ind2 in range(0, s2.shape[0]):
                dist[ind1, ind2] = np.sqrt(np.sum(pow(s1[ind1, :] - s2[ind2, :], 2)))

        return dist.mean()

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """
        return np.inf


class PenLogReg(Distance):
    """
    This class implements a distance measure based on the classification accuracy.

    The classification accuracy is calculated between two dataset d1 and d2 using
    lasso penalized logistics regression and return it as a distance. The lasso
    penalized logistic regression is done using glmnet package of Friedman et. al.
    [2]. While computing the distance, the algorithm automatically chooses
    the most relevant summary statistics as explained in Gutmann et. al. [1].
    The maximum value of the distance is 1.0.

    [1] Gutmann, M. U., Dutta, R., Kaski, S., & Corander, J. (2018). Likelihood-free inference via classification.
    Statistics and Computing, 28(2), 411-425.

    [2] Friedman, J., Hastie, T., and Tibshirani, R. (2010). Regularization
    paths for generalized linear models via coordinate descent. Journal of Statistical
    Software, 33(1), 1–22.
    """

    def __init__(self, statistics):
        """
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        """
        super(PenLogReg, self).__init__(statistics)

        self.n_folds = 10  # for cross validation in PenLogReg

    def distance(self, d1, d2):
        """Calculates the distance between two datasets.

        Parameters
        ----------
        d1: Python list
            Contains n1 data points.
        d2: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.float
            The distance between the two input data sets.
        """
        s1, s2 = self._calculate_summary_stat(d1, d2)
        self.n_simulate = s1.shape[0]

        if not s2.shape[0] == self.n_simulate:
            raise RuntimeError("The number of simulations in the two data sets should be the same in order for "
                               "the classification accuracy implemented in PenLogReg to be a proper distance. Please "
                               "check that `n_samples` in the `sample()` method for the sampler is equal to "
                               "the number of datasets in the observations.")

        # compute distance between the statistics
        training_set_features = np.concatenate((s1, s2), axis=0)
        label_s1 = np.zeros(shape=(len(s1), 1))
        label_s2 = np.ones(shape=(len(s2), 1))
        training_set_labels = np.concatenate((label_s1, label_s2), axis=0).ravel()

        groups = np.repeat(np.arange(self.n_folds), np.int(np.ceil(self.n_simulate / self.n_folds)))
        groups = groups[:self.n_simulate].tolist()
        groups += groups  # duplicate it as groups need to be defined for both datasets
        m = LogitNet(alpha=1, n_splits=self.n_folds)  # note we are not using random seed here!
        m = m.fit(training_set_features, training_set_labels, groups=groups)
        distance = 2.0 * (m.cv_mean_score_[np.where(m.lambda_path_ == m.lambda_max_)[0][0]] - 0.5)

        return distance

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """
        return 1.0


class LogReg(Distance):
    """This class implements a distance measure based on the classification
    accuracy [1]. The classification accuracy is calculated between two dataset d1 and d2 using
    logistics regression and return it as a distance. The maximum value of the distance is 1.0.

    [1] Gutmann, M. U., Dutta, R., Kaski, S., & Corander, J. (2018). Likelihood-free inference via classification.
    Statistics and Computing, 28(2), 411-425.
    """

    def __init__(self, statistics, seed=None):
        """
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        seed : integer, optionl
            Seed used to initialize the Random Numbers Generator used to determine the (random) cross validation split
            in the Logistic Regression classifier.
        """

        super(LogReg, self).__init__(statistics)
        # seed is used for a RandomState for the random split in the LogisticRegression classifier:
        self.rng = np.random.RandomState(seed=seed)

    def distance(self, d1, d2):
        """Calculates the distance between two datasets.

        Parameters
        ----------
        d1: Python list
            Contains n1 data points.
        d2: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.float
            The distance between the two input data sets.
        """

        s1, s2 = self._calculate_summary_stat(d1, d2)

        # compute distance between the statistics
        training_set_features = np.concatenate((s1, s2), axis=0)
        label_s1 = np.zeros(shape=(len(s1), 1))
        label_s2 = np.ones(shape=(len(s2), 1))
        training_set_labels = np.concatenate((label_s1, label_s2), axis=0).ravel()

        reg_inv = 1e5
        log_reg_model = linear_model.LogisticRegression(C=reg_inv, penalty='l1', max_iter=1000, solver='liblinear',
                                                        random_state=self.rng.randint(0, np.iinfo(np.uint32).max))
        log_reg_model.fit(training_set_features, training_set_labels)
        score = log_reg_model.score(training_set_features, training_set_labels)
        distance = 2.0 * (score - 0.5)
        return distance

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """
        return 1.0


class Wasserstein(Distance):
    """This class implements a distance measure based on the 2-Wasserstein distance, as used in [1]. This considers the 
    several simulations/observations in the datasets as iid samples from the model for a fixed parameter value/from the 
    data generating model, and computes the 2-Wasserstein distance between the empirical distributions those 
    simulations/observations define.  

    [1] Bernton, E., Jacob, P.E., Gerber, M. and Robert, C.P. (2019), Approximate Bayesian computation with the
    Wasserstein distance. J. R. Stat. Soc. B, 81: 235-269. doi:10.1111/rssb.12312
    """

    def __init__(self, statistics, num_iter_max=100000):
        """
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        num_iter_max : integer, optional
            The maximum number of iterations in the linear programming algorithm to estimate the Wasserstein distance.
            Default to 100000.
        """

        super(Wasserstein, self).__init__(statistics)

        self.num_iter_max = num_iter_max

    def distance(self, d1, d2):
        """Calculates the distance between two datasets.
        
        Parameters
        ----------
        d1: Python list
            Contains n1 data points.
        d2: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.float
            The distance between the two input data sets.
        """
        s1, s2 = self._calculate_summary_stat(d1, d2)

        # compute the Wasserstein distance between the empirical distributions:
        return wass_dist(samples_1=s1, samples_2=s2, num_iter_max=self.num_iter_max)

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf
