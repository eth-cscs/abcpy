import numpy as np
import warnings
from abc import ABCMeta, abstractmethod
from glmnet import LogitNet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import ot

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

        if self.s1.shape[1] != s2.shape[1]:
            raise ValueError("The dimension of summaries in the two datasets is different; check the dimension of the"
                             " provided observations and simulations.")

        return self.s1, s2


class Divergence(Distance, metaclass=ABCMeta):
    """This is an abstract class which subclasses Distance, and is used as a parent class for all divergence
    estimators; more specifically, it is used for all Distances which compare the empirical distribution of simulations
    and observations."""

    @abstractmethod
    def _estimate_always_positive(self):
        """This returns whether the implemented divergence always returns positive values or not. In fact, some 
        estimators may return negative values, which may break some inference algorithms"""
        raise NotImplementedError


class Euclidean(Distance):
    """
    This class implements the Euclidean distance between two vectors.

    The maximum value of the distance is np.inf.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.

    """

    def __init__(self, statistics_calc):
        super(Euclidean, self).__init__(statistics_calc)

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


class PenLogReg(Divergence):
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

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    """

    def __init__(self, statistics_calc):
        super(PenLogReg, self).__init__(statistics_calc)

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

    def _estimate_always_positive(self):
        return False


class LogReg(Divergence):
    """This class implements a distance measure based on the classification
    accuracy [1]. The classification accuracy is calculated between two dataset d1 and d2 using
    logistics regression and return it as a distance. The maximum value of the distance is 1.0.
    The logistic regression may not converge when using one single sample in each dataset (as for instance by putting
    n_samples_per_param=1 in an inference routine).

    [1] Gutmann, M. U., Dutta, R., Kaski, S., & Corander, J. (2018). Likelihood-free inference via classification.
    Statistics and Computing, 28(2), 411-425.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    seed : integer, optionl
        Seed used to initialize the Random Numbers Generator used to determine the (random) cross validation split
        in the Logistic Regression classifier.
    """

    def __init__(self, statistics_calc, seed=None):
        super(LogReg, self).__init__(statistics_calc)
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

    def _estimate_always_positive(self):
        return False


class Wasserstein(Divergence):
    """This class implements a distance measure based on the 2-Wasserstein distance, as used in [1]. This considers the
    several simulations/observations in the datasets as iid samples from the model for a fixed parameter value/from the
    data generating model, and computes the 2-Wasserstein distance between the empirical distributions those
    simulations/observations define.

    [1] Bernton, E., Jacob, P.E., Gerber, M. and Robert, C.P. (2019), Approximate Bayesian computation with the
    Wasserstein distance. J. R. Stat. Soc. B, 81: 235-269. doi:10.1111/rssb.12312

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    num_iter_max : integer, optional
        The maximum number of iterations in the linear programming algorithm to estimate the Wasserstein distance.
        Default to 100000.

    """

    def __init__(self, statistics_calc, num_iter_max=100000):
        super(Wasserstein, self).__init__(statistics_calc)

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

    def _estimate_always_positive(self):
        return True


class SlicedWasserstein(Divergence):
    """This class implements a distance measure based on the sliced 2-Wasserstein distance, as used in [1].
    This considers the several simulations/observations in the datasets as iid samples from the model for a fixed
    parameter value/from the data generating model, and computes the sliced 2-Wasserstein distance between the
    empirical distributions those simulations/observations define. Specifically, the sliced Wasserstein distance
    is a cheaper version of the Wasserstein distance which consists of projecting the multivariate data on 1d directions
    and computing the 1d Wasserstein distance, which is computationally cheap. The resulting sliced Wasserstein
    distance is obtained by averaging over a given number of projections.

    [1] Nadjahi, K., De Bortoli, V., Durmus, A., Badeau, R., & Şimşekli, U. (2020, May). Approximate bayesian
    computation with the sliced-wasserstein distance. In ICASSP 2020-2020 IEEE International Conference on Acoustics,
    Speech and Signal Processing (ICASSP) (pp. 5470-5474). IEEE.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    n_projections : int, optional
        Number of 1d projections used for estimating the sliced Wasserstein distance. Default value is 50.
    rng : np.random.RandomState, optional
        random number generators used to generate the projections. If not provided, a new one is instantiated.
    """

    def __init__(self, statistics_calc, n_projections=50, rng=np.random.RandomState()):
        super(SlicedWasserstein, self).__init__(statistics_calc)

        self.n_projections = n_projections
        self.rng = rng

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

        return ot.sliced_wasserstein_distance(X_s=s1, X_t=s2, n_projections=self.n_projections, seed=self.rng)

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf

    def _estimate_always_positive(self):
        return True


class GammaDivergence(Divergence):
    """
    This implements an empirical estimator of the gamma-divergence for ABC as suggested in [1]. In [1], the
    gamma-divergence was proposed as a divergence which is robust to outliers. The estimator is based on a nearest
    neighbor density estimate.
    Specifically, this considers the
    several simulations/observations in the datasets as iid samples from the model for a fixed parameter value/from the
    data generating model, and estimates the divergence between the empirical distributions those
    simulations/observations define.

    [1] Fujisawa, M., Teshima, T., Sato, I., & Sugiyama, M.
    γ-ABC: Outlier-robust approximate Bayesian computation based on a
    robust divergence estimator.
    In A. Banerjee and K. Fukumizu (Eds.), Proceedings of 24th
    International Conference on Artificial Intelligence and Statistics
    (AISTATS2021), Proceedings of Machine Learning Research, vol.130,
    pp.1783-1791, online, Apr. 13-15, 2021.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    k : int, optional
        nearest neighbor number for the density estimate. Default value is 1
    gam : float, optional
        the gamma parameter in the definition of the divergence. Default value is 0.1

    """

    def __init__(self, statistics_calc, k=1, gam=0.1):
        super(GammaDivergence, self).__init__(statistics_calc)

        self.k = k  # number of nearest neighbors used in the estimation algorithm
        self.gam = gam

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

        if s1.shape[0] > self.k or s2.shape[0] > self.k:
            assert ValueError(f"The provided value of k ({self.k}) is smaller or equal than the number of samples "
                              f"in one of the two datasets; that should instead be larger")

        # estimate the gamma divergence using the empirical distributions
        return self.skl_estimator_gamma_q(s1=s1, s2=s2, k=self.k, gam=self.gam)

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf

    @staticmethod
    def skl_estimator_gamma_q(s1, s2, k=1, gam=0.1):
        """ Gamma-Divergence estimator using scikit-learn's NearestNeighbours
            s1: (N_1,D) Sample drawn from distribution P
            s2: (N_2,D) Sample drawn from distribution Q
            k: Number of neighbours considered (default 1)
            return: estimated D(P|Q)

            Adapted from code provided by Masahiro Fujisawa (University of Tokyo / RIKEN AIP)
        """

        n, m = len(s1), len(s2)  # NOTE: here different convention of n, m wrt MMD and EnergyDistance
        d = float(s1.shape[1])

        radius = 10  # this is not used at all...
        s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, radius=radius, algorithm='kd_tree').fit(s1)
        s2_neighbourhood = NearestNeighbors(n_neighbors=k, radius=radius, algorithm='kd_tree').fit(s2)
        s3_neighbourhood = NearestNeighbors(n_neighbors=k + 1, radius=radius, algorithm='kd_tree').fit(s2)

        d_gam = d * gam

        s1_distances, indices = s1_neighbourhood.kneighbors(s1, k + 1)
        s2_distances, indices = s2_neighbourhood.kneighbors(s1, k)
        rho = s1_distances[:, -1]
        nu = s2_distances[:, -1]
        if np.any(rho == 0):
            warnings.warn(
                f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
                f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
                f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
                RuntimeWarning)
        # notice: the one below becomes 0 when one element in the s1 dataset is equal to one in the s2 dataset
        # and k=1 (as the distance between those two would be 0, which gives infinity when dividing)
        if np.any(nu == 0):
            warnings.warn(f"The distance between an element of the first dataset and its {k}-th NN in the second "
                          f"dataset is 0; this causes divergences in the code, and it is usually due to equal "
                          f"elements"
                          f" in the two datasets. Increasing the value of k usually solves this.", RuntimeWarning)
        second_term = np.sum(1 / (rho ** d_gam)) / (n * (n - 1) ** gam)
        fourth_term = np.sum(1 / (nu ** d_gam)) / (n * m ** gam)

        s3_distances, indices = s3_neighbourhood.kneighbors(s2, k + 1)
        rho_q = s3_distances[:, -1]

        if np.any(rho_q == 0):
            warnings.warn(
                f"The distance between an element of the second dataset and its {k}-th NN in the same dataset "
                f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
                f"{k + 1} times in the second dataset. Increasing the value of k usually solves this.",
                RuntimeWarning)

        third_term = np.sum(1 / (rho_q ** d_gam))
        # third_term /= m * (m ** gam)  # original code: I think the second term here should be m - 1
        third_term /= m * (m - 1) ** gam  # corrected version

        third_term = third_term ** gam
        fourth_term = fourth_term ** (1 + gam)
        D = (1 / (gam * (gam + 1))) * (np.log((second_term * third_term) / fourth_term))
        return D

    def _estimate_always_positive(self):
        return False


class KLDivergence(Divergence):
    """
    This implements an empirical estimator of the KL divergence for ABC as suggested in [1]. The estimator is based
    on a nearest neighbor density estimate.
    Specifically, this considers the
    several simulations/observations in the datasets as iid samples from the model for a fixed parameter value/from the
    data generating model, and estimates the divergence between the empirical distributions those
    simulations/observations define.

    [1] Jiang, B. (2018, March). Approximate Bayesian computation with Kullback-Leibler divergence as data discrepancy.
    In International Conference on Artificial Intelligence and Statistics (pp. 1711-1721). PMLR.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    k : int, optional
        nearest neighbor number for the density estimate. Default value is 1
    """

    def __init__(self, statistics_calc, k=1):
        super(KLDivergence, self).__init__(statistics_calc)

        self.k = k  # number of nearest neighbors used in the estimation algorithm

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

        if s1.shape[0] > self.k or s2.shape[0] > self.k:
            assert ValueError(f"The provided value of k ({self.k}) is smaller or equal than the number of samples "
                              f"in one of the two datasets; that should instead be larger")

        # estimate the KL divergence using the empirical distributions
        return self.skl_estimator_KL_div(s1=s1, s2=s2, k=self.k)

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf

    @staticmethod
    def skl_estimator_KL_div(s1, s2, k=1):
        """
        Adapted from https://github.com/nhartland/KL-divergence-estimators/blob/5473a23f5f13d7557100504611c57c9225b1a6eb/src/knn_divergence.py

        MIT license

        KL-Divergence estimator using scikit-learn's NearestNeighbours
            s1: (N_1,D) Sample drawn from distribution P
            s2: (N_2,D) Sample drawn from distribution Q
            k: Number of neighbours considered (default 1)
            return: estimated D(P|Q)
        """

        n, m = len(s1), len(s2)  # NOTE: here different convention of n, m wrt MMD and EnergyDistance
        d = float(s1.shape[1])

        radius = 10  # this is useless
        s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, radius=radius, algorithm='kd_tree').fit(s1)
        s2_neighbourhood = NearestNeighbors(n_neighbors=k, radius=radius, algorithm='kd_tree').fit(s2)

        s1_distances, indices = s1_neighbourhood.kneighbors(s1, k + 1)
        s2_distances, indices = s2_neighbourhood.kneighbors(s1, k)
        rho = s1_distances[:, -1]
        nu = s2_distances[:, -1]
        if np.any(rho == 0):
            warnings.warn(
                f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
                f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
                f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
                RuntimeWarning)
        D = np.sum(np.log(nu / rho))

        return (d / n) * D + np.log(m / (n - 1))  # this second term should be enough for it to be valid for m \neq n

    def _estimate_always_positive(self):
        return False


class MMD(Divergence):
    """
    This implements an empirical estimator of the MMD for ABC as suggested in [1]. This class implements a gaussian
    kernel by default but allows specifying different kernel functions. Notice that the original version in [1]
    suggested an unbiased estimate, which however can return negative values. We also provide a biased but provably
    positive estimator following the remarks in [2].
    Specifically, this considers the
    several simulations/observations in the datasets as iid samples from the model for a fixed parameter value/from the
    data generating model, and estimates the MMD between the empirical distributions those
    simulations/observations define.

    [1] Park, M., Jitkrittum, W., & Sejdinovic, D. (2016, May). K2-ABC: Approximate Bayesian computation with
    kernel embeddings. In Artificial Intelligence and Statistics (pp. 398-407). PMLR.
    [2] Nguyen, H. D., Arbel, J., Lü, H., & Forbes, F. (2020). Approximate Bayesian computation via the energy
    statistic. IEEE Access, 8, 131683-131698.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    kernel : str or callable
        Can be a string denoting the kernel, or a function. If a string, only gaussian is implemented for now; in
        that case, you can also provide an additional keyword parameter 'sigma' which is used as the sigma in the
        kernel. Default is the gaussian kernel.
    biased_estimator : boolean, optional
        Whether to use the biased (but always positive) or unbiased estimator; by default, it uses the biased one.
    kernel_kwargs
        Additional keyword arguments to be passed to the distance calculator.
    """

    def __init__(self, statistics_calc, kernel="gaussian", biased_estimator=False, **kernel_kwargs):
        super(MMD, self).__init__(statistics_calc)

        self.kernel_vectorized = False
        if not isinstance(kernel, str) and not callable(kernel):
            raise RuntimeError("'kernel' must be either a string or a function.")
        if isinstance(kernel, str):
            if kernel == "gaussian":
                self.kernel = self.def_gaussian_kernel(**kernel_kwargs)
                self.kernel_vectorized = True  # the gaussian kernel is vectorized
            else:
                raise NotImplementedError("The required kernel is not implemented.")
        else:
            self.kernel = kernel  # if kernel is a callable already

        self.biased_estimator = biased_estimator

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

        # compute the Gram matrix
        K11, K22, K12 = self.compute_Gram_matrix(s1, s2)

        # Estimate MMD
        if self.biased_estimator:
            return self.MMD_V_estimator(K11, K22, K12)
        else:
            return self.MMD_unbiased(K11, K22, K12)

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf

    @staticmethod
    def def_gaussian_kernel(sigma=1):
        # notice in the MMD paper they set sigma to a median value over the observation; check that.
        sigma_2 = 2 * sigma ** 2

        # def Gaussian_kernel(x, y):
        #     xy = x - y
        #     # assert np.allclose(np.dot(xy, xy), np.linalg.norm(xy) ** 2)
        #     return np.exp(- np.dot(xy, xy) / sigma_2)
        def Gaussian_kernel_vectorized(X, Y):
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)  # pairwise differences
            return np.exp(- np.einsum('xyi,xyi->xy', XY, XY) / sigma_2)

        return Gaussian_kernel_vectorized

    def compute_Gram_matrix(self, s1, s2):

        if self.kernel_vectorized:
            K11 = self.kernel(s1, s1)
            K22 = self.kernel(s2, s2)
            K12 = self.kernel(s1, s2)
        else:
            m = s1.shape[0]
            n = s2.shape[0]

            K11 = np.zeros((m, m))
            K22 = np.zeros((n, n))
            K12 = np.zeros((m, n))

            for i in range(m):
                # we assume the function to be symmetric; this saves some steps:
                for j in range(i, m):
                    K11[j, i] = K11[i, j] = self.kernel(s1[i], s1[j])

            for i in range(n):
                # we assume the function to be symmetric; this saves some steps:
                for j in range(i, n):
                    K22[j, i] = K22[i, j] = self.kernel(s2[i], s2[j])

            for i in range(m):
                for j in range(n):
                    K12[i, j] = self.kernel(s1[i], s2[j])

        # can we improve the above? Could use map but would not change too much likely.
        return K11, K22, K12

    @staticmethod
    def MMD_unbiased(Kxx, Kyy, Kxy):
        # from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        m = Kxx.shape[0]
        n = Kyy.shape[0]

        t1 = (1. / (m * (m - 1))) * np.sum(Kxx - np.diag(np.diagonal(Kxx)))
        t2 = (2. / (m * n)) * np.sum(Kxy)
        t3 = (1. / (n * (n - 1))) * np.sum(Kyy - np.diag(np.diagonal(Kyy)))

        MMDsquared = (t1 - t2 + t3)

        return MMDsquared

    @staticmethod
    def MMD_V_estimator(Kxx, Kyy, Kxy):
        # The estimate when distribution of x is not equal to y
        m = Kxx.shape[0]
        n = Kyy.shape[0]

        t1 = (1. / (m * m)) * np.sum(Kxx)
        t2 = (2. / (m * n)) * np.sum(Kxy)
        t3 = (1. / (n * n)) * np.sum(Kyy)

        MMDsquared = (t1 - t2 + t3)

        return MMDsquared

    def _estimate_always_positive(self):
        return self.biased_estimator


class EnergyDistance(MMD):
    """
    This implements an empirical estimator of the Energy Distance for ABC as suggested in [1].
    This class uses the Euclidean distance by default as a base distance, but allows to pass different distances.
    Moreover, when the Euclidean distance is specified, it is possible to pass an additional keyword argument `beta`
    which denotes the power of the distance to consider.
    In [1], the authors suggest to use a biased but provably positive estimator; we also provide an unbiased estimate,
    which however can return negative values.
    Specifically, this considers the
    several simulations/observations in the datasets as iid samples from the model for a fixed parameter value/from the
    data generating model, and estimates the MMD between the empirical distributions those
    simulations/observations define.

    [1] Nguyen, H. D., Arbel, J., Lü, H., & Forbes, F. (2020). Approximate Bayesian computation via the energy
    statistic. IEEE Access, 8, 131683-131698.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    base_distance : str or callable
        Can be a string denoting the kernel, or a function. If a string, only Euclidean distance is implemented
        for now; in that case, you can also provide an additional keyword parameter 'beta' which is the power
        of the distance to consider. By default, this uses the Euclidean distance.
    biased_estimator : boolean, optional
        Whether to use the biased (but always positive) or unbiased estimator; by default, it uses the biased one.
    base_distance_kwargs
        Additional keyword arguments to be passed to the distance calculator.
    """

    def __init__(self, statistics_calc, base_distance="Euclidean", biased_estimator=True, **base_distance_kwargs):
        if not isinstance(base_distance, str) and not callable(base_distance):
            raise RuntimeError("'base_distance' must be either a string or a function.")
        if isinstance(base_distance, str):
            if base_distance == "Euclidean":
                self.base_distance = self.def_Euclidean_distance(**base_distance_kwargs)
            else:
                raise NotImplementedError("The required kernel is not implemented.")
        else:
            self.base_distance = base_distance  # if base_distance is a callable already

        self.biased_estimator = biased_estimator

        def negative_distance(*args):
            return - self.base_distance(*args)

        super(EnergyDistance, self).__init__(statistics_calc, kernel=negative_distance,
                                             biased_estimator=self.biased_estimator)

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf

    @staticmethod
    def def_Euclidean_distance(beta=1):
        if beta <= 0 or beta > 2:
            raise RuntimeError("'beta' not in the right range (0,2]")

        if beta == 1:
            def Euclidean_distance(x, y):
                return np.linalg.norm(x - y)
        else:
            def Euclidean_distance(x, y):
                return np.linalg.norm(x - y) ** beta

        return Euclidean_distance


class SquaredHellingerDistance(Divergence):
    """
    This implements an empirical estimator of the squared Hellinger distance for ABC. Using the Hellinger distance was
    suggested originally in [1], but as that work did not provide originally any implementation details, this
    implementation is original. The estimator is based on a nearest neighbor density estimate.
    Specifically, this considers the
    several simulations/observations in the datasets as iid samples from the model for a fixed parameter value/from the
    data generating model, and estimates the divergence between the empirical distributions those
    simulations/observations define.

    [1] Frazier, D. T. (2020). Robust and Efficient Approximate Bayesian Computation: A Minimum Distance Approach.
    arXiv preprint arXiv:2006.14126.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    k : int, optional
        nearest neighbor number for the density estimate. Default value is 1
    """

    def __init__(self, statistics_calc, k=1):
        super(SquaredHellingerDistance, self).__init__(statistics_calc)

        self.k = k  # number of nearest neighbors used in the estimation algorithm

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

        if s1.shape[0] > self.k or s2.shape[0] > self.k:
            assert ValueError(f"The provided value of k ({self.k}) is smaller or equal than the number of samples "
                              f"in one of the two datasets; that should instead be larger")

        # estimate the gamma divergence using the empirical distributions
        return self.skl_estimator_squared_Hellinger_distance(s1=s1, s2=s2, k=self.k)

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        return 2

    @staticmethod
    def skl_estimator_squared_Hellinger_distance(s1, s2, k=1):
        """ Squared Hellinger distance estimator using scikit-learn's NearestNeighbours
            s1: (N_1,D) Sample drawn from distribution P
            s2: (N_2,D) Sample drawn from distribution Q
            k: Number of neighbours considered (default 1)
            return: estimated D(P|Q)

        """

        n, m = len(s1), len(s2)  # NOTE: here different convention of n, m wrt MMD and EnergyDistance
        d = float(s1.shape[1])
        d_2 = d / 2

        radius = 10  # this is not used at all...
        s1_neighbourhood_k1 = NearestNeighbors(n_neighbors=k + 1, radius=radius, algorithm='kd_tree').fit(s1)
        s1_neighbourhood_k = NearestNeighbors(n_neighbors=k, radius=radius, algorithm='kd_tree').fit(s1)
        s2_neighbourhood_k1 = NearestNeighbors(n_neighbors=k + 1, radius=radius, algorithm='kd_tree').fit(s2)
        s2_neighbourhood_k = NearestNeighbors(n_neighbors=k, radius=radius, algorithm='kd_tree').fit(s2)

        s1_distances, indices = s1_neighbourhood_k1.kneighbors(s1, k + 1)
        s2_distances, indices = s2_neighbourhood_k.kneighbors(s1, k)
        rho = s1_distances[:, -1]
        nu = s2_distances[:, -1]
        if np.any(rho == 0):
            warnings.warn(
                f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
                f"is 0; this is due to elements which are repeated "
                f"{k + 1} times in the first dataset, and may lead to a poor estimate of the distance. "
                f"Increasing the value of k usually solves this.",
                RuntimeWarning)

        if np.any(nu == 0):
            warnings.warn(f"The distance between an element of the first dataset and its {k}-th NN in the second "
                          f"dataset is 0; this causes divergences in the code, and it is usually due to equal "
                          f"elements"
                          f" in the two datasets. Increasing the value of k usually solves this.", RuntimeWarning)
        first_estimator = np.sum((rho / nu) ** d_2)
        first_estimator = 2 - 2 * np.sqrt((n - 1) / m) * first_estimator

        s2_distances, indices = s2_neighbourhood_k1.kneighbors(s2, k + 1)
        s1_distances, indices = s1_neighbourhood_k.kneighbors(s2, k)
        rho = s2_distances[:, -1]
        nu = s1_distances[:, -1]
        if np.any(rho == 0):
            warnings.warn(
                f"The distance between an element of the second dataset and its {k}-th NN in the same dataset "
                f"is 0; this is due to elements which are repeated "
                f"{k + 1} times in the second dataset, and may lead to a poor estimate of the distance. "
                f"Increasing the value of k usually solves this.",
                RuntimeWarning)
        # notice: the one below becomes 0 when one element in the s1 dataset is equal to one in the s2 dataset
        # and k=1 (as the distance between those two would be 0, which gives infinity when dividing)
        if np.any(nu == 0):
            warnings.warn(f"The distance between an element of the second dataset and its {k}-th NN in the first "
                          f"dataset is 0; this causes divergences in the code, and it is usually due to equal "
                          f"elements"
                          f" in the two datasets. Increasing the value of k usually solves this.", RuntimeWarning)
        second_estimator = np.sum((rho / nu) ** d_2)
        second_estimator = 2 - 2 * np.sqrt((m - 1) / n) * second_estimator

        # average the two estimators:
        final_estimator = 0.5 * (first_estimator + second_estimator)

        return final_estimator

    def _estimate_always_positive(self):
        return True
