from abc import ABCMeta, abstractmethod

import numpy as np
from glmnet import LogitNet
from sklearn.covariance import ledoit_wolf

from abcpy.graphtools import GraphTools


class Approx_likelihood(metaclass=ABCMeta):
    """This abstract base class defines the approximate likelihood
    function.
    """

    @abstractmethod
    def __init__(self, statistics_calc):
        """
        The constructor of a sub-class must accept a non-optional statistics
        calculator; then, it must call the __init__ method of the parent class. This ensures that the
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
        self.stat_obs = None
        self.data_set = None
        self.dataSame = False

    @abstractmethod
    def loglikelihood(self, y_obs, y_sim):
        """To be overwritten by any sub-class: should compute the approximate loglikelihood
        value given the observed data set y_obs and the data set y_sim simulated from
        model set at the parameter value.

        Parameters
        ----------
        y_obs: Python list
            Observed data set.
        y_sim: Python list
            Simulated data set from model at the parameter value.
            
        Returns
        -------
        float
            Computed approximate loglikelihood.
        """

        raise NotImplemented

    def likelihood(self, y_obs, y_sim):
        return np.exp(self.loglikelihood(y_obs, y_sim))

    def _calculate_summary_stat(self, y_obs, y_sim):
        if not isinstance(y_obs, list):
            raise TypeError('Observed data is not of allowed types')

        if not isinstance(y_sim, list):
            raise TypeError('simulated data is not of allowed types')

        # Check whether y_obs is same as the stored dataset.
        if self.data_set is not None:
            # check that the the observations have the same length; if not, they can't be the same:
            if len(y_obs) != len(self.data_set):
                self.dataSame = False
            elif len(np.array(y_obs[0]).reshape(-1, )) == 1:
                self.dataSame = self.data_set == y_obs
            else:  # otherwise it fails when y_obs[0] is array
                self.dataSame = all(
                    [(np.array(self.data_set[i]) == np.array(y_obs[i])).all() for i in range(len(y_obs))])

        if self.stat_obs is None or self.dataSame is False:
            self.stat_obs = self.statistics_calc.statistics(y_obs)
            self.data_set = y_obs

        # Extract summary statistics from the simulated data
        stat_sim = self.statistics_calc.statistics(y_sim)

        if self.stat_obs.shape[1] != stat_sim.shape[1]:
            raise ValueError("The dimension of summaries in the two datasets is different; check the dimension of the"
                             " provided observations and simulations.")

        return self.stat_obs, stat_sim


class SynLikelihood(Approx_likelihood):
    """This class implements the approximate likelihood function which computes the approximate
    likelihood using the synthetic likelihood approach described in Wood [1].
    For synthetic likelihood approximation, we compute the robust precision matrix using Ledoit and Wolf's [2]
    method.
    
    [1] S. N. Wood. Statistical inference for noisy nonlinear ecological 
    dynamic systems. Nature, 466(7310):1102–1104, Aug. 2010.
    
    [2] O. Ledoit and M. Wolf, A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices,
    Journal of Multivariate Analysis, Volume 88, Issue 2, pages 365-411, February 2004.
    """

    def __init__(self, statistics_calc):
        super(SynLikelihood, self).__init__(statistics_calc)

    def loglikelihood(self, y_obs, y_sim):
        stat_obs, stat_sim = self._calculate_summary_stat(y_obs, y_sim)

        # Compute the mean, robust precision matrix and determinant of precision matrix
        mean_sim = np.mean(stat_sim, 0)
        lw_cov_, _ = ledoit_wolf(stat_sim)
        robust_precision_sim = np.linalg.inv(lw_cov_)
        sign_logdet, robust_precision_sim_logdet = np.linalg.slogdet(robust_precision_sim)  # we do not need sign
        # print("DEBUG: combining.")
        # we may have different observation; loop on those now:
        # likelihoods = np.zeros(stat_obs.shape[0])
        # for i, single_stat_obs in enumerate(stat_obs):
        #     x_new = np.einsum('i,ij,j->', single_stat_obs - mean_sim, robust_precision_sim, single_stat_obs - mean_sim)
        #     likelihoods[i] = np.exp(-0.5 * x_new)
        # do without for loop:
        diff = stat_obs - mean_sim.reshape(1, -1)
        x_news = np.einsum('bi,ij,bj->b', diff, robust_precision_sim, diff)
        logliks = -0.5 * x_news
        logfactor = 0.5 * stat_obs.shape[0] * robust_precision_sim_logdet
        return np.sum(logliks) + logfactor  # compute the sum of the different loglikelihoods for each observation


class PenLogReg(Approx_likelihood, GraphTools):
    """This class implements the approximate likelihood function which computes the approximate
    likelihood up to a constant using penalized logistic regression described in
    Dutta et. al. [1]. It takes one additional function handler defining the 
    true model and two additional parameters n_folds and n_simulate correspondingly defining number
    of folds used to estimate prediction error using cross-validation and the number 
    of simulated dataset sampled from each parameter to approximate the likelihood
    function. For lasso penalized logistic regression we use glmnet of Friedman et.
    al. [2].
    
    [1] Thomas, O., Dutta, R., Corander, J., Kaski, S., & Gutmann, M. U. (2020).
    Likelihood-free inference by ratio estimation. Bayesian Analysis.
    
    [2] Friedman, J., Hastie, T., and Tibshirani, R. (2010). Regularization 
    paths for generalized linear models via coordinate descent. Journal of Statistical 
    Software, 33(1), 1–22.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    model : abcpy.models.Model
        Model object that conforms to the Model class.
    n_simulate : int
        Number of data points to simulate for the reference data set; this has to be the same as n_samples_per_param
        when calling the sampler. The reference data set is generated by drawing parameters from the prior and samples
        from the model when PenLogReg is instantiated.
    n_folds: int, optional
        Number of folds for cross-validation. The default value is 10.
    max_iter: int, optional
        Maximum passes over the data. The default is 100000.
    seed: int, optional
        Seed for the random number generator. The used glmnet solver is not
        deterministic, this seed is used for determining the cv folds. The default value is
        None.
    """

    def __init__(self, statistics_calc, model, n_simulate, n_folds=10, max_iter=100000, seed=None):

        super(PenLogReg, self).__init__(statistics_calc)  # call the super init to initialize correctly

        self.model = model
        self.n_folds = n_folds
        self.n_simulate = n_simulate
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.max_iter = max_iter
        # Simulate reference data and extract summary statistics from the reference data
        self.ref_data_stat = self._simulate_ref_data(rng=self.rng)[0]

    def loglikelihood(self, y_obs, y_sim):
        stat_obs, stat_sim = self._calculate_summary_stat(y_obs, y_sim)

        if not stat_sim.shape[0] == self.n_simulate:
            raise RuntimeError("The number of samples in the reference data set is not the same as the number of "
                               "samples in the generated data. Please check that `n_samples` in the `sample()` method"
                               "for the sampler is equal to `n_simulate` in PenLogReg.")

        # Compute the approximate likelihood for the y_obs given theta
        y = np.append(np.zeros(self.n_simulate), np.ones(self.n_simulate))
        X = np.array(np.concatenate((stat_sim, self.ref_data_stat), axis=0))
        # define here groups for cross-validation:
        groups = np.repeat(np.arange(self.n_folds), np.int(np.ceil(self.n_simulate / self.n_folds)))
        groups = groups[:self.n_simulate].tolist()
        groups += groups  # duplicate it as groups need to be defined for both datasets
        m = LogitNet(alpha=1, n_splits=self.n_folds, max_iter=self.max_iter, random_state=self.seed, scoring="log_loss")
        m = m.fit(X, y, groups=groups)
        result = -np.sum((m.intercept_ + np.sum(np.multiply(m.coef_, stat_obs), axis=1)), axis=0)

        return result

    def _simulate_ref_data(self, rng=np.random.RandomState()):
        """
        Simulate the reference data set. This code is run at the initialization of
        Penlogreg
        """

        ref_data_stat = [[None] * self.n_simulate for i in range(len(self.model))]
        self.sample_from_prior(rng=rng)
        for model_index, model in enumerate(self.model):
            ind = 0
            while ref_data_stat[model_index][-1] is None:
                data = model.forward_simulate(model.get_input_values(), 1, rng=rng)
                # this is wrong, it applies the computation of the statistic independently to the element of data[0]:
                # print("data[0]", data[0].tolist())
                # data_stat = self.statistics_calc.statistics(data[0].tolist())
                # print("stat of data[0]", data_stat)
                # print("data", data)
                data_stat = self.statistics_calc.statistics(data)
                # print("stat of data", data_stat)
                ref_data_stat[model_index][ind] = data_stat
                ind += 1
            ref_data_stat[model_index] = np.squeeze(np.asarray(ref_data_stat[model_index]))
        return ref_data_stat
