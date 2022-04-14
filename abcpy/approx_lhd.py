import numpy as np
from abc import ABCMeta, abstractmethod
from glmnet import LogitNet
from scipy.stats import gaussian_kde, rankdata, norm
from sklearn.covariance import ledoit_wolf

from jax import grad, vmap
import jax.numpy as jnp

from abcpy.graphtools import GraphTools
from abcpy.statistics import Identity


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
        """Computes the likelihood by taking the exponential of the loglikelihood method.

        Parameters
        ----------
        y_obs: Python list
            Observed data set.
        y_sim: Python list
            Simulated data set from model at the parameter value.

        Returns
        -------
        float
            Computed approximate likelihood.

        """
        return np.exp(self.loglikelihood(y_obs, y_sim))

    def _calculate_summary_stat(self, y_obs, y_sim):
        """Helper function that extracts the summary statistics s_obs and s_sim from y_obs and
        y_y_sim using the statistics object stored in self.statistics_calc. This stores s_obs for the purpose of checking
        whether that is repeated in next calls to the function, and avoiding computing the statitistics for the same
        dataset several times.

        Parameters
        ----------
        y_obs : array-like
            d1 contains n_obs data sets.
        y_sim : array-like
            d2 contains n_sim data sets.

        Returns
        -------
        tuple
            Tuple containing numpy.ndarray's with the summary statistics extracted from d1 and d2.
        """
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

    def __init__(self, statistics_calc):
        """This class implements the approximate likelihood function which computes the approximate
        likelihood using the synthetic likelihood approach described in Wood [1].
        For synthetic likelihood approximation, we compute the robust precision matrix using Ledoit and Wolf's [2]
        method.

        [1] S. N. Wood. Statistical inference for noisy nonlinear ecological
        dynamic systems. Nature, 466(7310):1102–1104, Aug. 2010.

        [2] O. Ledoit and M. Wolf, A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices,
        Journal of Multivariate Analysis, Volume 88, Issue 2, pages 365-411, February 2004.


        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        """

        super(SynLikelihood, self).__init__(statistics_calc)

    def loglikelihood(self, y_obs, y_sim):
        """Computes the loglikelihood.

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
        logfactor = 0.5 * self.stat_obs.shape[0] * robust_precision_sim_logdet
        return np.sum(logliks) + logfactor  # compute the sum of the different loglikelihoods for each observation


class SemiParametricSynLikelihood(Approx_likelihood):

    def __init__(self, statistics_calc, bw_method_marginals="silverman"):
        """
        This class implements the approximate likelihood function which computes the approximate
        likelihood using the semiparametric Synthetic Likelihood (semiBSL) approach described in [1]. Specifically, this
        represents the likelihood as a product of univariate marginals and the copula components (exploiting Sklar's
        theorem).
        The marginals are approximated from simulations using a Gaussian KDE, while the copula is assumed to be a Gaussian
        copula, whose parameters are estimated from data as well.

        This does not yet include shrinkage strategies for the correlation matrix.

        [1] An, Z., Nott, D. J., & Drovandi, C. (2020). Robust Bayesian synthetic likelihood via a semi-parametric approach.
        Statistics and Computing, 30(3), 543-557.

        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        bw_method_marginals : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth, passed to `scipy.stats.gaussian_kde`. Following the docs
            of that method, this can be 'scott', 'silverman', a scalar constant or a callable. If a scalar, this will be
            used directly as `kde.factor`. If a callable, it should take a `gaussian_kde` instance as only parameter
            and return a scalar. If None (default), 'silverman' is used. See the Notes in `scipy.stats.gaussian_kde`
            for more details.
        """
        super(SemiParametricSynLikelihood, self).__init__(statistics_calc)
        # create a dict in which store the denominator of the correlation matrix for the different n values;
        # this saves from repeating computations:
        self.corr_matrix_denominator = {}
        self.bw_method_marginals = bw_method_marginals  # the bw method to use in the gaussian_kde

    def loglikelihood(self, y_obs, y_sim):
        """Computes the loglikelihood. This implementation aims to be equivalent to the `BSL` R package,
        but the results are slightly different due to small differences in the way the KDE is performed

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

        stat_obs, stat_sim = self._calculate_summary_stat(y_obs, y_sim)
        n_obs, d = stat_obs.shape
        if d < 2:
            raise RuntimeError("The dimension of the statistics need to be at least 2 in order to apply semiBSL.")

        # first: estimate the marginal KDEs for each coordinate
        logpdf_obs = np.zeros_like(stat_obs)  # this will contain the estimated pdf at the various observation points
        u_obs = np.zeros_like(stat_obs)  # this instead will contain the transformed u's using the estimated CDF
        for j in range(d):
            # estimate the KDE using the data in stat_sim for coordinate j. This leads to slightly different results
            # from the R package implementation due to slightly different way to estimate the factor as well as
            # different way to evaluate the kernel (they use a weird interpolation there).
            kde = gaussian_kde(stat_sim[:, j], bw_method=self.bw_method_marginals)
            logpdf_obs[:, j] = kde.logpdf(stat_obs[:, j])
            for i in range(n_obs):  # loop over the different observations
                u_obs[i, j] = kde.integrate_box_1d(-np.infty, stat_obs[i, j])  # compute the CDF
        etas_obs = norm.ppf(u_obs)

        # second: estimate the correlation matrix for the gaussian copula using gaussian rank correlation
        R_hat = self._estimate_gaussian_correlation(stat_sim)
        R_hat_inv = np.linalg.inv(R_hat)
        R_sign_det, R_inv_logdet = np.linalg.slogdet(R_hat_inv)  # sign not used

        # third: combine the two to compute the loglikelihood;
        # for each observation:
        # logliks = np.zeros(n_obs)
        # for i in range(n_obs):
        #     logliks[i] = np.sum(logpdf_obs[i])  # sum along marginals along dimensions
        #     # add the copula density:
        #     logliks[i] += 0.5 * R_inv_logdet
        #     logliks[i] -= 0.5 * np.einsum("i,ij,j->", etas_obs[i], R_hat_inv - np.eye(d), etas_obs[i])

        # do jointly:
        loglik = np.sum(logpdf_obs)  # sum along marginals along dimensions
        # add the copula density:
        copula_density = -0.5 * np.einsum("bi,ij,bj->b", etas_obs, R_hat_inv - np.eye(d), etas_obs)
        loglik += np.sum(copula_density) + 0.5 * n_obs * R_inv_logdet

        return loglik

    def _estimate_gaussian_correlation(self, x):
        """Estimates the correlation matrix using data in `x` in the way described in [1]. This implementation
        gives the same results as the `BSL` R package.

        Parameters
        ----------
        x: np.ndarray
            Data set.

        Returns
        -------
        np.ndarray
            Estimated correlation matrix for the gaussian copula.
        """
        n, d = x.shape
        r = np.zeros_like(x)
        for j in range(d):
            r[:, j] = rankdata(x[:, j])

        rqnorm = norm.ppf(r / (n + 1))

        if n not in self.corr_matrix_denominator.keys():
            # compute the denominator:
            self.corr_matrix_denominator[n] = np.sum(norm.ppf((np.arange(n) + 1) / (n + 1)) ** 2)
        denominator = self.corr_matrix_denominator[n]

        R_hat = np.einsum('ki,kj->ij', rqnorm, rqnorm) / denominator

        return R_hat


class PenLogReg(Approx_likelihood, GraphTools):

    def __init__(self, statistics_calc, model, n_simulate, n_folds=10, max_iter=100000, seed=None):
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
            when calling the sampler. The reference data set is generated by drawing parameters from the prior and
            samples from the model when PenLogReg is instantiated.
        n_folds: int, optional
            Number of folds for cross-validation. The default value is 10.
        max_iter: int, optional
            Maximum passes over the data. The default is 100000.
        seed: int, optional
            Seed for the random number generator. The used glmnet solver is not
            deterministic, this seed is used for determining the cv folds. The default value is
            None.
        """

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
        """Computes the loglikelihood.

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
        stat_obs, stat_sim = self._calculate_summary_stat(y_obs, y_sim)

        if not stat_sim.shape[0] == self.n_simulate:
            raise RuntimeError("The number of samples in the reference data set is not the same as the number of "
                               "samples in the generated data. Please check that `n_samples` in the `sample()` method"
                               "for the sampler is equal to `n_simulate` in PenLogReg.")

        # Compute the approximate likelihood for the y_obs given theta
        y = np.append(np.zeros(self.n_simulate), np.ones(self.n_simulate))
        X = np.array(np.concatenate((stat_sim, self.ref_data_stat), axis=0))
        # define here groups for cross-validation:
        groups = np.repeat(np.arange(self.n_folds), int(np.ceil(self.n_simulate / self.n_folds)))
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

        Parameters
        ----------
        rng: Random number generator, optional
            Defines the random number generator to be used. If None, a newly initialized one is used

        Returns
        -------
        list
            The simulated list of datasets.

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


class ScoringRule(Approx_likelihood, metaclass=ABCMeta):
    """This is the abstract class for the ScoringRule which allows it to be used as an Approx_likelihood in ABCpy"""

    def __init__(self, statistics_calc, weight=1):
        """Needs to be called by each sub-class to correctly initialize the statistics_calc"""
        # call the super of Distance to initialize the statistics_calc stuff:
        Approx_likelihood.__init__(self, statistics_calc)
        self.weight = weight  # this is the weight used to multiply the scoring rule for the loglikelihood computation

    def loglikelihood(self, y_obs, y_sim):
        """Alias the score method to a loglikelihood method """
        return - self.weight * self.score(y_obs, y_sim)

    def score(self, observations, simulations):
        """
        Notice: here the score is assumed to be a "penalty"; we use therefore the sign notation of Dawid, not the one
        in Gneiting and Raftery (2007).
        To be overwritten by any sub-class. Here, `observations` and `simulations` are lists of length respectively `n_obs` and `n_sim`. Then,
        for each fixed observation the `n_sim` simulations are used to estimate the scoring rule. Subsequently, the
        values are summed over each of the `n_obs` observations.

        Parameters
        ----------
        observations: Python list
            Contains `n_obs` data points.
        simulations: Python list
            Contains `n_sim` data points.

        Returns
        -------
        float
            The score between the simulations and the observations.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """

        s_observations, s_simulations = self._calculate_summary_stat(observations, simulations)

        return self._estimate_score(s_observations, s_simulations)

    @abstractmethod
    def _estimate_score(self, s_observations, s_simulations):
        """
        This method needs to be implemented by each sub-class. It should return the score for the given data set.

        Parameters
        ----------
        s_observations: numpy array
            The summary statistics of the observed data set. Shape is (n_obs, n_summary_stat).
        s_simulations: numpy array
            The summary statistics of the simulated data set. Shape is (n_sim, n_summary_stat).
        """
        raise NotImplementedError

    @abstractmethod
    def score_max(self):
        """To be overwritten by any sub-class"""
        raise NotImplementedError


class UnivariateContinuousRankedProbabilityScoreEstimate(ScoringRule):
    """Estimates the Continuous Ranked Probability Score. Here, I assume the observations and simulations are lists of
    length respectively n_obs and n_sim. Then, for each fixed observation the n_sim simulations are used to estimate the
    scoring rule. Subsequently, the values are averaged over each of the n_obs observations.
    """

    def __init__(self, statistics_calc):
        super(UnivariateContinuousRankedProbabilityScoreEstimate, self).__init__(statistics_calc)

    def _estimate_score(self, s_observations, s_simulations):
        """
        Parameters
        ----------
        s_observations: numpy array
            The summary statistics of the observed data set. Shape is (n_obs, n_summary_stat).
        s_simulations: numpy array
            The summary statistics of the simulated data set. Shape is (n_sim, n_summary_stat).
        """
        scores = np.zeros(shape=(s_observations.shape[0]))
        # this for loop is not very efficient, can be improved; this is taken from the Euclidean distance.
        for ind1 in range(s_observations.shape[0]):
            scores[ind1] = self.estimate_CRPS_score(s_observations[ind1], s_simulations)
        return scores.sum()

    def score_max(self):
        # As the statistics are positive, the max possible value is 1
        return np.inf

    @staticmethod
    def estimate_CRPS_score(observation, simulations):
        """observation is a single value, while simulations is an array. We estimate this by building an empirical
         unbiased estimate of Eq. (1) in Ziel and Berk 2019"""
        diff_X_y = np.abs(observation - simulations)
        n_sim = simulations.shape[0]
        diff_X_tildeX = np.abs(simulations.reshape(1, -1) - simulations.reshape(-1, 1))

        return 2 * np.mean(diff_X_y) - np.sum(diff_X_tildeX) / (n_sim * (n_sim - 1))


class EnergyScore(ScoringRule):
    def __init__(self, statistics_calc, weight=1, beta=1, use_jax=False):
        """ Estimates the EnergyScore. Here, I assume the observations and simulations are lists of
        length respectively n_obs and n_sim. Then, for each fixed observation the n_sim simulations are used to estimate the
        scoring rule. Subsequently, the values are summed over each of the n_obs observations.

        Note this scoring rule is connected to the energy distance between probability distributions.
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        weight : int, optional.
            Weight used in defining the scoring rule posterior. Default is 1.
        beta : int, optional.
            Power used to define the energy score. Default is 1.
        use_jax : bool, optional.
            Whether to use JAX for the computation; in that case, you can compute unbiased gradient estimate
            of the score with respect to parameters. Default is False.
        """

        self.beta = beta
        self.beta_over_2 = 0.5 * beta
        self.use_jax = use_jax

        if use_jax:
            # define the gradient function with jax:
            self._grad_estimate_score = grad(self._estimate_score, argnums=1)
            self.np = jnp
        else:
            self.np = np

        super(EnergyScore, self).__init__(statistics_calc, weight=weight)

    def score_gradient(self, observations, simulations, simulations_gradients):
        """
        Computes gradient of the unbiased estimate of the score with respect to the parameters.

        Parameters
        ----------
        observations: Python list
            Contains n1 data points.
        simulations: Python list
            Contains n2 data points.
        simulations_gradients: Python list
            Contains n2 data points, each of which is the gradient of the simulations with respect to the
            parameters (and is therefore of shape (simulation_dim, n_params)).

        Returns
        -------
        numpy.ndarray
            The gradient of the score with respect to the parameters.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """
        if not self.use_jax:
            raise RuntimeError("The gradient of the energy score is only available with jax.")

        if not isinstance(self.statistics_calc,
                          Identity) or self.statistics_calc.degree != 1 or self.statistics_calc.cross:
            raise RuntimeError(
                "The gradient of the energy score is only available for the identity statistics with degree="
                "1 and cross=False.")

        if not len(simulations) == len(simulations_gradients):
            raise RuntimeError("The number of simulations and the number of gradients must be the same.")

        s_observations, s_simulations = self._calculate_summary_stat(observations, simulations)

        score_grad = self._grad_estimate_score(s_observations, s_simulations)
        # score grad contains the gradients of the score with respect to the simulation statistics; it is therefore of
        # shape (n_sim, simulation_dim)
        simulations_gradients = np.array(simulations_gradients)
        # simulations_gradients is of shape (n_sim, simulation_dim, n_params)

        if not simulations_gradients.shape[0:2] == score_grad.shape:
            raise RuntimeError("The shape of the score gradient must be the"
                               " same as the first two shapes of the simulations.")

        # then need to multiply the gradients for each simulation along the simulation_dim axis and then average
        # over n_dim; that leads to the gradient of the score with respect to the parameters:

        return np.einsum('ij,ijk->k', score_grad, simulations_gradients)

    def score_max(self):
        # As the statistics are positive, the max possible value is 1
        return np.inf

    def _estimate_score(self, s_observations, s_simulations):
        """
        We estimate this by building an empirical unbiased estimate of Eq. (2) in Ziel and Berk 2019

        Parameters
        ----------
        s_observations: numpy array
            The summary statistics of the observed data set. Shape is (n_obs, n_summary_stat).
        s_simulations: numpy array
            The summary statistics of the simulated data set. Shape is (n_sim, n_summary_stat).
        """
        n_obs = s_observations.shape[0]
        n_sim, p = s_simulations.shape
        diff_X_y = s_observations.reshape(n_obs, 1, -1) - s_simulations.reshape(1, n_sim, p)
        diff_X_y = self.np.einsum('ijk, ijk -> ij', diff_X_y, diff_X_y)

        diff_X_tildeX = s_simulations.reshape(1, n_sim, p) - s_simulations.reshape(n_sim, 1, p)

        # exclude diagonal elements which are zero:
        diff_X_tildeX = self.np.einsum('ijk, ijk -> ij', diff_X_tildeX, diff_X_tildeX)[~self.np.eye(n_sim, dtype=bool)]
        if self.beta_over_2 != 1:
            diff_X_y **= self.beta_over_2
            diff_X_tildeX **= self.beta_over_2

        return 2 * self.np.sum(self.np.mean(diff_X_y, axis=1)) - n_obs * self.np.sum(diff_X_tildeX) / (
                    n_sim * (n_sim - 1))


class KernelScore(ScoringRule):

    def __init__(self, statistics_calc, weight=1, kernel="gaussian", biased_estimator=False, use_jax=False,
                 **kernel_kwargs):
        """
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        weight : int, optional.
            Weight used in defining the scoring rule posterior. Default is 1.
        kernel : str or callable, optional
            Can be a string denoting the kernel, or a function. If a string, only gaussian is implemented for now; in
            that case, you can also provide an additional keyword parameter 'sigma' which is used as the sigma in the
            kernel. If a function is provided it should take two arguments; additionally, it needs to be written in
            jax if use_jax is True, otherwise gradient computation will not work.
        biased_estimator : bool, optional
            Whether to use the biased estimator or not. Default is False.
        use_jax : bool, optional.
            Whether to use JAX for the computation; in that case, you can compute unbiased gradient estimate
            of the score with respect to parameters. Default is False.
        **kernel_kwargs : dict, optional
            Additional keyword arguments for the kernel.
        """

        if not isinstance(kernel, str) and not callable(kernel):
            raise RuntimeError("'kernel' must be either a string or a function of two variables returning a scalar. "
                               "In that case, it must be written in JAX if use_jax is True.")

        super(KernelScore, self).__init__(statistics_calc, weight=weight)

        self.kernel_vectorized = False
        self.use_jax = use_jax

        if use_jax:
            # define the gradient function with jax:
            self._grad_estimate_score = grad(self._estimate_score, argnums=1)
            self.np = jnp
        else:
            self.np = np

        # set up the kernel
        if isinstance(kernel, str):
            if kernel == "gaussian":
                self.kernel = self._def_gaussian_kernel(**kernel_kwargs)
                self.kernel_vectorized = True  # the gaussian kernel is vectorized
            else:
                raise NotImplementedError("The required kernel is not implemented.")
        else:  # if kernel is a callable already
            if use_jax:
                self.kernel = self._all_pairs(kernel)  # this makes it a vectorized function
                self.kernel_vectorized = True  # the gaussian kernel is vectorized
            else:
                self.kernel = kernel

        self.biased_estimator = biased_estimator

    def score_gradient(self, observations, simulations, simulations_gradients):
        """
        Computes gradient of the unbiased estimate of the score with respect to the parameters.

        Parameters
        ----------
        observations: Python list
            Contains n1 data points.
        simulations: Python list
            Contains n2 data points.
        simulations_gradients: Python list
            Contains n2 data points, each of which is the gradient of the simulations with respect to the
            parameters (and is therefore of shape (simulation_dim, n_params)).

        Returns
        -------
        numpy.ndarray
            The gradient of the score with respect to the parameters.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """
        if not self.use_jax:
            raise RuntimeError("The gradient of the energy score is only available with jax.")

        if not isinstance(self.statistics_calc,
                          Identity) or self.statistics_calc.degree != 1 or self.statistics_calc.cross:
            raise RuntimeError(
                "The gradient of the energy score is only available for the identity statistics with degree="
                "1 and cross=False.")

        if not len(simulations) == len(simulations_gradients):
            raise RuntimeError("The number of simulations and the number of gradients must be the same.")

        s_observations, s_simulations = self._calculate_summary_stat(observations, simulations)

        score_grad = self._grad_estimate_score(s_observations, s_simulations)
        # score grad contains the gradients of the score with respect to the simulation statistics; it is therefore of
        # shape (n_sim, simulation_dim)
        simulations_gradients = np.array(simulations_gradients)
        # simulations_gradients is of shape (n_sim, simulation_dim, n_params)

        if not simulations_gradients.shape[1] == score_grad.shape[1]:
            raise RuntimeError("The shape of the score gradient must be the"
                               " same as the first two shapes of the simulations.")

        # then need to multiply the gradients for each simulation along the simulation_dim axis and then average
        # over n_dim; that leads to the gradient of the score with respect to the parameters:

        return np.einsum('ij,ijk->k', score_grad, simulations_gradients)

    def score_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf

    def _estimate_score(self, s_observations, s_simulations):
        """
        Parameters
        ----------
        s_observations: numpy array
            The summary statistics of the observed data set. Shape is (n_obs, n_summary_stat).
        s_simulations: numpy array
            The summary statistics of the simulated data set. Shape is (n_sim, n_summary_stat).
        """
        # compute the Gram matrix
        K_sim_sim, K_obs_sim = self._compute_Gram_matrix(s_observations, s_simulations)

        # Estimate MMD
        if self.biased_estimator:
            return self._MMD_V_estimator(K_sim_sim, K_obs_sim)
        else:
            return self._MMD_unbiased(K_sim_sim, K_obs_sim)

    def _def_gaussian_kernel(self, sigma=1):
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
            return self.np.exp(- self.np.einsum('xyi,xyi->xy', XY, XY) / sigma_2)

        return Gaussian_kernel_vectorized

    def _compute_Gram_matrix(self, s_observations, s_simulations):

        if self.kernel_vectorized:
            K_sim_sim = self.kernel(s_simulations, s_simulations)
            K_obs_sim = self.kernel(s_observations, s_simulations)
        else:
            # this should not happen in case self.use_jax is True
            n_obs = s_observations.shape[0]
            n_sim = s_simulations.shape[0]

            K_sim_sim = np.zeros((n_sim, n_sim))
            K_obs_sim = np.zeros((n_obs, n_sim))

            for i in range(n_sim):
                # we assume the function to be symmetric; this saves some steps:
                for j in range(i, n_sim):
                    K_sim_sim[j, i] = K_sim_sim[i, j] = self.kernel(s_simulations[i], s_simulations[j])

            for i in range(n_obs):
                for j in range(n_sim):
                    K_obs_sim[i, j] = self.kernel(s_observations[i], s_simulations[j])

        return K_sim_sim, K_obs_sim

    def _MMD_unbiased(self, K_sim_sim, K_obs_sim):
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * self.np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * (n_sim - 1))) * self.np.sum(K_sim_sim[~self.np.eye(n_sim, dtype=bool)])

        return n_obs * t_sim_sim - t_obs_sim

    def _MMD_V_estimator(self, K_sim_sim, K_obs_sim):
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * self.np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * n_sim)) * self.np.sum(K_sim_sim)

        return n_obs * t_sim_sim - t_obs_sim

    @staticmethod
    def _all_pairs(f):
        """Used to apply a function of two elements to all possible pairs."""
        f = vmap(f, in_axes=(None, 0))
        f = vmap(f, in_axes=(0, None))
        return f
