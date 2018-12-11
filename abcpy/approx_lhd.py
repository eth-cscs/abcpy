from abc import ABCMeta, abstractmethod

from abcpy.graphtools import GraphTools

import numpy as np
from sklearn.covariance import ledoit_wolf
from glmnet import LogitNet 


class Approx_likelihood(metaclass = ABCMeta):
    """This abstract base class defines the approximate likelihood 
    function.
    """

    @abstractmethod
    def __init__(self, statistics_calc):       
        """
        The constructor of a sub-class must accept a non-optional statistics
        calculator, which is stored to self.statistics_calc.

        Parameters
        ----------
        statistics_calc : abcpy.stasistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        """
        
        raise NotImplemented

    @abstractmethod
    def likelihood(y_obs, y_sim):
        """To be overwritten by any sub-class: should compute the approximate likelihood 
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
            Computed approximate likelihood.
        """
        
        raise NotImplemented


class SynLiklihood(Approx_likelihood):
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
        self.stat_obs = None
        self.data_set=None
        self.statistics_calc = statistics_calc


    def likelihood(self, y_obs, y_sim):
        # print("DEBUG: SynLiklihood.likelihood().")
        if not isinstance(y_obs, list):
            # print("type(y_obs) : ", type(y_obs), " , type(y_sim) : ", type(y_sim))
            # print("y_obs : ", y_obs)
            raise TypeError('Observed data is not of allowed types')

        if not isinstance(y_sim, list):
            raise TypeError('simulated data is not of allowed types')

        # Extract summary statistics from the observed data
        if(self.stat_obs is None or y_obs!=self.data_set):
            self.stat_obs = self.statistics_calc.statistics(y_obs)
            self.data_set=y_obs

        # Extract summary statistics from the simulated data
        stat_sim = self.statistics_calc.statistics(y_sim)
        
        # Compute the mean, robust precision matrix and determinant of precision matrix
        mean_sim = np.mean(stat_sim,0)
        lw_cov_, _ = ledoit_wolf(stat_sim)
        robust_precision_sim = np.linalg.inv(lw_cov_)
        robust_precision_sim_det = np.linalg.det(robust_precision_sim)
        # print("DEBUG: combining.")
        tmp1 = robust_precision_sim * np.array(self.stat_obs.reshape(-1,1) - mean_sim.reshape(-1,1)).T
        tmp2 = np.exp(np.sum(-0.5*np.sum(np.array(self.stat_obs-mean_sim) * np.array(tmp1).T, axis = 1)))
        tmp3 = pow(np.sqrt((1/(2*np.pi)) * robust_precision_sim_det),self.stat_obs.shape[0])
        return tmp2 * tmp3


class PenLogReg(Approx_likelihood, GraphTools):
    """This class implements the approximate likelihood function which computes the approximate
    likelihood up to a constant using penalized logistic regression described in
    Dutta et. al. [1]. It takes one additional function handler defining the 
    true model and two additional parameters n_folds and n_simulate correspondingly defining number
    of folds used to estimate prediction error using cross-validation and the number 
    of simulated dataset sampled from each parameter to approximate the likelihood
    function. For lasso penalized logistic regression we use glmnet of Friedman et.
    al. [2].
    
    [1] Reference: R. Dutta, J. Corander, S. Kaski, and M. U. Gutmann. Likelihood-free 
    inference by penalised logistic regression. arXiv:1611.10242, Nov. 2016.
    
    [2] Friedman, J., Hastie, T., and Tibshirani, R. (2010). Regularization 
    paths for generalized linear models via coordinate descent. Journal of Statistical 
    Software, 33(1), 1–22.

    Parameters
    ----------
    statistics_calc : abcpy.stasistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    model : abcpy.models.Model
        Model object that conforms to the Model class.
    n_simulate : int
        Number of data points in the simulated data set.
    n_folds: int, optional
        Number of folds for cross-validation. The default value is 10.
    max_iter: int, optional
        Maximum passes over the data. The default is 100000.
    seed: int, optional
        Seed for the random number generator. The used glmnet solver is not
        deterministic, this seed is used for determining the cv folds. The default value is
        None.
    """
    def __init__(self, statistics_calc, model, n_simulate, n_folds=10, max_iter = 100000, seed = None):

        self.model = model
        self.statistics_calc = statistics_calc
        self.n_folds = n_folds
        self.n_simulate = n_simulate
        self.seed = seed
        self.max_iter = max_iter
        # Simulate reference data and extract summary statistics from the reference data
        self.ref_data_stat = self._simulate_ref_data()[0]

        self.stat_obs = None
        self.data_set = None
        

        
    def likelihood(self, y_obs, y_sim):
        if not isinstance(y_obs, list):
            raise TypeError('Observed data is not of allowed types')
        
        if not isinstance(y_sim, list):
            raise TypeError('simulated data is not of allowed types')            
        
        # Extract summary statistics from the observed data
        if(self.stat_obs is None or self.data_set!=y_obs):
            self.stat_obs = self.statistics_calc.statistics(y_obs)
            self.data_set=y_obs
                
        # Extract summary statistics from the simulated data
        stat_sim = self.statistics_calc.statistics(y_sim)
        
        # Compute the approximate likelihood for the y_obs given theta
        y = np.append(np.zeros(self.n_simulate),np.ones(self.n_simulate))
        X = np.array(np.concatenate((stat_sim,self.ref_data_stat),axis=0))
        m = LogitNet(alpha = 1, n_splits = self.n_folds, max_iter = self.max_iter, random_state= self.seed)
        m = m.fit(X, y)
        result = np.exp(-np.sum((m.intercept_+np.sum(np.multiply(m.coef_,self.stat_obs),axis=1)),axis=0))
        
        return result


    def _simulate_ref_data(self, rng=np.random.RandomState()):
        """
        Simulate the reference data set. This code is run at the initialization of
        Penlogreg
        """

        ref_data_stat = [[None]*self.n_simulate for i in range(len(self.model))]
        self.sample_from_prior(rng=rng)
        for model_index, model in enumerate(self.model):
            ind=0
            while(ref_data_stat[model_index][-1] is None):
                data = model.forward_simulate(model.get_input_values(), 1, rng=rng)
                data_stat = self.statistics_calc.statistics(data[0].tolist())
                ref_data_stat[model_index][ind]= data_stat
                ind+=1
            ref_data_stat[model_index] = np.squeeze(np.asarray(ref_data_stat[model_index]))
        return ref_data_stat
