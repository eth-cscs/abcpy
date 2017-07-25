from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.covariance import ledoit_wolf
from glmnet import LogitNet 

class Approx_likelihood(metaclass = ABCMeta):
    """This abstract base class defines the approximate likelihood 
    function. To approximate the likelihood function at a parameter value given observed dataset, 
    we need to pass a dataset simulated from model set at the parameter value and the observed dataset. 
    """

    @abstractmethod
    def __init__(self, statistics_calc):       
        """    The constructor of a sub-class must accept a non-optional statistics
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
        value given the observed dataset y_obs and dataset y_sim simulated from
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
    """This class implements the aproximate likelihood function which computes the pproximate
    likelihood using the synthetic likelihood approach described in Wood [1].
    For synthetic likelihood approximation, we compute the robust precision matrix using Ledoit and Wolf's [2]
    method.
    
    [1] S. N. Wood. Statistical inference for noisy nonlinear ecological 
    dynamic systems. Nature, 466(7310):1102–1104, Aug. 2010.
    
    [2] O. Ledoit and M. Wolf, A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices,
    Journal of Multivariate Analysis, Volume 88, Issue 2, pages 365-411, February 2004.
    """
    def __init__(self, statistics_calc):
        self.statistics_calc = statistics_calc


    def likelihood(self, y_obs, y_sim):
        # print("DEBUG: SynLiklihood.likelihood().")
        if not isinstance(y_obs, list):
            raise TypeError('Observed data is not of allowed types')

        if not isinstance(y_sim, list):
            raise TypeError('simulated data is not of allowed types')

        # Extract summary statistics from the observed data
        stat_obs = self.statistics_calc.statistics(y_obs)

        # Extract summary statistics from the simulated data
        stat_sim = self.statistics_calc.statistics(y_sim)

        # Compute the mean, robust precision matrix and determinant of precision matrix
        # print("DEBUG: meansim computation.")
        mean_sim = np.mean(stat_sim,0)
        # print("DEBUG: robust_precision_sim computation.")
        lw_cov_, _ = ledoit_wolf(stat_sim)
        robust_precision_sim = np.linalg.inv(lw_cov_)
        # print("DEBUG: robust_precision_sim_det computation..")
        robust_precision_sim_det = np.linalg.det(robust_precision_sim)
        # print("DEBUG: combining.")
        result = pow(np.sqrt((1/(2*np.pi))*robust_precision_sim_det),stat_obs.shape[0])\
        *np.exp(np.sum(-0.5*np.sum(np.array(stat_obs-mean_sim)* \
        np.array(np.matrix(robust_precision_sim)*np.matrix(stat_obs-mean_sim).T).T, axis = 1)))

        return result


class PenLogReg(Approx_likelihood):
    """This class implements the aproximate likelihood function which computes the pproximate
    likelihood upto a constant using penalized logistic regression described in 
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
    """
    def __init__(self, statistics_calc, model, n_simulate, n_folds=10, max_iter = 100000, seed = None):
        """
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
        
        self.model = model
        self.statistics_calc = statistics_calc
        self.n_folds = n_folds
        self.n_simulate = n_simulate
        self.seed = seed
        self.max_iter = max_iter
        # Simulate reference data and extract summary statistics from the reffernce data      
        self.ref_data_stat = self._simulate_ref_data()
        

        
    def likelihood(self, y_obs, y_sim):
        if not isinstance(y_obs, list):
            raise TypeError('Observed data is not of allowed types')
        
        if not isinstance(y_sim, list):
            raise TypeError('simulated data is not of allowed types')            
        
        # Extract summary statistics from the observed data
        stat_obs = self.statistics_calc.statistics(y_obs)
                
        # Extract summary statistics from the simulated data
        stat_sim = self.statistics_calc.statistics(y_sim)
        
        # Compute the approximate likelihood for the y_obs given theta
        y = np.append(np.zeros(self.n_simulate),np.ones(self.n_simulate))
        X = np.array(np.concatenate((stat_sim,self.ref_data_stat),axis=0))
        m = LogitNet(alpha = 1, n_splits = self.n_folds, max_iter = self.max_iter, random_state= self.seed)
        m = m.fit(X, y)
        result = np.exp(-np.sum((m.intercept_+np.sum(np.multiply(m.coef_,stat_obs),axis=1)),axis=0))
        
        return result


    def _simulate_ref_data(self):
        """
        Simulate the reference dataset. This code is run at the initializtion of 
        Penlogreg
        """

        ref_data_stat = [None]*self.n_simulate
        for ind in range(0,self.n_simulate):        
            self.model.sample_from_prior()
            ref_data_stat[ind] = self.statistics_calc.statistics(self.model.simulate(1))
            
        return np.squeeze(np.asarray(ref_data_stat))            
