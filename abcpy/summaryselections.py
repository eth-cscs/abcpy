from abc import ABCMeta, abstractmethod

from abcpy.graphtools import GraphTools
from abcpy.acceptedparametersmanager import *

import numpy as np
from sklearn import linear_model


class Summaryselections(metaclass=ABCMeta):
    """This abstract base class defines a way to choose the summary statistics.
    """

    @abstractmethod
    def __init__(self, model, statistics_calc, backend, n_samples=1000, seed=None):
        """The constructor of a sub-class must accept a non-optional model, statistics calculator and 
        backend which are stored to self.model, self.statistics_calc and self.backend. Further it 
        accepts two optional parameters n_samples and seed defining the number of simulated dataset
        used for the pilot to decide the summary statistics and the integer to initialize the random 
        number generator.
    
        Parameters
        ----------
        model: abcpy.models.Model
            Model object that conforms to the Model class.
        statistics_cal: abcpy.statistics.Statistics
            Statistics object that conforms to the Statistics class.
        backend: abcpy.backends.Backend
            Backend object that conforms to the Backend class.
        n_samples: int, optional
            The number of (parameter, simulated data) tuple generated to learn the summary statistics in pilot step. 
            The default value is 1000.
        n_samples_per_param: int, optional
            Number of data points in each simulated data set.
        seed: integer, optional
            Optional initial seed for the random number generator. The default value is generated randomly.    
        """
        raise NotImplementedError

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['backend']
        return state

    @abstractmethod
    def transformation(self, statistics):
        raise NotImplementedError


class Semiautomatic(Summaryselections, GraphTools):
    """This class implements the semi automatic summary statistics choice described in Fearnhead and Prangle [1].
    
    [1] Fearnhead P., Prangle D. 2012. Constructing summary statistics for approximate
    Bayesian computation: semi-automatic approximate Bayesian computation. J. Roy. Stat. Soc. B 74:419–474.    
    """

    def __init__(self, model, statistics_calc, backend, n_samples=1000, n_samples_per_param = 1, seed=None):
        self.model = model
        self.statistics_calc = statistics_calc
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.n_samples_per_param = n_samples_per_param

        # An object managing the bds objects
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        self.accepted_parameters_manager.broadcast(self.backend, [])

        # main algorithm
        seed_arr = self.rng.randint(1, n_samples * n_samples, size=n_samples, dtype=np.int32)
        rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
        rng_pds = self.backend.parallelize(rng_arr)

        sample_parameters_statistics_pds = self.backend.map(self._sample_parameter_statistics, rng_pds)

        sample_parameters_and_statistics = self.backend.collect(sample_parameters_statistics_pds)
        sample_parameters, sample_statistics = [list(t) for t in zip(*sample_parameters_and_statistics)]
        sample_parameters = np.array(sample_parameters)
        sample_statistics = np.concatenate(sample_statistics)

        self.coefficients_learnt = np.zeros(shape=(sample_parameters.shape[1], sample_statistics.shape[1]))
        regr = linear_model.LinearRegression(fit_intercept=True)
        for ind in range(sample_parameters.shape[1]):
            regr.fit(sample_statistics, sample_parameters[:, ind])
            self.coefficients_learnt[ind, :] = regr.coef_

    def transformation(self, statistics):
        if not statistics.shape[1] == self.coefficients_learnt.shape[1]:
            raise ValueError('Mismatch in dimension of summary statistics')
        return np.dot(statistics, np.transpose(self.coefficients_learnt))

    def _sample_parameter_statistics(self, rng=np.random.RandomState()):
        """
        Samples a single model parameter and simulates from it until
        distance between simulated outcome and the observation is
        smaller than eplison.

        Parameters
        ----------
        seed: int
            value of a seed to be used for reseeding
        Returns
        -------
        np.array
            accepted parameter
        """

        self.sample_from_prior(rng=rng)
        parameter = self.get_parameters()
        y_sim = self.simulate(self.n_samples_per_param, rng=rng)
        if y_sim is not None:
            statistics = self.statistics_calc.statistics(y_sim)
        return (parameter, statistics)
