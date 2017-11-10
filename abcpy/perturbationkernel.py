from abc import ABCMeta, abstractmethod
from ProbabilisticModel import Continuous, Discrete
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gamma

class PerturbationKernel(metaclass = ABCMeta):
    """This abstract base class represents all perturbation kernels"""
    @abstractmethod
    def __init__(self, models):
        raise NotImplementedError

    @abstractmethod
    def update(self, weigths):
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x):
        raise NotImplementedError

class StandardKernel(PerturbationKernel):
    """Implementation of a standard kernel. All continuous parameters given are perturbed with a multivariate normal distribution, all discrete parameters are perturbed with a random walk.

    Parameters
    ----------
    models: list
        list of all probabilistic models that should be perturbed
        """
    def __init__(self, models):
        self.models = models

    #NOTE we cant do perturb c[0] and c[1] differently!
    def update(self, weights):
        """Perturbs all the parameters using the weights given.

        Parameters
        ----------
        weights: list
            The weights that should be used to calculate the covariance matrix.

        Returns
        -------
        list(tuple)
            A list containing tuples. For each tuple, the first entry corresponds to the probabilistic model to be considered, the second entry corresponds to the perturbed parameters associated with this model.
        """
        continuous_model_values = []
        discrete_model_values = []

        #find all the current values for the discrete and continous models
        for model in self.models:
            if(isinstance(model, Continuous)):
                for fixed_value in model.fixed_values:
                    continuous_model_values.append(fixed_value)
            else:
                for fixed_value in model.fixed_values:
                    discrete_model_values.append(fixed_value)

        # Perturb continuous parameters
        cov = np.cov(continuous_model_values, aweights=weights)
        perturbed_continuous_values = np.random.multivariate_normal(continuous_model_values, cov)

        # Perturb discrete parameters
        perturbed_discrete_values = []
        for discrete_value in discrete_model_values:
            perturbed_discrete_values.append(np.randint(discrete_value-1,discrete_value+2))

        #NOTE WE could also do that above
        # Merge the two lists
        perturbed_values_including_models = []
        index_in_continuous_models = 0
        index_in_discrete_models=0
        for model in self.models:
            if(isinstance(model, Continuous)):
                model_values = []
                for i in range(model.dimension):
                    model_values.append(perturbed_continuous_values[index_in_continuous_models])
                    index_in_continuous_models+=1
                perturbed_values_including_models.append((model,model_values))
            else:
                model_values=[]
                for i in range(model.dimension):
                    model_values.append(perturbed_discrete_values[index_in_discrete_models])
                    index_in_discrete_models+=1
                perturbed_values_including_models.append((model,model_values))

        return perturbed_values_including_models

    def pdf(self, x):
        return 1


