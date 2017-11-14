from abc import ABCMeta, abstractmethod
from ProbabilisticModel import Continuous
import numpy as np
from scipy.stats import multivariate_normal


# TODO ask rito how the pdf should be calculated here as well

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
    # TODO adjust docstring
    def update(self, accepted_parameters_manager, row_index, rng=np.random.RandomState()):
        """Perturbs all the parameters using the weights given.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            Defines the accepted parameters manager to be used to access parameter values.
        column_index: integer
            The column of the accepted parameters matrix that should be perturbed
        rng: random number generator
            Defines the random number generator to be used.

        Returns
        -------
        list(tuple)
            A list containing tuples. For each tuple, the first entry corresponds to the probabilistic model to be considered, the second entry corresponds to the perturbed parameters associated with this model.
        """

        all_model_values = accepted_parameters_manager.get_accepted_parameters_bds_values(self.models)
        # TODO this has the wrong dimensions, run test_pmcabc to see, weights have dimension 250, this should have two entries with each 250 values...


        continuous_model_values = [[] for i in range(len(all_model_values))]
        discrete_model_values = [[] for i in range(len(all_model_values))]

        model_values_index = 0

        contains_continuous = False
        contains_discrete = False

        # Find all the currently accepted values for the discrete and continous models
        for model in self.models:
            if(isinstance(model, Continuous)):
                contains_continuous = True
                for i in range(model.dimension):
                    for j in range(len(all_model_values)):
                        continuous_model_values[j].append(all_model_values[j][model_values_index])
                    model_values_index+=1
            else:
                contains_discrete = True
                for i in range(model.dimension):
                    for j in range(len(all_model_values)):
                        discrete_model_values[j].append(all_model_values[j][model_values_index])
                    model_values_index+=1

        # NOTE this flips the things again, this should not happen!
        # Perturb continuous parameters, if applicable
        # NOTE weights has lenght 250, continuous_model_values is in total 2x2
        if(contains_continuous):
            weights = accepted_parameters_manager.accepted_weights_bds.value()
            continuous_model_values = np.array(continuous_model_values)
            cov = np.cov(continuous_model_values, aweights=weights.reshape(-1), rowvar=False)
            #cov = accepted_parameters_manager.accepted_cov_mat_bds.value()
            perturbed_continuous_values = rng.multivariate_normal(continuous_model_values[row_index], cov)

        # Perturb discrete parameters, if applicable
        if(contains_discrete):
            perturbed_discrete_values = []
            discrete_model_values = np.array(discrete_model_values)[row_index]
            for discrete_value in discrete_model_values:
                perturbed_discrete_values.append(rng.randint(discrete_value-1,discrete_value+2))

        # Merge the two lists
        perturbed_values_including_models = []
        index_in_continuous_models = 0
        index_in_discrete_models=0

        for model in self.models:
            model_values = []
            if(isinstance(model, Continuous)):
                for i in range(model.dimension):
                    model_values.append(perturbed_continuous_values[index_in_continuous_models])
                    index_in_continuous_models+=1
                perturbed_values_including_models.append((model,model_values))
            else:
                for i in range(model.dimension):
                    model_values.append(perturbed_discrete_values[index_in_discrete_models])
                    index_in_discrete_models+=1
                perturbed_values_including_models.append((model,model_values))

        return perturbed_values_including_models

    def pdf(self, mean, cov, x):
        return multivariate_normal(mean,cov).pdf(x)


