import numpy as np

from abcpy.probabilisticmodels import ProbabilisticModel, Hyperparameter

class Normal(ProbabilisticModel):
    def __init__(self, parameters):
        super(Normal, self).__init__(parameters)
        self.dimension = 1

    def _check_parameters_at_initialization(self, parameters):
        if(len(parameters)!=2):
            raise IndexError('Normal should have exactly two input parameters.')
        variance, index = parameters[1]
        if(isinstance(variance, Hyperparameter) and variance.fixed_values[0]<0):
            raise ValueError('Variance has to be larger than 0.')

    def _check_parameters_before_sampling(self, parameters):
        if(parameters[1]<0):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        parameter_values = self.get_parameter_values()
        return_value = []

        return_value.append(self._check_parameters_before_sampling(parameter_values))

        if (return_value[0]):
            mu = parameter_values[0]
            sigma = parameter_values[1]
            return_value.append(np.array(rng.normal(mu, sigma, k)).reshape(-1))

        return return_value

    def pdf(self, x):
        parameter_values = self.get_parameter_values()
        mu = parameter_values[0]
        sigma = parameter_values[1]
        return norm(mu, sigma).pdf(x)