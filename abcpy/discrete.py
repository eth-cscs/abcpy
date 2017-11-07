from ProbabilisticModel import ProbabilisticModel, Discrete

import numpy as np
from scipy.special import comb
#TODO rewrite for new implementations
#NOTE tested using scipy.stats.binom, all functions give correct values
class Binomial(Discrete, ProbabilisticModel):
    """
    This class implements a probabilistic model following a binomial distribution.

    Parameters
    ----------
    parameters: list
        Contains the probabilistic models and hyperparameters from which the model derives. Note that the first entry of the list, n, has to be larger than or equal to 0, while the second entry, p, has to be in the interval [0,1].
    """
    def __init__(self, parameters):
        super(Binomial, self).__init__(parameters)
        self.dimension = 1
        #ensure that values sampled from other distributions will be of type int
        self.parameter_values[0] = int(self.parameter_values[0])

    def set_parameters(self, parameters, rng=np.random.RandomState()):
        """
        Sets the parameters of the model. If the first parameter is not an integer, it is casted to one.
        """
        if(super(Binomial, self).set_parameters(parameters, rng=rng)):
            self.parameter_values[0] = int(self.parameter_values[0])
            return True
        return False

    def _check_parameters(self, parameters):
        """
        Checks the parameter values at initialization. Returns False iff the first parameter is less than 0 or the second parameter does not lie in [0,1]
        """
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for Binomial has to be of type list.')
        if(parameters[0]<0):
            return False
        if(parameters[1]<0 or parameters[1]>1):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values given as fixed values. Returns False iff the number of free parameters of the model is not equal to the length of the parameters given, or the given parameter values do not lie within the accepted ranges.
        """
        length=0
        for parent in self.parents:
            if(super(Binomial, self).number_of_free_parameters()==len(parameters)):
                if(isinstance(self.parents[0], ProbabilisticModel) and parameters[0]<0):
                    return False
                if(isinstance(self.parents[1], ProbabilisticModel) and len(parameters)==1 and (parameters[0]<0 or parameters[0]>1)):
                    return False
                if(isinstance(self.parents[1], ProbabilisticModel) and len(parameters)==2 and (parameters[1]<0 or parameters[1]>1)):
                    return False
            return True
        return False

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        n = self.parameter_values[0]
        p = self.parameter_values[1]
        return rng.binomial(n, p, k)

    def pmf(self, x):
        #NOTE either x is for sure an int, or we round it?
        x = int(x)
        n = self.parameter_values[0]
        p = self.parameter_values[1]
        if(x>n):
            return 0
        return comb(n,x)*(p**x)*(1-p)**(n-x)