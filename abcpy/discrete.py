from ProbabilisticModel import ProbabilisticModel, Discrete

import numpy as np
from scipy.misc import comb

#NOTE tested using scipy.stats.binom, all functions give correct values
class Binomial(Discrete, ProbabilisticModel):
    """
    This class implements a probabilistic model following a binomial distribution.

    Parameters
    ----------
    parameters: list
        Contains the probabilistic models and hyperparameters from which the model derives. Note that the first entry of the list has to be larger than or equal to 0, while the second entry has to be in the interval [0,1].
    """
    def __init__(self, parameters):
        # NOTE super will call the constructor of Probabilistic model, as long as we do not specify a constructor for continuous/discrete
        super(Binomial, self).__init__(parameters)
        self.dimension = 1
        #ensure that values sampled from other distributions will be of type int
        self.parameter_values[0] = int(self.parameter_values[0])

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        if(super(Binomial, self).fix_parameters(parameters, rng=rng)):
            self.parameter_values[0] = int(self.parameter_values[0])
            return True
        return False

    def get_parameters(self):
        return super(Binomial, self).get_parameters()

    def _check_parameters(self, parameters):
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for Binomial has to be of type list.')
        if(parameters[0]<0):
            return False
        if(parameters[1]<0 or parameters[1]>1):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        length=0
        for parent in self.parents:
            if(isinstance(parent, ProbabilisticModel)):
                length+=parent.dimension
        if(length==len(parameters)):
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