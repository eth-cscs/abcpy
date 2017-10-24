import numpy as np
from ProbabilisticModel import ProbabilisticModel
from continuous import Continuous

"""Toughts on Kernel:

Imo, the only way is to collect all parameters are the root (what happens for multiple roots?). Kernel dependencies can go both ways. If there is an edge a->b, this means that for the kernel a<->b. Therefore, all parameters depend on all, or the kernel at root has the dimension of total free parameters.

This means that we need to be able to collect all values at the root, sample it there, and then distribute the appropriate values through the model.

However, it could be that some parameters go with the kernel, but others dont, there are for example two kernels (i.e. "all variances change the same and all means do, but means and variances dont". So, we need to specify for the kernel: all parameters which belong to it.

As far as I can see, a kernel is currently implemented as "kernel = MultiNormal" but we do not care about the values the user gives, always before the kernel is called for the first time, we call kernel.set_parameters.

So the user should ideally define:

what distribution he wants to use. And then which parameters should obey this kernel. All, only some, and so on. How we go through the graph is our problem.

So he should define
kernel = kernel.Multinormal(a,c,d)
-> this tells us when we do perturb: hey, these three need to be together, so group them
-> it should also be such that if we now say "and kernel.Multinormal(e,f,d) 
this either should not work at all, or in this case maybe prodcue a joint?
"""


class Uniform(ProbabilisticModel, Continuous):
    """
    This class implements a probabilistic model following a uniform distribution.

    Parameters
    ----------
    parameters: list
        Contains two lists. The first list specifies the probabilistic models and hyperparameters from which the lower         bound of the uniform distribution derive. The second list specifies the probabilistic models and hyperparameters        from which the upper bound derives.
    """
    def __init__(self, parameters):
        self._check_user_input(parameters)
        self.parent_length_lower = len(parameters[0])
        self.parent_length_upper = len(parameters[1])
        self.length = [0,0]
        joint_parameters = []
        for i in range(2):
            for j in range(len(parameters[i])):
                joint_parameters.append(parameters[i][j])
                if(isinstance(parameters[i][j], ProbabilisticModel)):
                    self.length[i]+=parameters[i][j].dimension
                else:
                    self.length[i]+=1
        super(Uniform, self).__init__(joint_parameters)
        self.lower_bound = self.parameter_values[:self.length[0]]
        self.upper_bound = self.parameter_values[self.length[0]:]
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = self.length[0]

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        #print(lower_bound)
        samples = np.zeros(shape=(k, len(self.lower_bound))) #this means: len columns, and each has k entries
        for j in range(0, len(self.lower_bound)):
            samples[:, j] = rng.uniform(self.lower_bound[j], self.upper_bound[j], k)
        return samples

    def _check_user_input(self, parameters):
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for Uniform has to be of type list.')
        if(len(parameters)<2):
            raise IndexError('Input to Uniform has to be at least of length 2.')
        if(not(isinstance(parameters[0], list))):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')
        if(not(isinstance(parameters[1], list))):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')

    def _check_parameters(self, parameters):
        if(self.length[0]!=self.length[1]):
            raise IndexError('Length of upper and lower bound have to be equal.')
        for i in range(self.length[0]):
            if(parameters[i]>parameters[i+self.length[0]]):
                return False
        return True


    def _check_parameters_fixed(self, parameters):
        length = [0,0]
        bounds = [[],[]]
        index=0
        for i in range(self.parent_length_lower):
            if(isinstance(self.parents[i], ProbabilisticModel)):
                length[0]+=self.parents[i].dimension
                for j in range(self.parents[i].dimension):
                    bounds[0].append(parameters[index])
                    index+=1
            else:
                bounds[0].append(self.parameter_values[i])
        for i in range(self.parent_length_lower, self.parent_length_lower+self.parent_length_upper):
            if(isinstance(self.parents[i], ProbabilisticModel)):
                length[1]+=self.parents[i].dimension
                for j in range(self.parents[i].dimension):
                    bounds[1].append(parameters[index])
                    index+=1
            else:
                bounds[1].append(self.parameter_values[i])
        if(length[0]+length[1]==len(parameters)):
            for i in range(len(bounds[0])):
                if(bounds[0][i]>bounds[1][i]):
                    return False
            return True
        return False

    def get_parameters(self):
        lb_parameters = []
        index=0
        for i in range(self.parent_length_lower):
            if(isinstance(self.parents[i], ProbabilisticModel)):
                for j in range(self.parents[i].dimension):
                    lb_parameters.append(self.parameter_values[index])
                    index+=1
            else:
                index+=1
        ub_parameters = []
        for i in range(self.parent_length_lower, self.parent_length_lower+self.parent_length_upper):
            if(isinstance(self.parents[i], ProbabilisticModel)):
                for j in range(self.parents[i].dimension):
                    ub_parameters.append(self.parameter_values[index])
            else:
                index+=1
        return [lb_parameters, ub_parameters]

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        if (not(parameters)):
            if(super(Uniform, self).fix_parameters(rng=rng)):
                self.updated = True
                return True
            else:
                return False
        else:
            joint_parameters =[]
            for i in range(2):
                for j in range(len(parameters[i])):
                    joint_parameters.append(parameters[i][j])
            if(super(Uniform, self).fix_parameters(joint_parameters)):
                self.lower_bound = self.parameter_values[:self.length[0]]
                self.upper_bound = self.parameter_values[self.length[0]:]
                return True
            return False


    def pdf(self, x):
        if (np.product(np.greater_equal(x, np.array(self.lower_bound)) * np.less_equal(x, np.array(self.upper_bound)))):
            pdf_value = 1. / np.product(np.array(self.upper_bound) - np.array(self.lower_bound))
        else:
            pdf_value = 0.
        return pdf_value