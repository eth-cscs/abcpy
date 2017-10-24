from abc import ABCMeta, abstractmethod
import numpy as np

from scipy.stats import multivariate_normal, norm
from scipy.special import gamma

class ProbabilisticModel(metaclass = ABCMeta):
    """This abstract class represents all probabilistic models.

        Parameters
        ----------
        parameters: list, each element can either be of type ProbabilisticModel or float
            Contains the probabilistic models and hyperparameters which define the parameters of the probabilistic model.
        is_uniform: boolean
            Set to true if the probabilistic model describes a uniform distribution.

    """
    def __init__(self, parameters):
        #Save all probabilistic models and hyperparameters from which the model derives.
        self.parents = parameters
        #Initialize list which will contain the values for all parameters associated with the model. If the parameters          derive from a probabilistic model, they will be sampled.
        self.parameter_values = []
        parameter_values_temp = []
        counter = 0
        while(counter<10):
            if(self.fix_parameters()):
                break
            counter += 1
        if(counter==10):
            raise ValueError('Cannot find appropriate parameters for probabilistic model.')
        self.updated = False #whether the node has been touched

    #NOTE the check_fixed assumes that the parameters are exactly of the lenght needed for that distribution only ->needs to be considered in case we want to be able to "fix on whole graph"
    @abstractmethod
    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        """
        Fixes the parameters associated with the probabilistic model.

        Parameters
        ----------
        parameters: None or list of length equal to the free parameters of the probabilistic model.
            If set to none, all free parameters are sampled from their distribution.
        rng: Random number generator
            Defines the random number generator to be used for sampling. The default value is initialized with a random             seed.
        Returns
        -------
        boolean:
            Tells the user whether the parameters were fixed. They will not be fixed if they do not conform to the                  required format of the underlying distribution.
        """
        parameter_values_temp = []
        if(not(parameters)):
            for i in range(len(self.parents)):
                if(isinstance(self.parents[i], ProbabilisticModel)):
                    parameter_value = self.parents[i].sample_from_distribution(1, rng=rng)
                    if(isinstance(parameter_value[0], (list, np.ndarray))):
                        parameter_value = parameter_value[0]
                    for j in range(len(parameter_value)):
                        parameter_values_temp.append(parameter_value[j])
                else:
                    parameter_values_temp.append(self.parents[i])
            if(self._check_parameters(parameter_values_temp)):
                self.parameter_values = parameter_values_temp
                self.updated = True
                return True
            else:
                return False
        else:
            if(not(self._check_parameters_fixed(parameters))):
                return False
            index=0
            i=0
            while(i<len(parameters)):
                while(not(isinstance(self.parents[index],ProbabilisticModel)) or self.parents[index].updated):
                    index+=1
                #NOTE this does currently not work for Uniform, because an empty list will be converted to 0 entries
                if(not(parameters[i]) and not(isinstance(self, Uniform))):
                    parameter_value = self.parents[index].sample_from_distribution(1, rng=rng)
                    if(isinstance(parameter_value[0], (list, np.ndarray))):
                        parameter_value = parameter_value[0]
                    for j in range(len(parameter_value)):
                        self.parameter_values[index+j] = parameter_value[j]
                    index+=len(parameter_value)
                    i+=len(parameter_value)
                else:
                    for j in range(self.parents[index].dimension):
                        self.parameter_values[index+j]=parameters[i]
                        i+=1
                    index+=self.parents[index].dimension
            self.updated = True
            return True

    @abstractmethod
    def get_parameters(self):
        """
        Returns the current values of the free parameters of the probabilistic model.

        Returns
        -------
        return_values: list
            The current values of the free parameters.
        """
        return_values = []
        index=0
        for i in range(len(self.parents)):
            if(isinstance(self.parents[i], ProbabilisticModel)):
                for j in range(self.parents[i].dimension):
                    return_values.append(self.parameter_values[index+j])
                index+=self.parents[i].dimension
            else:
                index+=1
        return return_values


    @abstractmethod
    def _check_parameters(self, parameters):
        """
        Checks parameters at initialization.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters which define the probabilistic model.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_parameters_fixed(self, parameters):
        """
        Checks parameters in the fix_parameters method.

        Parameters
        ----------
        parameters: list
            Contains the values to which the free parameters should be fixed.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from the distribution associated with the probabilistic model by using the current values for each              probabilistic model from which the model derives.

        Parameters
        ----------
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the                  generator.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x):
        """
        Calculates the probability density function of the model, if applicable.

        Parameters
        ----------
        x: float
            The location at which the probability density function should be evaluated.
        """
        raise NotImplementedError


class Uniform(ProbabilisticModel):
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