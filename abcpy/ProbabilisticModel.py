from abc import ABCMeta, abstractmethod
import numpy as np

#TODO both ricker and lorenz could support that you give some in a combined model, and some not, we should implement that, but not a priority

#NOTE we could call self.parents self.prior?

#NOTE do we maybe want to average over a couple of samples during initialization, rather than taking a single value? this is an issue for example for StudentT!



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
    #TODO MAYBE WE SHOULD CHECK WHETHER IT IS POSSIBLE TO OBTAIN THE PARAMETERS FROM THD PARENT DISTRIBUTIONS BEFORE ASSINGING?
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

    def pdf(self, x):
        if(isinstance(self, Discrete)):
            return self.pmf(x)
        else:
            raise NotImplementedError

class Continuous(metaclass = ABCMeta):
    """
    This abstract class represents all continuous probabilistic models.
    """
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

class Discrete(metaclass = ABCMeta):
    """
    This abstract class represents all discrete probabilistic models.
    """
    @abstractmethod
    def pmf(self, x):
        """
        Calculates the probability mass function of the model, if applicable.

        Parameters
        ----------
        x: float
            The location at which the probability mass function should be evaluated.
        """
        raise NotImplementedError
