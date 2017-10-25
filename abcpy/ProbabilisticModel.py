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
        #this loop samples multiple times in case a sampled parameter lies outside the accepted range for that parameter.
        counter = 0
        while(counter<10):
            if(self.sample_parameters()):
                break
            counter += 1
        if(counter==10):
            raise ValueError('Cannot find appropriate parameters for probabilistic model.')
        self.visited = False #whether the node has been touched


    def sample_parameters(self, rng=np.random.RandomState()):
        parameter_values_temp = []
        for i in range(len(self.parents)):
            if (isinstance(self.parents[i], ProbabilisticModel)):
                parameter_value = self.parents[i].sample_from_distribution(1, rng=rng)
                if (isinstance(parameter_value[0], (list, np.ndarray))):
                    parameter_value = parameter_value[0]
                for j in range(len(parameter_value)):
                    parameter_values_temp.append(parameter_value[j])
            else:
                parameter_values_temp.append(self.parents[i])
        if (self._check_parameters(parameter_values_temp)):
            # print('Fixed parameters of %s to %s' % (self.__str__(), parameter_values_temp.__str__()))
            self.parameter_values = parameter_values_temp
            self.visited = True
            return True
        else:
            return False

    def set_parameters(self, parameters, rng=np.random.RandomState()):
        if (not (self._check_parameters_fixed(parameters))):
            return False
        index = 0
        i = 0
        while (i < len(parameters)):
            while (not (isinstance(self.parents[index], ProbabilisticModel)) or self.parents[index].visited):
                index += 1
            # NOTE this does currently not work for Uniform, because an empty list will be converted to 0 entries
            if (not (parameters[i]) and not (isinstance(self, Uniform))):
                parameter_value = self.parents[index].sample_from_distribution(1, rng=rng)
                if (isinstance(parameter_value[0], (list, np.ndarray))):
                    parameter_value = parameter_value[0]
                for j in range(len(parameter_value)):
                    self.parameter_values[index + j] = parameter_value[j]
                index += len(parameter_value)
                i += len(parameter_value)
            else:
                for j in range(self.parents[index].dimension):
                    self.parameter_values[index + j] = parameters[i]
                    i += 1
                index += self.parents[index].dimension
        self.visited = True
        return True


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
        self._num_parameters = 0
        self.length = [0,
                       0]  # this is needed to check that lower and upper are of same length, just because the total length is even does not guarantee that
        joint_parameters = []
        for i in range(2):
            for j in range(len(parameters[i])):
                joint_parameters.append(parameters[i][j])
                if (isinstance(parameters[i][j], ProbabilisticModel)):
                    self.length[i] += parameters[i][j].dimension
                else:
                    self.length[i] += 1
        self._num_parameters = self.length[0] + self.length[1]
        self.dimension = int(self._num_parameters / 2)
        super(Uniform, self).__init__(joint_parameters)
        self.visited = False

        # Parameter specifying the dimension of the return values of the distribution.

    def num_parameters(self):
        return self._num_parameters

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        samples = np.zeros(shape=(k, self.dimension))  # this means: len columns, and each has k entries
        for j in range(0, self.dimension):
            samples[:, j] = rng.uniform(self.parameter_values[j], self.parameter_values[j + self.dimension], k)
        return samples

    def _check_user_input(self, parameters):
        if (not (isinstance(parameters, list))):
            raise TypeError('Input for Uniform has to be of type list.')
        if (len(parameters) < 2):
            raise IndexError('Input to Uniform has to be at least of length 2.')
        if (not (isinstance(parameters[0], list))):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')
        if (not (isinstance(parameters[1], list))):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')

    def _check_parameters(self, parameters):
        if (self.num_parameters() % 2 == 1):
            raise IndexError('Length of upper and lower bound have to be equal.')
        if (self.length[0] != self.length[1]):
            raise IndexError('Length of upper and lower bound have to be equal.')
        for i in range(self.dimension):
            if (parameters[i] > parameters[i + self.dimension]):
                return False
        return True

    def _check_parameters_fixed(self, parameters):
        i = 0
        index = 0
        index_paramter_values = 0
        bounds = [[], []]
        length_free = 0
        for j in range(2):
            length = 0
            while (length < self.length[j]):
                if (isinstance(self.parents[i], ProbabilisticModel)):
                    length += self.parents[i].dimension
                    for t in range(self.parents[i].dimension):
                        bounds[j].append(parameters[index])
                        index += 1
                        index_paramter_values += 1
                        length_free += 1
                else:
                    length += 1
                    bounds[j].append(self.parameter_values[index_paramter_values])
                    index_paramter_values += 1
                i += 1
        if (length_free == len(parameters)):
            for j in range(len(bounds[0])):
                if (bounds[0][j] > bounds[1][j]):
                    return False
            return True
        return False

    def pdf(self, x):
        lower_bound = self.parameter_values[:self.dimension]
        upper_bound = self.parameter_values[self.dimension:]
        if (np.product(np.greater_equal(x, np.array(lower_bound)) * np.less_equal(x, np.array(upper_bound)))):
            pdf_value = 1. / np.product(np.array(upper_bound) - np.array(lower_bound))
        else:
            pdf_value = 0.
        return pdf_value