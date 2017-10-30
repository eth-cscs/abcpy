from abc import ABCMeta, abstractmethod
import numpy as np


#TODO both ricker and lorenz could support that you give some in a combined model, and some not, we should implement that, but not a priority

#NOTE we could call self.parents self.prior?

#NOTE do we maybe want to average over a couple of samples during initialization, rather than taking a single value? this is an issue for example for StudentT!

#TODO in the constructor of probmodel: we go through all the parameters given: if they are not a prob model, we initialize them as a hyperparameter

class ProbabilisticModel(metaclass = ABCMeta):
    """This abstract class represents all probabilistic models.

        Parameters
        ----------
        parameters: list, each element can either be of type ProbabilisticModel or float
            Contains the probabilistic models and hyperparameters which define the parameters of the probabilistic model.

    """
    def __init__(self, parameters):
        #Save all probabilistic models and hyperparameters from which the model derives.
        self.parents = parameters
        #Initialize list which will contain the values for all parameters associated with the model. If the parameters          derive from a probabilistic model, they will be sampled.
        self.parameter_values = []

        try_finding_parameters = 0
        #if the probabilistic model depends on other probabilistic models, sampling might create parameters that lie outside of the accepted parameter range of the model. In this case, resampling is tried 10 times before it is concluded that no appropriate parameters can be found.
        while(try_finding_parameters<10):
            if(self.sample_parameters()):
                break
            try_finding_parameters += 1
        if(try_finding_parameters==10):
            raise ValueError('Cannot find appropriate parameters for probabilistic model.')
        self.visited = False #whether the node has been touched


    def sample_parameters(self, rng=np.random.RandomState()):
        """
        Samples parameters from their distribution, and saves them as parameter values for the current probabilistic model.
        This is commonly used at initialization of a probabilistic model.

        Parameters
        ----------
        rng: Random number generator
            Defines the random number generator to be used by the sampling function.

        Returns
        -------
        boolean
            whether it was possible to set the parameters to sampled values
        """

        #for each parent of the probabilistic model, a value is sampled from this parent. The values are saved to a temporary list
        parameter_values_temp = []
        for parent in self.parents:
            parameter_value = parent.sample_from_distribution(1, rng=rng)
            if (isinstance(parameter_value[0], (list, np.ndarray))):
                parameter_value = parameter_value[0]
            for parameter in parameter_value:
                parameter_values_temp.append(parameter)

        #the temporary list is checked for whether the values are valid for the probabilistic model and in case they are, the parameter_values attribute is fixed to these values
        if (self._check_parameters(parameter_values_temp)):
            # print('Fixed parameters of %s to %s' % (self.__str__(), parameter_values_temp.__str__()))
            self.parameter_values = parameter_values_temp
            self.visited = True
            return True
        else:
            return False

    def set_parameters(self, parameters, rng=np.random.RandomState()):
        """
        Sets the parameter values of the probabilistic model to the specified values.
        This method is commonly used to set new values after perturbing the old parameters.

        Parameters
        ----------
        parameters: python list
            list of the new parameter values
        rng: Random number generator
            Defines the random number generator to be used.

        Returns
        -------
        boolean
            Returns True if it was possible to set the values using the provided list
        """

        #The input is checked for whether it is a valid input for the probabilistic model
        if (not (self._check_parameters_fixed(parameters))):
            return False
        index = 0
        current_parameter_index = 0
        #for each parent, the corresponding parameter_value entry is set to the new value
        while (current_parameter_index < len(parameters)):
            #NOTE WHY DO WE CARE WHETHER IT HAS BEEN VISITED FOR SETTING VALUES?
            while (self.parents[index].visited):
                index += 1
            for j in range(self.parents[index].dimension):
                self.parameter_values[index] = parameters[current_parameter_index]
                current_parameter_index += 1
                index += 1

        #the probabilistic model gets marked as visited so that it does not get set multiple times while setting parameters on the whole graph
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
        for parent in self.parents:
            for j in range(parent.dimension):
                return_values.append(self.parameter_values[index+j])
            index+=parent.dimension
        return return_values

    def number_of_free_parameters(self):
        """
        Returns the number of free parameters of the probabilistic model.
        Commonly used to know how many parameters need to be taken from the total number of perturbed parameters to set new values.
        """
        return len(self.get_parameters())


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
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        x: list
            The point at which the pdf should be evaluated.
        """
        #if the probabilistic model is discrete, there is no probability density function, but a probability mass function. This check ensures that calling the pdf of such a model still works.
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

#NOTE the parameter_values will be a list, check everywhere whether it is okay to be used like that (for hyper not for in general)
class Hyperparameter(ProbabilisticModel):
    """
    This class represents all hyperparameters (i.e. fixed parameters).

    Parameters
    ----------
    parameters: list
        The values to which the hyperparameter should be set
    """
    def __init__(self, parameters):
        #a hyperparameter is defined by the fact that it does not have any parents
        self.parents = []
        self.parameter_values = parameters
        self.visited = False
        self.dimension = 0

    def sample_parameters(self, rng=np.random.RandomState()):
        self.visited = True
        return True

    def set_parameters(self, parameters, rng=np.random.RandomState()):
        self.visited = True
        return True

    def get_parameters(self):
        return []

    def _check_parameters(self, parameters):
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        return self.parameter_values*k

    def pdf(self, x):
        return 1

