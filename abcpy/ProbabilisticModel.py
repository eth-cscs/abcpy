from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import gamma

#TODO time the function calls once everything is implemented -> compare to if you use numba


#TODO both ricker and lorenz could support that you give some in a combined model, and some not, we should implement that, but not a priority

#NOTE we could call self.parents self.prior?

#NOTE it is not possible to give a hyperparameter as Hyperparameter([1]), only as 1, do we want the other possibility?

class ProbabilisticModel(metaclass = ABCMeta):
    """This abstract class represents all probabilistic models.

        Parameters
        ----------
        parameters: list, each element is either a tupel containing the parent as well as the output index to which this parameter corresponds, a ProbabilisticModel or a hyperparameter.
            Contains the probabilistic models and hyperparameters which define the parameters of the probabilistic model.

    """
    def __init__(self, parameters):
        #Save all probabilistic models and hyperparameters from which the model derives
        self.parents = []

        #Initialize a list that will later contain the current sampled values for this distribution
        self.fixed_values = [None]

        parents_temp = []

        #initialize the parents
        for parameter in parameters:
            if(not(isinstance(parameter, tuple))):
                #if an entry is a ProbabilisticModel, all the output values are saved in order in self.parents
                if(isinstance(parameter, ProbabilisticModel)):
                    for i in range(parameter.dimension):
                        parents_temp.append((parameter, i))
                #if an entry is not of type ProbabilisticModel or a tupel, it is a hyperparameter
                else:
                    parents_temp.append((Hyperparameter([parameter]),0))
            else:
                parents_temp.append(parameter)
        #check whether the suggested parameters are allowed for this probabilistic model
        if(self._check_parameters_at_initialization(parents_temp)):
            self.parents = parents_temp
        else:
            raise ValueError('Domains of the specified parents do not match the required range of this model.')

        self.visited = False #whether the node has been touched


    def __getitem__(self, item):
        """
        Overloads the access operator. If the access operator is called, a tupel of the ProbablisticModel that called the operator and the index at which it was called is returned.
        Commonly used at initialization of new probabilistic models to specify a mapping between model outputs and parameters.
        """
        return (self, item)

    def get_parameter_values(self):
        """
        Returns the values to be used by the model as parameters.
        Commonly used when sampling from the distribution.
        """
        return_value = []
        #saves the parameter values provided by the parents in the desired order specified in self.parents
        for parameter, index in self.parents:
            return_value.append(parameter.fixed_values[index])
        return return_value


    #todo: this should check somehow things to know whether to return true or false. Can it even happen that this is false as long we do not have a domain?
    #NOTE why did this return true/false?
    def sample_parameters(self, rng=np.random.RandomState()):
        """
        Samples the parameter value using the distribution associated with the model. Saves this value in self.fixed_values.
        Commonly used when sampling from the prior.

        Parameters
        ----------
        rng: Random number generator
            Defines the random number generator to be used by the sampling function.

        Returns
        -------
        boolean
            whether it was possible to set the parameters to sampled values
        """

        fixed_values_temp = self.sample_from_distribution(1, rng=rng)
        if(isinstance(fixed_values_temp[0], (list, np.ndarray))):
            fixed_values_temp = fixed_values_temp[0]
        self.fixed_values = fixed_values_temp
        return True

    #NOTE currently, only check_parameters of uniform can return False
    def set_parameters(self, parameters):
        """
        Sets the parameter values of the probabilistic model to the specified values.
        This method is commonly used to set new values after perturbing the old parameters.

        Parameters
        ----------
        parameters: python list
            list of the new parameter values

        Returns
        -------
        boolean
            Returns True if it was possible to set the values using the provided list
        """
        if(self._check_parameters_fixed(parameters)):
            self.fixed_values = parameters
            return True
        return False

    def get_parameters(self):
        """
        Returns the current sampled value of the probabilistic model.

        Returns
        -------
        return_values: list
            The current values of the model.
        """
        return self.fixed_values

    #todo this is wrong, but do we ever need this function anymore?
    def number_of_free_parameters(self):
        """
        Returns the number of free parameters of the probabilistic model.
        Commonly used to know how many parameters need to be taken from the total number of perturbed parameters to set new values.
        """
        return len(self.get_parameters())


    @abstractmethod
    def _check_parameters_at_initialization(self, parameters):
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
        Checks parameters in the fixed_parameters method.

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
        self.fixed_values = parameters
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

    def _check_parameters_at_initialization(self, parameters):
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        return self.fixed_values*k

    def pdf(self, x):
        #Mathematically, the expression for the pdf of a hyperparameter should be: if(x==self.fixed_parameters) return 1; else return 0; However, since the pdf is called recursively for the whole model structure, and pdfs multiply, this would mean that all pdfs become 0. Setting the return value to 1 ensures proper calulation of the overall pdf.
        return 1

