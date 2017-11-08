from abc import ABCMeta, abstractmethod
import numpy as np

#TODO ricker and lorenz implementations

#NOTE should we make hyperparameter prviate? the user should never call it as it would break behavior

#NOTE we never check whether the number of parameters we give is equal to that of the number of free parameters associated with a model. But I dont think that is necessary


class ProbabilisticModel(metaclass = ABCMeta):
    """This abstract class represents all probabilistic models.

        Parameters
        ----------
        parameters: list, each element is either a tupel containing the parent as well as the output index to which this parameter corresponds, a ProbabilisticModel or a hyperparameter.
    """
    def __init__(self, parameters):
        # Save all probabilistic models and hyperparameters from which the model derives
        self.parents = []

        # Initialize a list that will later contain the current sampled values for this distribution
        self.fixed_values = [None]

        parents_temp = []

        # Initialize the parents
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
        # Check whether the suggested parameters are allowed for this probabilistic model
        self._check_parameters_at_initialization(parents_temp)
        self.parents = parents_temp

        # A flag containing whether the probabilistic model has been touched during a recursive operation
        self.visited = False

    def __getitem__(self, item):
        """
        Overloads the access operator. If the access operator is called, a tupel of the ProbablisticModel that called the operator and the index at which it was called is returned.
        Commonly used at initialization of new probabilistic models to specify a mapping between model outputs and parameters.

        Parameters
        ----------
        item: integer
            The index in the output of the parent model which should be linked to the parameter being defined.
        """
        # Ensure the specified index does not lie outside the range of the return value of the model
        if(item>=self.dimension):
            raise IndexError('The specified index lies out of range for probabilistic model %s.'%(self.__class__.__name__))
        return self, item

    def get_parameter_values(self):
        """
        Returns the values to be used by the model as parameters.
        Commonly used when sampling from the distribution.
        """
        return_value = []
        # Saves the parameter values provided by the parents in the desired order specified in self.parents
        for parameter, index in self.parents:
            return_value.append(parameter.fixed_values[index])
        return return_value

    def sample_parameters(self, rng=np.random.RandomState()):
        """
        Samples from the distribution associated with the probabilistic model and assigns the result to fixed_values, if applicable.
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

        sample_result = self.sample_from_distribution(1, rng=rng)
        # Sample_result will contain two entries, the first being True, iff the fixed_values from the parent models are an allowed input to the current model
        if(sample_result[0]):
            fixed_values_temp=sample_result[1]
            if(isinstance(fixed_values_temp[0], (list, np.ndarray))):
                fixed_values_temp = fixed_values_temp[0]
            self.fixed_values = fixed_values_temp
            return True
        return False

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
    def _check_parameters_before_sampling(self, parameters):
        """
        Checks parameters before sampling from the distribution.

        Parameters
        ----------
        parameters: list
            Contains the current sampled values for all parents of the current probabilistic model.
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
        # If the probabilistic model is discrete, there is no probability density function, but a probability mass function. This check ensures that calling the pdf of such a model still works.
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
        return True

    def get_parameters(self):
        return []

    def _check_parameters_at_initialization(self, parameters):
        return True

    def _check_parameters_before_sampling(self, parameters):
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        return [True, self.fixed_values*k]

    def pdf(self, x):
        # Mathematically, the expression for the pdf of a hyperparameter should be: if(x==self.fixed_parameters) return 1; else return 0; However, since the pdf is called recursively for the whole model structure, and pdfs multiply, this would mean that all pdfs become 0. Setting the return value to 1 ensures proper calulation of the overall pdf.
        return 1

