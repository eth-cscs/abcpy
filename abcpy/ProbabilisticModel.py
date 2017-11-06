from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import gamma


#TODO both ricker and lorenz could support that you give some in a combined model, and some not, we should implement that, but not a priority

#NOTE we could call self.parents self.prior?


#NOTE according to rito, if we specify: a[0], a[0], a[1], we want that a[0]s are the same! --> our code already does the right thing.

#todo do we need some kind of axis like marcel specified?
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

        # Initialize list which will contain the values for all parameters associated with the model. If the parameters          derive from a probabilistic model, they will be sampled.
        self.fixed_parameters = [None]

        #initialize the parents
        for parameter in parameters:
            if(not(isinstance(parameter, tuple))):
                #if an entry is a ProbabilisticModel, all the output values are saved in order in self.parents
                if(isinstance(parameter, ProbabilisticModel)):
                    for i in range(parameter.dimension):
                        self.parents.append((parameter, i))
                #if an entry is not of type ProbabilisticModel or a tupel, it is a hyperparameter
                else:
                    self.parents.append(Hyperparameter([parameter]))
            else:
                self.parents.append(parameter)

        #initialize all fixed_parameters to None, so that fixed_parameters has the correct length
        #NOTE we could instead also just set the dimension over self.parameter_index, and leave fixed_parameters empty until we sample?
        self.fixed_parameters*=len(self.parents)


        self.visited = False #whether the node has been touched


    def __getitem__(self, item):
        """
        Overloads the access operator. If the access operator is called, a tupel of the ProbablisticModel that called the operator and the index at which it was called is returned.
        Commonly used at initialization of new probabilistic models to specify a mapping between model outputs and parameters.
        """
        return (self, item)


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
        #TODO rewrite this once we know whether rito now really wants values saved at the nodes!!!

        #for each parent of the probabilistic model, a value is sampled from this parent. The values are saved to a list
        parent_values = []
        for parent in self.parents:
            fixed_parameters = parent.sample_from_distribution(1, rng=rng)
            if (isinstance(fixed_parameters[0], (list, np.ndarray))):
                fixed_parameters = fixed_parameters[0]
            parent_values.append(fixed_parameters)

        #use the mapping provided in parameter_index to assign the proper parameter values to each parameter in a temporary list
        fixed_parameters_temp = []
        for parameter_index in self.parameter_index:
            fixed_parameters_temp.append(parent_values[parameter_index[0]][parameter_index[1]])

        #the temporary list is checked for whether the values are valid for the probabilistic model and in case they are, the fixed_parameters attribute is fixed to these values
        if (self._check_parameters(fixed_parameters_temp)):
            # print('Fixed parameters of %s to %s' % (self.__str__(), fixed_parameters_temp.__str__()))
            self.fixed_parameters = fixed_parameters_temp
            return True
        else:
            return False

    #TODO this function should work differently in case we save values at nodes
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
        fixed_parameters_index=0
        current_parameters_index=0
        #iterate over all parameter_index. If the parent is not a hyperparameter, set the corresponding value
        for parameter_index in self.parameter_index:
            #NOTE why do we need to check whether it has been visited?
            if(not(parameter_index[0].visited) and parameter_index[0].dimension!=0):
                self.fixed_parameters[fixed_parameters_index] = parameters[current_parameters_index]
                current_parameters_index+=1
            fixed_parameters_index+=1
        return True

    #NOTE THIS GIVES BACK IN ORDER OF INPUT, IE NORMAL(A[1],A[0]) -> A[1],A[0]
    #TODO as well, if we save values at nodes, this should work differently
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
        #Append all the parameter values which do not correspond to a hyperparameter
        for parameter_index in self.parameter_index:
            if(parameter_index[0].dimension!=0):
                return_values.append(self.fixed_parameters[index])
            index+=1
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

#NOTE the fixed_parameters will be a list, check everywhere whether it is okay to be used like that (for hyper not for in general)
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
        self.fixed_parameters = parameters
        self.visited = False
        self.dimension = 0
        self.children_index = []
        self.parameter_index = []

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
        return self.fixed_parameters*k

    def pdf(self, x):
        #Mathematically, the expression for the pdf of a hyperparameter should be: if(x==self.fixed_parameters) return 1; else return 0; However, since the pdf is called recursively for the whole model structure, and pdfs multiply, this would mean that all pdfs become 0. Setting the return value to 1 ensures proper calulation of the overall pdf.
        return 1

