from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import gamma


#TODO both ricker and lorenz could support that you give some in a combined model, and some not, we should implement that, but not a priority

#NOTE we could call self.parents self.prior?

#NOTE do we maybe want to average over a couple of samples during initialization, rather than taking a single value? this is an issue for example for StudentT!

#TODO in the constructor of probmodel: we go through all the parameters given: if they are not a prob model, we initialize them as a hyperparameter

#NOTE WE COULD ALSO IMPLEMENT IT SUCH THAT GET AND SET USE THE ORDER OF THE PARENTS AS THEY APPEAR --> EASIER TO DO WITH INFERENCES.PY, BUT WILL HAVE TO IMPLEMENT AN EXTRA FUNCTION FOR USER OUTPUT MAYBE?


#NOTE CURRENTLY, IF WE GIVE TWICE THE SAME MODEL, WE GET THE SAME VALUE FOR BOTH PARAMETERS! WE SHOULD RESAMPLE!
class ProbabilisticModel(metaclass = ABCMeta):
    """This abstract class represents all probabilistic models.

        Parameters
        ----------
        parameters: list, each element can either be of type ProbabilisticModel or float
            Contains the probabilistic models and hyperparameters which define the parameters of the probabilistic model.

    """
    def __init__(self, parameters):
        #Save all probabilistic models and hyperparameters from which the model derives without duplicates.
        self.parents = []

        # Initialize list which will contain the values for all parameters associated with the model. If the parameters          derive from a probabilistic model, they will be sampled.
        self.parameter_values = []

        #NOTE probably needs renaming
        #Initialize a list which will contain the order in which the output of a model should be assigned to the parameter values of a model derived from the current model
        self.children_index = []

        #Initialize a counter which specifies the current index to be considered in the children_index list.
        self.index = 0

        #NOTE probably needs renaming
        #Initialize a list which will contain a mapping of parameter values to corresponding parents as well as index of a sampled output of the parent
        self.parameter_index = []

        #boolean to mark whether a parent has been included in the list of parents before
        has_been_used=False

        #loop over all given parameters, and set the corresponding parameter_index entry to a tupel of the correct parent and index in the output of this parent
        for parameter in parameters:
            has_been_used=False
            for index, parent in enumerate(self.parents):
                #if the parameter is already contained in the parents-list, it gets marked
                if(parameter==parent):
                    has_been_used = True
                    current_parent=index
                    break

            #if the parameter is not in the parents list yet, it gets added to it
            if(not(has_been_used)):
                self.parents.append(parameter)
                current_parent = len(self.parents)-1

            #set the parameter_index value, depending on whether the access operator was used or not
            if(not(parameter.children_index)):
                #have this outside the loop to ensure that hyperparameters are initialized as well
                self.parameter_index.append((current_parent, 0))
                for j in range(1, parameter.dimension):
                    self.parameter_index.append((current_parent,j))
            else:
                self.parameter_index.append((current_parent, parameter.children_index[parameter.index]))
                parameter.index+=1

        #clear all children_index and index values
        for parameter in parameters:
            parameter.children_index=[]
            parameter.index=0


        try_finding_parameters = 0
        #if the probabilistic model depends on other probabilistic models, sampling might create parameters that lie outside of the accepted parameter range of the model. In this case, resampling is tried 10 times before it is concluded that no appropriate parameters can be found.
        while(try_finding_parameters<10):
            if(self.sample_parameters()):
                break
            try_finding_parameters += 1
        if(try_finding_parameters==10):
            raise ValueError('Cannot find appropriate parameters for probabilistic model.')
        self.visited = False #whether the node has been touched


    def __getitem__(self, item):
        """
        Overloads the access operator. If the access operator is called, the specified index is saved in the children_index list.
        Commonly used at initialization of new probabilistic models to specify a mapping between model outputs and parameters.
        """
        self.children_index.append(item)
        return self


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

        #for each parent of the probabilistic model, a value is sampled from this parent. The values are saved to a list
        parent_values = []
        for parent in self.parents:
            parameter_value = parent.sample_from_distribution(1, rng=rng)
            #NOTE THIS CHECKER NOW BREAKS FOR 2D COV MATRICES, SINCE BELOW WE ACCESS THE 0TH ELEMENT AFTERWARDS -> needs additional [] around cov to work properly...
            if (isinstance(parameter_value[0], (list, np.ndarray))):
                parameter_value = parameter_value[0]
            parent_values.append(parameter_value)

        #use the mapping provided in parameter_index to assign the proper parameter values to each parameter in a temporary list
        parameter_values_temp = []
        for parameter_index in self.parameter_index:
            parameter_values_temp.append(parent_values[parameter_index[0]][parameter_index[1]])

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
        parameter_values_index=0
        current_parameters_index=0
        #iterate over all parameter_index. If the parent is not a hyperparameter, set the corresponding value
        for parameter_index in self.parameter_index:
            #NOTE why do we need to check whether it has been visited?
            if(not(parameter_index[0].visited) and parameter_index[0].dimension!=0):
                self.parameter_values[parameter_values_index] = parameters[current_parameters_index]
                current_parameters_index+=1
            parameter_values_index+=1

        #the probabilistic model gets marked as visited so that it does not get set multiple times while setting parameters on the whole graph
        self.visited = True
        return True

    #NOTE THIS GIVES BACK IN ORDER OF INPUT, IE NORMAL(A[1],A[0]) -> A[1],A[0]
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
                return_values.append(self.parameter_values[index])
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
        return self.parameter_values*k

    def pdf(self, x):
        return 1

class Normal(ProbabilisticModel, Continuous):
    """
    This class implements a probabilistic model following a normal distribution with mean mu and variance sigma.

    Parameters
    ----------
    parameters: list
        Contains the probabilistic models and hyperparameters from which the model derives. If the list only has 1             entry, this entry is required to be a 2-dimensional ProbabilisticModel. Note that the second value of the list is not allowed to be smaller than 0.
    """
    def __init__(self, parameters):
        super(Normal, self).__init__(parameters)
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = 1

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from a normal distribution using the current values for each probabilistic model from which the model derives.

        Parameters
        ----------
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the                  generator.
        """
        mu = self.parameter_values[0]
        sigma = self.parameter_values[1]
        return np.array(rng.normal(mu, sigma, k).reshape(-1))

    def _check_parameters(self, parameters):
        """
        Checks parameter values sampled from the parents of the normal distribution at initialization. Returns False iff the variance is smaller than 0.
        """
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for Normal has to be of type list.')
        if(parameters[1]<=0):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values that are given as fixed values. Returns False iff the variance would be set to 0 or the length of the input and free parameters do not agree.
        """
        if(super(Normal, self).number_of_free_parameters()==len(parameters)):
            if(len(parameters)==2 and parameters[1]<=0):
                return False
            return True
        return False

    def pdf(self, x):
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        x: list
            The point at which the pdf should be evaluated.
        """
        mu = self.parameter_values[0]
        sigma = self.parameter_values[1]
        return norm(mu,sigma).pdf(x)

class MultivariateNormal(ProbabilisticModel, Continuous):
    """
    This class implements a probabilistic model following a multivariate normal distribution with mean and covariance matrix.

    Parameters
    ----------
    parameters: list of at least length 2
        Contains the probabilistic models and hyperparameters from which the model derives. The last entry defines the         covariance matrix, while all other entries define the mean. Note that if the mean is n dimensional, the                covariance matrix is required to be of dimension nxn. The covariance matrix is required to be positive-definite.
    """
    def __init__(self,parameters):
        super(MultivariateNormal, self).__init__(parameters)
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parameter_values)-1

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
    Samples from a multivariate normal distribution using the current values for each probabilistic model from which the model derives.

    Parameters
    ----------
    k: integer
        The number of samples that should be drawn.
    rng: Random number generator
        Defines the random number generator to be used. The default value uses a random seed to initialize the                  generator.
    """
        mean = self.parameter_values[:len(self.parameter_values)-1]
        cov = self.parameter_values[len(self.parameter_values)-1]
        return rng.multivariate_normal(mean, cov, k)

    def _check_parameters(self, parameters):
        """
        Checks parameter values sampled from the parents at initialization. Returns False iff the covariance matrix is not symmetric or not positive definite.
        """
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for MultivariateNormal has to be of type list.')
        if(len(parameters)<2):
            raise IndexError('Input for MultivariateNormal has to be of at least length 2.')
        length = len(parameters)-1
        cov = np.array(parameters[len(parameters)-1])
        if(length!=len(cov[0])):
            raise IndexError('Length of mean and covariance matrix have to match.')

        #check whether the covariance matrix is symmetric
        if(not(np.allclose(cov, cov.T, atol=1e-3))):
            return False
        #check whether the covariance matrix is positive definite
        try:
            is_pos = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values that are given as fixed values. Returns False iff the number of parameters given and the number of free parameters are not equal.
        """
        if(len(parameters)!=super(MultivariateNormal, self).number_of_free_parameters()):
            return False
        return True

    def pdf(self, x):
        """
       Calculates the probability density function at point x.
       Commonly used to determine whether perturbed parameters are still valid according to the pdf.

       Parameters
       ----------
       x: list
           The point at which the pdf should be evaluated.
       """
        mean= self.parameter_values[:len(self.parameter_values)-1]
        cov = self.parameter_values[len(self.parameter_values)-1]
        return multivariate_normal(mean, cov).pdf(x)
