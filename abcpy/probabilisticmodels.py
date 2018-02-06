from abc import ABCMeta, abstractmethod
import numpy as np

class ProbabilisticModel(metaclass = ABCMeta):
    """This abstract class represents all probabilistic models.
    """
    def __init__(self, parameters, name=''):
        """This constructor should be called from any derived class. 

        It requires as input all parameters (random variables) on which the current
        model depends. These input parameters can be specified in different ways:

        1. as a tuple (ProbabilisticModel | Hyperparameter, int) to use specific output parameters of a model
        2. as ProbabilisticModel to use all output parameters of a model
        3. as int | float for hyperparameters

        In the first case the current model depends on a single output parameter
        (second tuple element) of a probabilistic model (first tuple
        element). In the second case on all output parameters of a probabilistic
        model, and, in the third case, on a fixed valued hyperparameter. Note
        that internally, the constructor will rewrite all input parameters to
        the first case.

        Parameters
        ----------
        parameters: list
            A list of input parameters.

        name: string
            A human readible name for the model. Can be the variable name for example.
        """

        # set name
        self.name = name

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
                    parents_temp.append((Hyperparameter([parameter]), 0))
            else:
                parents_temp.append(parameter)
        # Check whether the suggested parameters are allowed for this probabilistic model
        self._check_parameters_at_initialization(parents_temp)
        self.parents = parents_temp

        # A flag containing whether the probabilistic model has been touched during a recursive operation
        self.visited = False

        self.calculated_pdf = None

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
            Contains the probabilistic models and hyperparameters which define the probabilistic model as tupels.
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

        Returns
        -------
        boolean
            Whether it is possible to sample from the distribution, given the parent parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_parameters_fixed(self, parameters):
        """
        Checks parameters in the set_parameters method. Should return False iff the parameters cannot come from the distribution of the probabilistic model.

        Parameters
        ----------
        parameters: list
            Contains the values to which the free parameters should be fixed.

        Returns
        -------
        boolean:
            Whether the given parameters could have been sampled from this distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_from_distribution(self, k, rng):
        """
        Samples from the distribution associated with the probabilistic model by using the current values for each              probabilistic model from which the model derives.

        Parameters
        ----------
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the                  generator.

        Returns
        -------
        list:
            The first entry is a boolean, specifying whether it was possible to sample from the distribution. The second entry is a numpy array, containing k elements, each of length self.dimension, the k samples from the distribution.
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

        Returns
        -------
        float:
            The pdf evaluated at point x.
        """
        # If the probabilistic model is discrete, there is no probability density function, but a probability mass function. This check ensures that calling the pdf of such a model still works.
        if(isinstance(self, Discrete)):
            return self.pmf(x)
        else:
            raise NotImplementedError

    def __add__(self, other):
        """Overload the + operator for probabilistic models.

        Parameters
        ----------
        other: probabilistic model or Hyperparameter
            The model to be added to self.

        Returns
        -------
        SummationModel
            A probabilistic model describing a model coming from summation.
        """
        return SummationModel([self,other])

    def __radd__(self, other):
        """Overload the + operator from the righthand side to support addition of Hyperparameters from the left.

        Parameters
        ----------
        Other: Hyperparameter
            The hyperparameter to be added to self.

        Returns
        -------
        SummationModel
            A probabilistic model describgin a model coming from summation.
        """
        return SummationModel([other, self])

    def __sub__(self, other):
        """Overload the - operator for probabilistic models.

        Parameters
        ----------
        other: probabilistic model or Hyperparameter
            The model to be subtracted from self.

        Returns
        -------
        SubtractionModel
            A probabilistic model describing a model coming from subtraction.
        """
        return SubtractionModel([self,other])

    def __rsub__(self, other):
        """Overload the - operator from the righthand side to support subtraction of Hyperparameters from the left.

        Parameters
        ----------
        Other: Hyperparameter
            The hyperparameter to be subtracted from self.

        Returns
        -------
        SubtractionModel
            A probabilistic model describing a model coming from subtraction.
        """
        return SubtractionModel([other,self])

    def __mul__(self, other):
        """Overload the * operator for probabilistic models.

        Parameters
        ----------
        other: probabilistic model or Hyperparameter
            The model to be multiplied with self.

        Returns
        -------
        MultiplicationModel
            A probabilistic model describing a model coming from multiplication.
        """
        return MultiplicationModel([self,other])

    def __rmul__(self, other):
        """Overload the * operator from the righthand side to support subtraction of Hyperparameters from the left.

                Parameters
                ----------
                Other: Hyperparameter
                    The hyperparameter to be subtracted from self.

                Returns
                -------
                MultiplicationModel
                    A probabilistic model describing a model coming from multiplication.
                """
        return MultiplicationModel([other,self])

    def __truediv__(self, other):
        """Overload the / operator for probabilistic models.

        Parameters
        ----------
        other: probabilistic model or Hyperparameter
            The model to be divide self.

        Returns
        -------
        DivisionModel
            A probabilistic model describing a model coming from division.
        """
        return DivisionModel([self, other])

    def __rtruediv__(self, other):
        """Overload the / operator from the righthand side to support subtraction of Hyperparameters from the left.

        Parameters
        ----------
        Other: Hyperparameter
            The hyperparameter to be subtracted from self.

        Returns
        -------
        DivisionModel
            A probabilistic model describing a model coming from division.
        """
        return DivisionModel([other, self])

    def __pow__(self, power, modulo=None):
        return ExponentialModel([self, power])

    def __rpow__(self, other):
        return RExponentialModel([other, self])


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

    """
    def __init__(self, parameters, name='Hyperparameter'):
        """

        Parameters
        ----------
        parameters: list
            The values to which the hyperparameter should be set
        """
        # A hyperparameter is defined by the fact that it does not have any parents
        self.parents = []
        self.name = name
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
        return 1.


class ModelResultingFromOperation(ProbabilisticModel):
    """This class implements probabilistic models returned after performing an operation on two probabilistic models
        """

    def __init__(self, parameters, name=''):
        """

        Parameters
        ----------
        parameters: list
            List of probabilistic models that should be added together.

        """
        super(ModelResultingFromOperation, self).__init__(parameters, name)

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        raise NotImplementedError

    def _check_parameters_at_initialization(self, parameters):
        """Checks whether the parameters are valid at initialization.

        Parameters
        ----------
        parameters: list of tupels
            """
        parent_1 = parameters[0][0]
        parent_2 = parameters[-1][0]
        length_parent_1 = 0
        length_parent_2 = 0
        for parent, parent_index in parameters:
            if (parent == parent_1):
                length_parent_1 += 1
            else:
                length_parent_2 += 1
        if (length_parent_1 == length_parent_2):
            self.dimension = length_parent_1
            return
        raise ValueError('The provided models are not of equal dimension.')

    def _check_parameters_before_sampling(self, parameters):
        """Checks parameters before sampling. Provided due to inheritance."""
        return True

    def _check_parameters_fixed(self, parameters):
        """Checks parameters while setting them. Provided due to inheritance."""
        return True

    def pdf(self, x):
        """Calculates the probability density function at point x.

        Parameters
        ----------
        x: float or list
            The point at which the pdf should be evaluated.

        Returns
        -------
        float
            The probability density function evaluated at point x.
        """
        # Since the nodes provided as input have to be independent, the resulting pdf will be pdf(parent 1)*pfd(parent 2). During the recursive graph action, this is calculated automatically, so the pdf at this node is expected to be 1
        return 1.


class SummationModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from an addition of two probabilistic models"""

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """Adds the sampled values of both parent distributions.

        Parameters
        ----------
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the sum of the parents values.
        """
        return_value = []

        # we need to obtain new samples of the parents for each sample (if we just use get_parameter_values, we will have k identical samples)
        for i in range(k):
            # make sure each parent is only sampled once (for use of access operator or similar)
            visited_parents = [False for i in range(len(self.parents))]
            parameter_values = [0 for i in range(len(self.parents))]

            # sample from each parent and associate the sampled values with the correct positions in parameter_vaulues
            for parent_loc, parent in enumerate(self.parents):
                parent = parent[0]
                if(not(visited_parents[parent_loc])):
                    sample_of_parent = parent.sample_from_distribution(1, rng=rng)
                    if(sample_of_parent[0]):
                        for parent_loc_tmp, parent_tmp in enumerate(self.parents):
                            if(parent==parent_tmp[0]):
                                visited_parents[parent_loc_tmp]=True
                                parameter_values[parent_loc_tmp]=sample_of_parent[1][parent_tmp[1]]
                    else:
                        return [False]

            # add the corresponding parameter_values
            sample_value = []
            for j in range(self.dimension):
                sample_value.append(parameter_values[j]+parameter_values[j+self.dimension])
            if(len(sample_value)==1):
                sample_value=sample_value[0]
            return_value.append(sample_value)

        return [True, np.array(return_value)]


class SubtractionModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from an subtraction of two probabilistic models"""

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """Adds the sampled values of both parent distributions.

        Parameters
        ----------
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the difference of the parents values.
        """
        return_value = []
        sample_value = []

        # we need to obtain new samples of the parents for each sample (if we just use get_parameter_values, we will have k identical samples)
        for i in range(k):
            # make sure each parent is only sampled once (for use of access operator or similar)
            visited_parents = [False for i in range(len(self.parents))]
            parameter_values = [0 for i in range(len(self.parents))]

            # sample from each parent and associate the sampled values with the correct positions in parameter_vaulues
            for parent_loc, parent in enumerate(self.parents):
                parent = parent[0]
                if (not (visited_parents[parent_loc])):
                    sample_of_parent = parent.sample_from_distribution(1, rng=rng)
                    if (sample_of_parent[0]):
                        for parent_loc_tmp, parent_tmp in enumerate(self.parents):
                            if (parent == parent_tmp[0]):
                                visited_parents[parent_loc_tmp] = True
                                parameter_values[parent_loc_tmp] = sample_of_parent[1][parent_tmp[1]]
                    else:
                        return [False]

            # subtract the corresponding parameter_values
            sample_value = []
            for j in range(self.dimension):
                sample_value.append(parameter_values[j] - parameter_values[j + self.dimension])
            if(len(sample_value)==1):
                sample_value=sample_value[0]
            return_value.append(sample_value)

        return [True, np.array(return_value)]


class MultiplicationModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from a multiplication of two probabilistic models"""
    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """Multiplies the sampled values of both parent distributions element wise.

        Parameters
        ----------
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the product of the parents values.
            """
        return_value = []

        # we need to obtain new samples of the parents for each sample (if we just use get_parameter_values, we will have k identical samples)
        for i in range(k):
            # make sure each parent is only sampled once (for use of access operator or similar)
            visited_parents = [False for i in range(len(self.parents))]
            parameter_values = [0 for i in range(len(self.parents))]

            # sample from each parent and associate the sampled values with the correct positions in parameter_vaulues
            for parent_loc, parent in enumerate(self.parents):
                parent = parent[0]
                if (not (visited_parents[parent_loc])):
                    sample_of_parent = parent.sample_from_distribution(1, rng=rng)
                    if (sample_of_parent[0]):
                        for parent_loc_tmp, parent_tmp in enumerate(self.parents):
                            if (parent == parent_tmp[0]):
                                visited_parents[parent_loc_tmp] = True
                                parameter_values[parent_loc_tmp] = sample_of_parent[1][parent_tmp[1]]
                    else:
                        return [False]

            # multiply the corresponding parameter_values
            sample_value = []

            for j in range(self.dimension):
                sample_value.append(parameter_values[j]*parameter_values[j+self.dimension])
            if (len(sample_value) == 1):
                sample_value = sample_value[0]
            return_value.append(sample_value)

        return [True, np.array(return_value)]


class DivisionModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from a division of two probabilistic models"""
    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """Divides the sampled values of both parent distributions.

        Parameters
        ----------
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the fraction of the parents values.
        """
        return_value = []

        # we need to obtain new samples of the parents for each sample (if we just use get_parameter_values, we will have k identical samples)
        for i in range(k):
            # make sure each parent is only sampled once (for use of access operator or similar)
            visited_parents = [False for i in range(len(self.parents))]
            parameter_values = [0 for i in range(len(self.parents))]

            # sample from each parent and associate the sampled values with the correct positions in parameter_vaulues
            for parent_loc, parent in enumerate(self.parents):
                parent = parent[0]
                if (not (visited_parents[parent_loc])):
                    sample_of_parent = parent.sample_from_distribution(1, rng=rng)
                    if (sample_of_parent[0]):
                        for parent_loc_tmp, parent_tmp in enumerate(self.parents):
                            if (parent == parent_tmp[0]):
                                visited_parents[parent_loc_tmp] = True
                                parameter_values[parent_loc_tmp] = sample_of_parent[1][parent_tmp[1]]
                    else:
                        return [False]

            # divide the corresponding parameter_values
            sample_value = []

            for j in range(self.dimension):
                sample_value.append(parameter_values[j]/parameter_values[j + self.dimension])
            return_value.append(sample_value)

        return [True, np.array(return_value)]


class ExponentialModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from an exponentiation of two probabilistic models"""

    def _check_parameters_at_initialization(self, parameters):
        """Raises an error iff the exponent has more than 1 dimension."""
        parent_power = parameters[-1][0]
        number_of_parent_power = 0

        for parent_parameters, parent_parameters_index in parameters:
            if(parent_parameters==parent_power):
                number_of_parent_power+=1

        if(number_of_parent_power>1):
            raise ValueError('The exponent can only be 1 dimensional.')

        self.dimension = len(parameters[:-1])

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """Raises the sampled values of the base by the exponent.

        Parameters
        ----------
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the exponential of the parents values.
        """
        result = []

        for i in range(k):
            # make sure each parent is only sampled once (for use of access operator or similar)
            visited_parents = [False for i in range(len(self.parents))]
            parameter_values = [1. for j in range(len(self.parents))]

            # sample from each parent and associate the sampled values with the correct positions in parameter_vaulues
            for parent_loc, parent in enumerate(self.parents):
                parent = parent[0]
                if (not (visited_parents[parent_loc])):
                    sample_of_parent = parent.sample_from_distribution(1, rng=rng)
                    if (sample_of_parent[0]):
                        for parent_loc_tmp, parent_tmp in enumerate(self.parents):
                            if (parent == parent_tmp[0]):
                                visited_parents[parent_loc_tmp] = True
                                parameter_values[parent_loc_tmp] = sample_of_parent[1][parent_tmp[1]]
                    else:
                        return [False]

            power = parameter_values[-1]

            sample_value = []

            for j in range(self.dimension):
                sample_value.append(parameter_values[j]**power)
            result.append(sample_value)

        return [True, np.array(result)]


class RExponentialModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from an exponentiation of a Hyperparameter by another probabilistic model."""

    def _check_parameters_at_initialization(self, parameters):
        """Raises an error iff the exponent has more than 1 dimension."""

        parent_power = parameters[0][0]
        number_of_parent_power = 0

        for parent_parameteres, parent_parameters_index in parameters:
            if(parent_parameteres==parent_power):
                number_of_parent_power+=1

        if(number_of_parent_power>1):
            raise ValueError('The exponent can only be 1 dimensional.')

        self.dimension = len(parameters[1:])

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """Raises the base by the sampled value of the exponent.

        Parameters
        ----------
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the exponential of the parents values.
        """
        result = []

        for i in range(k):
            # make sure each parent is only sampled once (for use of access operator or similar)
            visited_parents = [False for i in range(len(self.parents))]
            parameter_values = [1. for j in range(len(self.parents))]

            # sample from each parent and associate the sampled values with the correct positions in parameter_vaulues
            for parent_loc, parent in enumerate(self.parents):
                parent = parent[0]
                if (not (visited_parents[parent_loc])):
                    sample_of_parent = parent.sample_from_distribution(1, rng=rng)
                    if (sample_of_parent[0]):
                        for parent_loc_tmp, parent_tmp in enumerate(self.parents):
                            if (parent == parent_tmp[0]):
                                visited_parents[parent_loc_tmp] = True
                                parameter_values[parent_loc_tmp] = sample_of_parent[1][parent_tmp[1]]
                    else:
                        return [False]

            power = parameter_values[0]

            sample_value = []

            for j in range(self.dimension):
                sample_value.append(parameter_values[j] ** power)
            result.append(sample_value)

        return [True, np.array(result)]

