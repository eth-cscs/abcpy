from abc import ABCMeta, abstractmethod
from numbers import Number
import numpy as np


class InputConnector():
    def __init__(self, dimension):
        """
        Creates input parameters of given dimensionality. Each dimension needs to be specified using the set method.

        Parameters
        ----------
        dimension: int
            Dimensionality of the input parameters.
        """

        self._all_indices_specified = False
        self._dimension = dimension
        self._models = [None]*dimension
        self._model_indices = [None]*dimension


    def from_number(number):
        """
        Convenient initializer that converts a number to a hyperparameter input parameter.

        Parameters
        ----------
        number

        Returns
        -------
        InputConnector
        """

        if isinstance(number, Number):
            input_parameters = InputConnector(1)
            input_parameters.set(0, Hyperparameter(number), 0)
            return input_parameters
        else:
            raise TypeError('Unsupported type.')


    def from_model(model):
        """
        Convenient initializer that converts the full output of a model to input parameters.

        Parameters
        ----------
        ProbabilisticModel

        Returns
        -------
        InputConnector
        """

        if isinstance(model, ProbabilisticModel):
            input_parameters = InputConnector(model.get_output_dimension())
            for i in range(model.get_output_dimension()):
                input_parameters.set(i, model, i)
            return input_parameters
        else:
            raise TypeError('Unsupported type.')


    def from_list(parameters):
        """
        Creates an InputParameters object from a list of ProbabilisticModels.

        In this case, number of input parameters equals the sum of output dimensions of all models in the parameter
        list. Further, the output and models are connected to the input parameters in the order they appear in the
        parameter list.

        For convenience,
        - the parameter list can contain nested lists
        - the method also accepts numbers instead of models, which are automatically converted to hyper parameters.

        Parameters
        ----------
        parameters: list
            A list of ProbabilisticModels

        Returns
        -------
        InputConnector
        """

        if isinstance(parameters, list):
            unnested_parameters = []
            parameters_count = 0
            for item in parameters:
                input_parameters_from_item = item
                if isinstance(item, list):
                    input_parameters_from_item = InputConnector.from_list(item)
                elif isinstance(item, (Hyperparameter, ProbabilisticModel)):
                    input_parameters_from_item = InputConnector.from_model(item)
                elif isinstance(item, Number):
                    input_parameters_from_item = InputConnector.from_number(item)
                elif not isinstance(item, InputConnector):
                    raise TypeError('Unsupported type.')

                unnested_parameters.append(input_parameters_from_item)
                parameters_count += input_parameters_from_item.get_parameter_count()

            # here, unnested_parameters is a list of InputConnector and parameters_count hold the total number of
            # parameters in this list
            input_parameters = InputConnector(parameters_count)
            index = 0
            for param in unnested_parameters:
                for pi in range(0, param.get_parameter_count()):
                    input_parameters.set(index, param._models[pi], param._model_indices[pi])
                    index += 1
            return input_parameters
        else:
            raise TypeError('Input is not a list')



    def __getitem__(self, index):
        """
        For the input models, return those fixed value(s) that are specified by index.

        In case a value of an input model is not (yet) fixed, None is returned.

        In other words, if the input models are interpreted as random variables, this method returns a realization for
        all random variables used as input parameter, which is specified by 'index'.

        Parameters
        ----------
        index: int, slice

        Returns
        -------
        int, float, list, None
        """

        # index is a single number
        if isinstance (index, Number):
            model = self._models[index]
            model_index = self._model_indices[index]
            if model.get_stored_output_values() is None:
                return None
            else:
                output_values = model.get_stored_output_values()
                return output_values[model_index]

        # index is a slice
        elif isinstance(index, slice):
            result = []
            for i in range(index.start, index.stop):
                val = self[i]
                if val != None:
                    result.append(val)
                else:
                    return None
            return result


    def get_values(self):
        """
        Returns the fixed values of all input models.

        Returns
        -------
        np.array
        """

        result = [0]*self._dimension
        for i in range(0, self._dimension):
            result[i] = self.__getitem__(i)
        return result


    def get_models(self):
        """
        Returns a list of all models.

        Returns
        -------
        list
        """

        return self._models


    def get_model(self, index):
        """
        Returns the model at index.

        Returns
        -------
        ProbabilisticModel
        """

        return self._models[index]


    def get_parameter_count(self):
        """
        Returns the number of parameters.

        Returns
        -------
        int
        """

        return self._dimension


    def set(self, index, model, model_index):
        """
        Sets for an input parameter index the input model and the model index to use.

        For convenience, model can also be a number, which is automatically casted to a hyper parameter.

        Parameters
        ----------
        index: int
            Index of the input parameter to be set.
        model: ProbabilisticModel, Number
            The model to be set for the input parameter.
        model_index: int
            Index of model's output to be used as input parameter.
        """

        if isinstance(model, Number):
            model = Hyperparameter(model)

        self._models[index] = model
        self._model_indices[index] = model_index
        if (self._models != None):
            self._all_indices_specified = True


    def all_models_fixed_values(self):
        """
        Checks whether all input models have fixed an output value (pseudo data).

        In order get a fixed output value (a realization of the random variable described by the model) a model has to
        run a forward simulation, which is not done automatically upon initialization.

        Returns
        -------
        boolean
        """

        for model in self._models:
            if model.get_stored_output_values() is None:
                return False
        return True


class ProbabilisticModel(metaclass = ABCMeta):
    """
    This abstract class represents all probabilistic models.
    """

    def __init__(self, input_connector, name=''):
        """
        This initializer *must be* called from any derived class to properly connect it to its input models.

        It accepts as input an InputConnector object that fully specifies how to connect all parent models to the
        current model.

        Parameters
        ----------
        input_connector: list
            A list of input parameters.

        name: string
            A human readable name for the model. Can be the variable name for example.
        """


        # set name
        self.name = name

        # parameters is of type InputConnector
        if isinstance(input_connector, InputConnector):
            if input_connector.all_models_fixed_values() and self._check_input(input_connector.get_values()) == False:
                raise ValueError('Input parameters are not compatible with current model.')
            self._parameters = input_connector
        else:
            raise TypeError('Input parameters are of wrong type.')

        self._fixed_values = None

        # A flag indicating whether the model has been touched during a recursive operation
        self.visited = False
        self.calculated_pdf = None


    def __getitem__(self, item):
        """
        Overloads the access operator. If the access operator is called, a tupel of the ProbablisticModel that called
        the operator and the index at which it was called is returned. Commonly used at initialization of new
        probabilistic models to specify a mapping between model outputs and parameters.

        Parameters
        ----------
        item: integer
            The index in the output of the parent model which should be linked to the parameter being defined.
        """

        if isinstance(item, Number):
            if(item>=self.get_output_dimension()):
                raise IndexError('The specified index lies out of range for probabilistic model %s.'%(self.__class__.__name__))
            input_parameters = InputConnector(1)
            input_parameters.set(0, self, item)
            return input_parameters
        else:
            raise TypeError('Input of unsupported type.')


    def get_input_values(self):
        """
        Returns the input values from the parent models as a list.
        Commonly used when sampling from the distribution.

        Returns
        -------
        list
        """

        return self.get_input_connector().get_values()


    def get_input_models(self):
        """
        Returns a list of all input models.

        Returns
        -------
        list
        """

        input_connector = self.get_input_connector()
        return input_connector.get_models()


    def get_stored_output_values(self):
        """
        Returns the stored sampled value of the probabilistic model after setting the values explicitly.

        At initialization the function should return None.

        Returns
        -------
        numpy.array or None.
        """

        return self._fixed_values


    def get_input_connector(self):
        """
        Returns the input connector object that connecects the current model to its parents.

        In case of no dependencies, this function should return None.

        Returns
        -------
        InputConnector, None
        """

        return self._parameters


    def get_input_dimension(self):
        """
        Returns the input dimension of the current model.

        Returns
        -------
        int

        """

        return self._parameters._dimension


    def set_output_values(self, values):
        """
        Sets the output values of the model. This method is commonly used to set new values after perturbing the old
        ones.


        Parameters
        ----------
        values: numpy array or dimension equal to output dimension.

        Returns
        -------
        boolean
            Returns True if it was possible to set the values, false otherwise.
        """
        if not isinstance(values, np.ndarray):
            raise TypeError('Elements of input list are not of type numpy array.')
        if values.shape[0] != self.get_output_dimension():
            raise IndexError('Size of input list not equal to number output dimensions.')

        if self._check_output(values):
            self._fixed_values = values
            return True
        return False


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
            A probabilistic model describing a model coming from summation.
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
        return SubtractionModel([self, other])


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


    def _forward_simulate_and_store_output(self, rng=np.random.RandomState()):
        """
        Samples from the model associated and assigns the result to fixed_values, if applicable. Commonly used when
        sampling from the prior.

        Parameters
        ----------
        rng: Random number generator
            Defines the random number generator to be used by the sampling function.

        Returns
        -------
        boolean
            Check whether it was possible to set the parameters to sampled values.
        """

        parameters_are_valid = self._check_input(self.get_input_values())
        if(parameters_are_valid):
            sample_result = self.forward_simulate(self.get_input_values(), 1, rng=rng)
            if sample_result != None:
                self.set_output_values(sample_result[0])
                return True
        return False


    def pdf(self, input_values, x):
        """
        Calculates the probability density function at point x.

        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        x: list
            The point at which the pdf should be evaluated.

        Returns
        -------
        float:
            The pdf evaluated at point x.
        """
        # If the probabilistic model is discrete, there is no probability density function, but a probability mass function. This check ensures that calling the pdf of such a model still works.
        if(isinstance(self, Discrete)):
            return self.pmf(input_values, x)
        else:
            raise NotImplementedError


    def calculate_and_store_pdf_if_needed(self, x):
        """
        Calculates the probability density function at point x and stores the result internally for later use.

        This function is intended to be used within the inference computation.

        Parameters
        ----------
        x: list
            The point at which the pdf should be evaluated.
        """

        if self._calculated_pdf == None:
            self._calculated_pdf = self.pdf(self.get_input_values(), x)


    def flush_stored_pdf(self):
        """
        This function flushes the internally stored value of a previously computed pdf.
        """

        self._calculated_pdf = None


    def get_stored_pdf(self):
        """
        Retrieves the value of a previously calculated pdf.

        Returns
        -------
        number
        """

        return self._calculated_pdf


    @abstractmethod
    def _check_input(self, input_values):
        """
        Check whether the input parameters are compatible with the underlying model.

        The following behavior is expected:

        1. If the input is of wrong type or has the wrong format, this method should *raise an exception*. For example,
        if the number of parameters does not match what the model expects.

        2. If the values of the input models are not compatible, this method should return False. For example, if an
        input value is not from the expected domain.

        Background information: Many inference schemes modify the input slightly by applying a small perturbation during
        sampling. This method is called to check whether the perturbation yields a reasonable input to the current
        model. In case this function returns False, the inference schemes re-perturb the input and try again. If the
        check is not done properly, the inference computation might **crash** or **not terminate**.

        Parameters
        ----------
        input_values: list
            A list of numbers that are the concatenation of all parent model outputs in the order specified by the
            InputConnector object that was passed during initialization.

        Returns
        -------
        boolean
            True if the fixed value of the parameters can be used as input for the current model. False otherwise.
        """

        raise NotImplementedError


    @abstractmethod
    def _check_output(self, values):
        """
        Checks whether values contains a reasonable output of the current model.

        Parameters
        ----------
        values: numpy array
            Array of shape (get_output_dimension(),) that contains the model output.

        Returns
        -------
        boolean
            Return false if values cannot possibly be generated from the model and true otherwise.
        """

        raise NotImplementedError


    @abstractmethod
    def forward_simulate(self, input_values, k, rng, mpi_comm=None):
        """
        Provides the output (pseudo data) from a forward simulation of the current model.

        In case the model is intended to be used as input for another model, a forward simulation **must** return a list
        of k numpy arrays with shape (get_output_dimension(),).

        In case the model is directly used for inference, and not as input for another model, a forward simulation
        also must return a list, but the elements can be arbitrarily defined. In this case it is only important that the
        used statistics and distance functions can read the input.

        Parameters
        ----------
        input_values: list
            A list of numbers that are the concatenation of all parent model outputs in the order specified by the
            InputConnector object that was passed during initialization.
        k: integer
            The number of forward simulations that should be run
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the
            generator.
        mpi_comm: MPI communicator object
        	    Defines the MPI communicator object for MPI parallelization. The default value is None,
        	    meaning the forward simulation is not MPI-parallelized.

        Returns
        -------
        list
            A list of *k* elements, where each element is of type numpy arary and represents the result of a single
            forward simulation.
        """

        raise NotImplementedError


    @abstractmethod
    def get_output_dimension(self):
        """
        Provides the output dimension of the current model.

        This function is in particular important if the current model is used as an input for other models. In such a
        case it is assumed that the output is always a vector of int or float. The length of the vector is the dimension
        that should be returned here.

        Returns
        -------
        int:
            The dimension of the output vector of a single forward simulation.
        """

        raise NotImplementedError


class Continuous(metaclass = ABCMeta):
    """
    This abstract class represents all continuous probabilistic models.
    """

    @abstractmethod
    def pdf(self, input_values, x):
        """
        Calculates the probability density function of the model.

        Parameters
        ----------
        input_values: list
            A list of numbers that are the concatenation of all parent model outputs in the order specified by the
            InputConnector object that was passed during initialization.
        x: float
            The location at which the probability density function should be evaluated.
        """

        raise NotImplementedError


class Discrete(metaclass = ABCMeta):
    """
    This abstract class represents all discrete probabilistic models.
    """

    @abstractmethod
    def pmf(self, input_values, x):
        """
        Calculates the probability mass function of the model.

        Parameters
        ----------
        input_values: list
            A list of numbers that are the concatenation of all parent model outputs in the order specified by the
            InputConnector object that was passed during initialization.
        x: float
            The location at which the probability mass function should be evaluated.
        """

        raise NotImplementedError


class Hyperparameter(ProbabilisticModel):
    """
    This class represents all hyperparameters (i.e. fixed parameters).

    """
    def __init__(self, value, name='Hyperparameter'):
        """

        Parameters
        ----------
        value: list
            The values to which the hyperparameter should be set
        """

        # A hyperparameter is defined by the fact that it does not have any parents
        self.name = name
        self._fixed_values = np.array([value])
        self.visited = False


    def _forward_simulate_and_store_output(self, rng=np.random.RandomState()):
        self.visited = True
        return True


    def _check_input(self, input_values):
        """
        Hyperparameters have no input, thus we only accept None.
        """

        if not isinstance(input_values, list):
            raise TypeError
        if len(input_values) == 0:
            return True
        return False


    def _check_output(self, values):
        return False


    def set_output_values(self, values, rng=np.random.RandomState()):
        if not isinstance(values, np.ndarray):
            raise TypeError('Input not a numpy.array.')
        if values.shape[0] != 1:
            raise IndexError('Dimensions not matching.')
        return False


    def get_input_dimension(self):
        return 0;

    def get_output_dimension(self):
        return 1;


    def get_input_connector(self):
        return None


    def get_input_models(self):
        return []


    def get_input_values(self):
        return []


    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        return [np.array(self._fixed_values) for _ in range(k)]


    def pdf(self, input_values, x):
        # Mathematically, the expression for the pdf of a hyperparameter should be: if(x==self.fixed_parameters) return
        # 1; else return 0; However, since the pdf is called recursively for the whole model structure, and pdfs
        # multiply, this would mean that all pdfs become 0. Setting the return value to 1 ensures proper calulation of
        # the overall pdf.
        return 1.


class ModelResultingFromOperation(ProbabilisticModel):
    """This class implements probabilistic models returned after performing an operation on two probabilistic models
        """

    def __init__(self, parameters, name=''):
        """
        Parameters
        ----------
        parameters: list
            List containing two probabilistic models that should be added together.
        """

        if len(parameters) != 2:
            raise TypeError('Input list does not contain two models.')

        # here, parameters contains exactly two elements
        model_output_dim = [0, 0]
        for i, model in enumerate(parameters):
            if isinstance(model, ProbabilisticModel):
                model_output_dim[i] = model.get_output_dimension()
            elif isinstance(model, Number):
                model_output_dim[i] = 1
            else:
                raise TypeError('Unsupported type.')

        # here, model_output_dim contains the dim of both input models
        if model_output_dim[0] != model_output_dim[1]:
            raise ValueError('The provided models are not of equal dimension.')

        self._dimension = 1
        input_parameters = InputConnector.from_list(parameters)
        super(ModelResultingFromOperation, self).__init__(input_parameters, name)


    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        raise NotImplementedError


    def _check_input(self, input_values):
        return True


    def _check_output(self, parameters):
        """Checks parameters while setting them. Provided due to inheritance."""
        return True


    def get_output_dimension(self):
        return self._dimension


    def pdf(self, input_values, x):
        """Calculates the probability density function at point x.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        x: float or list
            The point at which the pdf should be evaluated.

        Returns
        -------
        float
            The probability density function evaluated at point x.
        """
        # Since the nodes provided as input have to be independent, the resulting pdf will be pdf(parent 1)*pfd(parent 2). During the recursive graph action, this is calculated automatically, so the pdf at this node is expected to be 1
        return 1.

    def sample_from_input_models(self, k, rng=np.random.RandomState()):
        """
        Return for each input model k samples.

        Parameters
        ----------
        k: int
            Specifies the number of samples to generate from each input model.

        Returns
        -------
        dict
            A dictionary of type ProbabilisticModel:[], where the list contains k samples of the corresponding model.
        """

        model_samples = {}

        # Store the visited state of all input models
        visited_state = [False] * self.get_input_dimension()
        for i in range(0, self.get_input_dimension()):
            visited_state[i] = self.get_input_connector().get_model(i).visited

        # Set visited flag of all input models to False
        for i in range(0, self.get_input_dimension()):
            self.get_input_connector().get_model(i).visited = False

        # forward simulate each input model to get fixed input for the current model
        for i in range(0, self.get_input_dimension()):
            model = self.get_input_connector().get_model(i)
            if not model.visited:
                model_has_valid_parameters = model._check_input(model.get_input_values())
                if model_has_valid_parameters:
                    model_samples[model] = model.forward_simulate(model.get_input_values(), k, rng=rng)
                    model.visited = True
                else:
                    raise ValueError('Model %s has invalid input parameters.' % parent.name)

        # Restore the visited state of all input models
        for i in range(0, self.get_input_dimension()):
            self.get_input_connector().get_model(i).visited = visited_state[i]

        return model_samples


class SummationModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from an addition of two probabilistic models"""

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """Adds the sampled values of both parent distributions.

        Parameters
        ----------
        input_values: list
            List of input values
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.
        mpi_comm: MPI communicator object
            Defines the MPI communicator object for MPI parallelization. The default value is None,
            meaning the forward simulation is not MPI-parallelized.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the
            sum of the parents values.
        """
        return_value = []

        # we need to obtain new samples of the parents for each sample (if we just use get_input_values, we will
        # have k identical samples)
        model_samples = self.sample_from_input_models(k, rng)

        for i in range(k):
            parameter_values = [0 for i in range(self.get_input_dimension())]
            for j in range(0, self.get_input_dimension()):
                model = self.get_input_connector().get_model(j)
                parameter_values[j] = model_samples[model][i]

            # add the corresponding parameter_values
            sample_value = []
            for j in range(self.get_output_dimension()):
                sample_value.append(parameter_values[j]+parameter_values[j+self.get_output_dimension()])
            if(len(sample_value)==1):
                sample_value=sample_value[0]
            return_value.append(sample_value)

        return return_value


class SubtractionModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from an subtraction of two probabilistic models"""

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """Adds the sampled values of both parent distributions.

        Parameters
        ----------
        input_values: list
            List of input values
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.
        mpi_comm: MPI communicator object
            Defines the MPI communicator object for MPI parallelization. The default value is None,
            meaning the forward simulation is not MPI-parallelized.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the
            difference of the parents values.
        """
        return_value = []
        sample_value = []

        model_samples = self.sample_from_input_models(k, rng)

        for i in range(k):
            parameter_values = [0 for i in range(self.get_input_dimension())]
            for j in range(0, self.get_input_dimension()):
                model = self.get_input_connector().get_model(j)
                parameter_values[j] = model_samples[model][i]

            # subtract the corresponding parameter_values
            sample_value = []
            for j in range(self.get_output_dimension()):
                sample_value.append(parameter_values[j] - parameter_values[j + self.get_output_dimension()])
            if(len(sample_value)==1):
                sample_value=sample_value[0]
            return_value.append(sample_value)

        return return_value


class MultiplicationModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from a multiplication of two probabilistic models"""
    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """Multiplies the sampled values of both parent distributions element wise.

        Parameters
        ----------
        input_values: list
            List of input values
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.
        mpi_comm: MPI communicator object
            Defines the MPI communicator object for MPI parallelization. The default value is None,
            meaning the forward simulation is not MPI-parallelized.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the product of the parents values.
        """
        return_value = []

        model_samples = self.sample_from_input_models(k, rng)

        for i in range(k):
            parameter_values = [0 for i in range(self.get_input_dimension())]
            for j in range(0, self.get_input_dimension()):
                model = self.get_input_connector().get_model(j)
                parameter_values[j] = model_samples[model][i]

            # multiply the corresponding parameter_values
            sample_value = []

            for j in range(self.get_output_dimension()):
                sample_value.append(parameter_values[j] * parameter_values[j+self.get_output_dimension()])
            if (len(sample_value) == 1):
                sample_value = sample_value[0]
            return_value.append(sample_value)

        return return_value


class DivisionModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from a division of two probabilistic models"""

    def forward_simulate(self, input_valus, k, rng=np.random.RandomState(), mpi_comm=None):
        """Divides the sampled values of both parent distributions.

        Parameters
        ----------
        input_values: list
            List of input values
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.
        mpi_comm: MPI communicator object
            Defines the MPI communicator object for MPI parallelization. The default value is None,
            meaning the forward simulation is not MPI-parallelized.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the fraction of the parents values.
        """
        return_value = []

        model_samples = self.sample_from_input_models(k, rng)

        for i in range(k):
            parameter_values = [0 for i in range(self.get_input_dimension())]
            for j in range(0, self.get_input_dimension()):
                model = self.get_input_connector().get_model(j)
                parameter_values[j] = model_samples[model][i]

            # divide the corresponding parameter_values
            sample_value = []

            for j in range(self.get_output_dimension()):
                sample_value.append(parameter_values[j]/parameter_values[j + self.get_output_dimension()])
            return_value += sample_value

        return return_value


class ExponentialModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from an exponentiation of two probabilistic models"""

    def __init__(self, parameters, name=''):
        """
        Specific initializer for exponential models that does additional checks.

        Parameters
        ----------
        parameters: list
            List of probabilistic models that should be added together.
        """

        exp = parameters[1]
        if isinstance(exp, ProbabilisticModel):
            if exp.get_output_dimension() != 1:
                raise ValueError('The exponent can only be 1 dimensional.')

        super(ExponentialModel, self).__init__(parameters, name)


    def _check_input(self, input_values):
        return True


    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """Raises the sampled values of the base by the exponent.

        Parameters
        ----------
        input_values: list
            List of input values
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.
        mpi_comm: MPI communicator object
            Defines the MPI communicator object for MPI parallelization. The default value is None,
            meaning the forward simulation is not MPI-parallelized.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the exponential of the parents values.
        """
        result = []

        model_samples = self.sample_from_input_models(k, rng)

        for i in range(k):
            parameter_values = [0 for i in range(self.get_input_dimension())]
            for j in range(0, self.get_input_dimension()):
                model = self.get_input_connector().get_model(j)
                parameter_values[j] = model_samples[model][i]

            power = parameter_values[-1]

            sample_value = []

            for j in range(self.get_output_dimension()):
                sample_value.append(parameter_values[j]**power)
            result.append(np.array(sample_value))

        return result


class RExponentialModel(ModelResultingFromOperation):
    """This class represents all probabilistic models resulting from an exponentiation of a Hyperparameter by another probabilistic model."""

    def __init__(self, parameters, name=''):
        """
        Specific initializer for exponential models that does additional checks.

        Parameters
        ----------
        parameters: list
            List of probabilistic models that should be added together.
        """

        exp = parameters[1]
        if isinstance(exp, ProbabilisticModel):
            if exp.get_output_dimension() != 1:
                raise ValueError('The exponent can only be 1 dimensional.')
        super(RExponentialModel, self).__init__(parameters, name)


    def _check_input(self, input_values):
        return True


    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """Raises the base by the sampled value of the exponent.

        Parameters
        ----------
        input_values: list
            List of input values
        k: integer
            The number of samples that should be sampled
        rng: random number generator
            The random number generator to be used.
        mpi_comm: MPI communicator object
            Defines the MPI communicator object for MPI parallelization. The default value is None,
            meaning the forward simulation is not MPI-parallelized.

        Returns
        -------
        list:
            The first entry is True, it is always possible to sample, given two parent values. The second entry is the exponential of the parents values.
        """
        result = []

        model_samples = self.sample_from_input_models(k, rng)

        for i in range(k):
            parameter_values = [0 for i in range(self.get_input_dimension())]
            for j in range(0, self.get_input_dimension()):
                model = self.get_input_connector().get_model(j)
                parameter_values[j] = model_samples[model][i]

            power = parameter_values[0]

            sample_value = []

            for j in range(self.get_output_dimension()):
                sample_value.append(parameter_values[j] ** power)
            result.append(sample_value)

        return [np.array(result)]

