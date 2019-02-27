from abcpy.probabilisticmodels import ProbabilisticModel, Discrete, Hyperparameter, InputConnector

import numpy as np
from scipy.special import comb
from scipy.stats import poisson, bernoulli


class Bernoulli(Discrete, ProbabilisticModel):
    def __init__(self, parameters, name='Bernoulli'):
        """This class implements a probabilistic model following a bernoulli distribution.

        Parameters
        ----------
        parameters: list
             A list containing one entry, the probability of the distribution.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for Bernoulli has to be of type list.')
        if len(parameters)!=1:
            raise ValueError('Input for Bernoulli has to be of length 1.')

        self._dimension = len(parameters)
        input_parameters = InputConnector.from_list(parameters)
        super(Bernoulli, self).__init__(input_parameters, name)
        self.visited = False


    def _check_input(self, input_values):
        """
        Checks parameter values sampled from the parents.
        """
        if len(input_values) > 1:
            return False

        # test whether probability is in the interval [0,1]
        if input_values[0]<0 or input_values[0]>1:
           return False

        return True


    def _check_output(self, parameters):
        """
        Checks parameter values given as fixed values. Returns False iff it is not an integer
        """
        if not isinstance(parameters[0], (int, np.int32, np.int64)):
            return False
        return True


    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from the bernoulli distribution associtated with the probabilistic model.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        k: integer
            The number of samples to be drawn.
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list: [np.ndarray]
            A list containing the sampled values as np-array.
        """

        result = np.array(rng.binomial(1, input_values[0], k))
        return [np.array([x]) for x in result]


    def get_output_dimension(self):
        return self._dimension


    def pmf(self, input_values, x):
        """Evaluates the probability mass function at point x.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        x: float
            The point at which the pmf should be evaluated.

        Returns
        -------
        float:
            The pmf evaluated at point x.
        """
        probability = input_values[0]
        pmf = bernoulli(probability).pmf(x)
        self.calculated_pmf = pmf
        return pmf


class Binomial(Discrete, ProbabilisticModel):
    def __init__(self, parameters, name='Binomial'):
        """
        This class implements a probabilistic model following a binomial distribution.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives. Note that the first
            entry of the list, n, an integer and has to be larger than or equal to 0, while the second entry, p, has to be in the
            interval [0,1].

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for Binomial has to be of type list.')
        if len(parameters)!=2:
            raise ValueError('Input for Binomial has to be of length 2.')

        self._dimension = 1
        input_parameters = InputConnector.from_list(parameters)
        super(Binomial, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        """Raises an Error iff:
        - The number of trials is less than 0
        - The number of trials is not an integer
        - The success probability is not in [0,1]
        """

        if len(input_values) != 2:
            raise TypeError('Number of input parameters is exactly 2.')

        # test whether number of trial is an integer
        if not isinstance(input_values[0], (int, np.int32, np.int64)):
            raise TypeError('Input parameter for number of trials has to be an integer.')

        # test whether probability is in the interval [0,1]
        if input_values[1] < 0 or input_values[1] > 1:
            return False

        # test whether number of trial less than 0
        if input_values[0] < 0:
            return False

        return True


    def _check_output(self, parameters):
        if not isinstance(parameters[0], (int, np.int32, np.int64)):
            return False
        return True


    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a binomial distribution using the current values for each probabilistic model from which the model derives.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the generator.

        Returns
        -------
        list: [np.ndarray]
            A list containing the sampled values as np-array.
        """

        result = rng.binomial(input_values[0], input_values[1], k)
        return [np.array([x]) for x in result]


    def get_output_dimension(self):
        return self._dimension


    def pmf(self, input_values, x):
        """
        Calculates the probability mass function at point x.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        x: list
            The point at which the pmf should be evaluated.

        Returns
        -------
        Float
            The evaluated pmf at point x.
        """

        # If the provided point is not an integer, it is converted to one
        x = int(x)
        n = input_values[0]
        p = input_values[1]
        if(x>n):
            pmf = 0
        else:
            pmf = comb(n,x)*pow(p,x)*pow((1-p),(n-x))
        self.calculated_pmf = pmf
        return pmf


class Poisson(Discrete, ProbabilisticModel):
    def __init__(self, parameters, name='Poisson'):
        """This class implements a probabilistic model following a poisson distribution.

        Parameters
        ----------
        parameters: list
            A list containing one entry, the mean of the distribution.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for Poisson has to be of type list.')
        if len(parameters)!=1:
            raise ValueError('Input for Poisson has to be of length 1.')

        self._dimension = 1
        input_parameters = InputConnector.from_list(parameters)
        super(Poisson, self).__init__(input_parameters, name)
        self.visited = False


    def _check_input(self, input_values):
        """Raises an error iff more than one parameter are given or the parameter given is smaller than 0."""

        if len(input_values) > 1:
            return False

        # test whether the parameter is smaller than 0
        if input_values[0]<0:
           return False

        return True


    def _check_output(self, parameters):
        if not isinstance(parameters[0], (int, np.int32, np.int64)):
            return False
        return True


    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples k values from the defined possion distribution.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        k: integer
            The number of samples.
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list: [np.ndarray]
            A list containing the sampled values as np-array.


        """

        result = rng.poisson(int(input_values[0]), k)
        return [np.array([x]) for x in result]


    def get_output_dimension(self):
        return self._dimension


    def pmf(self, input_values, x):
        """Calculates the probability mass function of the distribution at point x.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        x: integer
            The point at which the pmf should be evaluated.

        Returns
        -------
        Float
            The evaluated pmf at point x.
        """

        pmf = poisson(int(input_values[0])).pmf(x)
        self.calculated_pmf = pmf
        return pmf



class DiscreteUniform(Discrete, ProbabilisticModel):
    def __init__(self, parameters, name='DiscreteUniform'):
        """This class implements a probabilistic model following a Discrete Uniform distribution.

        Parameters
        ----------
        parameters: list
             A list containing two entries, the upper and lower bound of the range.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for Discrete Uniform has to be of type list.')
        if len(parameters) != 2:
            raise ValueError('Input for Discrete Uniform has to be of length 2.')

        self._dimension = 1
        input_parameters = InputConnector.from_list(parameters)
        super(DiscreteUniform, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 2:
            raise ValueError('Number of parameters of FloorField model must be 2.')

        # Check whether input is from correct domain
        lowerbound = input_values[0]  # Lower bound
        upperbound = input_values[1]  # Upper bound

        if not isinstance(lowerbound, (int, np.int64, np.int32, np.int16)) or not isinstance(upperbound, (int, np.int64, np.int32, np.int16)) or lowerbound >= upperbound:
            return False
        return True

    def _check_output(self, parameters):
        """
        Checks parameter values given as fixed values. Returns False iff it is not an integer
        """
        if not isinstance(parameters[0], (int, np.int32, np.int64)):
            return False
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        """
        Samples from the Discrete Uniform distribution associated with the probabilistic model.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        k: integer
            The number of samples to be drawn.
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list: [np.ndarray]
            A list containing the sampled values as np-array.
        """
        result = np.array(rng.randint(input_values[0], input_values[1]+1, size=k, dtype=np.int64))
        return [np.array([x]).reshape(-1,) for x in result]

    def get_output_dimension(self):
        return self._dimension

    def pmf(self, input_values, x):
        """Evaluates the probability mass function at point x.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        x: float
            The point at which the pmf should be evaluated.

        Returns
        -------
        float:
            The pmf evaluated at point x.
        """
        lowerbound, upperbound = input_values[0], input_values[1]
        if x >= lowerbound and x <= upperbound:
            pmf = 1. / (upperbound - lowerbound + 1)
        else:
            pmf = 0
        self.calculated_pmf = pmf
        return pmf

