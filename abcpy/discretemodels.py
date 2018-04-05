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


    def forward_simulate(self, k, rng=np.random.RandomState()):
        """Samples from the bernoulli distribution associtated with the probabilistic model.

        Parameters
        ----------
        k: integer
            The number of samples to be drawn.
        rng: random number generator
            The random number generator to be used.
        """
        parameter_values = self.get_input_values()
        result = np.array(rng.binomial(1, parameter_values[0], k))
        return [np.array([x]) for x in result]


    def get_output_dimension(self):
        return self._dimension


    def pmf(self, x):
        """Evaluates the probability mass function at point x.

        Parameters
        ----------
        x: float
            The point at which the pmf should be evaluated.

        Returns
        -------
        float:
            The pmf evaluated at point x.
        """
        parameter_values = self.get_input_values()
        probability = parameter_values[0]
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


    def forward_simulate(self, k, rng=np.random.RandomState()):
        """
        Samples from a binomial distribution using the current values for each probabilistic model from which the model derives.

        Parameters
        ----------
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the                  generator.

        Returns
        -------
        list: [boolean, np.ndarray]
            A list containing whether it was possible to sample values from the distribution and if so, the sampled values.
        """

        parameter_values = self.get_input_values()
        result = rng.binomial(parameter_values[0], parameter_values[1], k)
        return [np.array([x]) for x in result]

    def get_output_dimension(self):
        return self._dimension


    def pmf(self, x):
        """
        Calculates the probability mass function at point x.

        Parameters
        ----------
        x: list
            The point at which the pmf should be evaluated.
        """
        parameter_values = self.get_input_values()

        # If the provided point is not an integer, it is converted to one
        x = int(x)
        n = parameter_values[0]
        p = parameter_values[1]
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


    def forward_simulate(self, k, rng=np.random.RandomState()):
        """Samples k values from the defined possion distribution.

        Parameters
        ----------
        k: integer
            The number of samples.
        rng: random number generator
            The random number generator to be used.
        """

        parameter_values = self.get_input_values()

        result = rng.poisson(int(parameter_values[0]), k)

        return [np.array([x]) for x in result]

    def get_output_dimension(self):
        return self._dimension


    def pmf(self, x):
        """Calculates the probability mass function of the distribution at point x.

        Parameters
        ----------
        x: integer
            The point at which the pmf should be evaluated.

        Returns
        -------
        Float
            The evaluated pmf at point x.
        """
        parameter_values = self.get_input_values()

        pmf = poisson(int(parameter_values[0])).pmf(x)

        return pmf
