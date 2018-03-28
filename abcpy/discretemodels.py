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
            raise TypeError('Input for Uniform has to be of type list.')
        if len(parameters)!=1:
            raise ValueError('Input for Uniform has to be of length 2.')

        self._dimension = len(parameters)
        input_parameters = InputConnector.from_list(parameters)
        super(Bernoulli, self).__init__(input_parameters, name)
        self.visited = False


    def _check_input(self, input_connector):
        """
        Checks parameter values sampled from the parents.
        """
        if(input_connector.get_parameter_count() > 1):
            return False

        # test whether lower bound is not greater than upper bound
        if input_connector[0]<0 or input_connector[0]>1:
           return False

        return True


    def _check_output(self, parameters):
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
        return_values = []
        return_values.append(self._check_parameters_before_sampling(parameter_values))
        if(return_values[0]):
            return_values.append(rng.binomial(1, parameter_values[0], k))
        return return_values


    def get_output_dimension(self):
        return 1


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
        return bernoulli(parameter_values[0]).pmf(x)


class Binomial(Discrete, ProbabilisticModel):
    def __init__(self, parameters, name='Binomial'):
        """
        This class implements a probabilistic model following a binomial distribution.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives. Note that the first
            entry of the list, n, has to be larger than or equal to 0, while the second entry, p, has to be in the
            interval [0,1].

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        # Rewrite user input
        input_parameters = InputConnector.from_list(parameters)
        super(Binomial, self).__init__(input_parameters, name)


    def _check_input(self, parameters):
        """Raises an Error iff:
        - The number of trials is smaller than 0
        - The number of trials is not an integer
        - The success probability is not in [0,1]
        """

        if(parameters.get_parameter_count() == 2):
            if isinstance(parameters.get_model(0), Hyperparameter):
                if not isinstance(parameters[0], int):
                    raise ValueError('The number of trials has to be of type integer.')
        else:
            raise ValueError('There have to be exactly two input parameters.')

        if parameters[0] < 0:
            return False

        if parameters[1] < 0 or parameters[1] > 1:
            return False
        return True


    def _check_output(self, parameters):
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
        n = parameter_values[0]
        p = parameter_values[1]
        result = rng.binomial(n, p, k)
        return [np.array([x]) for x in result]


    def get_output_dimension(self):
        return 1


    def pmf(self, x):
        """
        Calculates the probability mass function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pmf.

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
            return 0
        return comb(n,x)*(p**x)*(1-p)**(n-x)


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
        # TODO: No tests for Poisson distribution
        super(Poisson, self).__init__(parameters, name)


    def _check_input(self, parameters):
        """Raises an error iff more than one parameter are given or the parameter given is smaller than 0."""
        if(len(parameters)>1):
            raise IndexError('The probabilistic model associated with the poisson distribution only takes 1 parameter as input.')
        if(isinstance(parameters[0][0], Hyperparameter)):
            if(parameters[0][0].fixed_values[0]<=0):
                raise ValueError('The mean of the poisson distribution has to be larger than 0.')
            if(not(isinstance(parameters[0][0].fixed_values[0], int))):
                raise ValueError('The mean has to be of type integer.')


    def _check_output(self, parameters):
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
        parameter_values[0] = int(parameter_values[0])
        return_values = []
        return_values.append(self._check_parameters_before_sampling(parameter_values))

        if(return_values[0]):
            return_values.append(rng.poisson(parameter_values[0], k))

        return return_values


    def get_output_dimension(self):
        return 1


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
        parameter_values[0] = int(parameter_values[0])
        return poisson(parameter_values[0]).pmf(x)
