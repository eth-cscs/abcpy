from abcpy.probabilisticmodels import ProbabilisticModel, Discrete, Hyperparameter

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
            :param name:
        """
        super(Bernoulli, self).__init__(parameters, name)


    def _check_parameters_at_initialization(self, parameters):
        """Raises an error if more than one parameters are given, or if the probability is not in the interval (0,1)."""
        if(len(parameters)>1):
            raise IndexError('The probabilistic model of the bernoulli distribution only takes one parameter.')
        if(isinstance(parameters[0][0], Hyperparameter)):
            if(parameters[0][0].fixed_values[0]<=0 or parameters[0][0].fixed_values[0]>=1):
                raise ValueError('The probability has to be in the interval (0,1).')

    def _check_parameters_fixed(self, parameters):
        return True

    def _check_parameters_before_sampling(self, parameters):
        """Returns False iff the probability is not in the interval (0,1)."""
        if(parameters[0]<=0 or parameters[0]>=1):
            return False
        return True

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """Samples from the bernoulli distribution associtated with the probabilistic model.

        Parameters
        ----------
        k: integer
            The number of samples to be drawn.
        rng: random number generator
            The random number generator to be used.
        """
        parameter_values = self.get_parameter_values()
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
        parameter_values = self.get_parameter_values()
        return bernoulli(parameter_values[0]).pmf(x)


class Binomial(Discrete, ProbabilisticModel):
    def __init__(self, parameters, name='Binomial'):
        """
        This class implements a probabilistic model following a binomial distribution.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives. Note that the first entry of the list, n, has to be larger than or equal to 0, while the second entry, p, has to be in the interval [0,1].

        name: string
            The name that should be given to the probabilistic model in the journal file.
            :param name:
        """

        # Rewrite user input
        input_parameters = []
        for parameter in parameters:
            if (isinstance(parameter, list)):
                input_parameters.append(parameter[0])
            else:
                input_parameters.append(parameter)

        super(Binomial, self).__init__(input_parameters, name)


    def _check_parameters_at_initialization(self, parameters):
        """Raises an Error iff:
        - The number of trials is smaller than 0
        - The number of trials is not an integer
        - The success probability is not in [0,1]
        """
        if(len(parameters)==2):
            parameter_1, index_1 = parameters[0]
            parameter_2, index_2 = parameters[1]

            if(isinstance(parameter_1, Hyperparameter)):
                if(parameter_1.fixed_values[0]<0):
                    raise ValueError('The number of trials has to be larger than or equal to 0.')
                if(not(isinstance(parameter_1.fixed_values[0], int))):
                    raise ValueError('The number of trials has to be of type integer.')
            if(isinstance(parameter_2, Hyperparameter)):
                if(parameter_2.fixed_values[0]<0 or parameter_2.fixed_values[0]>1):
                    raise ValueError('The success probability has to be in the interval [0,1]')

    def _check_parameters_before_sampling(self, parameters):
        """
        Returns False iff the first parameter is smaller than 0 or the second parameter is not in the interval [0,1]
        """
        if(parameters[0]<0):
            return False
        if(parameters[1]<0 or parameters[1]>1):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
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
        parameter_values = self.get_parameter_values()
        return_value = []
        return_value.append(self._check_parameters_before_sampling(parameter_values))

        if(return_value[0]):
            n = parameter_values[0]
            p = parameter_values[1]
            return_value.append(rng.binomial(n, p, k))

        return return_value

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
        parameter_values = self.get_parameter_values()

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
            :param name:
        """
        super(Poisson, self).__init__(parameters, name)


    def _check_parameters_at_initialization(self, parameters):
        """Raises an error iff more than one parameter are given or the parameter given is smaller than 0."""
        if(len(parameters)>1):
            raise IndexError('The probabilistic model associated with the poisson distribution only takes 1 parameter as input.')
        if(isinstance(parameters[0][0], Hyperparameter)):
            if(parameters[0][0].fixed_values[0]<=0):
                raise ValueError('The mean of the poisson distribution has to be larger than 0.')
            if(not(isinstance(parameters[0][0].fixed_values[0], int))):
                raise ValueError('The mean has to be of type integer.')

    def _check_parameters_before_sampling(self, parameters):
        """Returns True iff the mean is larger than 0."""
        if(parameters[0]<=0):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """Samples k values from the defined possion distribution.

        Parameters
        ----------
        k: integer
            The number of samples.
        rng: random number generator
            The random number generator to be used.
        """
        parameter_values = self.get_parameter_values()
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
        parameter_values = self.get_parameter_values()
        parameter_values[0] = int(parameter_values[0])
        return poisson(parameter_values[0]).pmf(x)
