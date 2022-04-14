import numpy as np
from scipy.special import gamma
from scipy.stats import multivariate_normal, norm, lognorm, expon

from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True
    from torchtyping import TensorType, patch_typeguard

    patch_typeguard()


class Uniform(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Uniform'):
        """
        This class implements a probabilistic model following an uniform distribution.

        Parameters
        ----------
        parameters: list
            Contains two lists. The first list specifies the probabilistic models and hyperparameters from which the
            lower bound of the uniform distribution derive. The second list specifies the probabilistic models and
            hyperparameters from which the upper bound derives.

        name: string, optional
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for Uniform has to be of type list.')
        if len(parameters) < 2:
            raise ValueError('Input for Uniform has to be of length 2.')
        if not isinstance(parameters[0], list):
            raise TypeError('Each boundary for Uniform has to be of type list.')
        if not isinstance(parameters[1], list):
            raise TypeError('Each boundary for Uniform has to be of type list.')
        if len(parameters[0]) != len(parameters[1]):
            raise ValueError('Length of upper and lower bound have to be equal.')

        self._dimension = len(parameters[0])
        input_parameters = InputConnector.from_list(parameters)
        super(Uniform, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        """
        Checks parameter values sampled from the parents.
        """
        if len(input_values) % 2 != 0:
            return False

        # test whether lower bound is not greater than upper bound
        for j in range(self.get_output_dimension()):
            if input_values[j] > input_values[j + self.get_output_dimension()]:
                return False
        return True

    def _check_output(self, parameters):
        """
        Checks parameter values given as fixed values. Returns False iff a lower bound value is larger than a
        corresponding upper bound value.
        """

        for i in range(self.get_output_dimension()):
            lower_value = self.get_input_connector()[i]
            upper_value = self.get_input_connector()[i + self.get_output_dimension()]
            if parameters[i] < lower_value or parameters[i] > upper_value:
                return False
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a uniform distribution using the current values for each probabilistic model from which the model derives.

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

        samples = np.zeros(shape=(k, self.get_output_dimension()))
        for j in range(0, self.get_output_dimension()):
            samples[:, j] = rng.uniform(input_values[j], input_values[j + self.get_output_dimension()], k)
        return [np.array(x).reshape(-1, ) for x in samples]

    def get_output_dimension(self):
        return self._dimension

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
        Float:
            The evaluated pdf at point x.
        """

        lower_bound = input_values[:self.get_output_dimension()]
        upper_bound = input_values[self.get_output_dimension():]

        if np.product(np.greater_equal(x, np.array(lower_bound)) * np.less_equal(x, np.array(upper_bound))):
            pdf_value = 1. / np.product(np.array(upper_bound) - np.array(lower_bound))
        else:
            pdf_value = 0.
        self.calculated_pdf = pdf_value
        return pdf_value


class Normal(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Normal'):
        """
        This class implements a probabilistic model following a normal distribution with mean mu and variance sigma.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives.
            The list has two entries: from the first entry mean of the distribution and from the second entry variance is derived.
            Note that the second value of the list is strictly greater than 0.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for Normal has to be of type list.')
        if len(parameters) < 2:
            raise ValueError('Input for Normal has to be of length 2.')

        input_parameters = InputConnector.from_list(parameters)
        super(Normal, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        """
        Returns True if the standard deviation is negative.
        """
        if len(input_values) != 2:
            return False

        if input_values[1] <= 0:
            return False
        return True

    def _check_output(self, parameters):
        """
        Checks parameter values that are given as fixed values.
        """
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a normal distribution using the current values for each probabilistic model from which the model derives.

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

        mu = input_values[0]
        sigma = input_values[1]
        result = np.array(rng.normal(0, 1, k) * sigma + mu)
        return [np.array([x]).reshape(-1, ) for x in result]

    def forward_simulate_and_gradient(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        mu = input_values[0]
        sigma = input_values[1]
        z = rng.normal(0, 1, k)
        result = np.array(z * sigma + mu)
        grad = np.ones((k, 2))
        grad[:, 1] = z

        return [np.array([x]).reshape(-1, ) for x in result], [np.array([x]).reshape(1, 2) for x in grad]

    def get_output_dimension(self):
        return 1
        # Why does the following not work here?
        # return self._dimension

    def pdf(self, input_values, x):
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        input_values: list
            List of input parameters of the from [mu, sigma]
        x: list
            The point at which the pdf should be evaluated.

        Returns
        -------
        Float:
            The evaluated pdf at point x.
        """

        mu = input_values[0]
        sigma = input_values[1]
        pdf = norm(mu, sigma).pdf(x)
        self.calculated_pdf = pdf
        return pdf


class StudentT(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='StudentT'):
        """
        This class implements a probabilistic model following the Student's T-distribution.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives.
            The list has two entries: from the first entry mean of the distribution and from the second entry degrees of freedom is derived.
            Note that the second value of the list is strictly greater than 0.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for StudentT has to be of type list.')
        if len(parameters) < 2:
            raise ValueError('Input for StudentT has to be of length 2.')

        input_parameters = InputConnector.from_list(parameters)
        super(StudentT, self).__init__(input_parameters, name)
        self.visited = False

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a Student's T-distribution using the current values for each probabilistic model from which the model derives.

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

        mean = input_values[0]
        df = input_values[1]
        result = np.array((rng.standard_t(df, k) + mean))
        return [np.array([x]).reshape(-1, ) for x in result]

    def _check_input(self, input_values):
        """
        Checks parameter values sampled from the parents of the probabilistic model. Returns False iff the degrees of freedom are smaller than or equal to 0.
        """
        if len(input_values) != 2:
            return False

        if input_values[1] <= 0:
            return False

        return True

    def _check_output(self, parameters):
        """
        Checks parameter values given as fixed values.
        """
        return True

    def get_output_dimension(self):
        return 1
        # Why does the following not work here?
        # return self._dimension

    def pdf(self, input_values, x):
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        input_values: list
            List of input parameters
        x: list
            The point at which the pdf should be evaluated.

        Returns
        -------
        Float:
            The evaluated pdf at point x.
        """

        df = input_values[1]
        x -= input_values[0]  # divide by std dev if we include that
        pdf = gamma((df + 1) / 2) / (np.sqrt(df * np.pi) * gamma(df / 2) * (1 + x ** 2 / df) ** ((df + 1) / 2))
        self.calculated_pdf = pdf
        return pdf


class MultivariateNormal(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Multivariate Normal'):
        """
        This class implements a probabilistic model following a multivariate normal distribution with mean and
        covariance matrix.

        Parameters
        ----------
        parameters: list of at length 2
            Contains the probabilistic models and hyperparameters from which the model derives. The first entry defines
            the mean, while the second entry defines the Covariance matrix. Note that if the mean is n dimensional, the
            covariance matrix is required to be of dimension nxn, symmetric and
            positive-definite.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        # convert user input to InputConnector object

        if not isinstance(parameters, list):
            raise TypeError('Input for Multivariate Normal has to be of type list.')
        if len(parameters) < 2:
            raise ValueError('Input for Multivariate Normal has to be of length 2.')

        mean = parameters[0]
        if isinstance(mean, list):
            self._dimension = len(mean)
        elif isinstance(mean, ProbabilisticModel):
            self._dimension = mean.get_output_dimension()
        input_parameters = InputConnector.from_list(parameters)

        super(MultivariateNormal, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        """
        Checks parameter values sampled from the parents at initialization. Returns False iff the covariance matrix is
        not symmetric or not positive definite.
        """
        # Test whether input in compatible
        dim = self._dimension
        param_ctn = len(input_values)
        if param_ctn != dim + dim ** 2:
            return False

        cov = np.array(input_values[dim:dim + dim ** 2]).reshape((dim, dim))

        # Check whether the covariance matrix is symmetric
        if not np.allclose(cov, cov.T, atol=1e-3):
            return False

        # Check whether the covariance matrix is positive definite
        try:
            is_pos = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False

        return True

    def _check_output(self, parameters):
        """
        Checks parameter values that are given as fixed values.
        """

        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a multivariate normal distribution using the current values for each probabilistic model from which the
        model derives.

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

        dim = self.get_output_dimension()
        mean = np.array(input_values[0:dim])
        cov = np.array(input_values[dim:dim + dim ** 2]).reshape((dim, dim))
        result = rng.multivariate_normal(mean, cov, k)
        return [np.array([result[i, :]]).reshape(-1, ) for i in range(k)]

    def get_output_dimension(self):
        return self._dimension

    def pdf(self, input_values, x):
        """
        Calculates the probability density function at point x. Commonly used to determine whether perturbed parameters
        are still valid according to the pdf.

        Parameters
        ----------
        input_values: list
            List of input parameters
        x: list
           The point at which the pdf should be evaluated.

        Returns
        -------
        Float:
            The evaluated pdf at point x.
        """

        dim = self._dimension
        # Extract parameters
        mean = np.array(input_values[0:dim])
        cov = np.array(input_values[dim:dim + dim ** 2]).reshape((dim, dim))

        pdf = multivariate_normal(mean, cov).pdf(x)
        self.calculated_pdf = pdf
        return pdf


class MultiStudentT(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='MultiStudentT'):
        """
        This class implements a probabilistic model following the multivariate Student-T distribution.

        Parameters
        ----------
        parameters: list
            All but the last two entries contain the probabilistic models and hyperparameters from which the model
            derives. The second to last entry contains the covariance matrix. If the mean is of dimension n, the
            covariance matrix is required to be nxn dimensional. The last entry contains the degrees of freedom.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for Multivariate StudentT has to be of type list.')
        if len(parameters) < 3:
            raise ValueError('Input for Multivariate Student T has to be of length 3.')
        if not isinstance(parameters[0], list):
            raise TypeError('Input for mean of Multivariate Student T has to be of type list.')
        if not isinstance(parameters[1], list):
            raise TypeError('Input for covariance of Multivariate Student T has to be of type list.')

        mean = parameters[0]
        if isinstance(mean, list):
            self._dimension = len(mean)
            input_parameters = InputConnector.from_list(parameters)
        elif isinstance(mean, ProbabilisticModel):
            self._dimension = mean.get_output_dimension()
            input_parameters = parameters

        super(MultiStudentT, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        """
        Returns False iff the degrees of freedom are less than or equal to 0, the covariance matrix is not symmetric or
        the covariance matrix is not positive definite.
        """

        dim = self._dimension
        param_ctn = len(input_values)
        if param_ctn > dim + dim ** 2 + 1 or param_ctn < dim + dim ** 2 + 1:
            return False

        # Extract parameters
        mean = np.array(input_values[0:dim])
        cov = np.array(input_values[dim:dim + dim ** 2]).reshape((dim, dim))
        df = input_values[-1]

        # Check whether the covariance matrix is symmetric
        if not np.allclose(cov, cov.T, atol=1e-3):
            return False

        # Check whether the covariance matrix is positive definite
        try:
            is_pos = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False

        # Check whether the degrees of freedom are <=0
        if df <= 0:
            return False

        return True

    def _check_output(self, parameters):
        """
        Checks parameter values given as fixed values.
        """
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a multivariate Student's T-distribution using the current values for each probabilistic model from
        which the model derives.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the
            generator.

        Returns
        -------
        list: [np.ndarray]
            A list containing the sampled values as np-array.
        """

        # Extract input_parameters
        dim = self.get_output_dimension()
        mean = np.array(input_values[0:dim])
        cov = np.array(input_values[dim:dim + dim ** 2]).reshape((dim, dim))
        df = input_values[-1]

        if df == np.inf:
            chisq = 1.0
        else:
            chisq = rng.chisquare(df, k) / df
            chisq = chisq.reshape(-1, 1).repeat(dim, axis=1)
        mvn = rng.multivariate_normal(np.zeros(dim), cov, k)
        result = (mean + np.divide(mvn, np.sqrt(chisq)))
        return [np.array([result[i, :]]).reshape(-1, ) for i in range(k)]

    def get_output_dimension(self):
        return self._dimension

    def pdf(self, input_values, x):
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        input_values: list
            List of input parameters
        x: list
           The point at which the pdf should be evaluated.

        Returns
        -------
        Float:
            The evaluated pdf at point x.
        """

        dim = self.get_output_dimension()

        # Extract parameters
        mean = np.array(input_values[0:dim])
        cov = np.array(input_values[dim:dim + dim ** 2]).reshape((dim, dim))
        df = input_values[-1]

        p = len(mean)
        numerator = gamma((df + p) / 2)
        denominator = gamma(df / 2) * pow(df * np.pi, p / 2.) * np.sqrt(abs(np.linalg.det(cov)))
        normalizing_const = numerator / denominator
        tmp = 1 + 1 / df * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))
        density = normalizing_const * pow(tmp, -((df + p) / 2.))
        self.calculated_pdf = density
        return density


class LogNormal(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='LogNormal'):
        """
        This class implements a probabilistic model following a Lognormal distribution with mean mu and variance sigma.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives.
            The list has two entries: from the first entry mean of the underlying normal distribution and from the second entry variance of the underlying normal
            distribution is derived.
            Note that the second value of the list is strictly greater than 0.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for LogNormal has to be of type list.')
        if len(parameters) < 2:
            raise ValueError('Input for LogNormal has to be of length 2.')

        input_parameters = InputConnector.from_list(parameters)
        super(LogNormal, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        """
        Returns True if the standard deviation is negative.
        """
        if len(input_values) != 2:
            return False

        if input_values[1] <= 0:
            return False
        return True

    def _check_output(self, parameters):
        """
        Checks parameter values that are given as fixed values.
        """
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a normal distribution using the current values for each probabilistic model from which the model derives.

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

        mu = input_values[0]
        sigma = input_values[1]
        result = np.array(rng.lognormal(mu, sigma, k))
        return [np.array([x]).reshape(-1, ) for x in result]

    def get_output_dimension(self):
        return 1
        # Why does the following not work here?
        # return self._dimension

    def pdf(self, input_values, x):
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        input_values: list
            List of input parameters of the from [mu, sigma]
        x: list
            The point at which the pdf should be evaluated.

        Returns
        -------
        Float:
            The evaluated pdf at point x.
        """

        mu = input_values[0]
        sigma = input_values[1]
        pdf = lognorm(scale=np.exp(mu), s=sigma).pdf(x)
        self.calculated_pdf = pdf
        return pdf


class Exponential(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Exponential'):
        """
        This class implements a probabilistic model following a normal distribution with mean mu and variance sigma.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives.
            The list has one entry: the rate :math:`\lambda` of the exponential distribution, that has therefore pdf:
            :math:`f(x; \lambda) = \lambda \exp(-\lambda x )`

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError('Input for Exponential has to be of type list.')
        if len(parameters) != 1:
            raise ValueError('Input for Exponential has to be of length 1.')

        input_parameters = InputConnector.from_list(parameters)
        super(Exponential, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        """
        Returns True if the standard deviation is negative.
        """
        if len(input_values) != 1:
            return False

        if input_values[0] <= 0:
            return False
        return True

    def _check_output(self, parameters):
        """
        Checks parameter values that are given as fixed values.
        """
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a normal distribution using the current values for each probabilistic model from which the model derives.

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

        rate = input_values[0]
        scale = 1 / rate
        result = np.array(rng.exponential(scale, k))
        return [np.array([x]).reshape(-1, ) for x in result]

    def get_output_dimension(self):
        return 1
        # Why does the following not work here?
        # return self._dimension

    def pdf(self, input_values, x):
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        input_values: list
            List of input parameters of the from [rate]
        x: list
            The point at which the pdf should be evaluated.

        Returns
        -------
        Float:
            The evaluated pdf at point x.
        """

        rate = input_values[0]
        scale = 1 / rate
        pdf = expon(scale=scale).pdf(x)
        self.calculated_pdf = pdf
        return pdf


class GenerativeNeuralNetworkWrap(ProbabilisticModel, Continuous):

    def __init__(self, parameters, net, name='GenerativeNeuralNetworkWrap'):
        """
        This class wraps a generative neural network for use in the ABCpy API.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives.
        net : torch.nn object
            the generative neural network. Its forward method should take two inputs: the statistical parameters (theta)
            and an argument "n_simulations" that defines the number of simulations. It needs to generate the
            noise inside the `net` itself. Reproducibility is preserved by fixing the torch random seed before calling
            the net forward method.
        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not has_torch:
            raise ImportError(
                "Pytorch is required to instantiate an element of the {} class, in order to handle "
                "neural networks. Please install it. ".format(self.__class__.__name__))

        if not isinstance(parameters, list):
            raise TypeError('Input for GenerativeNeuralNetworkWrap has to be of type list.')
        self.net = net

        input_parameters = InputConnector.from_list(parameters)
        super(GenerativeNeuralNetworkWrap, self).__init__(input_parameters, name)

    def _check_input(self, input_values):
        """
        Returns True. Not very useful for the moment.
        """
        return True

    def _check_output(self, parameters):
        """
        Useless for the moment.
        """
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a generative neural network using the provided parameter values.

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

        # need to transform input_values to a torch tensor
        input_values = torch.atleast_2d(torch.from_numpy(np.array(input_values).astype(np.float32)))

        # set torch seed using the rng by drawing an integer from the rng
        torch.random.manual_seed(rng.randint(0, 2 ** 32))

        # forward pass
        result = self.net(input_values, n_simulations=k)[0].detach().numpy()

        return [x for x in result]

    def forward_simulate_and_gradient(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a generative neural network using the provided parameter values and computes the gradient of the
        output with respect to the input parameters.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the
            generator.

        Returns
        -------
        list: [np.ndarray]
            A list containing the sampled values as np-array.
        """

        # need to transform input_values to a torch tensor
        input_values = torch.atleast_2d(torch.from_numpy(np.array(input_values).astype(np.float32)))
        input_values.requires_grad = True

        # set torch seed using the rng by drawing an integer from the rng
        torch.random.manual_seed(rng.randint(0, 2 ** 32))

        # forward pass
        result = self.net(input_values, n_simulations=k)[0]

        # compute gradient of result with respect to input_values
        # result is of shape ["number_generations", "output_size"]
        # input_values is of shape [1, "num_parameters"]
        grad = self.jacobian(input_values, result)

        result = result.detach().numpy()

        return [x for x in result], [x for x in grad]

    def get_output_dimension(self):
        return 1  # this is not correct; it should check the output network size.

    @staticmethod
    def jacobian(input: TensorType[1, "num_parameters"], output: TensorType["number_generations", "output_size"]) -> \
            TensorType["number_generations", "output_size", "num_parameters"]:
        '''
        Returns the Jacobian matrix of the function that produced the output evaluated at the input

        This function is quite inefficient but cannot think of a better way to do it.
        '''
        J = torch.zeros((output.shape[0], output.shape[1], input.shape[1]))

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                J[i, j] = torch.autograd.grad(output[i, j], input, retain_graph=True)[0][0]

        return J
