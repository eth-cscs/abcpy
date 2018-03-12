from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, Hyperparameter, InputParameters
import numpy as np

from numbers import Number
from scipy.stats import multivariate_normal, norm
from scipy.special import gamma



class Normal(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Normal'):
        """
        This class implements a probabilistic model following a normal distribution with mean mu and variance sigma.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives. Note that the second value of the list is not allowed to be smaller than 0.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if isinstance(parameters, list):
            input_parameters = InputParameters.from_list(parameters)
            super(Normal, self).__init__(input_parameters, name)
        else:
            raise TypeError('Input type not supported')



    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from a normal distribution using the current values for each probabilistic model from which the model derives.

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
        mu = parameter_values[0]
        sigma = parameter_values[1]
        return np.array(rng.normal(mu, sigma, k)).reshape((k)).tolist()


    def _check_parameters(self, parameters):
        """
        Returns True if the standard deviation is negative.
        """
        if parameters.get_parameter_count() != 2:
            raise ValueError('Number of parameters is not two.')

        if(parameters[1] < 0):
            return False
        return True


    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values that are given as fixed values.
        """
        return True


    def get_output_dimension(self):
        return 1


    def pdf(self, x):
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        x: list
            The point at which the pdf should be evaluated.
        """
        parameter_values = self.get_parameter_values()
        mu = parameter_values[0]
        sigma = parameter_values[1]
        pdf = norm(mu,sigma).pdf(x)
        self.calculated_pdf = pdf
        return pdf



class MultivariateNormal(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Multivariate Normal'):
        """
        This class implements a probabilistic model following a multivariate normal distribution with mean and
        covariance matrix.

        Parameters
        ----------
        parameters: list of at least length 2
            Contains the probabilistic models and hyperparameters from which the model derives. The last entry defines
            the covariance matrix, while all other entries define the mean. Note that if the mean is n dimensional, the
            covariance matrix is required to be of dimension nxn. The covariance matrix is required to be
            positive-definite.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        # convert user input to InputParameters object
        mean = parameters[0]
        if isinstance(mean, list):
            self._dimension = len(mean)
            input_parameters = InputParameters.from_list(parameters)
        elif isinstance(mean, ProbabilisticModel):
            self._dimension = mean.get_output_dimension()
            input_parameters = parameters

        super(MultivariateNormal, self).__init__(input_parameters, name)


    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from a multivariate normal distribution using the current values for each probabilistic model from which the
        model derives.

        Parameters
        ----------
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the generator.

        Returns
        -------
        list: [boolean, np.ndarray]
            A list containing whether it was possible to sample values from the distribution and if so, the sampled values.
        """

        dim = self._dimension
        parameter_values = self.get_parameter_values()
        mean = np.array(parameter_values[0:dim])
        cov = np.array(parameter_values[dim:dim+dim**2]).reshape((dim, dim))
        return rng.multivariate_normal(mean, cov, k).reshape(k,-1).tolist()


    def _check_parameters(self, parameters):
        """
        Checks parameter values sampled from the parents at initialization. Returns False iff the covariance matrix is
        not symmetric or not positive definite.
        """
        # Test whether input in compatible
        dim = self._dimension
        param_ctn = parameters.get_parameter_count()
        if param_ctn != dim+dim**2:
            return False

        cov = np.array(parameters[dim:dim+dim**2]).reshape((dim,dim))

        # Check whether the covariance matrix is symmetric
        if not np.allclose(cov, cov.T, atol=1e-3):
            return False

        # Check whether the covariance matrix is positive definite
        try:
            is_pos = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False

        return True


    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values that are given as fixed values.
        """

        return True


    def get_output_dimension(self):
        return self._dimension


    def pdf(self, x):
        """
        Calculates the probability density function at point x. Commonly used to determine whether perturbed parameters
        are still valid according to the pdf.

        Parameters
        ----------
        x: list
           The point at which the pdf should be evaluated.
       """
        parameter_values = self.get_parameter_values()
        mean= parameter_values[:-1]
        cov = parameter_values[-1]
        pdf = multivariate_normal(mean, cov).pdf(x)
        self.calculated_pdf = pdf
        return pdf


class MixtureNormal(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Mixture Normal'):
        """
        This class implements a probabilistic model following a mixture normal distribution.

        Parameters
        ----------
        parameters: list
            Contains all the probabilistic models and hyperparameters from which the model derives.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """
        # TODO: docstring documentation needs to be improved.

        self._dimension = len(parameters)
        input_parameters = InputParameters.from_list(parameters)
        super(MixtureNormal, self).__init__(input_parameters, name)


    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from a multivariate normal distribution using the current values for each probabilistic model from which the model derives.

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
        mean = parameter_values
        # There is no check of the parameter_values because the mixture normal will accept all parameters

        # Generate k lists from mixture_normal
        Data_array = [None] * k
        dimension = self.get_output_dimension()
        index_array = rng.binomial(1, 0.5, k)
        for i in range(k):
            # Initialize the time-series
            index = index_array[i]
            Data = index * rng.multivariate_normal(mean=mean, cov=np.identity(dimension)) \
                   + (1 - index) * rng.multivariate_normal(mean=mean, cov=0.01 * np.identity(dimension))
            Data_array[i] = Data
        return np.array(Data_array).tolist()

    def _check_parameters(self, parameters):
        """
        Checks the values for the parameters sampled from the parents of the probabilistic model at initialization.
        """
        return True

    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values given as fixed values.
        """
        return True

    def get_output_dimension(self):
        return self._dimension


    def pdf(self, x):
        """
       Calculates the probability density function at point x.
       Commonly used to determine whether perturbed parameters are still valid according to the pdf.

       Parameters
       ----------
       x: list
           The point at which the pdf should be evaluated.
       """
        mean = self.get_parameter_values()
        cov_1 = np.identity(self.get_output_dimension())
        cov_2 = 0.01*cov_1
        pdf = 0.5*(multivariate_normal(mean, cov_1).pdf(x))+0.5*(multivariate_normal(mean, cov_2).pdf(x))
        self.calculated_pdf = pdf
        return pdf


class StudentT(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='StudentT'):
        """
        This class implements a probabilistic model following the Student's T-distribution.

        Parameters
        ----------
        parameters: list
            If the list has two entries, the first entry contains the mean of the distribution, while the second entry             contains the degrees of freedom.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        input_parameters = InputParameters.from_list(parameters)
        super(StudentT, self).__init__(input_parameters, name)


    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from a Student's T-distribution using the current values for each probabilistic model from which the model derives.

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
        mean = parameter_values[0]
        df = parameter_values[1]
        return np.array((rng.standard_t(df,k)+mean).reshape(k,-1)).tolist()


    def _check_parameters(self, parameters):
        """
        Checks parameter values sampled from the parents of the probabilistic model. Returns False iff the degrees of freedom are smaller than or equal to 0.
        """
        if(parameters.get_parameter_count() > 2 or parameters.get_parameter_count() < 2):
            raise IndexError('Input to StudentT has to be of length 2.')

        if parameters[1] <= 0:
            return False

        return True


    def _check_parameters_before_sampling(self, parameters):
        """
        Returns False iff the degrees of freedom are smaller than or equal to 0.
        """
        if(parameters[1]<=0):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values given as fixed values.
        """
        return True

    def get_output_dimension(self):
        return 1


    def pdf(self, x):
        """
       Calculates the probability density function at point x.
       Commonly used to determine whether perturbed parameters are still valid according to the pdf.

       Parameters
       ----------
       x: list
           The point at which the pdf should be evaluated.
       """
        parameter_values = self.get_parameter_values()
        df = parameter_values[1]
        x-=parameter_values[0] #divide by std dev if we include that
        pdf = gamma((df+1)/2)/(np.sqrt(df*np.pi)*gamma(df/2)*(1+x**2/df)**((df+1)/2))
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

        mean = parameters[0]
        if isinstance(mean, list):
            self._dimension = len(mean)
            input_parameters = InputParameters.from_list(parameters)
        elif isinstance(mean, ProbabilisticModel):
            self._dimension = mean.get_output_dimension()
            input_parameters = parameters

        super(MultiStudentT, self).__init__(input_parameters, name)


    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from a multivariate Student's T-distribution using the current values for each probabilistic model from
        which the model derives.

        Parameters
        ----------
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the
            generator.

        Returns
        -------
        list: [boolean, np.ndarray]
            A list containing whether it was possible to sample values from the distribution and if so, the sampled
            values.
        """

        # Extract parameters
        parameters = self.get_parameter_values()
        dim = self.get_output_dimension()
        mean = np.array(parameters[0:dim])
        cov = np.array(parameters[dim:dim+dim**2]).reshape((dim, dim))
        df = parameters[-1]

        if (df == np.inf):
            chisq = 1.0
        else:
            chisq = rng.chisquare(df, k) / df
            chisq = chisq.reshape(-1, 1).repeat(dim, axis=1)
        mvn = rng.multivariate_normal(np.zeros(dim), cov, k)
        result = (mean + np.divide(mvn, np.sqrt(chisq)))

        return result.tolist()


    def _check_parameters(self, parameters):
        """
        Returns False iff the degrees of freedom are less than or equal to 0, the covariance matrix is not symmetric or
        the covariance matrix is not positive definite.
        """

        dim = self.get_output_dimension()
        expected_param_cnt = dim + dim**2 + 1
        observed_param_cnt = parameters.get_parameter_count()
        if( observed_param_cnt > expected_param_cnt or observed_param_cnt < expected_param_cnt ):
            raise ValueError('Wrong number of input parameters.')

        # Extract parameters
        mean = np.array(parameters[0:dim])
        cov = np.array(parameters[dim:dim+dim**2]).reshape((dim, dim))
        df = parameters[-1]

        # Check whether the covariance matrix is symmetric
        if not np.allclose(cov, cov.T, atol=1e-3):
            return False

        # Check whether the covariance matrix is positive definit
        try:
            is_pos = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False

        # Check whether the degrees of freedom are <=0
        if df <= 0:
            return False

        return True


    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values given as fixed values.
        """
        return True

    def get_output_dimension(self):
        return self._dimension


    def pdf(self, x):
        """
       Calculates the probability density function at point x.
       Commonly used to determine whether perturbed parameters are still valid according to the pdf.

       Parameters
       ----------
       x: list
           The point at which the pdf should be evaluated.
       """
        parameter_values = self.get_parameter_values()
        mean = parameter_values[:-2]
        cov = parameter_values[-2]
        v = parameter_values[-1]
        mean = np.array(mean)
        cov = np.array(cov)
        p=len(mean)
        numerator = gamma((v + p) / 2)
        denominator = gamma(v / 2) * pow(v * np.pi, p / 2.) * np.sqrt(abs(np.linalg.det(cov)))
        normalizing_const = numerator / denominator
        tmp = 1 + 1 / v * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))
        density = normalizing_const * pow(tmp, -((v + p) / 2.))
        self.calculated_pdf = density
        return density


class Uniform(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Uniform'):
        """
        This class implements a probabilistic model following a uniform distribution.

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
        if len(parameters)<2:
            raise ValueError('Input to Uniform has to be at least of length 2.')
        if not isinstance(parameters[0], list):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')
        if not isinstance(parameters[1], list):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')
        if len(parameters[0]) != len(parameters[1]):
            raise ValueError('Length of upper and lower bound have to be equal.')

        self._dimension = len(parameters[0])
        input_parameters = InputParameters.from_list(parameters)
        super(Uniform, self).__init__(input_parameters, name)
        self.visited = False


    def num_parameters(self):
        return self._num_parameters


    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from a uniform distribution using the current values for each probabilistic model from which the model derives.

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

        samples = np.zeros(shape=(k, self.get_output_dimension()))
        for j in range(0, self.get_output_dimension()):
            samples[:, j] = rng.uniform(parameter_values[j], parameter_values[j+self.get_output_dimension()], k)

        return samples.tolist()


    def _check_parameters(self, parameters):
        """
        Checks parameter values sampled from the parents.
        """
        if(parameters.get_parameter_count() % 2 != 0):
            raise IndexError('Number of input parameters is odd.')

        # test whether lower bound is not greater than upper bound
        for j in range(self.get_output_dimension()):
            if (parameters[j] > parameters[j+self.get_output_dimension()]):
                return False
        return True


    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values given as fixed values. Returns False iff a lower bound value is larger than a corresponding upper bound value.
        """
        for i in range(self.get_output_dimension()):
            parent_lower, index_lower = self.parents[i]
            parent_upper, index_upper = self.parents[i+self.get_output_dimension()]
            lower_value = parent_lower.fixed_values[index_lower]
            upper_value = parent_upper.fixed_values[index_upper]
            if(parameters[i]<lower_value or parameters[i]>upper_value):
                return False
        return True


    def get_output_dimension(self):
        return self._dimension


    def pdf(self, x):
        """
       Calculates the probability density function at point x.
       Commonly used to determine whether perturbed parameters are still valid according to the pdf.

       Parameters
       ----------
       x: list
           The point at which the pdf should be evaluated.
       """
        parameter_values = self.get_parameter_values()
        lower_bound = parameter_values[:self.get_output_dimension()]
        upper_bound = parameter_values[self.get_output_dimension():]
        if (np.product(np.greater_equal(x, np.array(lower_bound)) * np.less_equal(x, np.array(upper_bound)))):
            pdf_value = 1. / np.product(np.array(upper_bound) - np.array(lower_bound))
        else:
            pdf_value = 0.
        self.calculated_pdf = pdf_value
        return pdf_value


