from ProbabilisticModel import ProbabilisticModel, Continuous, Hyperparameter
import numpy as np

from scipy.stats import multivariate_normal, norm
from scipy.special import gamma

#TODO document somewhere that sample_parameters needs to be called for standalone distributions


class Normal(ProbabilisticModel, Continuous):
    """
    This class implements a probabilistic model following a normal distribution with mean mu and variance sigma.

    Parameters
    ----------
    parameters: list
        Contains the probabilistic models and hyperparameters from which the model derives. Note that the second value of the list is not allowed to be smaller than 0.
    """
    def __init__(self, parameters):
        # Rewrite user input
        input_parameters = []
        for parameter in parameters:
            if(isinstance(parameter, list)):
                input_parameters.append(parameter[0])
            else:
                input_parameters.append(parameter)

        super(Normal, self).__init__(input_parameters)
        # Parameter specifying the dimension of the return values of the distribution.
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

        Returns
        -------
        list: [boolean, np.ndarray]
            A list containing whether it was possible to sample values from the distribution and if so, the sampled values.
        """
        parameter_values = self.get_parameter_values()
        return_value = []
        # Check the parameters for whether they are a valid input for the normal distribution
        return_value.append(self._check_parameters_before_sampling(parameter_values))

        if(return_value[0]):
            mu = parameter_values[0]
            sigma = parameter_values[1]
            return_value.append(np.array(rng.normal(mu, sigma, k)).reshape(-1))

        return return_value

    def _check_parameters_at_initialization(self, parameters):
        """
        Returns True iff the second parameter was not a hyperparameter or was a hyperparameter and was >=0
        """
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for Normal has to be of type list.')
        parameter, index = parameters[1]

        #NOTE this check will be replaced by a loop as soon as we have domain implementations
        # Check whether in case the second parameter is a hyperparameter, it is not smaller than 0
        if(isinstance(parameter, Hyperparameter) and parameter.fixed_values[index]<0):
            raise ValueError('The specified standard deviation is less than 0.')

    def _check_parameters_before_sampling(self, parameters):
        """
        Returns False iff the standard deviation is smaller than 0.
        """
        if(parameters[1]<0):
            return False
        return True

    #NOTE this should check whether the parameters can come from this distribution --> if there are domains, this function might change
    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values that are given as fixed values.
        """
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
        parameter_values = self.get_parameter_values()
        mu = parameter_values[0]
        sigma = parameter_values[1]
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
        # The user input will contain two lists, a list for the mean, and a list for the covariance matrix. Put this into the appropriate format used by the super constructor.
        parameters_temp = []
        for parameter in parameters[0]:
            parameters_temp.append(parameter)
        parameters_temp.append(parameters[1])

        super(MultivariateNormal, self).__init__(parameters_temp)

        # Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parents)-1

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
        return_value = []
        return_value.append(self._check_parameters_before_sampling(parameter_values))

        if(return_value[0]):
            mean = parameter_values[:-1]
            cov = parameter_values[-1]
            return_value.append(rng.multivariate_normal(mean, cov, k))

        return return_value

    def _check_parameters_at_initialization(self, parameters):
        """
        Checks parameter values sampled from the parents at initialization. Returns False iff the covariance matrix is not symmetric or not positive definite.
        """
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for MultivariateNormal has to be of type list.')

        if(len(parameters)<2):
            raise IndexError('Input for MultivariateNormal has to be of at least length 2.')

        length = len(parameters)-1
        cov, index = parameters[-1]

        #NOTE if we allow nonhyperparameter initialization, we replace this by a domain check
        if(isinstance(cov, Hyperparameter)):
            cov = np.array(cov.fixed_values[index])
            if(length!=len(cov[0])):
                raise IndexError('Length of mean and covariance matrix have to match.')

            # Check whether the covariance matrix is symmetric
            if(not(np.allclose(cov, cov.T, atol=1e-3))):
                raise ValueError('Covariance matrix is not symmetric.')
            # Check whether the covariance matrix is positive definite
            try:
                is_pos = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                raise ValueError('Covariance matrix is not positive definite.')

    def _check_parameters_before_sampling(self, parameters):
        """
        Returns True iff the covariance matrix provided is symmetric and positive definite.
        """

        cov = np.array(parameters[-1])
        # Check whether the covariance matrix is symmetric
        if(not(np.allclose(cov, cov.T, atol=1e-3))):
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
        mean= parameter_values[:-1]
        cov = parameter_values[-1]
        return multivariate_normal(mean, cov).pdf(x)


class MixtureNormal(ProbabilisticModel, Continuous):
    """
    This class implements a probabilistic model following a mixture normal distribution.

    Parameters
    ----------
    parameters: list
        Contains all the probabilistic models and hyperparameters from which the model derives.
    """
    def __init__(self, parameters):
        super(MixtureNormal, self).__init__(parameters)
        # Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parents)

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
        dimension = len(mean)
        index_array = rng.binomial(1, 0.5, k)
        for i in range(k):
            # Initialize the time-series
            index = index_array[i]
            Data = index * rng.multivariate_normal(mean=mean, cov=np.identity(dimension)) \
                   + (1 - index) * rng.multivariate_normal(mean=mean, cov=0.01 * np.identity(dimension))
            Data_array[i] = Data

        return [True, np.array(Data_array)]

    #NOTE is there any restriction on mixture normal parameters/a domain?
    def _check_parameters_at_initialization(self, parameters):
        """
        Checks the values for the parameters sampled from the parents of the probabilistic model at initialization.
        """
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for MixtureNormal has to be of type list.')

    def _check_parameters_before_sampling(self, parameters):
        return True

    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values given as fixed values.
        """
        return True

    #TODO ASK RITO WHETHER THIS IS CORRECT
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
        cov_1 = np.identity(self.dimension)
        cov_2 = 0.01*cov_1
        return 0.5*(multivariate_normal(mean, cov_1).pdf(x))+0.5*(multivariate_normal(mean, cov_2).pdf(x))


class StudentT(ProbabilisticModel, Continuous):
    """
    This class implements a probabilistic model following the Student's T-distribution.

    Parameters
    ----------
    parameters: list
        If the list has two entries, the first entry contains the mean of the distribution, while the second entry             contains the degrees of freedom.
    """
    def __init__(self, parameters):
        # Rewrite user input
        input_parameters = []
        for parameter in parameters:
            if (isinstance(parameter, list)):
                input_parameters.append(parameter[0])
            else:
                input_parameters.append(parameter)

        super(StudentT, self).__init__(input_parameters)
        # Parameter specifying the dimension of the return values of the distribution.
        self.dimension = 1

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
        return_value = []
        return_value.append(self._check_parameters_before_sampling(parameter_values))

        if(return_value[0]):
            mean = parameter_values[0]
            df = parameter_values[1]
            return_value.append(np.array((rng.standard_t(df,k)+mean).reshape(-1)))

        return return_value

    def _check_parameters_at_initialization(self, parameters):
        """
        Checks parameter values sampled from the parents of the probabilistic model. Returns False iff the degrees of freedom are smaller than or equal to 0.
        """
        if(not(isinstance(parameters, list))):
            raise TypeError('Input to StudentT has to be of type list.')
        if(len(parameters)>2):
            raise IndexError('Input to StudentT has to be of length 2.')

        parameter, index = parameters[1]
        if(isinstance(parameter, Hyperparameter) and parameter.fixed_values[index]<=0):
            raise ValueError('The sampled values for the model lie outside its domain.')

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
        return gamma((df+1)/2)/(np.sqrt(df*np.pi)*gamma(df/2)*(1+x**2/df)**((df+1)/2))


class MultiStudentT(ProbabilisticModel, Continuous):
    """
    This class implements a probabilistic model following the multivariate Student-T distribution.

    Parameters
    ----------
    parameters: list
        All but the last two entries contain the probabilistic models and hyperparameters from which the model derives.        The second to last entry contains the covariance matrix. If the mean is of dimension n, the covariance matrix          is required to be nxn dimensional. The last entry contains the degrees of freedom.
    """
    def __init__(self, parameters):
        # The user input contains a list for the mean. Change this to be compatible with the format required by the super constructor.
        parameters_temp = []
        for parameter in parameters[0]:
            parameters_temp.append(parameter)
        parameters_temp.append(parameters[1])
        parameters_temp.append(parameters[2])

        super(MultiStudentT, self).__init__(parameters_temp)

        # Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parents)-2

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from a multivariate Student's T-distribution using the current values for each probabilistic model from which the model derives.

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
            mean = parameter_values[:-2]
            cov = parameter_values[-2]
            df = parameter_values[-1]
            p = len(mean)
            if (df == np.inf):
                chis1 = 1.0
            else:
                chisq = rng.chisquare(df, k) / df
                chisq = chisq.reshape(-1, 1).repeat(p, axis=1)
            mvn = rng.multivariate_normal(np.zeros(p), cov, k)
            result = (mean + np.divide(mvn, np.sqrt(chisq)))
            return_value.append(result)

        return return_value

    def _check_parameters_at_initialization(self, parameters):
        """
        Checks parameter values sampled from the parents of the probabilistic model. Returns False iff the degrees of freedom are less than or equal to 0, the covariance matrix is not symmetric or the covariance matrix is not positive definite.
        """
        length = len(parameters)-2
        cov, index = parameters[-2]

        if(isinstance(cov, Hyperparameter)):
            cov = np.array(cov.fixed_values[index])
            if(not(length==len(cov[0]))):
                raise IndexError('Mean and covariance matrix have to be of same length.')
            # Check whether the covariance matrix is symmetric
            if (not (np.allclose(cov, cov.T, atol=1e-3))):
                raise ValueError('Covariance matrix is not symmetric.')
            # Check whether the covariance matrix is positive definiet
            try:
                is_pos = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                raise ValueError('Covariance matrix is not positive definite.')

        df, index = parameters[-1]
        # Check whether the degrees of freedom are <=0
        if(df.fixed_values[index]<=0):
            raise ValueError('Degrees of freedom are required to be larger than 0.')

    def _check_parameters_before_sampling(self, parameters):
        """
        Returns False iff the covariance matrix is not symmetric or not positive definite, or the degrees of freedom are smaller than or equal to 0.
        """
        df = parameters[-1]
        cov = np.array(parameters[-2])

        if(not(np.allclose(cov, cov.T, atol=1e-3))):
            return False
        try:
            is_pos = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False

        if(df<=0):
            return False

        return True

    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values given as fixed values.
        """
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
        return density


class Uniform(ProbabilisticModel, Continuous):
    """
    This class implements a probabilistic model following a uniform distribution.

    Parameters
    ----------
    parameters: list
        Contains two lists. The first list specifies the probabilistic models and hyperparameters from which the lower         bound of the uniform distribution derive. The second list specifies the probabilistic models and hyperparameters from which the upper bound derives.
    """
    def __init__(self, parameters):
        # The user input is checked, since the input has to be rewritten internally before sending it to the constructor of the probabilistic model
        self._check_user_input(parameters)

        # The total number of parameters is initialized
        self._num_parameters = 0

        # Stores the length of the parameter values of the lower and upper bound. This is needed to check that lower and upper are of same length, just because the total length is even does not guarantee that
        self.length = [0,0]
        joint_parameters = []

        # Rewrite the user input to be useable by the constructor of probabilistic model and set the length of upper and lower bound
        for i in range(2):
            for parameter in parameters[i]:
                joint_parameters.append(parameter)
                self.length[i]+=1
                # If the parameter is not a hyperparameter, the length of the bound has to be equal to the parameter dimension. We cannot simply add the parameters dimension since the dimension of a hyperparameter is 0.
                if(not(isinstance(parameter, tuple)) and isinstance(parameter, ProbabilisticModel)):
                    for j in range(1,parameter.dimension):
                        self.length[i]+=1

        self._num_parameters=self.length[0]+self.length[1]

        # Parameter specifying the dimension of the return values of the distribution.
        self.dimension = int(self._num_parameters/2)

        super(Uniform, self).__init__(joint_parameters)
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
        return_value = []
        return_value.append(self._check_parameters_before_sampling(parameter_values))

        if(return_value[0]):
            samples = np.zeros(shape=(k, self.dimension))
            for j in range(0, self.dimension):
                samples[:, j] = rng.uniform(parameter_values[j], parameter_values[j+self.dimension], k)
            return_value.append(samples)

        return return_value

    def _check_user_input(self, parameters):
        """
        Checks the users input before it is rewritten to work with the probabilistic model constructor.
        """
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for Uniform has to be of type list.')
        if(len(parameters)<2):
            raise IndexError('Input to Uniform has to be at least of length 2.')
        if(not(isinstance(parameters[0], list))):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')
        if(not(isinstance(parameters[1], list))):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')

    #NOTE right now it would only be possible to compare bounds if they are both hyperparameters. For more functionality we need the domains
    def _check_parameters_at_initialization(self, parameters):
        """
        Checks parameter values sampled from the parents.
        """
        if(self.length[0]!=self.length[1]):
            raise IndexError('Length of upper and lower bound have to be equal.')

    def _check_parameters_before_sampling(self, parameters):
        """
        Returns False iff for some pair of lower and upper bound, the lower bound is larger than the upper bound.
        """
        for j in range(self.dimension):
            if(parameters[j]>parameters[j+self.dimension]):
                return False
        return True


    def _check_parameters_fixed(self, parameters):
        """
        Checks parameter values given as fixed values. Returns False iff a lower bound value is larger than a corresponding upper bound value.
        """
        for i in range(self.dimension):
            parent_lower, index_lower = self.parents[i]
            parent_upper, index_upper = self.parents[i+self.dimension]
            lower_value = parent_lower.fixed_values[index_lower]
            upper_value = parent_upper.fixed_values[index_upper]
            if(parameters[i]<lower_value or parameters[i]>upper_value):
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
        parameter_values = self.get_parameter_values()
        lower_bound = parameter_values[:self.dimension]
        upper_bound = parameter_values[self.dimension:]
        if (np.product(np.greater_equal(x, np.array(lower_bound)) * np.less_equal(x, np.array(upper_bound)))):
            pdf_value = 1. / np.product(np.array(upper_bound) - np.array(lower_bound))
        else:
            pdf_value = 0.
        return pdf_value

#NOTE NO NEW IMPLEMENTATIONS BELOW THIS POINT

class StochLorenz95(ProbabilisticModel, Continuous):
    """Generates time dependent 'slow' weather variables following forecast model of Wilks [1],
        a stochastic reparametrization of original Lorenz model Lorenz [2].

        [1] Wilks, D. S. (2005). Effects of stochastic parametrizations in the lorenz ’96 system.
        Quarterly Journal of the Royal Meteorological Society, 131(606), 389–407.

        [2] Lorenz, E. (1995). Predictability: a problem partly solved. In Proceedings of the
        Seminar on Predictability, volume 1, pages 1–18. European Center on Medium Range
        Weather Forecasting, Europe

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives.
        initial_state: numpy.ndarray, optional
            Initial state value of the time-series, The default value is None, which assumes a previously computed
            value from a full Lorenz model as the Initial value.
        n_timestep: int, optional
            Number of steps for a time between [0,4], where 4 corresponds to 20 days. The default value is 160 steps.
        """
    def __init__(self, parameters, initial_state= None, n_timestep=160):
        super(StochLorenz95, self).__init__(parameters)
        self.n_timestep = n_timestep
        # Assign initial state
        if not initial_state == None:
            self.initial_state = initial_state
        else:
            self.initial_state = np.array([6.4558, 1.1054, -1.4502, -0.1985, 1.1905, 2.3887, 5.6689, 6.7284, 0.9301, \
                                           4.4170, 4.0959, 2.6830, 4.7102, 2.5614, -2.9621, 2.1459, 3.5761, 8.1188,
                                           3.7343, 3.2147, 6.3542, \
                                           4.5297, -0.4911, 2.0779, 5.4642, 1.7152, -1.2533, 4.6262, 8.5042, 0.7487,
                                           -1.3709, -0.0520, \
                                           1.3196, 10.0623, -2.4885, -2.1007, 3.0754, 3.4831, 3.5744, 6.5790])
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.initial_state)
        # Other parameters kept fixed
        self.F = 10
        self.sigma_e = 1
        self.phi = 0.4

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        # Generate n_simulate time-series of weather variables satisfying Lorenz 95 equations
        timeseries_array = [None] * k

        # Initialize timesteps
        time_steps = np.linspace(0, 4, self.n_timestep)

        for k in range(0, k):
            # Define a parameter object containing parameters which is needed
            # to evaluate the ODEs
            # Stochastic forcing term
            eta = self.sigma_e * np.sqrt(1 - pow(self.phi, 2)) * rng.normal(0, 1, self.initial_state.shape[0])

            # Initialize the time-series
            timeseries = np.zeros(shape=(self.initial_state.shape[0], self.n_timestep), dtype=np.float)
            timeseries[:, 0] = self.initial_state
            # Compute the timeseries for each time steps
            for ind in range(0, self.n_timestep - 1):
                # parameters to be supplied to the ODE solver
                parameter = [eta, np.array(self.fixed_parameters)]
                # Each timestep is computed by using a 4th order Runge-Kutta solver
                x = self._rk4ode(self._l95ode_par, np.array([time_steps[ind], time_steps[ind + 1]]), timeseries[:, ind],
                                 parameter)
                timeseries[:, ind + 1] = x[:, -1]
                # Update stochastic foring term
                eta = self.phi * eta + self.sigma_e * np.sqrt(1 - pow(self.phi, 2)) * rng.normal(0, 1)
            timeseries_array[k] = timeseries
        # return an array of objects of type Timeseries
        return timeseries_array

    def _l95ode_par(self, t, x, parameter):
        """
        The parameterized two-tier lorenz 95 system defined by a set of symmetic
        ordinary differential equation. This function evaluates the differential
        equations at a value x of the time-series

        Parameters
        ----------
        x: numpy.ndarray of dimension px1
            The value of timeseries where we evaluate the ODE
        parameter: Python list
            The set of parameters needed to evaluate the function
        Returns
        -------
        numpy.ndarray
            Evaluated value of the ode at a fixed timepoint
        """
        # Initialize the array containing the evaluation of ode
        dx = np.zeros(shape=(x.shape[0],))
        eta = parameter[0]
        theta = parameter[1]
        # Deterministic parameterization for fast weather variables
        # ---------------------------------------------------------
        # assumed to be polynomial, degree of the polynomial same as the
        # number of columns in closure parameter
        degree = theta.shape[0]
        X = np.ones(shape=(x.shape[0], 1))
        for ind in range(1, degree):
            X = np.column_stack((X, pow(x, ind)))

        # deterministic reparametrization term
        # ------------------------------------
        gu = np.sum(X * theta, 1)

        # ODE definition of the slow variables
        # ------------------------------------
        dx[0] = -x[-2] * x[-1] + x[-1] * x[1] - x[0] + self.F - gu[0] + eta[0]
        dx[1] = -x[-1] * x[0] + x[0] * x[2] - x[1] + self.F - gu[1] + eta[1]
        for ind in range(2, x.shape[0] - 2):
            dx[ind] = -x[ind - 2] * x[ind - 1] + x[ind - 1] * x[ind + 1] - x[ind] + self.F - gu[ind] + eta[ind]
        dx[-1] = -x[-3] * x[-2] + x[-2] * x[1] - x[-1] + self.F - gu[-1] + eta[-1]

        return dx

    def _rk4ode(self, ode, timespan, timeseries_initial, parameter):
        """
        4th order runge-kutta ODE solver.

        Parameters
        ----------
        ode: function
            The function defining Ordinary differential equation
        timespan: numpy.ndarray
            A numpy array containing the timepoints where the ode needs to be solved.
            The first time point corresponds to the initial value
        timeseries_init: np.ndarray of dimension px1
            Intial value of the time-series, corresponds to the first value of timespan
        parameter: Python list
            The parameters needed to evaluate the ode
        Returns
        -------
        np.ndarray
            Timeseries initiated at timeseries_init and satisfying ode solved by this solver.
        """

        timeseries = np.zeros(shape=(timeseries_initial.shape[0], timespan.shape[0]))
        timeseries[:, 0] = timeseries_initial

        for ind in range(0, timespan.shape[0] - 1):
            time_diff = timespan[ind + 1] - timespan[ind]
            time_mid_point = timespan[ind] + time_diff / 2
            k1 = time_diff * ode(timespan[ind], timeseries_initial, parameter)
            k2 = time_diff * ode(time_mid_point, timeseries_initial + k1 / 2, parameter)
            k3 = time_diff * ode(time_mid_point, timeseries_initial + k2 / 2, parameter)
            k4 = time_diff * ode(timespan[ind + 1], timeseries_initial + k3, parameter)
            timeseries_initial = timeseries_initial + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            timeseries[:, ind + 1] = timeseries_initial
        # Return the solved timeseries at the values in timespan
        return timeseries

    def _check_parameters(self, parameters):
        if(not(isinstance(parameters, list))):
            raise TypeError('Input to StochLorenz95 has to be of type list.')
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def pdf(self, x):
        raise NotImplementedError


class Ricker(ProbabilisticModel, Continuous):
    """Ecological model that describes the observed size of animal population over time
        described in [1].

        [1] S. N. Wood. Statistical inference for noisy nonlinear ecological
        dynamic systems. Nature, 466(7310):1102–1104, Aug. 2010.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives.
        n_timestep: int, optional
        Number of timesteps. The default value is 100.
    """
    def __init__(self, parameters, n_timestep=100):
        super(Ricker, self).__init__(parameters)
        self.n_timestep = n_timestep
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = n_timestep

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        timeseries_array = [None] * k
        # Initialize local parameters
        log_r = self.fixed_parameters[0]
        sigma = self.fixed_parameters[1]
        phi = self.fixed_parameters[2]
        for k in range(0, k):
            # Initialize the time-series
            timeseries_obs_size = np.zeros(shape=(self.n_timestep), dtype=np.float)
            timeseries_true_size = np.ones(shape=(self.n_timestep), dtype=np.float)
            for ind in range(1, self.n_timestep - 1):
                timeseries_true_size[ind] = np.exp(log_r + np.log(timeseries_true_size[ind - 1]) - timeseries_true_size[
                    ind - 1] + sigma * rng.normal(0, 1))
                timeseries_obs_size[ind] = rng.poisson(phi * timeseries_true_size[ind])
            timeseries_array[k] = timeseries_obs_size
        # return an array of objects of type Timeseries
        return timeseries_array

    def _check_parameters(self, parameters):
        if(not(isinstance(parameters,list))):
            raise TypeError('Input to Ricker has to be of type list.')
        if(len(parameters)>3):
            raise IndexError('Input to Ricker can be at most of length 3.')
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def pdf(self, x):
        raise NotImplementedError