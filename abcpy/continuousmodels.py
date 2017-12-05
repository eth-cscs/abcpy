from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, Hyperparameter
import numpy as np

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

        # Save the name given by the user for the journal output
        self.name = name

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
        pdf = norm(mu,sigma).pdf(x)
        self.calculated_pdf = pdf
        return pdf


class MultivariateNormal(ProbabilisticModel, Continuous):
    def __init__(self,parameters, name='Multivariate Normal'):
        """
        This class implements a probabilistic model following a multivariate normal distribution with mean and covariance matrix.

        Parameters
        ----------
        parameters: list of at least length 2
            Contains the probabilistic models and hyperparameters from which the model derives. The last entry defines the         covariance matrix, while all other entries define the mean. Note that if the mean is n dimensional, the                covariance matrix is required to be of dimension nxn. The covariance matrix is required to be positive-definite.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        # The user input will contain two lists, a list for the mean, and a list for the covariance matrix. Put this into the appropriate format used by the super constructor.
        parameters_temp = []
        for parameter in parameters[0]:
            parameters_temp.append(parameter)
        parameters_temp.append(parameters[1])

        super(MultivariateNormal, self).__init__(parameters_temp)

        # Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parents)-1

        # Save the name given by the user for the journal output
        self.name = name

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
        pdf = multivariate_normal(mean, cov).pdf(x)
        self.calculated_pdf = pdf
        return pdf


class MixtureNormal(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Mixture normal'):
        """
        This class implements a probabilistic model following a mixture normal distribution.

        Parameters
        ----------
        parameters: list
            Contains all the probabilistic models and hyperparameters from which the model derives.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """
        super(MixtureNormal, self).__init__(parameters)
        # Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parents)

        # Save the name given by the user for the journal output
        self.name = name

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
        pdf = 0.5*(multivariate_normal(mean, cov_1).pdf(x))+0.5*(multivariate_normal(mean, cov_2).pdf(x))
        self.calculated_pdf = pdf
        return pdf


class StudentT(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Student T'):
        """
        This class implements a probabilistic model following the Student's T-distribution.

        Parameters
        ----------
        parameters: list
            If the list has two entries, the first entry contains the mean of the distribution, while the second entry             contains the degrees of freedom.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

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

        # Save the name given by the user for the journal output
        self.name = name

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
        pdf = gamma((df+1)/2)/(np.sqrt(df*np.pi)*gamma(df/2)*(1+x**2/df)**((df+1)/2))
        self.calculated_pdf = pdf
        return pdf


class MultiStudentT(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Multivariate Student T'):
        """
        This class implements a probabilistic model following the multivariate Student-T distribution.

        Parameters
        ----------
        parameters: list
            All but the last two entries contain the probabilistic models and hyperparameters from which the model derives.        The second to last entry contains the covariance matrix. If the mean is of dimension n, the covariance matrix          is required to be nxn dimensional. The last entry contains the degrees of freedom.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        # The user input contains a list for the mean. Change this to be compatible with the format required by the super constructor.
        parameters_temp = []
        for parameter in parameters[0]:
            parameters_temp.append(parameter)
        parameters_temp.append(parameters[1])
        parameters_temp.append(parameters[2])

        super(MultiStudentT, self).__init__(parameters_temp)

        # Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parents)-2

        # Save the name given by the user for the journal output
        self.name = name

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
                chisq = 1.0
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
        self.calculated_pdf = density
        return density


class Uniform(ProbabilisticModel, Continuous):
    def __init__(self, parameters, name='Uniform'):
        """
        This class implements a probabilistic model following a uniform distribution.

        Parameters
        ----------
        parameters: list
            Contains two lists. The first list specifies the probabilistic models and hyperparameters from which the lower         bound of the uniform distribution derive. The second list specifies the probabilistic models and hyperparameters from which the upper bound derives.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

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

        # Save the name given by the user for the journal output
        self.name = name


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
        self.calculated_pdf = pdf_value
        return pdf_value


