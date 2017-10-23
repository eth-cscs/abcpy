from abc import ABCMeta, abstractmethod
import numpy as np

from scipy.stats import multivariate_normal, norm
from scipy.special import gamma

#TODO both ricker and lorenz could support that you give some in a combined model, and some not, we should implement that, but not a priority

#NOTE we could call self.parents self.prior?

#NOTE do we maybe want to average over a couple of samples during initialization, rather than taking a single value? this is an issue for example for StudentT!

class ProbabilisticModel(metaclass = ABCMeta):
    """This abstract class represents all probabilistic models.

        Parameters
        ----------
        parameters: list, each element can either be of type ProbabilisticModel or float
            Contains the probabilistic models and hyperparameters which define the parameters of the probabilistic model.
        is_uniform: boolean
            Set to true if the probabilistic model describes a uniform distribution.

    """
    def __init__(self, parameters, is_uniform = False):
        #Save all probabilistic models and hyperparameters from which the model derives.
        self.parents = parameters
        #Initialize list which will contain the values for all parameters associated with the model. If the parameters          derive from a probabilistic model, they will be sampled.
        self.parameter_values = []
        parameter_values_temp = []
        if(not is_uniform):
            counter = 0
            while(counter<10):
                counter += 1
                parameter_values_temp = []
                for i in range(len(self.parents)):
                    if (isinstance(self.parents[i], ProbabilisticModel)):
                        parameter_value = self.parents[i].sample_from_distribution(1)
                        if(isinstance(parameter_value[0], (list, np.ndarray))):
                            parameter_value = parameter_value[0]
                        if (isinstance(parameter_value, (list, np.ndarray))):
                            for j in range(len(parameter_value)):
                                parameter_values_temp.append(parameter_value[j])
                        else:
                            parameter_values_temp.append(parameter_value)
                    else:
                        parameter_values_temp.append(self.parents[i])
                if(self._check_parameters(parameter_values_temp)):
                    self.parameter_values = parameter_values_temp
                    break
            if(counter==10 and not(self._check_parameters(parameter_values_temp))):
                raise ValueError('Cannot find appropriate parameter values for probabilistic model.')
        else:
            counter =0
            while(counter<10):
                counter+=1
                lb = []
                ub = []
                for i in range(len(self.parents[0])):
                    if(isinstance(self.parents[0][i], ProbabilisticModel)):
                        lb_values = self.parents[0][i].sample_from_distribution(1)
                        if(isinstance(lb_values[0], (list, np.ndarray))):
                            lb_values = lb_values[0]
                        if(isinstance(lb_values, (list, np.ndarray))):
                            for j in range(len(lb_values)):
                                lb.append(lb_values[j])
                        else:
                            lb.append(lb_values)
                    else:
                        lb.append(self.parents[0][i])
                for i in range(len(self.parents[1])):
                    if(isinstance(self.parents[1][i], ProbabilisticModel)):
                        ub_values = self.parents[1][i].sample_from_distribution(1)
                        if(isinstance(ub_values[0], (list, np.ndarray))):
                            ub_values = ub_values[0]
                        if(isinstance(ub_values, (list, np.ndarray))):
                            for j in range(len(ub_values)):
                                ub.append(ub_values[j])
                        else:
                            ub.append(ub_values)
                    else:
                        ub.append(self.parents[1][i])
                if(self._check_parameters([lb, ub])):
                    self.parameter_values = [lb, ub]
                    break
            if(counter==10 and not(self._check_parameters([lb, ub]))):
                raise ValueError('Cannot find appropriate parameter values for probabilistic model.')
        self.updated = False #whether the node has been touched

    #NOTE the check_fixed assumes that the parameters are exactly of the lenght needed for that distribution only ->needs to be considered in case we want to be able to "fix on whole graph"
    @abstractmethod
    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        """
        Fixes the parameters associated with the probabilistic model.

        Parameters
        ----------
        parameters: None or list of length equal to the free parameters of the probabilistic model.
            If set to none, all free parameters are sampled from their distribution.
        rng: Random number generator
            Defines the random number generator to be used for sampling. The default value is initialized with a random             seed.
        Returns
        -------
        boolean:
            Tells the user whether the parameters were fixed. They will not be fixed if they do not conform to the                  required format of the underlying distribution.
        """
        index = 0
        #NOTE DO WE NEED SELF.UPDATED HERE?
        if(not(parameters)):
            for i in range(len(self.parents)):
                if(isinstance(self.parents[i], ProbabilisticModel)):
                    parameter_value = self.parents[i].sample_from_distribution(1, rng=rng)
                    if(isinstance(parameter_value[0], (list, np.ndarray))):
                        parameter_value = parameter_value[0]
                    for j in range(len(parameter_value)):
                        self.parameter_values[index]=parameter_value[j]
                        index+=1
                else:
                    index+=1
            self.updated = True
        else:
            if(not(self._check_parameters_fixed(parameters))):
                return False
            index=0
            i=0
            while(i<len(parameters)):
                while(not(isinstance(self.parents[index],ProbabilisticModel)) or self.parents[index].updated):
                    index+=1
                if(not(parameters[i])):
                    parameter_value = self.parents[index].sample_from_distribution(1, rng=rng)
                    if(isinstance(parameter_value[0], (list, np.ndarray))):
                        parameter_value = parameter_value[0]
                    for j in range(len(parameter_value)):
                        self.parameter_values[index+j] = parameter_value[j]
                    index+=len(parameter_value)
                    i+=len(parameter_value)
                else:
                    for j in range(self.parents[index].dimension):
                        self.parameter_values[index+j]=parameters[i]
                        i+=1
                    index+=self.parents[index].dimension
            self.updated = True
            return True

    @abstractmethod
    def get_parameters(self):
        """
        Returns the current values of the free parameters of the probabilistic model.

        Returns
        -------
        return_values: list
            The current values of the free parameters.
        """
        return_values = []
        index=0
        for i in range(len(self.parents)):
            if(isinstance(self.parents[i], ProbabilisticModel)):
                for j in range(self.parents[i].dimension):
                    return_values.append(self.parameter_values[index+j])
                index+=self.parents[i].dimension
            else:
                index+=1
        return return_values


    @abstractmethod
    def _check_parameters(self, parameters):
        """
        Checks parameters at initialization.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters which define the probabilistic model.
        """
        raise NotImplementedError

    @abstractmethod
    def _check_parameters_fixed(self, parameters):
        """
        Checks parameters in the fix_parameters method.

        Parameters
        ----------
        parameters: list
            Contains the values to which the free parameters should be fixed.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        """
        Samples from the distribution associated with the probabilistic model by using the current values for each              probabilistic model from which the model derives.

        Parameters
        ----------
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the                  generator.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x):
        """
        Calculates the probability density function of the model, if applicable

        Parameters
        ----------
        x: float
            The location at which the probability density function should be evaluated.
        """
        raise NotImplementedError



class Normal(ProbabilisticModel):
    """
    This class implements a probabilistic model following a normal distribution with mean mu and variance sigma.

    Parameters
    ----------
    parameters: list
        Contains the probabilistic models and hyperparameters from which the model derives. If the list only has 1             entry, this entry is required to be a 2-dimensional ProbabilisticModel. Note that sigma is not allowed to be           smaller than 0.
    """
    def __init__(self, parameters):
        super(Normal, self).__init__(parameters)
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = 1

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        mu = self.parameter_values[0]
        sigma = self.parameter_values[1]
        return np.array(rng.normal(mu, sigma, k).reshape(-1))

    def _check_parameters(self, parameters):
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for Normal has to be of type list.')
        if(parameters[1]<=0):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        length=0
        for parent in self.parents:
            if(isinstance(parent, ProbabilisticModel)):
                length+=parent.dimension
        if(length==len(parameters)):
            if(len(parameters)==2 and parameters[1]<=0):
                return False
            return True
        return False

    def get_parameters(self):
        return super(Normal, self).get_parameters()

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        return super(Normal, self).fix_parameters(parameters, rng)

    def pdf(self, x):
        mu = self.parameter_values[0]
        sigma = self.parameter_values[1]
        return norm(mu,sigma).pdf(x)


class MultivariateNormal(ProbabilisticModel):
    """
    This class implements a probabilistic model following a multivariate normal distribution with mean and covariance matrix.

    Parameters
    ----------
    parameters: list of at least length 2
        Contains the probabilistic models and hyperparameters from which the model derives. The last entry defines the         covariance matrix, while all other entries define the mean. Note that if the mean is n dimensional, the                covariance matrix is required to be of dimension nxn. The covariance matrix is required to be positive-definite.
    """
    def __init__(self,parameters):
        super(MultivariateNormal, self).__init__(parameters)
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parameter_values)-1

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        mean = self.parameter_values[:len(self.parameter_values)-1]
        cov = self.parameter_values[len(self.parameter_values)-1]
        return rng.multivariate_normal(mean, cov, k)

    def _check_parameters(self, parameters):
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for MultivariateNormal has to be of type list.')
        if(len(parameters)<2):
            raise IndexError('Input for MultivariateNormal has to be of at least length 2.')
        length = len(parameters)-1
        cov = np.array(parameters[len(parameters)-1])
        if(length!=len(cov[0])):
            raise IndexError('Length of mean and covariance matrix have to match.')
        if(not(np.allclose(cov, cov.T, atol=1e-3))):
            return False
        try:
            is_pos = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        length=0
        for parent in self.parents:
            if(isinstance(parent, ProbabilisticModel)):
                length+=parent.dimension
        if(len(parameters)!=length):
            return False
        return True

    def get_parameters(self):
        return super(MultivariateNormal, self).get_parameters()

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        return super(MultivariateNormal, self).fix_parameters(parameters, rng)

    def pdf(self, x):
        mean= self.parameter_values[:len(self.parameter_values)-1]
        cov = self.parameter_values[len(self.parameter_values)-1]
        return multivariate_normal(mean, cov).pdf(x)


class MixtureNormal(ProbabilisticModel):
    """
    This class implements a probabilistic model following a mixture normal distribution.

    Parameters
    ----------
    parameters: list
        Contains all the probabilistic models and hyperparameters from which the model derives.
    """
    def __init__(self, parameters):
        super(MixtureNormal, self).__init__(parameters)
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parameter_values)

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        mean = self.parameter_values
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
        return np.array(Data_array)

    def _check_parameters(self, parameters):
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for MixtureNormal has to be of type list.')
        return True

    def _check_parameters_fixed(self, parameters):
        return True

    def get_parameters(self):
        return super(MixtureNormal, self).get_parameters()

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        return super(MixtureNormal, self).fix_parameters(parameters, rng)

    def pdf(self, x):
        mean= self.parameter_values[:len(self.parameter_values)-1]
        cov_1 = np.identity(self.dimension)
        cov_2 = 0.01*cov_1
        return 0.5*(multivariate_normal(mean, cov_1).pdf(x))+0.5*(multivariate_normal(mean, cov_2).pdf(x))


class StudentT(ProbabilisticModel):
    """
    This class implements a probabilistic model following the Student-T distribution.

    Parameters
    ----------
    parameters: list
        If the list has two entries, the first entry contains the mean of the distribution, while the second entry             contains the degrees of freedom.
    """
    def __init__(self, parameters):
        super(StudentT, self).__init__(parameters)
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = 1

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        mean = self.parameter_values[0]
        df = self.parameter_values[1]
        return np.array((rng.standard_t(df,k)+mean).reshape(-1))

    def _check_parameters(self, parameters):
        if(not(isinstance(parameters, list))):
            raise TypeError('Input to StudentT has to be of type list.')
        if(len(parameters)>2):
            raise IndexError('Input to StudentT has to be of length 2 or smaller.')
        if(parameters[1]<=0):
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        length=0
        for parent in self.parents:
            if(isinstance(parent, ProbabilisticModel)):
                length+=parent.dimension
        if(length==len(parameters)):
            if(len(parameters)==2 and parameters[1]<=0):
                return False
            return True
        return False

    def get_parameters(self):
        return super(StudentT, self).get_parameters()

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        return super(StudentT, self).fix_parameters(parameters, rng)

    def pdf(self, x):
        df = self.parameter_values[1]
        x-=self.parameter_values[0] #divide by std dev if we include that
        return gamma((df+1)/2)/(np.sqrt(df*np.pi)*gamma(df/2)*(1+x**2/df)**((df+1)/2))


class MultiStudentT(ProbabilisticModel):
    """
    This class implements a probabilistic model following the multivariate Student-T distribution.

    Parameters
    ----------
    parameters: list
        All but the last two entries contain the probabilistic models and hyperparameters from which the model derives.        The second to last entry contains the covariance matrix. If the mean is of dimension n, the covariance matrix          is required to be nxn dimensional. The last entry contains the degrees of freedom.
    """
    def __init__(self, parameters):
        super(MultiStudentT, self).__init__(parameters)
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(self.parameter_values)-2

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        mean = self.parameter_values[:len(self.parameter_values)-2]
        cov = self.parameter_values[len(self.parameter_values)-2]
        df = self.parameter_values[len(self.parameter_values)-1]
        p = len(mean)
        if (df == np.inf):
            chis1 = 1.0
        else:
            chisq = rng.chisquare(df, k) / df
            chisq = chisq.reshape(-1, 1).repeat(p, axis=1)
        mvn = rng.multivariate_normal(np.zeros(p), cov, k)
        result = (mean + np.divide(mvn, np.sqrt(chisq)))
        return result

    def _check_parameters(self, parameters):
        length = len(parameters)-2
        cov = np.array(parameters[len(parameters)-2])
        if(not(length==len(cov[0]))):
            raise IndexError('Mean and covariance matrix have to be of same length.')
        if(parameters[len(parameters)-1]<=0):
            return False
        cov = np.array(cov)
        if(not(np.allclose(cov, cov.T, atol = 1e-3))):
            return False
        try:
            is_pos = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return False
        return True

    def _check_parameters_fixed(self, parameters):
        length = 0
        for parent in self.parents:
            if(isinstance(parent, ProbabilisticModel)):
                length+=parent.dimension
        if(length==len(parameters)):
            return True
        return False

    def get_parameters(self):
        return super(MultiStudentT, self).get_parameters()

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        return super(MultiStudentT, self).fix_parameters(parameters, rng)

    def pdf(self, x):
        mean = self.parameter_values[:len(self.parameter_values)-2]
        cov = self.parameter_values[len(self.parameter_values)-2]
        v = self.parameter_values[len(self.parameter_values)-1]
        mean = np.array(mean)
        cov = np.array(cov)
        p=len(mean)
        numerator = gamma((v + p) / 2)
        denominator = gamma(v / 2) * pow(v * np.pi, p / 2.) * np.sqrt(abs(np.linalg.det(cov)))
        normalizing_const = numerator / denominator
        tmp = 1 + 1 / v * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))
        density = normalizing_const * pow(tmp, -((v + p) / 2.))
        return density


class Uniform(ProbabilisticModel):
    """
    This class implements a probabilistic model following a uniform distribution

    Parameters
    ----------
    parameters: list
        Contains two lists. The first list specifies the probabilistic models and hyperparameters from which the lower         bound of the uniform distribution derive. The second list specifies the probabilistic models and hyperparameters        from which the upper bound derives.
    """
    def __init__(self, parameters):
        super(Uniform, self).__init__(parameters, True)
        #Parameter specifying the dimension of the return values of the distribution.
        self.dimension = len(parameters[0])

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        lower_bound = self.parameter_values[0]
        upper_bound = self.parameter_values[1]
        #print(lower_bound)
        samples = np.zeros(shape=(k, len(lower_bound))) #this means: len columns, and each has k entries
        for j in range(0, len(lower_bound)):
            samples[:, j] = rng.uniform(lower_bound[j], upper_bound[j], k)
        return samples

    def _check_parameters(self, parameters):
        if(not(isinstance(parameters, list))):
            raise TypeError('Input for Uniform has to be of type list.')
        if(len(parameters)<2):
            raise IndexError('Input to Uniform has to be at least of length 2.')
        if(not(isinstance(parameters[0], list))):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')
        if(not(isinstance(parameters[1], list))):
            raise TypeError('Each boundary for Uniform ahs to be of type list.')
        if(len(parameters[0])!=len(parameters[1])):
            raise IndexError('Lower and upper bound have to be of same length.')
        for i in range(len(parameters[0])):
            if(parameters[0][i]>parameters[1][i]):
                return False
        return True

    def _check_parameters_fixed(self, parameters):
        lb = []
        ub= []
        length_lb = 0
        length_ub = 0
        index=0
        for parent in self.parents[0]:
            if(isinstance(parent, ProbabilisticModel)):
                length_lb+=parent.dimension
                for j in range(parent.dimension):
                    lb.append(parameters[0][index])
                    index+=1
            else:
                lb.append(parent)
        index=0
        for parent in self.parents[1]:
            if(isinstance(parent, ProbabilisticModel)):
                length_ub+=parent.dimension
                for j in range(parent.dimension):
                    ub.append(parameters[1][index])
                    index+=1
            else:
                ub.append(parent)
        if(length_lb==len(parameters[0]) and length_ub==len(parameters[1])):
            for i in range(len(lb)):
                if(lb[i]>ub[i]):
                    return False
            return True
        return False

    def get_parameters(self):
        index=0
        lb_parameters = []
        for parent in self.parents[0]:
            if(isinstance(parent, ProbabilisticModel)):
                for j in range(parent.dimension):
                    lb_parameters.append(self.parameter_values[0][index])
                    index+=1
            else:
                index+=1
        index=0
        ub_parameters = []
        for parent in self.parents[1]:
            if(isinstance(parent, ProbabilisticModel)):
                for j in range(parent.dimension):
                    ub_parameters.append(self.parameter_values[1][index])
                    index+=1
            else:
                index+=1
        return [lb_parameters, ub_parameters]

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        if (not (parameters)):
            for s in range(2):
                index=0
                for i in range(len(self.parents[s])):
                    if (isinstance(self.parents[s][i], ProbabilisticModel)):
                        parameter_value = self.parents[s][i].sample_from_distribution(1, rng=rng)
                        if(isinstance(parameter_value[0], (list, np.ndarray))):
                            parameter_value=parameter_value[0]
                        for j in range(len(parameter_value)):
                            self.parameter_values[s][index] = parameter_value[j]
                            index += 1
                    else:
                        index += 1
            self.updated = True
        else:
            if(not(self._check_parameters_fixed(parameters))):
                return False
            index = 0
            i = 0
            for s in range(2):
                index=0
                i=0
                while (i < len(parameters[s])):
                    while (not (isinstance(self.parents[s][index], ProbabilisticModel)) or self.parents[s][index].updated):
                        index += 1
                    #NOTE THIS IS NOT POSSIBLE, BECAUSE AN EMPTY LIST IS GIVEN FOR FIXED WHOLE LISTS!
                    '''
                    if (not(parameters[i])):
                        parameter_value = self.parents[s][index].sample_from_distribution(1, rng=rng)
                        if(isinstance(parameter_value[0], (list, np.ndarray))):
                            parameter_value=parameter_value[0]
                        for j in range(len(parameter_value)):
                            self.parameter_values[s][index + j] = parameter_value[j]
                        index += len(parameter_value)
                        i += len(parameter_value)'''
                    for j in range(self.parents[s][index].dimension):
                        self.parameter_values[s][index + j] = parameters[s][i]
                        i += 1
                    index += self.parents[s][index].dimension
            self.updated = True
            return True


    def pdf(self, x):
        lower_bound = self.parents[0]
        upper_bound = self.parents[1]
        if (np.product(np.greater_equal(x, np.array(lower_bound)) * np.less_equal(x, np.array(upper_bound)))):
            pdf_value = 1. / np.product(np.array(upper_bound) - np.array(lower_bound))
        else:
            pdf_value = 0.
        return pdf_value

class StochLorenz95(ProbabilisticModel):
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
            Number of timesteps between [0,4], where 4 corresponds to 20 days. The default value is 160.
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
                parameter = [eta, np.array(self.parameter_values)]
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

    def get_parameters(self):
        return super(StochLorenz95, self).get_parameters()

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        return super(StochLorenz95, self).fix_parameters(parameters, rng=rng)

    def pdf(self, x):
        raise NotImplementedError


class Ricker(ProbabilisticModel):
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
        log_r = self.parameter_values[0]
        sigma = self.parameter_values[1]
        phi = self.parameter_values[2]
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

    def get_parameters(self):
        return super(Ricker, self).get_parameters()

    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        return super(Ricker, self).fix_parameters(parameters, rng=rng)

    def pdf(self, x):
        raise NotImplementedError