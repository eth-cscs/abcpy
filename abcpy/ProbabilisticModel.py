from abc import ABCMeta, abstractmethod
import numpy as np


class ProbabilisticModel(metaclass = ABCMeta):
    def __init__(self, parameters):
        self.parents = parameters #all parents of the node
        self.updated = False #whether the node has been touched

    #self.value can be multidimensional, ie 3 or more values
    def fix_parameters(self, parameters=None, rng=np.random.RandomState()):
        #if we have no input parameters, we want to set our own value to a newly sampled one
        if(not(parameters)):
            self.value = self.sample_from_distribution(1, rng)
            if(isinstance(self.value[0],np.ndarray)):
                self.value = self.value[0]
        else:
            index=0
            i=0
            while(i<len(parameters)):
                while(not(isinstance(self.parents[index],ProbabilisticModel)) or self.parents[index].updated):
                    index+=1
                if(not(parameters[i])):
                    self.parents[index].fix_parameters(rng=rng)
                    self.parents[index].updated = True
                    index+=1
                    i+=1
                else:
                    for j in range(len(self.parents[index].value)):
                        self.parents[index].value[j] = parameters[i]
                        i+=1
                    self.parents[index].updated = True
                    index+=1

    def get_parameters(self):
        parameters = []
        for i in range(len(self.parents)):
            if(isinstance(self.parents[i],ProbabilisticModel)):
                for j in range(len(self.parents[i].value)):
                    parameters.append(self.parents[i].value[j])
        return parameters



    @abstractmethod
    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        raise NotImplementedError


class Normal(ProbabilisticModel):
    def __init__(self, parameters):
        super(Normal, self).__init__(parameters)
        self.value = self.sample_from_distribution(1)

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        if(len(self.parents)==1):
            mu = self.parents[0].value[0]
            sigma = self.parents[0].value[1]
        else:
            if(isinstance(self.parents[0], ProbabilisticModel)):
                mu = self.parents[0].value[0]
            else:
                mu = self.parents[0]
            if(isinstance(self.parents[1],ProbabilisticModel)):
                sigma = self.parents[1].value[0]
            else:
                sigma = self.parents[1]
        return np.array(rng.normal(mu, sigma, k).reshape(-1))


class MultivariateNormal(ProbabilisticModel):
    def __init__(self,parameters):
        super(MultivariateNormal, self).__init__(parameters)
        self.value = self.sample_from_distribution(1)[0] #else [[]], is this correct?

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        mean = []
        for i in range(len(self.parents)-1):
            if(isinstance(self.parents[i], ProbabilisticModel)):
                helper = self.parents[i].value #we get back the value, but the value is a list, containing an array?
                for element in helper:
                    mean.append(element)
            else:
                mean.append(self.parents[i])
        if(isinstance(self.parents[len(self.parents)-1],ProbabilisticModel)):
            cov = self.parents[len(self.parents)-1].value
        else:
            cov = self.parents[len(self.parents)-1]
        return rng.multivariate_normal(mean, cov, k)


class MixtureNormal(ProbabilisticModel):
    def __init__(self, parameters):
        super(MixtureNormal, self).__init__(parameters)
        self.value = self.sample_from_distribution(1)[0]

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        mean = []
        for i in range(len(self.parents)):
            if(isinstance(self.parents[i], ProbabilisticModel)):
                helper = self.parents[i].value
                for element in helper:
                    mean.append(element)
            else:
                mean.append(self.parents[i])
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


class StudentT(ProbabilisticModel):
    def __init__(self, parameters):
        super(StudentT, self).__init__(parameters)
        self.value = self.sample_from_distribution(1)

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        if(len(self.parents)==1):
            mean = self.parents[0].value[0]
            df = self.parents[0].value[1]
        else:
            if(isinstance(self.parents[0],ProbabilisticModel)):
                mean = self.parents[0].value[0]
            else:
                mean = self.parents[0]
            if(isinstance(self.parents[1], ProbabilisticModel)):
                df = self.parents[1].value[0]
            else:
                df = self.parents[1]
        return np.array((rng.standard_t(df,k)+mean).reshape(-1))


class MultiStudentT(ProbabilisticModel):
    def __init__(self, parameters):
        super(MultiStudentT, self).__init__(parameters)
        self.value = self.sample_from_distribution(1)[0]

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        mean = []
        for i in range(len(self.parents)-2):
            if(isinstance(self.parents[i],ProbabilisticModel)):
                helper = self.parents[i].value
                for element in helper:
                    mean.append(element)
            else:
                mean.append(self.parents[i])
        if(isinstance(self.parents[len(self.parents)-2],ProbabilisticModel)):
            cov = self.parents[len(self.parents)-2].value
        else:
            cov = self.parents[len(self.parents)-2]
        if(isinstance(self.parents[len(self.parents)-1],ProbabilisticModel)):
            df = self.parents[len(self.parents)-1].value
        else:
            df = self.parents[len(self.parents)-1]
        p = len(mean)
        if (df == np.inf):
            chis1 = 1.0
        else:
            chisq = rng.chisquare(df, k) / df
            chisq = chisq.reshape(-1, 1).repeat(p, axis=1)
        mvn = rng.multivariate_normal(np.zeros(p), cov, k)
        result = (mean + np.divide(mvn, np.sqrt(chisq)))
        return result


class Uniform(ProbabilisticModel):
    def __init__(self, parameters):
        super(Uniform, self).__init__(parameters)
        self.value = self.sample_from_distribution(1)[0]

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        lower_bound = []
        for i in range(len(self.parents[0])):
            if(isinstance(self.parents[0][i], ProbabilisticModel)):
                helper = self.parents[0][i].value
                for element in helper:
                    lower_bound.append(element)
            else:
                lower_bound.append(self.parents[0][i])
        upper_bound = []
        for i in range(len(self.parents[1])):
            if(isinstance(self.parents[1][i], ProbabilisticModel)):
                helper = self.parents[1][i].value
                for element in helper:
                    upper_bound.append(element)
            else:
                upper_bound.append(self.parents[1][i])
        samples = np.zeros(shape=(k, len(lower_bound)))
        for j in range(0, len(lower_bound)):
            samples[:, j] = rng.uniform(lower_bound[j], upper_bound[j], k)
        return samples

#TODO cannot find where self.value or values of parents would be used???
class StochLorenz95(ProbabilisticModel):
    def __init__(self, parameters, initial_state= None, n_timestep=160):
        super(StochLorenz95, self).__init__(parameters)
        self.value = self.sample_from_distribution(1)
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
        # Other parameters kept fixed
        self.F = 10
        self.sigma_e = 1
        self.phi = 0.4

    #TODO this uses get and set parameters -> we need to fix/change those because this will give something different I think?
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
                parameter = [eta, self.get_parameters()]
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


class Ricker(ProbabilisticModel):
    def __init__(self, parameters, n_timestep=100):
        super(Ricker, self).__init__(parameters)
        self.n_timestep = n_timestep
        self.value = self.sample_from_distribution(1)

    def sample_from_distribution(self, k, rng=np.random.RandomState()):
        timeseries_array = [None] * k
        # Initialize local parameters
        if(isinstance(self.parents[0], ProbabilisticModel)):
            log_r = self.parents[0].value[0]
        else:
            log_r = self.parents[0]
        if(isinstance(self.parents[1], ProbabilisticModel)):
            sigma = self.parents[1].value[0]
        else:
            sigma = self.parents[1]
        if(isinstance(self.parents[2],ProbabilisticModel)):
            phi = self.parents[2].value[0]
        else:
            phi = self.parents[2]
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