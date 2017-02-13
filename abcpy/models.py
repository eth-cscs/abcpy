from abc import ABCMeta, abstractmethod

import numpy as np
from abcpy.distributions import Distribution

class Model(metaclass = ABCMeta):
    """
    This abstract class represents the model and forces the
    implementation of certain methods required by the framework.
    """

    @abstractmethod
    def __init__(self, prior, seed = None): 
        """The constructor must be overwritten by a sub-class to initialize the model
        with a given prior.

        The standard behaviour is that concrete model parameters are sampled
        from the provided prior. However, it is alo possible for the constructor
        to provide optional (!) model parameters. In the latter case, the model
        should be initialized by the provided parameters instead from sampling
        from the prior.
    
        Parameters
        ----------
        prior: abcpy.distributions.Distribution
            A prior distribution
        seed: int, optional
            Optional initial seed for the random number generator that can be used in the model. The
            default value is generated randomly.
        """
        raise NotImplemented


    @abstractmethod
    def set_parameters(self, theta):
        """This method properly sets the parameters of the model and must be overwritten by a sub-class.

        Notes
        -----
        Make sure to test whether the provided parameters are
        compatible with the model. Return true if the parameters are accepted
        by the model and false otherwise. This behavior is expected e.g. by
        the inference schemes.

        Parameters
        ----------
        theta:
            An array-like structure containing the p parameter of the model,
            where theta[0] is the first and theta[p-1] is the last parameter.

        Returns
        -------
        boolean
            TRUE if model accepts the provided parameters, FALSE otherwise
        """

        raise NotImplemented


    @abstractmethod
    def sample_from_prior():
        """To be overwritten by any sub-class: should resample the model parameters
        from the prior distribution.
        """        
        raise NotImplemented

    
    @abstractmethod
    def simulate(self, k):
        """To be overwritten by any sub-class: should create k possible outcomes of
        the model using the current model parameters.

        Parameters
        ----------
        k: int
            Number of model outcomes to simulate
            
        Returns
        -------
        Python list 
            An array containing k realizations of the model
        """

        raise NotImplemented
    

    @abstractmethod
    def get_parameters(self):
        """To be overwritten by any sub-class: should extract the parameters from the model. 

        Returns
        -------
        numpy.ndarray 
            An array containing the p parameters of the model
        """

        raise NotImplemented


class Gaussian(Model):
    """This class implements the Gaussian model with unknown mean \
    :math:`\mu` and unknown standard deviation :math:`\sigma`.
    """

    def __init__(self, prior, mu = None, sigma = None, seed=None):
        """
        Parameters
        ----------
        prior: abcpy.distributions.Distribution
            Prior distribution
        mu: float, optional
            Mean of the Gaussian distribution. If the parameters is omitted, sampled
            from the prior.
        sigma: float, optional
            Standard deviation of the Gaussian distribution. If the parameters is omitted, sampled
            from the prior.
        seed: int, optional
            Initial seed. The default value is generated randomly.
        """

        # test prior
        if np.shape(prior.sample(1)) == (1,2):
            self.prior = prior
        else:
            raise ValueError("Prior generates values outside the model "
                             "parameter domain. ")

        # test provided model parameters
        if (mu == None) != (sigma == None):
            raise ValueError("Both or neither of the model parameters have to be provided.")

        # set model parameters directly if specified
        if mu != None and sigma != None:
            if self.set_parameters(np.array([mu, sigma])) == False:
                raise ValueError("The parameter values are out of the model parameter domain.")
        else:
            self.sample_from_prior()


        # set random number generator
        self.rng = np.random.RandomState(seed)

    def set_parameters(self, theta):
        if isinstance(theta, (list,np.ndarray)):
            theta = np.array(theta)
        else:
            raise TypeError('Theta is not of allowed types')

        if theta.shape[0] > 2: return False
        if theta[1] <= 0: return False
        self.mu = theta[0]
        self.sigma = theta[1]
        return True

    def get_parameters(self):
        return np.array([self.mu, self.sigma])

    def sample_from_prior(self):
        sample = self.prior.sample(1).reshape(-1)
        if self.set_parameters(sample) == False:
            raise ValueError("Prior generates values that are out the model parameter domain.")

    def simulate(self, k):
        return list((self.rng.normal(self.mu, self.sigma, k)).reshape(-1))



class Student_t(Model):
    """This class implements the Student_t distribution with unknown mean :math:`\mu` and unknown degrees of freedom.
    """
    def __init__(self, prior, mu=None, df=None, seed=None):
        """
        Parameters
        ----------
        prior: abcpy.distributions.Distribution
            Prior distribution        
        mu: float, optional
            Mean of the Stundent_t distribution. If the parameters is omitted, sampled
            from the prior.
        df: float, optional
            The degrees of freedom of the Student_t distribution. If the parameters is omitted, sampled
            from the prior.
        seed: int, optional
            Initial seed. The default value is generated randomly. 
        """
        assert(not (mu == None) != (df == None))
        self.prior = prior
        if mu == None and df == None:
            self.sample_from_prior()
        else:
            if self.set_parameters(np.array([mu, df])) == False:
                raise ValueError("The parameter values are out of the model parameter domain.")
                
        if not isinstance(prior, Distribution):
            raise TypeError('Prior is not of our defined Prior class type')

        self.rng = np.random.RandomState(seed)
        

    def sample_from_prior(self):
        sample = self.prior.sample(1).reshape(-1)
        if self.set_parameters(sample) == False:
            raise ValueError("Prior generates values that are out the model parameter domain.")         
            
    def simulate(self,k):
        return list((self.rng.standard_t(self.df,k)+self.mu).reshape(-1))

    def get_parameters(self):
        return np.array([self.mu, self.df])

    def set_parameters(self,theta):
        if isinstance(theta, (list,np.ndarray)):
            theta = np.array(theta)
        else:
            raise TypeError('Theta is not of allowed types')

        if theta.shape[0] > 2: return False
        if theta[1] <= 0: return False        
        self.mu = theta[0]
        self.df = theta[1]                      
        return True
        
class MixtureNormal(Model):
    """This class implements the Mixture of multivariate normal ditribution with unknown mean \
    :math:`\mu` described as following,
    :math:`x|\mu \sim 0.5\mathcal{N}(\mu,I_p)+0.5\mathcal{N}(\mu,0.01I_p)`, where :math:`x=(x_1,x_2,\ldots,x_p)` is the 
    dataset simulated from the model and mean is :math:`\mu=(\mu_1,\mu_2,\ldots,\mu_p)`.            """
    def __init__(self, prior, mu, seed = None):
        """
        Parameters
        ----------
        prior: abcpy.distributions.Distribution
            Prior distribution        
        mu: numpy.ndarray or list, optional   
            Mean of the mixture normal. If the parameter is omitted, sampled
            from the prior. 
        seed: int, optional
            Initial seed. The default value is generated randomly. 
        """    
        # Assign prior
        if not isinstance(prior, Distribution):
            raise TypeError('Prior is not of our defined Prior class type') 
        else:
            self.prior = prior
        # Assign parameters
        if isinstance(mu, (list,np.ndarray)):
            if self.set_parameters(mu) == False:
                raise ValueError("The parameter values are out of the model parameter domain.")   
        else:
            raise TypeError('The parameter theta is not of allowed types')
        # Initialize random number generator with provided seed, if None initialize with present time.
        self.rng = np.random.RandomState(seed)
                    
    def sample_from_prior(self):
        sample = self.prior.sample(1).reshape(-1)
        if self.set_parameters(sample) == False:
            raise ValueError("Prior generates values that are out the model parameter domain.")                

    def simulate(self, n_simulate):
        # Generate n_simulate np.ndarray from mixture_normal
        Data_array = [None]*n_simulate
        # Initialize local parameters
        dimension = self.mu.shape[0]
        index_array = self.rng.binomial(1, 0.5, n_simulate)
        for k in range(0,n_simulate):   
            # Initialize the time-series
            index = index_array[k]
            Data = index*self.rng.multivariate_normal(mean = self.mu, cov = np.identity(dimension)) \
                + (1-index)*self.rng.multivariate_normal(mean = self.mu, cov = 0.01*np.identity(dimension))
            Data_array[k] = Data
        # return an array of objects of type Timeseries
        return Data_array
    
    def get_parameters(self):
        return self.mu

    def set_parameters(self, mu):
        if isinstance(mu, (list,np.ndarray)):
            self.mu = np.array(mu)
        else:
            raise TypeError('The parameter value is not of allowed types')
            
        return True        
        
class StochLorenz95(Model):
    """Generates time dependent 'slow' weather variables following forecast model of Wilks [1], 
    a stochastic reparametrization of original Lorenz model Lorenz [2]. 
    
    [1] Wilks, D. S. (2005). Effects of stochastic parametrizations in the lorenz ’96 system. 
    Quarterly Journal of the Royal Meteorological Society, 131(606), 389–407.     

    [2] Lorenz, E. (1995). Predictability: a problem partly solved. In Proceedings of the 
    Seminar on Predictability, volume 1, pages 1–18. European Center on Medium Range
    Weather Forecasting, Europe             
    """    
    
    def __init__(self, prior, theta, initial_state = None, n_timestep = 160, seed = None):
        """
        Parameters
        ----------
        prior: abcpy.distributions.Distribution
            Prior distribution          
        theta: list or numpy.ndarray, optional       
            Closure parameters. If the parameter is omitted, sampled
            from the prior. 
        initial_state: numpy.ndarray, optional
            Initial state value of the time-series, The default value is None, which assumes a previously computed 
            value from a full Lorenz model as the Initial value. 
        n_timestep: int, optional
            Number of timesteps between [0,4], where 4 corresponds to 20 days. The default value is 160.
        seed: int, optional
            Initial seed. The default value is generated randomly.     
        """        
        
        # Assign prior
        if not isinstance(prior, Distribution):
            raise TypeError('Prior is not of our defined Prior class type') 
        else:
            self.prior = prior
        # Assign number of tmestep    
            self.n_timestep = n_timestep
        # Assign initial state    
        if not initial_state == None:
            self.initial_state = initial_state
        else:
            self.initial_state = np.array([6.4558,1.1054,-1.4502,-0.1985,1.1905,2.3887,5.6689,6.7284,0.9301, \
            4.4170,4.0959,2.6830,4.7102,2.5614,-2.9621,2.1459,3.5761,8.1188,3.7343,3.2147,6.3542, \
            4.5297,-0.4911,2.0779,5.4642,1.7152,-1.2533,4.6262,8.5042,0.7487,-1.3709,-0.0520, \
            1.3196,10.0623,-2.4885,-2.1007,3.0754,3.4831,3.5744,6.5790])
        # Assign closure parameters
        if isinstance(theta, (list, np.ndarray)):
            if self.set_parameters(theta) == False:
                raise ValueError("The parameter values are out of the model parameter domain.")
        else:
            raise TypeError('The parameter theta is not of allowed types')
        # Other parameters kept fixed
        self.F = 10
        self.sigma_e = 1
        self.phi = 0.4                               
        # Initialize random number generator with provided seed, if None initialize with present time.
        self.rng = np.random.RandomState(seed)     
                   
    def sample_from_prior(self): 
        sample = self.prior.sample(1).reshape(-1)
        if self.set_parameters(sample) == False:
            raise ValueError("Prior generates values that are out the model parameter domain.")                

    def simulate(self, n_simulate):
        # Generate n_simulate time-series of weather variables satisfying Lorenz 95 equations
        timeseries_array = [None]*n_simulate   
        
        # Initialize timesteps
        time_steps = np.linspace(0, 4, self.n_timestep)
        
        for k in range(0,n_simulate):   
            # Define a parameter object containing parameters which is needed 
            # to evaluate the ODEs
            # Stochastic forcing term 
            eta = self.sigma_e*np.sqrt(1-pow(self.phi,2))*self.rng.normal(0, 1, self.initial_state.shape[0])

            # Initialize the time-series
            timeseries = np.zeros(shape = (self.initial_state.shape[0], self.n_timestep), dtype=np.float)
            timeseries[:,0] = self.initial_state
            # Compute the timeseries for each time steps
            for ind in range(0, self.n_timestep-1):
                # parameters to be supplied to the ODE solver            
                parameter = [eta, self.get_parameters()]
                # Each timestep is computed by using a 4th order Runge-Kutta solver
                x = self._rk4ode(self._l95ode_par, np.array([time_steps[ind], time_steps[ind+1]]), timeseries[:,ind], parameter)
                timeseries[:,ind+1] = x[:,-1]
                # Update stochastic foring term
                eta = self.phi*eta+self.sigma_e*np.sqrt(1-pow(self.phi,2))*self.rng.normal(0, 1)
            timeseries_array[k] = timeseries
        # return an array of objects of type Timeseries
        return timeseries_array
    
    def get_parameters(self):
        return self.theta

    def set_parameters(self, theta):
        if isinstance(theta, (list,np.ndarray)):
            self.theta = np.array(theta)
        else:
            raise TypeError('The parameter value is not of allowed types')
        
        return True
        
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
        X = np.ones(shape = (x.shape[0],1))          
        for ind in range(1,degree):
            X = np.column_stack((X,pow(x,ind)))
            
        # deterministic reparametrization term
        # ------------------------------------    
        gu = np.sum(X*theta,1)
      
        # ODE definition of the slow variables
        # ------------------------------------        
        dx[0] = -x[-2]*x[-1] + x[-1]*x[1] - x[0] + self.F - gu[0] + eta[0]
        dx[1] = -x[-1]*x[0] + x[0]*x[2] - x[1] + self.F - gu[1] + eta[1]
        for ind in range(2, x.shape[0] - 2):
            dx[ind] = -x[ind-2]*x[ind-1] + x[ind-1]*x[ind+1] - x[ind] + self.F - gu[ind] + eta[ind]
        dx[-1] = -x[-3]*x[-2]+x[-2]*x[1] - x[-1] + self.F - gu[-1] + eta[-1]

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
        
        timeseries = np.zeros(shape = (timeseries_initial.shape[0], timespan.shape[0]))
        timeseries[:,0] = timeseries_initial

        for ind in range(0, timespan.shape[0]-1):
            time_diff = timespan[ind+1]-timespan[ind] 
            time_mid_point = timespan[ind] + time_diff/2
            k1 = time_diff*ode(timespan[ind], timeseries_initial, parameter)
            k2 = time_diff*ode(time_mid_point, timeseries_initial + k1/2, parameter)
            k3 = time_diff*ode(time_mid_point, timeseries_initial + k2/2, parameter)
            k4 = time_diff*ode(timespan[ind+1], timeseries_initial + k3, parameter)
            timeseries_initial = timeseries_initial + (k1 + 2*k2 + 2*k3 + k4)/6
            timeseries[:,ind+1] = timeseries_initial
        # Return the solved timeseries at the values in timespan    
        return timeseries    
class Ricker(Model):
    """Ecological model that describes the observed size of animal population over time 
    described in [1].
        
    [1] S. N. Wood. Statistical inference for noisy nonlinear ecological 
    dynamic systems. Nature, 466(7310):1102–1104, Aug. 2010.
    """

    def __init__(self, prior, theta=None, n_timestep = 100, seed = None):
        """
        Parameters
        ----------
        prior: abcpy.distributions.Distribution
            Prior distribution          
        theta: list or numpy.ndarray, optional       
            The parameter is a vector consisting of three numbers \
            :math:`\log r` (real number), :math:`\sigma` (positive real number, > 0), :math:`\phi` (positive real number > 0)
            If the parameter is ommitted, sampled from the prior.
        n_timestep: int, optional
            Number of timesteps. The default value is 100.
        seed: int, optional
            Initial seed. The default value is generated randomly.             
        """    
        # Assign prior
        if not isinstance(prior, Distribution):
            raise TypeError('Prior is not of our defined Prior class type') 
        else:
            self.prior = prior
        # Assign number of tmestep    
            self.n_timestep = n_timestep
        # Assign parameters
        if isinstance(theta, (list, np.ndarray)):
            if self.set_parameters(theta) == False:
                raise ValueError("The parameter values are out of the model parameter domain.")   
        # Initialize random number generator with provided seed, if None initialize with random seed.
        self.rng = np.random.RandomState(seed)
                    
    def sample_from_prior(self):
        sample = self.prior.sample(1).reshape(-1)
        if self.set_parameters(sample) == False:
            raise ValueError("Prior generates values that are out the model parameter domain.")                

    def simulate(self, n_simulate):
        timeseries_array = [None]*n_simulate
        # Initialize local parameters
        log_r = self.theta[0]
        sigma = self.theta[1]
        phi = self.theta[2]
        for k in range(0,n_simulate):   
            # Initialize the time-series
            timeseries_obs_size = np.zeros(shape = (self.n_timestep), dtype=np.float)
            timeseries_true_size = np.ones(shape = (self.n_timestep), dtype=np.float)
            for ind in range(1,self.n_timestep-1):
                timeseries_true_size[ind] = np.exp(log_r+np.log(timeseries_true_size[ind-1])-timeseries_true_size[ind-1]+sigma*self.rng.normal(0, 1))
                timeseries_obs_size[ind] = self.rng.poisson(phi*timeseries_true_size[ind])
            timeseries_array[k] = timeseries_obs_size
        # return an array of objects of type Timeseries
        return timeseries_array
    
    def get_parameters(self):
        return self.theta

    def set_parameters(self, theta):
        if isinstance(theta, (list,np.ndarray)):
                theta = np.array(theta)
        else:
            raise TypeError('The parameter value is not of allowed types')            
            
        if theta.shape[0] > 3: return False
        if theta[1] <= 0: return False
        if theta[2] <= 0: return False
        self.theta = theta
        return True
