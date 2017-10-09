from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import gamma



class Distribution(metaclass=ABCMeta):
    """
    This abstract base class represents a distribution. It can be used e.g. as a
     for models.

    """

    @abstractmethod
    def set_parameters(self, params):
        """To be overwritten by any sub-class: should set the parameters of the distribution.

        Parameters
        ----------
        params: list
            Contains all the distributions parameters.

        """

        raise NotImplementedError

    @abstractmethod
    def reseed(self, seed):
        """To be overwritten by any sub-class: reseed the random number generator with provided seed.

        Parameters
        ----------
        seed: integer
            New seed for the random number generator

        """

        raise NotImplementedError

    @abstractmethod
    def simulate(self, k, reset=0):
        """To be overwritten by any sub-class: should simulate k points from the implemented distribution.

        Parameters
        ----------
        k: integer
            The number of points to be sampled
        Returns
        -------
        np.ndarray
            kxp matrix containing k samples of p-dimensional points

        """

        raise NotImplementedError

    @abstractmethod
    def pdf(self, x):
        """To be overwritten by any sub-class: calculate the -denisty at *x*, where
        *x* is a parameter of dimension p.

        Parameters
        ----------
        x: np.ndarray
            A p-dimensional point from the support of the distribution
        Returns
        -------
        float
            The probability density for point x

        """

        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError



class Normal(Distribution):
    '''
    Parameters
    ----------
    mean: float or 1D distribution
        The mean of the distribution.
    var: float or 1D distribution
        Sigma of the distribution
    seed: int
        The seed to be used by the random number generator.
    '''

    def __init__(self, mean, var, seed=None):
        if(not(self.check_parameters(mean, var))):
            raise IndexError('Mean and var do not have matching dimensions.')
        self.mean = mean
        self.var = var
        self.rng = np.random.RandomState(seed)

    def set_parameters(self, params):
        '''
        Sets the parameters for the distribution.
        Parameters
        ----------
        params: list
            The first element of the list specifies the mean of the distribution in float or as a 1D distribution. The second element specifies the sigma of the distribution in float or as a 1D distribution.
        '''
        if(not(self.check_parameters(params[0],params[1]))):
            raise IndexError('Mean and var do not have matching dimensions.')
        self.mean = params[0]
        self.var = params[1]

    def reseed(self, seed):
        self.rng.seed(seed)

    def simulate(self, k, reset=0):
        '''
        Samples k values from the distribution.
        Parameters
        ----------
        k: int
            The number of samples which should be returned.
        reset: 0 or 1
            Specify whether the the random number generator should be reset after sampling
        Returns
        -------
        np.ndarray:
            The results of the sampling.
        '''
        if(isinstance(self.mean, Distribution)):
            mean = self.mean.simulate(1)[0]
        else:
            mean = self.mean
        if(isinstance(self.var, Distribution)):
            var = self.var.simulate(1)[0]
        else:
            var = self.var
        if(reset==1):
            rng_state = self.rng.get_state()
        result = (self.rng.normal(mean, var, k)).reshape(-1)
        if(reset == 1):
            self.rng.set_state(rng_state)
        return np.array(result)

    def pdf(self, x):
        if(isinstance(self.mean, Distribution) or isinstance(self.var, Distribution)):
            raise TypeError('Mean and Variance are not allowed to be of type distribution')
        else:
            return norm(self.mean, self.var).pdf(x)

    def get_parameters(self):
        if(isinstance(self.mean,Distribution)):
            simulated_mean = self.mean.simulate(10)
            simulated_mean = np.mean(simulated_mean)
        else:
            simulated_mean = self.mean
        if(isinstance(self.var, Distribution)):
            simulated_var = self.var.simulate(10)
        else:
            simulated_var = self.var
        return np.array([simulated_mean, simulated_var])

    def check_parameters(self, mean, var):
        if(hasattr(mean, '__len__')):
            if(len(mean)==1):
                return True
            return False
        return True

class MultiNormal(Distribution):
    '''
    Parameters
    ----------
    mean: p-dimensional list or distribution
        Defines the mean of the distribution.
    cov: pxp dimensional list
        Defines the covariance matrix of the distribution
    seed: int
        The initial seed to be used by the random number generator
    '''
    def __init__(self, mean, cov, seed=None):
        if(not(self.check_parameters(mean, cov))):
            return IndexError('Mean and cov do not have matching dimensions')
        self.mean = mean
        self.cov = cov
        self.rng = np.random.RandomState(seed)

    def set_parameters(self, params):
        '''
        Sets the parameters for the distribution.
        Parameters
        ----------
        params: list
            The first element of the list specifies the mean of the distribution as a p-dimensional list or distribution. The second element specifies the sigma of the distribution as a pxp dimensional list.
        '''
        if(not(self.check_parameters(params[0], params[1]))):
            return IndexError('Mean and cov do not have matching dimensions')
        self.mean = params[0]
        self.cov = params[1]

    def reseed(self,seed):
        self.rng.seed(seed)

    def simulate(self, k, reset=0):
        '''
        Samples k values from the distribution.
        Parameters
        ----------
        k: int
            The number of samples which should be returned.
        reset: 0 or 1
            Specify whether the the random number generator should be reset after sampling
        Returns
        -------
        np.ndarray:
            The results of the sampling.
        '''
        mean = []
        cov = [[0] * len(self.cov[0]) for i in range(len(self.cov[0]))]
        if(isinstance(self.mean, Distribution)):
            mean = self.mean.simulate(1)[0]
        else:
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i],Distribution)):
                    next_element = (self.mean[i].simulate(1)[0])
                    if(isinstance(next_element,np.ndarray)):
                        for j in range(len(next_element)):
                            mean.append(next_element[j])
                    else:
                        mean.append(next_element)
                else:
                    mean.append(self.mean[i])
        for i in range(len(self.cov)):
            for j in range(len(self.cov[i])):
                if(isinstance(self.cov[i][j], Distribution)):
                    cov[i][j] = self.cov[i][j].simulate(1)[0]
                else:
                    cov[i][j] = self.cov[i][j]

        #why did we reshape for the normal, but not for this, this is a multidimensional list
        #samples = [[0]*k for i in range len(mean)]
        #for i in range(len(mean)):
        #    samples[i] = self.rng.multivariate_normal(mean,cov, k)
        if(reset==1):
            rng_state = self.rng.get_state()
        result = self.rng.multivariate_normal(mean, cov, k)
        if(reset==1):
            self.rng.set_state(rng_state)
        return result

    def get_parameters(self):
        if(isinstance(self.mean,Distribution)):
            simulated_mean = self.mean.simulate(10)
            simulated_mean = np.mean(simulated_mean, axis=0)
        else:
            simulated_mean = []
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i], Distribution)):
                    simulated_value = self.mean.simulate(10)
                    simulated_value = np.mean(simulated_value, axis=0)
                    simulated_mean.append(simulated_value)
                else:
                    simulated_mean.append(self.mean[i])
        return np.array(simulated_mean)#np.array([simulated_mean, self.cov])

    def pdf(self, x):
        if(not(isinstance(self.mean, Distribution))):
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i],Distribution)):
                    raise TypeError('All elements of mean are not allowed to be of type distribution')
        for i in range(len(self.cov)):
            for j in range(len(self.cov[i])):
                if(isinstance(self.cov[i][j], Distribution)):
                    raise TypeError('All elements of cov are not allowed to be of type distribution')
        return multivariate_normal(self.mean, self.cov).pdf(x)

    def check_parameters(self, mean, cov):
        if(not(isinstance(mean,Distribution))):
            length = 0
            for i in range(len(mean)):
                if(isinstance(mean[i], Distribution)):
                    value = mean[i].simulate(1,reset=1)[0]
                    if(isinstance(value,np.ndarray)):
                        length+=len(value)
                    else:
                        length+=1
                else:
                    length+=1
            if(length==len(cov)):
                return True
        if(isinstance(mean,Distribution)):
            if(len(mean.simulate(1,reset=1)[0])==len(cov)):
                return True
        return False

#NOTE can we really give a distribution for the degrees of freedom, and if so, can we do the same thing for the multistudenT?
class StudentT(Distribution):
    '''
    Parameters
    ----------
    mu: float or 1D distribution
        Defines the mean of the distribution
    df: int
        Defines the degrees of freedom of the distribution
    seed: int
        The initial seed to be used by the random number generator.
    '''
    def __init__(self, mu, df, seed=None):
        self.mean = mu
        self.df = df
        self.rng = np.random.RandomState(seed)

    def set_parameters(self, params):
        self.mean = params[0]
        self.df = params[1]

    def reseed(self, seed):
        self.rng.seed(seed)

    def simulate(self, k, reset=0):
        if(isinstance(self.mean, Distribution)):
            mean = self.mean.simulate(1)[0]
            if(isinstance(mean,np.ndarray)):
                mean = mean[0]
        else:
            mean = self.mean
        if(isinstance(self.df, Distribution)):
            df = self.df.simulate(1)[0]
        else:
            df = self.df
        if(reset==1):
            rng_state = self.rng.get_state()
        result = (self.rng.standard_t(df, k) + mean).reshape(-1)
        if(reset==1):
            self.rng.set_state(rng_state)
        return np.array(result)

    def get_parameters(self):
        if(isinstance(self.mean, Distribution)):
            simulated_mean = self.mean.simulate(10)
            simulated_mean = np.mean(simulated_mean)
        else:
            simulated_mean = self.mean
        return np.array([simulated_mean, self.df])

    def pdf(self, x):
        if(isinstance(self.df, Distribution)):
            return TypeError("Degrees of freedom is not allowed to be of type distribution")
        return gamma((self.df+1)/2.)/(np.sqrt(self.df*np.pi)*gamma(self.df/2.))*(1+float(x**2)/self.df)**(-(self.df+1)/2.)

class MultiStudentT(Distribution):
    def __init__(self, mean, cov, df, seed=None):
        if(not(self.check_parameters(mean, cov))):
            raise IndexError('Mean and cov do not have matching dimensions.')
        self.mean = mean
        self.cov = cov
        self.df = df
        self.rng = np.random.RandomState(seed)

    def set_parameters(self, params):
        if(not(self.check_parameters(params[0], params[1]))):
            raise IndexError('Mean and cov do not have matching dimensions.')
        self.mean = params[0]
        self.cov = params[1]

        if(len(params)==3):
            self.df = params[2]

    def reseed(self, seed):
        self.rng.seed(seed)

    def simulate(self, k, reset=0):
        mean = []
        if(isinstance(self.mean, Distribution)):
            mean = self.mean.simulate(1)[0]
        else:
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i],Distribution)):
                    next_element = (self.mean[i].simulate(1)[0])
                    if(isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            mean.append(next_element[j])
                    else:
                        mean.append(next_element)
                else:
                    mean.append(self.mean[i])
        cov = [[0] * len(self.cov[0]) for i in range(len(self.cov[0]))]
        for i in range(len(self.cov)):
            for j in range(len(self.cov[i])):
                if(isinstance(self.cov[i][j], Distribution)):
                    cov[i][j] = self.cov[i][j].simulate(1)[0]
                else:
                    cov[i][j] = self.cov[i][j]
        p = len(mean)
        if (self.df == np.inf):
            chis1 = 1.0
        else:
            chisq = self.rng.chisquare(self.df, k) / self.df
            chisq = chisq.reshape(-1, 1).repeat(p, axis=1)
        mvn = self.rng.multivariate_normal(np.zeros(p), cov, k)
        if(reset==1):
            rng_state = self.rng.get_state()
        result = (mean + np.divide(mvn, np.sqrt(chisq)))
        if(reset==1):
            self.rng.set_state(rng_state)
        return result

    def get_parameters(self):
        if(isinstance(self.mean, Distribution)):
            simulated_mean = self.mean.simulate(10)
            simulated_mean = np.mean(simulated_mean, axis=0)
        else:
            simulated_mean = []
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i], Distribution)):
                    simulated_value = self.mean[i].simulate(10)
                    simulated_value = np.mean(simulated_value, axis=0)
                    simulated_mean.append(simulated_value)
                else:
                    simulated_mean.append(self.mean[i])
        return np.array(simulated_mean)

    def pdf(self, x):
        if(not(isinstance(self.mean, Distribution))):
            for i in range(len(self.mean)):
                if (isinstance(self.mean[i], Distribution)):
                    print("Mean is not allowed to be of type Distribution")
        cov = [[0] * len(self.cov[0]) for i in range(len(self.cov[0]))]
        for i in range(len(self.cov)):
            for j in range(len(self.cov[i])):
                if (isinstance(self.cov[i][j], Distribution)):
                    print("None of the elements of the covariance matrix are allowed to be of type distribution")

        mean = self.mean
        cov = self.cov
        v = self.df
        p = len(mean)

        numerator = gamma((v + p) / 2)
        denominator = gamma(v / 2) * pow(v * np.pi, p / 2.) * np.sqrt(abs(np.linalg.det(cov)))
        normalizing_const = numerator / denominator
        tmp = 1 + 1 / v * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))
        density = normalizing_const * pow(tmp, -((v + p) / 2.))
        return density

    def check_parameters(self, mean, cov):
        if(isinstance(mean, Distribution)):
            value = mean.simulate(1,reset=1)[0]
            if(isinstance(value, np.ndarray)):
                if(len(value)==len(cov)):
                    return True
            else:
                return len(cov)==1
        else:
            length = 0
            for i in range(len(mean)):
                if(isinstance(mean[i],Distribution)):
                    value = mean[i].simulate(1,reset=1)[0]
                    if(isinstance(value,np.ndarray)):
                        length+=len(value)
                    else:
                        length+=1
                else:
                    length += 1
            if(length==len(cov)):
                return True
        return False


class Uniform(Distribution):
    def __init__(self, lb, ub, seed=None):
        if(not(self.check_parameters(lb,ub))):
            raise IndexError('Lower and upper bound do not have matching dimensions.')
        self.lb = lb
        self.ub = ub
        self.rng = np.random.RandomState(seed)

    def set_parameters(self, params):
        if(not(self.check_parameters(params[0],params[1]))):
            raise IndexError('Lower and upper bound do not have matching dimensions.')
        self.lb = params[0]
        self.ub = params[1]

    def reseed(self, seed):
        self.rng.seed(seed)

    def simulate(self, k, reset=0):
        lb = []
        ub = []
        if(isinstance(self.lb, Distribution)):
            lb = self.lb.simulate(1)[0]
        else:
            for i in range(len(self.lb)):
                if(isinstance(self.lb[i], Distribution)):
                    next_element = (self.lb[i].simulate(1)[0])
                    if(isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            lb.append(next_element[j])
                    else:
                        lb.append(next_element)
                else:
                    lb.append(self.lb[i])

        if(isinstance(self.ub, Distribution)):
            ub = self.ub.simulate(1)[0]
        else:
            for i in range(len(self.ub)):
                if(isinstance(self.ub[i], Distribution)):
                    next_element = (self.ub[i].simulate(1)[0])
                    if(isinstance(next_element,np.ndarray)):
                        for j in range(len(next_element)):
                            ub.append(next_element[j])
                    else:
                        ub.append(next_element)
                else:
                    ub.append(self.ub[i])
        samples = [[0]*len(lb) for i in range(k)]
        if(reset==1):
            rng_state = self.rng.get_state()
        for i in range(k):
            for j in range(len(lb)):
                samples[i][j] = self.rng.uniform(lb[j],ub[j],1).tolist()[0]
        if(reset==1):
            self.rng.set_state(rng_state)
        return np.array(samples)

    def get_parameters(self):
        if(isinstance(self.lb,Distribution)):
            simulated_lower_tmp = self.lb.simulate(10)
            simulated_lower = np.mean(simulated_lower_tmp, axis=0)
        else:
            simulated_lower = []
            for i in range(len(self.lb)):
                if(isinstance(self.lb[i], Distribution)):
                    simulated_value = self.lb.simulate(10)
                    simulated_lower.append(np.mean(simulated_value, axis=0))
                else:
                    simulated_lower.append(self.lb[i])
        if(isinstance(self.ub, Distribution)):
            simulated_upper = self.ub.simulate(10)
            simulated_upper = np.mean(simulated_upper, axis=0)
        else:
            simulated_upper = []
            for i in range(len(self.ub)):
                if(isinstance(self.ub[i],Distribution)):
                    simulated_value = self.ub.simulate(10)
                    simulated_upper.append(np.mean(simulated_value, axis=0))
                else:
                    simulated_upper.append(self.ub[i])
        return np.array([simulated_lower, simulated_upper])

    def pdf(self, x):
        lb = []
        ub = []
        #NOTE WE NORMALLY SAID WE DONT SIMULATE FOR THE PDF, WHAT IS DESIRED?
        for i in range(len(lb)):
            if(isinstance(self.lb[i], Distribution)):
                lb.append(self.lb[i].simulate(1)[0])
            else:
                lb.append(self.lb[i])
            if(isinstance(self.ub[i], Distribution)):
                ub.append(self.ub[i].simulate(1)[0])
            else:
                ub.append(self.ub[i])

        if(np.product(np.greater_equal(x, self.lb)*np.less_equal(x, self.ub))):
            pdf_value = 1./np.product(self.ub-self.lb)
        else:
            pdf_value = 0.

        return pdf_value

    def check_parameters(self, lb, ub):
        length_lb=0
        if(isinstance(lb,Distribution)):
            simulated_lb = lb.simulate(1,reset=1)[0]
            if(isinstance(simulated_lb, np.ndarray)):
                length_lb += len(simulated_lb)
            else:
                length_lb+=1
        else:
            for i in range(len(lb)):
                if(isinstance(lb[i], Distribution)):
                    simulated_lb = lb[i].simulate(1,reset=1)[0]
                    if(isinstance(simulated_lb, np.ndarray)):
                        length_lb += len(simulated_lb)
                    else:
                        length_lb += 1
                else:
                    length_lb+=1
        length_ub = 0
        if(isinstance(ub,Distribution)):
            simulated_ub = ub.simulate(1,reset=1)[0]
            if(isinstance(simulated_ub, np.ndarray)):
                length_ub += len(simulated_ub)
            else:
                length_ub+=1
        else:
            for i in range(len(ub)):
                if(isinstance(ub[i], Distribution)):
                    simulated_ub = ub[i].simulate(1,reset=1)[0]
                    if(isinstance(simulated_ub, np.ndarray)):
                        length_ub+=len(simulated_ub)
                    else:
                        length_ub+=1
                else:
                    length_ub += 1
        return length_lb == length_ub

class MixtureNormal(Distribution):
    def __init__(self, mu, seed=None):
        self.mean = mu
        self.rng = np.random.RandomState(seed)

    def set_parameters(self, params):
        self.mean = params[0]

    def reseed(self, seed):
        self.rng.seed(seed)

    def simulate(self, k, reset=0):
        #Generate k lists from mixture_normal
        Data_array = [None]*k
        #Initialize local parameters
        mean =[]
        if(isinstance(self.mean, Distribution)):
            mean = self.mean.simulate(1)[0]
        else:
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i], Distribution)):
                    next_element = ((self.mean[i].simulate(1)[0]))
                    if(isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            mean.append(next_element[j])
                    else:
                        mean.append(next_element)
                else:
                    mean.append(self.mean[i])
        dimension = len(mean)
        if(reset==1):
            rng_state = self.rng.get_state()
        index_array = self.rng.binomial(1, 0.5, k)
        for i in range(k):
            #Initialize the time-series
            index = index_array[i]
            Data = index*self.rng.multivariate_normal(mean=mean, cov = np.identity(dimension)) \
            + (1-index)*self.rng.multivariate_normal(mean=mean, cov=0.01*np.identity(dimension))
            Data_array[i] = Data
        if(reset==1):
            self.rng.set_state(rng_state)
        return np.array(Data_array)

    def get_parameters(self):
        if(isinstance(self.mean, Distribution)):
            simulated_mean = self.mean.simulate(10)
            simulated_mean = np.mean(simulated_mean, axis=0)
        else:
            simulated_mean = []
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i], Distribution)):
                    simulated_mean_tmp = self.mean[i].simulate(10)
                    simulated_mean_tmp = np.mean(simulated_mean_tmp, axis=0)
                    simulated_mean.append(simulated_mean_tmp)
                else:
                    simulated_mean.append(self.mean[i])
        return np.array(simulated_mean)

    def pdf(self, x):
        pass

#class StochLorenz95(Distribution):
    #Im confused: what is the  here? we can simulate theta from it, but










