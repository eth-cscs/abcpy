from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import gamma


class Distribution(metaclass = ABCMeta):
    """
    This abstract base class represents a distribution. It can be used e.g. as a
    prior for models.

    """


    @abstractmethod
    def set_parameters(self, params):
        """To be overwritten by any sub-class: should set the parameters of the distribution.

        Parameters
        ----------
        theta: list
            Contains all the distributions parameters.
            
        """

        raise NotImplemented


    
    @abstractmethod
    def reseed(self, seed):
        """To be overwritten by any sub-class: reseed the random number generator with provided seed.

        Parameters
        ----------
        seed: integer
            New seed for the random number generator
            
        """

        raise NotImplemented

    
    
    @abstractmethod
    def sample(self, k):
        """To be overwritten by any sub-class: should sample k points from the implemented distribution.

        Parameters
        ----------
        k: integer
            The number of points to be sampled
        Returns
        -------
        numpy.ndarray
            kxp matrix containing k samples of p-dimensional points
            
        """
        
        raise NotImplemented
    

    @abstractmethod
    def pdf(self, x):
        """To be overwritten by any sub-class: calculate the prior-denisty at *x*, where
        *x* is a parameter of dimension p.

        Parameters
        ----------
        x: numpy.ndarray
            A p-dimensional point from the support of the distribution
        Returns
        -------
        float
            The probability density for point x
            
        """
        
        raise NotImplemented
        

class MultiNormal(Distribution):
    """
    This class implements a p-dimensional multivariate Normal distribution.
    """    
    def __init__(self, mean, cov, seed=None):
        """        
        Defines and mean covariance of multivariate normal distribution.
        
        Parameters
        ----------
        mean: numpy.ndarray
            p-dimensional vector containing p means
        cov: numpy.ndarray
            pxp covariance matrix 
        seed: integer
            Initial seed for the random number generator

        """

        self.rng = np.random.RandomState(seed)
        self.mean = mean
        self.cov = cov



    def set_parameters(self, params):      
        self.mean = params[0]
        self.cov = params[1]



    def reseed(self, seed):
        self.rng.seed(seed)
        

        
    def sample(self, k):
        #samples = self.distribution.rvs(k).reshape(k,p)
        samples = self.rng.multivariate_normal(self.mean, self.cov, k)
        return samples
    
    def pdf(self, x):
        return multivariate_normal(self.mean, self.cov).pdf(x)



class Uniform(Distribution):
    """
    This class implements a p-dimensional uniform Prior distribution in a closed interval.
    """    
    def __init__(self, lb : np.ndarray, ub : np.ndarray, seed=None):
        """
        Defines the upper and lower bounds of a p-dimensional uniform Prior distribution in a closed interval.

        Parameters
        ----------
        lb: numpy.ndarray or a list
            Vector containing p lower bounds
        ub: numpy.ndarray or a list
            Vector containing p upper bounds
        seed: integer
            Initial seed for the random number generator

        """

        self.lb, self.ub = self._check_parameters(lb, ub)
        self.rng = np.random.RandomState(seed)

        
    def set_parameters(self, params):
        lb = params[0]
        ub = params[1]
        self.lb, self.ub = self._check_parameters(lb, ub)


    def reseed(self, seed):
        self.rng.seed(seed)
        

    def sample(self, k):
        samples = np.zeros(shape=(k,len(self.lb)))
        for j in range(0,len(self.lb)):
            samples[:,j] = self.rng.uniform(self.lb[j], self.ub[j], k)
        
        return samples
    
        
    def pdf(self,x):
        if np.product(np.greater_equal(x,self.lb)*np.less_equal(x,self.ub)):
            pdf_value = 1/np.product(self.ub-self.lb)
        else:
            pdf_value = 0            
        return pdf_value
        

    def _check_parameters(self, lb, ub):
        new_lb = new_ub = None
        if isinstance(lb, (list,np.ndarray)):
            new_lb = np.array(lb)
        else:
            raise TypeError('The lower bound is not of allowed types')

        if isinstance(ub, (list,np.ndarray)):
            new_ub = np.array(ub)
        else:
            raise TypeError('The upper bound is not of allowed types')
                
        if new_lb.shape != new_ub.shape:
            raise BaseException('Dimension of lower bound and upper bound is not same.')

        return (new_lb, new_ub)
        



class MultiStudentT(Distribution):
    """
    This class implements a p-dimensional multivariate Student T distribution.
    
    """
    
    def __init__(self, mean, cov, df, seed=None):
        """Defines the mean, co-variance and degrees of freedom a p-dimensional multivariate Student T distribution.

        Parameters
        ----------
        mean: numpy.ndarray
            Vector containing p means, one for every dimension        
        cov: numpy.ndarray
            pxp matrix containing the co-variance matrix        
        df: np.uint
            Degrees of freedom
 
        """

        MultiStudentT._check_parameters(mean, cov, df)
        
        self.mean = mean
        self.cov = cov
        self.df = df
        self.rng = np.random.RandomState(seed)


        
    def set_parameters(self, params):      
        mean = params[0]
        cov = params[1]
        df = self.df
        
        if len(params) == 3:
            df = params[2]
        
        try:
            MultiStudentT._check_parameters(mean, cov, df)
        except TypeError or ValueError:
            return False

        self.mean = mean
        self.cov = cov
    

        
    def reseed(self, seed):
        self.rng.seed(seed)


    
    def sample(self, k):
        p = len(self.mean)
        if self.df == np.inf:
            chisq = 1.0
        else:
            chisq = self.rng.chisquare(self.df, k) / self.df
            chisq = chisq.reshape(-1,1).repeat(p, axis=1)
        mvn = self.rng.multivariate_normal(np.zeros(p), self.cov, k)
        return self.mean + np.divide(mvn, np.sqrt(chisq))

    
    
    def pdf(self, x):
        m = self.mean
        cov = self.cov
        v = self.df
        p = len(self.mean)
        
        numerator = gamma((v + p) / 2)
        denominator = gamma(v/2) * pow(v * np.pi, p/2.) * np.sqrt(abs(np.linalg.det(cov)))
        normalizing_const = numerator / denominator
        tmp = 1 + 1/v * np.dot(np.dot(np.transpose(x - m), np.linalg.inv(cov)), (x-m))
        density = normalizing_const * pow(tmp, -((v+p)/2.))
        return density
        

    
    @classmethod
    def _check_parameters(cls, mean, cov, df):
        """
        Private function that checks whether the parameters make sense for the distribution.
        """

        if not isinstance(mean, np.ndarray):
            raise TypeError("Parameter mean is not of type numpy.array.")

        if not isinstance(cov, np.ndarray):
            raise TypeError("Parameter cov is not of type numpy.array.")

        if mean.ndim != 1:
            raise ValueError("Parameter mean has dimension larger than 1.")

        if cov.ndim != 2:
            raise ValueError("Parameter cov is not of dimensionality 2.")

        if df <= 0:
            raise ValueError("Parameter df is smaller than 0.")
            
class Normal(Distribution):
    """
    This class implements a 1-dimensional Normal distribution.
    """    
    def __init__(self, mean, var, seed=None):
        """        
        Defines and mean and variance of normal distribution.
        
        Parameters
        ----------
        mean: numpy.ndarray
            mean
        var: numpy.ndarray
            variance
        seed: integer
            Initial seed for the random number generator

        """

        self.rng = np.random.RandomState(seed)
        self.mean = mean
        self.var = var



    def set_parameters(self, params):      
        self.mean = params[0]
        self.var = params[1]



    def reseed(self, seed):
        self.rng.seed(seed)
        

        
    def sample(self, k):
        #samples = self.distribution.rvs(k).reshape(k,p)
        samples = self.rng.normal(self.mean, self.var, k)
        return samples.reshape((k,1))
    
    def pdf(self, x):
        return norm(self.mean, self.var).pdf(x)
            
            
