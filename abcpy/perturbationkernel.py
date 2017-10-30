from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gamma


class PerturbationKernel(metaclass = ABCMeta):
    """
    This abstract base class represents all preturbation kernels.
    """
    @abstractmethod
    def perturb(self, parameters, cov):
        """
        Perturbs the parameters with a given covariance matrix, according to the perturbation kernel.
        Commonly used in inference methods to perturb accepted parameters.

        Parameters
        ----------
        parameters: python list
            The list of parameters to be perturbed.
        cov: 2D list
            The covariance matrix of the parameters.

        Returns
        -------
        list
            The perturbed parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x):
        """
        Calculates the probability density function at point x.

        Parameters
        ----------
        x: list
            The point at which the probability density function should be evaluated.
        Returns
        -------
        float
            The evaluated pdf.
        """
        raise NotImplementedError


class MultivariateNormalKernel(PerturbationKernel):
    """
    This class implements a multivariate normal kernel.

    Parameters
    ----------
    rng: Random number generator
        The random number generator to be used by the kernel.
    """
    def __init__(self, rng=np.random.RandomState()):
        self.rng = rng

    def perturb(self, parameters, cov):
        return self.rng.multivariate_normal(parameters, cov)

    def pdf(self, mean, cov, x):
        return multivariate_normal(mean, cov).pdf(x)


class MultiStudentTKernel(PerturbationKernel):
    """
    This class implements a multivariate Student's T kernel.

    Parameters
    ----------
    df: integer
        The degrees of freedom of the kernel.
    rng: Random number generator
        The random number generator to be used by the kernel.
    """
    def __init__(self, df, rng=np.random.RandomState()):
        self.rng = rng
        self.df = df

    def perturb(self, parameters, cov):
        p = len(parameters)
        if (self.df == np.inf):
            chis1 = 1.0
        else:
            chisq = self.rng.chisquare(self.df, 1) / self.df
            chisq = chisq.reshape(-1, 1).repeat(p, axis=1)
        mvn = self.rng.multivariate_normal(np.zeros(p), cov, 1)
        result = (parameters + np.divide(mvn, np.sqrt(chisq)))
        return result

    def pdf(self, mean, cov, x):
        mean = np.array(mean)
        cov = np.array(cov)
        p = len(mean)
        numerator = gamma((self.df + p) / 2)
        denominator = gamma(self.df / 2) * pow(self.df * np.pi, p / 2.) * np.sqrt(abs(np.linalg.det(cov)))
        normalizing_const = numerator / denominator
        tmp = 1 + 1 / self.df * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))
        density = normalizing_const * pow(tmp, -((self.df + p) / 2.))
        return density