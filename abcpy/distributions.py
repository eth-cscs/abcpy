from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import gamma

#NOTE ALL THE GET AND SET PARAMETERS HAVE BEEN TESTED, THEY SHOULD IN THEORY WORK

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
    def sample_from_prior(self):
        """To be overwritten by any sub-class: samples from the prior distribution and sets the current parameter values to the sampled values."""
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
    def sample(self, k, reset=0):
        """To be overwritten by any sub-class: should sample k points from the implemented distribution.

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
        """To be overwritten by any sub-class: returns the current parameters of the distribution."""
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
        if(not(isinstance(self.mean, Distribution))):
            self.mean_value = self.mean
        else:
            self.mean_value = self.mean.sample(1)[0]
            if(isinstance(self.mean_value,np.ndarray)):
                self.mean_value = self.mean_value[0]
        if(not(isinstance(self.var, Distribution))):
            self.var_value = self.var
        else:
            self.var_value = self.var.sample(1)[0]
            if(isinstance(self.var_value,np.ndarray)):
                self.var_value = self.var_value[0]

    def set_parameters(self, params):
        '''
        Sets the parameters for the distribution.
        Parameters
        ----------
        params: list
            The first element of the list specifies the mean of the distribution in float or as a 1D distribution. The second element specifies the sigma of the distribution in float or as a 1D distribution.
        '''

        if(not(isinstance(params,list))):
            raise TypeError('params must be of type list.')
        if(len(params)!=2):
            return False
        if(isinstance(params[1],list)and params[1][1]<=0):
            return False
        if(not(isinstance(params[1],list)) and params[1]<=0):
            return False
        if(isinstance(params[0],list)):
            if(len(params[0])==2):
                self.mean_value = params[0][0]
                self.mean.set_parameters(params[0][1])
            else:
                self.mean_value = params[0][0]
        else:
            self.mean_value = params[0]
        if(isinstance(params[1],list)):
            if(len(params[1])==2):
                self.var_value = params[1][0]
                self.var.set_parameters(params[1][1])
            else:
                self.var_value = params[1][0]
        else:
            self.var_value = params[1]
        return True

    def sample_from_prior(self):
        if(isinstance(self.mean, Distribution)):
            self.mean.sample_from_prior()
            sample_mean = self.mean.sample(1)[0]
            if(isinstance(sample_mean,np.ndarray)):
                sample_mean = sample_mean[0]
        else:
            sample_mean = self.mean
        if(isinstance(self.var, Distribution)):
            self.mean.sample_from_prior()
            sample_var = self.var.sample(1)[0]
            if(isinstance(sample_var, np.ndarray)):
                sample_var = sample_var[0]
        else:
            sample_var = self.var
        if(self.set_parameters([sample_mean, sample_var])==False):
            raise ValueError("Prior generates values that are out the model parameter domain.")


    def reseed(self, seed):
        self.rng.seed(seed)

    def sample(self, k, reset=0):
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
        if(reset==1):
            rng_state = self.rng.get_state()
        result = (self.rng.normal(self.mean_value, self.var_value, k)).reshape(-1)
        if(reset == 1):
            self.rng.set_state(rng_state)
        return np.array(result)

    def pdf(self, x):
        if(isinstance(self.mean, Distribution) or isinstance(self.var, Distribution)):
            raise TypeError('Mean and Variance are not allowed to be of type distribution')
        else:
            return norm(self.mean_value, self.var_value).pdf(x)

    def get_parameters(self):
        if(isinstance(self.mean, Distribution)):
            l1 = [self.mean_value, self.mean.get_parameters()]
        else:
            l1 = self.mean_value
        if(isinstance(self.var,Distribution)):
            l2 = [self.var_value, self.var.get_parameters()]
        else:
            l2 = self.var_value
        return [l1,l2]

    def check_parameters(self, mean, var):
        if(hasattr(mean, '__len__') and hasattr(var, '__len__')):
            if(len(mean)==1 and len(var)==1):
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
        if(isinstance(self.mean, Distribution)):
            self.mean_value = self.mean.sample(1)[0].tolist()
        else:
            mean_value = []
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i], Distribution)):
                    next_element = self.mean[i].sample(1)[0]
                    if(isinstance(next_element,np.ndarray)):
                        for j in range(len(next_element)):
                            mean_value.append(next_element[j])
                    else:
                        mean_value.append(next_element)
                else:
                    mean_value.append(self.mean[i])
            self.mean_value = mean_value

    def set_parameters(self, params):
        '''
        Sets the parameters for the distribution.
        Parameters
        ----------
        params: list
            The first element of the list specifies the mean of the distribution as a p-dimensional list or distribution. The second element specifies the sigma of the distribution as a pxp dimensional list.
        '''
        #if(not(self.check_parameters(params[0],params[1]))):
        #    return False
        for i in range(len(params[0])):
            if(isinstance(params[0][i],list)):
                if(isinstance(params[0][i][0],list)):
                    for j in range(len(params[0][i][0])):
                        self.mean_value[i+j] = params[0][i][0][j]
                    if(len(params[0][i][0])==len(self.mean_value)):
                        flag = True
                    else:
                        flag = False
                else:
                    flag=False
                    self.mean_value[i]=params[0][i][0]
                if(flag):
                    self.mean.set_parameters(params[0][i][1])
                else:
                    self.mean[i].set_parameters(params[0][i][1])
            else:
                self.mean_value[i] = params[0][i]
        self.cov = params[1]
        return True

    def sample_from_prior(self):
        if (isinstance(self.mean, Distribution)):
            self.mean.sample_from_prior()
            self.mean_value = self.mean.sample(1)[0].tolist()
        else:
            mean_value = []
            for i in range(len(self.mean)):
                if (isinstance(self.mean[i], Distribution)):
                    self.mean[i].sample_from_prior()
                    next_element = self.mean[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            mean_value.append(next_element[j])
                    else:
                        mean_value.append(next_element)
                else:
                    mean_value.append(self.mean[i])
        if(self.set_parameters([mean_value, self.cov])==False):
            raise ValueError("Prior generates values that are out the model parameter domain.")


    def reseed(self,seed):
        self.rng.seed(seed)

    def sample(self, k, reset=0):
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
        if(reset==1):
            rng_state = self.rng.get_state()
        result = self.rng.multivariate_normal(self.mean_value, self.cov, k)
        if(reset==1):
            self.rng.set_state(rng_state)
        return result

    def get_parameters(self):
        if(isinstance(self.mean, Distribution)):
            l1 = [[self.mean_value, self.mean.get_parameters()]]
        else:
            l1 = []
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i],Distribution)):
                    helper_samples = self.mean[i].sample(1,reset=1)[0]
                    if(isinstance(helper_samples, list)):
                        l_helper = []
                        for j in range(len(helper_samples)):
                            l_helper.append(self.mean_value[i+j])
                    else:
                        l_helper = self.mean_value[i]
                    l_helper_2 = self.mean[i].get_parameters()
                    l1.append([l_helper, l_helper_2])
                else:
                    l1.append(self.mean_value[i])
        return [(l1), self.cov]

    def pdf(self, x):
        if(not(isinstance(self.mean, Distribution))):
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i],Distribution)):
                    raise TypeError('All elements of mean are not allowed to be of type distribution')
        elif(isinstance(self.mean, Distribution)):
            raise TypeError('Mean is not allowed to be of type distribution')
        return multivariate_normal(self.mean_value, self.cov).pdf(x)

    def check_parameters(self, mean, cov):
        if(not(isinstance(mean,Distribution))):
            length = 0
            for i in range(len(mean)):
                if(isinstance(mean[i], Distribution)):
                    value = mean[i].sample(1,reset=1)[0]
                    if(isinstance(value,np.ndarray)):
                        length+=len(value)
                    else:
                        length+=1
                else:
                    length+=1
            if(length==len(cov)):
                return True
        if(isinstance(mean,Distribution)):
            if(len(mean.sample(1,reset=1)[0])==len(cov)):
                return True
        return False


class StudentT(Distribution):
    '''
    Parameters
    ----------
    mu: float or 1D distribution
        Defines the mean of the distribution
    df: int or 1D distribution
        Defines the degrees of freedom of the distribution
    seed: int
        The initial seed to be used by the random number generator.
    '''
    #NOTE THIS GIVES BACK WEIRD VALUES... IF WE SAMPLE WE GET LIKE 400000 FOR A DISTRIBUTION WITH 1 AND 4....
    def __init__(self, mu, df, seed=None):
        self.mean = mu
        self.df = df
        self.rng = np.random.RandomState(seed)
        if(isinstance(self.mean,Distribution)):
            self.mean_value = self.mean.sample(1)[0]
            if(isinstance(self.mean_value,np.ndarray)):
                self.mean_value = self.mean_value[0]
        else:
            self.mean_value = self.mean
        if(isinstance(self.df, Distribution)):
            self.df_value = self.df.sample(1)[0]
            if(isinstance(self.df_value, np.ndarray)):
                self.df_value = self.df_value[0]
        else:
            self.df_value = self.df

    def set_parameters(self, params):
        if(not(isinstance(params,list))):
            raise TypeError('params has to be of type list.')
        if(len(params)!=2):
            return False
        #if(isinstance(params[1],list) and params[1][1]<=1): #this works for I think normal distributions, but not if it is uniform?
        #    return False
        #if(not(isinstance(params[1],list)) and params[1]<=0):
        #    return False
        if(isinstance(params[0],list)):
            if(len(params[0])==2):
                self.mean_value = params[0][0]
                self.mean.set_parameters(params[0][1])
            else:
                self.mean_value = params[0][0]
        else:
            self.mean_value = params[0]
        if(isinstance(params[1],list)):
            if(len(params[1])==2):
                self.df_value = params[1][0]
                self.df.set_parameters(params[1][1])
        else:
            self.df_value = params[1]
        return True

    def sample_from_prior(self):
        mean_value = 0
        if(isinstance(self.mean, Distribution)):
            self.mean.sample_from_prior()
            mean_value = self.mean.sample(1)[0]
            if(isinstance(mean_value, np.ndarray)):
                mean_value = mean_value[0]
        else:
            mean_value = self.mean
        df_value = 0
        if(isinstance(self.df, Distribution)):
            self.df.sample_from_prior()
            df_value = self.df.sample(1)[0]
            if(isinstance(df_value, np.ndarray)):
                df_value = df_value[0]
        else:
            df_value = self.df
        if(self.set_parameters([mean_value, df_value])==False):
            raise ValueError("Prior generates values that are out the model parameter domain.")

    def reseed(self, seed):
        self.rng.seed(seed)

    def sample(self, k, reset=0):
        if(reset==1):
            rng_state = self.rng.get_state()
        result = (self.rng.standard_t(self.df_value, k) + self.mean_value).reshape(-1)
        if(reset==1):
            self.rng.set_state(rng_state)
        return np.array(result)

    def get_parameters(self):
        if(isinstance(self.mean, Distribution)):
            l1 = [self.mean_value, self.mean.get_parameters()]
        else:
            l1 = self.mean_value
        if(isinstance(self.df, Distribution)):
            l2 = [self.df_value, self.df.get_parameters()]
        else:
            l2 = self.df_value
        return [l1, l2]

    def pdf(self, x):
        if(isinstance(self.df, Distribution)):
            return TypeError("Degrees of freedom is not allowed to be of type distribution")
        return gamma((self.df_value+1)/2.)/(np.sqrt(self.df_value*np.pi)*gamma(self.df_value/2.))*(1+float(x**2)/self.df_value)**(-(self.df_value+1)/2.)

class MultiStudentT(Distribution):
    '''
    Parameters
    ----------
    mean: p-dimensional list or distribution
        The mean of the distribution
    cov: pxp dimensional list
        The covariance matrix of the distribution
    df: int or 1D distribution
        The degrees of freedom of the distribution
    seed: int
        The initial seed to be used by the random number generator.
    '''
    def __init__(self, mean, cov, df, seed=None):
        if(not(self.check_parameters(mean, cov))):
            raise IndexError('Mean and cov do not have matching dimensions.')
        self.mean = mean
        self.cov = cov
        self.df = df
        self.rng = np.random.RandomState(seed)
        if (isinstance(self.mean, Distribution)):
            self.mean_value = self.mean.sample(1)[0].tolist()
        else:
            mean_value = []
            for i in range(len(self.mean)):
                if (isinstance(self.mean[i], Distribution)):
                    next_element = self.mean[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            mean_value.append(next_element[j])
                    else:
                        mean_value.append(next_element)
                else:
                    mean_value.append(self.mean[i])
            self.mean_value = mean_value
        if(isinstance(self.df, Distribution)):
            self.df_value = self.df.sample(1)[0]
            if(isinstance(self.df_value, np.ndarray)):
                self.df_value = self.df_value[0]
        else:
            self.df_value = self.df

    def set_parameters(self, params):
        #if(not(self.check_parameters(params[0],params[1]))):
        #    return False
        if(len(params)>3 or len(params)<2):
            return False
        for i in range(len(params[0])):
            if(isinstance(params[0][i],list)):
                if(isinstance(params[0][i][0],list)):
                    for j in range(len(params[0][i][0])):
                        self.mean_value[i+j] = params[0][i][0][j]
                    if(len(params[0][i][0])==len(self.mean_value)):
                        flag = True
                    else:
                        flag = False
                else:
                    flag=False
                    self.mean_value[i]=params[0][i][0]
                if(flag):
                    self.mean.set_parameters(params[0][i][1])
                else:
                    self.mean[i].set_parameters(params[0][i][1])
            else:
                self.mean_value[i] = params[0][i]
        self.cov = params[1]
        if(len(params)==3):
            if(isinstance(params[2],list)):
                self.df_value=params[2][0]
                self.df.set_parameters(params[2][1])
            else:
                self.df_value = params[2]
        return True

    def sample_from_prior(self):
        if (isinstance(self.mean, Distribution)):
            self.mean.sample_from_prior()
            mean_value = self.mean.sample(1)[0].tolist()
        else:
            mean_value = []
            for i in range(len(self.mean)):
                if (isinstance(self.mean[i], Distribution)):
                    self.mean[i].sample_from_prior()
                    next_element = self.mean[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            mean_value.append(next_element[j])
                    else:
                        mean_value.append(next_element)
                else:
                    mean_value.append(self.mean[i])
        if(isinstance(self.df, Distribution)):
            self.df.sample_from_prior()
            df_value = self.df.sample(1)[0]
            if(isinstance(df_value, np.ndarray)):
                df_value = df_value[0]
        else:
            df_value = self.df
        if(self.set_parameters([mean_value, self.cov, df_value])==False):
            raise ValueError("Prior generates values that are out the model parameter domain.")


    def reseed(self, seed):
        self.rng.seed(seed)

    def sample(self, k, reset=0):
        mean = self.mean_value
        cov = self.cov
        p = len(mean)
        df = self.df_value
        if (df == np.inf):
            chis1 = 1.0
        else:
            chisq = self.rng.chisquare(df, k) / df
            chisq = chisq.reshape(-1, 1).repeat(p, axis=1)
        mvn = self.rng.multivariate_normal(np.zeros(p), cov, k)
        if(reset==1):
            rng_state = self.rng.get_state()
        result = (mean + np.divide(mvn, np.sqrt(chisq)))
        if(reset==1):
            self.rng.set_state(rng_state)
        return result

    def get_parameters(self):
        if (isinstance(self.mean, Distribution)):
            l1 = [[self.mean_value, self.mean.get_parameters()]]
        else:
            l1 = []
            for i in range(len(self.mean)):
                if (isinstance(self.mean[i], Distribution)):
                    helper_samples = self.mean[i].sample(1, reset=1)[0]
                    if (isinstance(helper_samples, np.ndarray)):
                        l_helper = []
                        for j in range(len(helper_samples)):
                            l_helper.append(self.mean_value[i + j])
                    else:
                        l_helper = self.mean_value[i]
                    l_helper_2 = self.mean[i].get_parameters()
                    l1.append([l_helper, l_helper_2])
                else:
                    l1.append(self.mean_value[i])
        l2 = self.cov
        if(isinstance(self.df, Distribution)):
            l3 = [self.df_value, self.df.get_parameters()]
        else:
            l3 = self.df_value
        return [l1, l2, l3]

    def pdf(self, x):
        if(not(isinstance(self.mean, Distribution))):
            for i in range(len(self.mean)):
                if (isinstance(self.mean[i], Distribution)):
                    print("Mean is not allowed to be of type Distribution")
        if(isinstance(self.mean, Distribution)):
            raise TypeError('Mean is not allowed to be of type Distribution')
        if(isinstance(self.df, Distribution)):
            raise TypeError('Degrees of freedom is not allowed to be of type Distribution.')
        mean = np.array(self.mean_value)
        cov = self.cov
        v = self.df_value
        p = len(mean)

        numerator = gamma((v + p) / 2)
        denominator = gamma(v / 2) * pow(v * np.pi, p / 2.) * np.sqrt(abs(np.linalg.det(cov)))
        normalizing_const = numerator / denominator
        tmp = 1 + 1 / v * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))
        density = normalizing_const * pow(tmp, -((v + p) / 2.))
        return density

    def check_parameters(self, mean, cov):
        if(isinstance(mean, Distribution)):
            value = mean.sample(1,reset=1)[0]
            if(isinstance(value, np.ndarray)):
                if(len(value)==len(cov)):
                    return True
            else:
                return len(cov)==1
        else:
            length = 0
            for i in range(len(mean)):
                if(isinstance(mean[i],Distribution)):
                    value = mean[i].sample(1,reset=1)[0]
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
    '''
    Parameters
    ----------
    lb: p-dimensional list or distribution
        The p lower bounds of the distribution
    ub: p-dimensional list or distribution
        The p upper bounds of the distribution
    seed: int
        The inital seed to be used by the random number generator
    '''
    def __init__(self, lb, ub, seed=None):
        if(not(self.check_parameters(lb,ub))):
            raise IndexError('Lower and upper bound do not have matching dimensions.')
        self.lb = lb
        self.ub = ub
        self.rng = np.random.RandomState(seed)
        if (isinstance(self.lb, Distribution)):
            self.lb_value = self.lb.sample(1)[0].tolist()
        else:
            lb = []
            for i in range(len(self.lb)):
                if (isinstance(self.lb[i], Distribution)):
                    next_element = self.lb[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            lb.append(next_element[j])
                    else:
                        lb.append(next_element)
                else:
                    lb.append(self.lb[i])
            self.lb_value = lb
        if (isinstance(self.ub, Distribution)):
            self.ub_value = self.ub.sample(1)[0].tolist()
        else:
            ub = []
            for i in range(len(self.ub)):
                if (isinstance(self.ub[i], Distribution)):
                    next_element = self.ub[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            ub.append(next_element[j])
                    else:
                        ub.append(next_element)
                else:
                    ub.append(self.ub[i])
            self.ub_value = ub

    #NOTE this doesnt include if one value is from a multid distribution!!!
    #as written above, multid is not yet supported, code that in
    def set_parameters(self, params):
        #if(not(self.check_parameters(params[0],params[1]))):
         #   return False
        for i in range(len(params[0])):
            if(isinstance(params[0][i],list)):
                if(isinstance(params[0][i][0],list)):
                    for j in range(len(params[0][i][0])):
                        self.lb_value[i+j]=params[0][i][0][j]
                    if(len(params[0][i][0])==len(self.lb_value)):
                        flag = True
                    else:
                        flag = False
                else:
                    self.lb_value[i]=params[0][i][0]
                    flag = False
                if(flag):
                    self.lb[0].set_parameters(params[0][i][1])
                else:
                    self.lb[i].set_parameters(params[0][i][1])
            else:
                self.lb_value[i]=params[0][i]
        for i in range(len(params[1])):
            if(isinstance(params[1][i],list)):
                if(isinstance(params[1][i][0],list)):
                    for j in range(len(params[1][i][0])):
                        self.ub_value[i+j]=params[1][i][0][j]
                    if(len(params[1][i][0])==len(self.ub_value)):
                        flag = True
                    else:
                        flag = False
                else:
                    self.ub_value[i]=params[1][i][0]
                    flag = False
                if(flag):
                    self.ub[0].set_parameters(params[1][i][1])
                else:
                    self.ub[i].set_parameters(params[1][i][1])
            else:
                self.ub_value[i]=params[1][i]
        return True

    def sample_from_prior(self):
        if (isinstance(self.lb, Distribution)):
            self.lb.sample_from_prior()
            lb_value = self.lb.sample(1)[0].tolist()
        else:
            lb = []
            for i in range(len(self.lb)):
                if (isinstance(self.lb[i], Distribution)):
                    self.lb[i].sample_from_prior()
                    next_element = self.lb[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            lb.append(next_element[j])
                    else:
                        lb.append(next_element)
                else:
                    lb.append(self.lb[i])
            lb_value = lb
        if (isinstance(self.ub, Distribution)):
            self.ub.sample_from_prior()
            ub_value = self.ub.sample(1)[0].tolist()
        else:
            ub = []
            for i in range(len(self.ub)):
                if (isinstance(self.ub[i], Distribution)):
                    self.ub[i].sample_from_prior()
                    next_element = self.ub[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            ub.append(next_element[j])
                    else:
                        ub.append(next_element)
                else:
                    ub.append(self.ub[i])
            ub_value = ub
        if(self.set_parameters([lb_value, ub_value])==False):
            raise ValueError("Prior generates values that are out the model parameter domain.")

    def reseed(self, seed):
        self.rng.seed(seed)

    def sample(self, k, reset=0):
        samples = np.zeros(shape=(k, len(self.lb_value)))
        for j in range(0, len(self.lb_value)):
            samples[:, j] = self.rng.uniform(self.lb_value[j], self.ub_value[j], k)
        return samples

    def get_parameters(self):
        if(isinstance(self.lb, Distribution)):
            l1 = [[self.lb_value, self.lb.get_parameters()]]
        else:
            l1=[]
            for i in range(len(self.lb)):
                if(isinstance(self.lb[i], Distribution)):
                    samples_helper = self.lb[i].sample(1,reset=1)[0]
                    l_helper = []
                    if(isinstance(samples_helper, np.ndarray)):
                        for j in range(len(samples_helper)):
                            l_helper.append(self.lb_value[i+j])
                    else:
                        l_helper = self.lb_value[i]
                    l1.append([l_helper, self.lb[i].get_parameters()])
                else:
                    l1.append(self.lb_value[i])
        if(isinstance(self.ub, Distribution)):
            l2 = [self.ub_value, self.ub.get_parameters()]
        else:
            l2 = []
            for i in range(len(self.ub)):
                if(isinstance(self.ub[i], Distribution)):
                    samples_helper = self.ub[i].sample(1,reset=1)[0]
                    l_helper =[]
                    if(isinstance(samples_helper, np.ndarray)):
                        for j in range(len(samples_helper)):
                            l_helper.append(self.ub_value[i+j])
                    else:
                        l_helper = self.ub_value[i]
                    l2.append([l_helper, self.ub[i].get_parameters()])
                else:
                    l2.append(self.ub_value[i])
        return [l1,l2]

    def pdf(self, x):
        for i in range(len(self.lb)):
            if(isinstance(self.lb[i],Distribution)):
                raise TypeError('None of the elements of the lower bound are allowed to be of type Distribution.')
        for i in range(len(self.ub)):
            if(isinstance(self.ub[i],Distribution)):
                raise TypeError('None of the elements of the upper bound are allowed to be of type Distribution.')
        if(np.product(np.greater_equal(x, np.array(self.lb_value))*np.less_equal(x, np.array(self.ub_value)))):
            pdf_value = 1./np.product(np.array(self.ub_value)-np.array(self.lb_value))
        else:
            pdf_value = 0.

        return pdf_value

    def check_parameters(self, lb, ub):
        length_lb=0
        if(isinstance(lb,Distribution)):
            simulated_lb = lb.sample(1,reset=1)[0]
            if(isinstance(simulated_lb, np.ndarray)):
                length_lb += len(simulated_lb)
            else:
                length_lb+=1
        else:
            for i in range(len(lb)):
                if(isinstance(lb[i], Distribution)):
                    simulated_lb = lb[i].sample(1,reset=1)[0]
                    if(isinstance(simulated_lb, np.ndarray)):
                        length_lb += len(simulated_lb)
                    else:
                        length_lb += 1
                else:
                    length_lb+=1
        length_ub = 0
        if(isinstance(ub,Distribution)):
            simulated_ub = ub.sample(1,reset=1)[0]
            if(isinstance(simulated_ub, np.ndarray)):
                length_ub += len(simulated_ub)
            else:
                length_ub+=1
        else:
            for i in range(len(ub)):
                if(isinstance(ub[i], Distribution)):
                    simulated_ub = ub[i].sample(1,reset=1)[0]
                    if(isinstance(simulated_ub, np.ndarray)):
                        length_ub+=len(simulated_ub)
                    else:
                        length_ub+=1
                else:
                    length_ub += 1
        return length_lb == length_ub

class MixtureNormal(Distribution):
    '''
    Parameters
    ----------
    mu: p-dimensional list or distribution
        The mean of the distribution
    seed: int
        The initial seed to be used by the random number generator
    '''
    def __init__(self, mu, seed=None):
        self.mean = mu
        self.rng = np.random.RandomState(seed)
        if(isinstance(self.mean, Distribution)):
            self.mean_value = self.mean.sample(1)[0].tolist()
        else:
            mean_value = []
            for i in range(len(self.mean)):
                if(isinstance(self.mean[i], Distribution)):
                    next_element = self.mean[i].sample(1)[0]
                    if(isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            mean_value.append(next_element[j])
                    else:
                        mean_value.append(next_element)
                else:
                    mean_value.append(self.mean[i])
            self.mean_value = mean_value

    def set_parameters(self, params):
        if(not(isinstance(params,list)) or len(params)!=1):
            return False
        for i in range(len(params[0])):
            if(isinstance(params[0][i],list)):
                if(isinstance(params[0][i][0],list)):
                    for j in range(len(params[0][i][0])):
                        self.mean_value[i+j] = params[0][i][0][j]
                    if(len(params[0][i][0])==len(self.mean_value)):
                        flag = True
                    else:
                        flag = False
                else:
                    flag=False
                    self.mean_value[i]=params[0][i][0]
                if(flag):
                    self.mean.set_parameters(params[0][i][1])
                else:
                    self.mean[i].set_parameters(params[0][i][1])
            else:
                self.mean_value[i] = params[0][i]
        return True

    def sample_from_prior(self):
        if (isinstance(self.mean, Distribution)):
            self.mean.sample_from_prior()
            mean_value = self.mean.sample(1)[0].tolist()
        else:
            mean_value = []
            for i in range(len(self.mean)):
                if (isinstance(self.mean[i], Distribution)):
                    self.mean[i].sample_from_prior()
                    next_element = self.mean[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            mean_value.append(next_element[j])
                    else:
                        mean_value.append(next_element)
                else:
                    mean_value.append(self.mean[i])
        if(self.set_parameters([mean_value])==False):
            raise ValueError("Prior generates values that are out the model parameter domain.")

    def reseed(self, seed):
        self.rng.seed(seed)

    def sample(self, k, reset=0):
        #Generate k lists from mixture_normal
        Data_array = [None]*k
        #Initialize local parameters
        mean = self.mean_value
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
        if (isinstance(self.mean, Distribution)):
            l1 = [[self.mean_value, self.mean.get_parameters()]]
        else:
            l1 = []
            for i in range(len(self.mean)):
                if (isinstance(self.mean[i], Distribution)):
                    helper_samples = self.mean[i].sample(1, reset=1)[0]
                    if (isinstance(helper_samples, np.ndarray)):
                        l_helper = []
                        for j in range(len(helper_samples)):
                            l_helper.append(self.mean_value[i + j])
                    else:
                        l_helper = self.mean_value[i]
                    l_helper_2 = self.mean[i].get_parameters()
                    l1.append([l_helper, l_helper_2])
                else:
                    l1.append(self.mean_value[i])
        return [l1]

    def pdf(self, x):
        raise TypeError('Mixture Normal does not have a likelihood.')

#TODO THE REMAPPING TO LISTS CREATES SOME ISSUES HERE
class StochLorenz95(Distribution):
    """Generates time dependent 'slow' weather variables following forecast model of Wilks [1],
    a stochastic reparametrization of original Lorenz model Lorenz [2].

    [1] Wilks, D. S. (2005). Effects of stochastic parametrizations in the lorenz ’96 system.
    Quarterly Journal of the Royal Meteorological Society, 131(606), 389–407.

    [2] Lorenz, E. (1995). Predictability: a problem partly solved. In Proceedings of the
    Seminar on Predictability, volume 1, pages 1–18. European Center on Medium Range
    Weather Forecasting, Europe

    Parameters
    ----------
    theta: p-dimensional list or distribution
        The parameters of the model.
    initial_state: numpy.ndarray, optional
        Initial state value of the time-series, The default value is None, which assumes a previously computed
        value from a full Lorenz model as the Initial value.
    n_timestep: int, optional
        Number of timesteps between [0,4], where 4 corresponds to 20 days. The default value is 160.
    seed: int, optional
        Initial seed. The default value is generated randomly.
    """

    def __init__(self, theta, initial_state=None, n_timestep=160, seed=None):

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
        # Assign closure parameters
        self.theta = theta
        if(isinstance(self.theta, Distribution)):
            self.theta_value = self.theta.sample(1)[0].tolist()
        else:
            theta_value = []
            for i in range(len(self.theta)):
                if(isinstance(self.theta[i],Distribution)):
                    next_element = self.theta[i].sample(1)[0]
                    if(isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            theta_value.append(next_element[j])
                    else:
                        theta_value.append(next_element)
                else:
                    theta_value.append(self.theta[i])
            self.theta_value=np.array(theta_value)
        # Other parameters kept fixed
        self.F = 10
        self.sigma_e = 1
        self.phi = 0.4
        # Initialize random number generator with provided seed, if None initialize with present time.
        self.rng = np.random.RandomState(seed)

    def reseed(self, seed):
        self.rng.seed(seed)

    def sample_from_prior(self):
        if (isinstance(self.theta, Distribution)):
            self.theta.sample_from_prior()
            theta_value = self.theta.sample(1)[0].tolist()
        else:
            theta_value = []
            for i in range(len(self.theta)):
                if (isinstance(self.theta[i], Distribution)):
                    self.theta[i].sample_from_prior()
                    next_element = self.theta[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            theta_value.append(next_element[j])
                    else:
                        theta_value.append(next_element)
                else:
                    theta_value.append(self.theta[i])
        if self.set_parameters(theta_value) == False:
            raise ValueError("Prior generates values that are out the model parameter domain.")

    def sample(self, n_simulate):
        # Generate n_simulate time-series of weather variables satisfying Lorenz 95 equations
        timeseries_array = [None] * n_simulate

        # Initialize timesteps
        time_steps = np.linspace(0, 4, self.n_timestep)

        for k in range(0, n_simulate):
            # Define a parameter object containing parameters which is needed
            # to evaluate the ODEs
            # Stochastic forcing term
            eta = self.sigma_e * np.sqrt(1 - pow(self.phi, 2)) * self.rng.normal(0, 1, self.initial_state.shape[0])

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
                eta = self.phi * eta + self.sigma_e * np.sqrt(1 - pow(self.phi, 2)) * self.rng.normal(0, 1)
            timeseries_array[k] = timeseries
        # return an array of objects of type Timeseries
        return timeseries_array

    def get_parameters(self):
        if(isinstance(self.theta, Distribution)):
            l1 = [[self.theta_value, self.theta.get_parameters()]]
        else:
            l1 = []
            for i in range(len(self.theta)):
                if(isinstance(self.theta[i],Distribution)):
                    samples_helper = self.theta[i].sample(1,reset=1)[0]
                    l_helper=[]
                    if(isinstance(samples_helper,np.ndarray)):
                        for j in range(len(samples_helper)):
                            l_helper.append(self.theta_value[i+j])
                    else:
                        l_helper = self.theta_value[i]
                    l1.append([l_helper, self.theta[i].get_parameters()])
                else:
                    l1.append(self.theta_value[i])
        return [l1]

    def set_parameters(self, theta):
        for i in range(len(theta)):
            if(isinstance(theta[i], list)):
                if(isinstance(theta[i][0],list)):
                    for j in range(len(theta[i][0])):
                        self.theta_value[i+j]=theta[i][0][j]
                    if(len(theta[i][0])==len(self.theta_value)):
                        flag = True
                    else:
                        flag = False
                else:
                    self.theta_value[i]=theta[i][0]
                    flag=False
                if(flag):
                    self.theta.set_parameters(theta[i][1])
                else:
                    self.theta[i].set_parameters(theta[i][1])
            else:
                self.theta_value[i]=theta[i]
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

    def pdf(self):
        raise TypeError('Lorenz95 does not have a likelihood.')


class Ricker(Distribution):
    """Ecological model that describes the observed size of animal population over time
    described in [1].

    [1] S. N. Wood. Statistical inference for noisy nonlinear ecological
    dynamic systems. Nature, 466(7310):1102–1104, Aug. 2010.

    Parameters
    ----------
    theta: 3 dimensional list or distribution
        The parameter is a vector consisting of three numbers \
        :math:`\log r` (real number), :math:`\sigma` (positive real number, > 0), :math:`\phi` (positive real number > 0)
    n_timestep: int, optional
        Number of timesteps. The default value is 100.
    seed: int, optional
        Initial seed. The default value is generated randomly.
    """

    def __init__(self, theta, n_timestep=100, seed=None):
        self.n_timestep = n_timestep
        self.theta = theta
        self.rng = np.random.RandomState(seed)

        if(isinstance(self.theta, Distribution)):
            self.theta_value = self.theta.sample(1)[0].tolist()
        else:
            theta_value = []
            for i in range(len(self.theta)):
                if(isinstance(self.theta[i],Distribution)):
                    next_element = self.theta[i].sample(1)[0]
                    if(isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            theta_value.append(next_element[j])
                    else:
                        theta_value.append(next_element)
                else:
                    theta_value.append(self.theta[i])
            self.theta_value = np.array(theta_value)

    def sample_from_prior(self):
        if (isinstance(self.theta, Distribution)):
            self.theta.sample_from_prior()
            theta_value = self.theta.sample(1)[0].tolist()
        else:
            theta_value = []
            for i in range(len(self.theta)):
                if (isinstance(self.theta[i], Distribution)):
                    self.theta[i].sample_from_prior()
                    next_element = self.theta[i].sample(1)[0]
                    if (isinstance(next_element, np.ndarray)):
                        for j in range(len(next_element)):
                            theta_value.append(next_element[j])
                    else:
                        theta_value.append(next_element)
                else:
                    theta_value.append(self.theta[i])
        if self.set_parameters(theta_value) == False:
            raise ValueError("Prior generates values that are out the model parameter domain.")

    def sample(self, n_simulate):
        timeseries_array = [None] * n_simulate
        # Initialize local parameters
        log_r = self.theta_value[0]
        sigma = self.theta_value[1]
        phi = self.theta_value[2]
        for k in range(0, n_simulate):
            # Initialize the time-series
            timeseries_obs_size = np.zeros(shape=(self.n_timestep), dtype=np.float)
            timeseries_true_size = np.ones(shape=(self.n_timestep), dtype=np.float)
            for ind in range(1, self.n_timestep - 1):
                timeseries_true_size[ind] = np.exp(log_r + np.log(timeseries_true_size[ind - 1]) - timeseries_true_size[
                    ind - 1] + sigma * self.rng.normal(0, 1))
                timeseries_obs_size[ind] = self.rng.poisson(phi * timeseries_true_size[ind])
            timeseries_array[k] = timeseries_obs_size
        # return an array of objects of type Timeseries
        return timeseries_array

    def get_parameters(self):
        if (isinstance(self.theta, Distribution)):
            l1 = [[self.theta_value, self.theta.get_parameters()]]
        else:
            l1 = []
            for i in range(len(self.theta)):
                if (isinstance(self.theta[i], Distribution)):
                    samples_helper = self.theta[i].sample(1, reset=1)[0]
                    l_helper = []
                    if (isinstance(samples_helper, np.ndarray)):
                        for j in range(len(samples_helper)):
                            l_helper.append(self.theta_value[i + j])
                    else:
                        l_helper = self.theta_value[i]
                    l1.append([l_helper, self.theta[i].get_parameters()])
                else:
                    l1.append(self.theta_value[i])
        return [l1]

    def set_parameters(self, theta):
        #TODO CHECK LENGHT, USW
        for i in range(len(theta)):
            if(isinstance(theta[i], list)):
                if(isinstance(theta[i][0],list)):
                    for j in range(len(theta[i][0])):
                        self.mean_value[i+j]=theta[i][0][j]
                    if(len(theta[i][0])==len(self.theta_value)):
                        flag = True
                    else:
                        flag = False
                else:
                    self.mean_value[i]=theta[i][0]
                    flag=False
                if(flag):
                    self.mean.set_parameters(theta[i][1])
                else:
                    self.mean[i].set_parameters(theta[i][1])
            else:
                self.mean_value[i]=theta[i]
        return True

    def pdf(self):
        raise TypeError('Ricker does not have a likelihood.')

    def reseed(self, seed):
        self.rng.seed(seed)







