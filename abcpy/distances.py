from abc import ABCMeta, abstractmethod

import numpy as np
from glmnet import LogitNet
from sklearn import linear_model

#TODO  we could give an option to initialize the default distance with either of the three other distances. Then, we would not expect the other threes to be ever called by the user and could remove the [] in the default/d1[0] in the rest.

class Distance(metaclass = ABCMeta):
    """This abstract base class defines how the distance between the observed and
    simulated data should be implemented.    
    """
    
    @abstractmethod
    def __init__(self, statistics_calc):
        """The constructor of a sub-class must accept a non-optional statistics
        calculator as a parameter. If stored to self.statistics_calc, the
        private helper method _calculate_summary_stat can be used.

        Parameters
        ----------
        statistics_calc : abcpy.stasistics.Statistics 
            Statistics extractor object that conforms to the Statistics class.
        """
        
        raise NotImplementedError

    
    @abstractmethod
    def distance(d1, d2):
        """To be overwritten by any sub-class: should calculate the distance between two
        sets of data d1 and d2 using their respective statistics.

        Notes
        -----
        The data sets d1 and d2 are array-like structures that contain n1 and n2 data
        points each.  An implementation of the distance function should work along
        the following steps:
        
        1. Transform both input sets dX = [ dX1, dX2, ..., dXn ] to sX = [sX1, sX2,
        ..., sXn] using the statistics object. See _calculate_summary_stat method.
        
        2. Calculate the mutual desired distance, here denoted by -, between the
        statstics dist = [s11 - s21, s12 - s22, ..., s1n - s2n].
        
        Important: any sub-class must not calculate the distance between data sets
        d1 and d2 directly. This is the reason why any sub-class must be
        initialized with a statistics object.
        
        Parameters
        ----------
        d1: Python list
            Contains n1 data points.
        d2: Python list
            Contains n2 data points.
        
        Returns
        -------
        numpy.ndarray
            The distance between the two input data sets.
        """
                
        raise NotImplementedError

    
    @abstractmethod
    def dist_max(self):
        """To be overwritten by sub-class: should return maximum possible value of the
        desired distance function.
 
        Examples
        --------
        If the desired distance maps to :math:`\mathbb{R}`, this method should return numpy.inf.

        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """
        
        raise NotImplementedError


    def _calculate_summary_stat(self,d1,d2):
        """Helper function that extracts the summary statistics s1 and s2 from d1 and
        d2 using the statistics object stored in self.statistics_calc.

        Parameters
        ----------
        d1 : array-like
            d1 contains n data sets.
        d2 : array-like
            d2 contains n data sets.

        Returns
        -------
        numpy.ndarray
            The summary statistics extracted from d1 and d2.

        """
        s1 = self.statistics_calc.statistics(d1)
        s2 = self.statistics_calc.statistics(d2)
        return (s1,s2)


class Euclidean(Distance):
    """
    This class implements the Euclidean distance between two vectors.

    The maximum value of the distance is np.inf.
    """
    
    def __init__(self, statistics):
        self.statistics_calc = statistics

        # Since the observations do always stay the same, we can save the summary statistics of them and not recalculate it each time
        self.s1 = None

        
    def distance(self, d1, d2):
        """Calculates the distance between two datasets.

        Parameters
        ----------
        d1, d2: list
            A list, containing a list describing the data set
        """
        d1 = d1[0]
        d2 = d2[0]
        if len(d1) != len(d2):
            raise BaseException("Input data sets have different sizes: {} vs {}".format(len(d1), len(d2)))
        if(self.s1 is None):
            self.s1 = self.statistics_calc.statistics(d1)
        s2 = self.statistics_calc.statistics(d2)

        # compute distance between the statistics
        dist = np.zeros(shape=(self.s1.shape[0],self.s1.shape[0]))
        for ind1 in range(0, self.s1.shape[0]):
            for ind2 in range(0, s2.shape[0]):
                dist[ind1,ind2] = np.sqrt(np.sum(pow(self.s1[ind1,:]-s2[ind2,:],2)))
                
        return dist.mean()

    
    def dist_max(self):
        return np.inf




class PenLogReg(Distance):
    """
    This class implements a distance mesure based on the classification accuracy.

    The classification accuracy is calculated between two dataset d1 and d2 using 
    lasso penalized logistics regression and return it as a distance. The lasso 
    penalized logistic regression is done using glmnet package of Friedman et. al.
    [2]. While computing the distance, the algorithm automatically chooses 
    the most relevant summary statistics as explained in Gutmann et. al. [1].
    The maximum value of the distance is 1.0.
       
    [1] Gutmann, M., Dutta, R., Kaski, S., and Corander, J. (2014). Statistical 
    inference of intractable generative models via classification. arXiv:1407.4981.
    
    [2] Friedman, J., Hastie, T., and Tibshirani, R. (2010). Regularization 
    paths for generalized linear models via coordinate descent. Journal of Statistical 
    Software, 33(1), 1–22.
    """

    def __init__(self, statistics):
        self.statistics_calc = statistics

        # Since the observations do always stay the same, we can save the summary statistics of them and not recalculate it each time
        self.s1 = None
        
    def distance(self, d1, d2):
        """Calculates the distance between two datasets.

        Parameters
        ----------
        d1, d2: list
            A list, containing a list describing the data set
        """
        # Extract summary statistics from the dataset
        if(self.s1 is None):
            self.s1 = self.statistics_calc.statistics(d1)
        s2 = self.statistics_calc.statistics(d2)
         
        # compute distnace between the statistics 
        training_set_features = np.concatenate((self.s1, s2), axis=0)
        label_s1 = np.zeros(shape=(len(self.s1), 1))
        label_s2 = np.ones(shape=(len(s2), 1))
        training_set_labels = np.concatenate((label_s1, label_s2), axis=0).ravel()

        m = LogitNet(alpha = 1, n_splits = 10)
        m = m.fit(training_set_features, training_set_labels)
        distance = 2.0 * (m.cv_mean_score_[np.where(m.lambda_path_== m.lambda_max_)[0][0]] - 0.5)
    
        return distance

    def dist_max(self):
        return 1.0
         
    
    

class LogReg(Distance):
    """This class implements a distance measure based on the classification
    accuracy [1]. The classification accuracy is calculated between two dataset d1 and d2 using 
    logistics regression and return it as a distance. The maximum value of the distance is 1.0.

    [1] Gutmann, M., Dutta, R., Kaski, S., and Corander, J. (2014). Statistical 
    inference of intractable generative models via classification. arXiv:1407.4981.
    """
    
    def __init__(self, statistics):
        self.statistics_calc = statistics

        # Since the observations do always stay the same, we can save the summary statistics of them and not recalculate it each time
        self.s1 = None
        
    def distance(self, d1, d2):
        """Calculates the distance between two datasets.

        Parameters
        ----------
        d1, d2: list
            A list, containing a list describing the data set
        """
        d1 = d1[0]
        d2 = d2[0]
        # Extract summary statistics from the dataset
        if(self.s1 is None):
            self.s1 = self.statistics_calc.statistics(d1)
        s2 = self.statistics_calc.statistics(d2)
        
        # compute distance between the statistics
        training_set_features = np.concatenate((self.s1, s2), axis=0)
        label_s1 = np.zeros(shape=(len(self.s1), 1))
        label_s2 = np.ones(shape=(len(s2), 1))
        training_set_labels = np.concatenate((label_s1, label_s2), axis=0).ravel()

        reg_inv = 1e5
        log_reg_model = linear_model.LogisticRegression(C=reg_inv, penalty='l1')
        log_reg_model.fit(training_set_features, training_set_labels)
        score = log_reg_model.score(training_set_features, training_set_labels)
        distance = 2.0 * (score - 0.5)
        return distance

    def dist_max(self):
        return 1.0


class DefaultJointDistance(Distance):
    """This class implements a default distance to be used when multiple root models exist. It uses LogReg as the distance calculator for each root model, and adds all individual distances.

    Parameters
    ----------
    statistics: abcpy.statistics object
        The statistics calculator to be used
    number_of_models: integer
        The number of root models on which the distance will act.
    """
    def __init__(self, statistics):
        self.statistics_calc = statistics
        self.distance_calc = LogReg(self.statistics_calc)

    def distance(self, d1, d2):
        total_distance = 0
        for observed_data, simulated_data in zip(d1,d2):
            total_distance+=self.distance_calc.distance([observed_data], [simulated_data])
        total_distance/=len(d2)
        return total_distance

    def dist_max(self):
        return 1.0
         
    
