from abc import ABCMeta, abstractmethod

import numpy as np

class JointDistance(metaclass = ABCMeta):
    """This abstract base class defines how the combination of distances computed on the observed and
    simulated datasets corresponding to different root models should be implemented.
    """
    
    @abstractmethod
    def __init__(self, models, distances):
        """The constructor of a sub-class must accept non-optional models and corresponding distances
        as parameters.

        Parameters
        ----------
        models : list
            A list of root models which are of each of type abcpy.probabilisticmodels
        distances: list
            A list of distances which are of each of type abcpy.distances and they should be
            in the same order as corresponding root models for which it would be used to compute the
            distance
        """
        
        raise NotImplementedError

    
    @abstractmethod
    def distance(d1, d2):
        """To be overwritten by any sub-class: should calculate the distance between two
        sets of data d1 and d2.

        Notes
        -----
        The data sets d1 and d2 are lists that contain the datasets corresponding to the root models.
        Both d1 and d2 should have the datasets in the same order as the root models are.
        
        Parameters
        ----------
        d1: Python list
            Contains lists which are datasets corresponding to root models.
        d2: Python list
            Contains lists which are datasets corresponding to root models.
        
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

class LinearCombination(JointDistance):
    """
    This class implements the linear combination of different distances computed on different datasets corresponding to
    different root models

    The maximum value of the distance is linear combination of the maximum value of the different distances it combines.
    """
    
    def __init__(self, models, distances, weights=None):
        """Combine the distances between different datasets.

        Parameters
        ----------
        weights: list
            A list, containing the weights (for linear combination) corresponding to each of the distances. Should be
            the same length of models. The default value if None, for which we assign equal weights to all distances
        """
        if len(models)!=len(distances):
            raise ValueError('Number of root models and Number of assigned distances are not same')

        if weights is None:
            self.weights = weights
            self.weights = np.ones(shape=(len(distances,)))/len(distances)
        else:
            if len(distances) != len(weights):
                raise ValueError('Number of distances and Number of weights are not same')
            else:
                weights = np.array(weights)
                self.weights = np.array(weights/sum(weights)).reshape(-1,)

        self.models = models
        self.distances = distances


    def distance(self, d1, d2):
        """Combine the distances between different datasets.

        Parameters
        ----------
        d1, d2: list
            A list, containing lists describing the different data sets
        """
        if not isinstance(d1, list):
            raise TypeError('Data is not of allowed types')
        if not isinstance(d2, list):
            raise TypeError('Data is not of allowed types')
        if len(d1)!=len(d2):
            raise ValueError('Both the datasets should contain dataset for each of the root models')

        combined_distance = 0.0
        for ind in range(len(self.distances)):
            combined_distance += self.weights[ind]*self.distances[ind].distance(d1[ind], d2[ind])

        return combined_distance

    
    def dist_max(self):
        combined_distance_max = 0.0
        for ind in range(len(self.distances)):
            combined_distance_max += self.weights[ind]*self.distances[ind].dist_max()
        return combined_distance_max
