from abc import ABCMeta, abstractmethod


class JointApprox_likelihood(metaclass = ABCMeta):
    """This abstract base class defines how the combination of distances computed on the observed and
    simulated datasets corresponding to different root models should be implemented.
    """
    
    @abstractmethod
    def __init__(self, models, approx_lhds):
        """The constructor of a sub-class must accept non-optional models and corresponding distances
        as parameters.

        Parameters
        ----------
        models : list
            A list of root models which are of each of type abcpy.probabilisticmodels
        approx_lhds: list
            A list of approximate likelihoods which are of each of type abcpy.approx_lhd and they should be
            in the same order as corresponding root models for which it would be used to compute the
            approximate likelihood
        """
        
        raise NotImplementedError

    @abstractmethod
    def likelihood(d1, d2):
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
        float
            Computed approximate likelihood.
        """

        raise NotImplemented

class ProductCombination(JointApprox_likelihood):
    """
    This class implements the product combination of different approximate likelihoods computed on different datasets corresponding to
    different root models

    """
    
    def __init__(self, models, approx_lhds):

        if len(models)!=len(approx_lhds):
            raise ValueError('Number of root models and Number of assigned approximate likelihoods are not same')

        self.models = models
        self.approx_lhds = approx_lhds


    def likelihood(self, d1, d2):
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

        combined_likelihood = 1.0
        for ind in range(len(self.approx_lhds)):
            combined_likelihood *= self.approx_lhds[ind].likelihood(d1[ind], d2[ind])

        return combined_likelihood