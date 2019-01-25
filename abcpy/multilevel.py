from abc import ABCMeta, abstractmethod

import numpy as np
from glmnet import LogitNet
from sklearn import linear_model


class Multilevel(metaclass=ABCMeta):
    """This abstract base class defines how the distance between the observed and
    simulated data should be implemented.
    """

    @abstractmethod
    def __init__(self, backend, data_thinner, criterion_calculator):
        """The constructor of a sub-class must accept a non-optional data thinner and criterion
        calculator as parameters.

        Parameters
        ----------
        backend: abcpy.backend
            Backend object
        data_thinner : object
            Object that operates on data and thins it
        criterion_calculator: object
            Object that operates on n_samples_per_param data and computes the criterion
        """

        self.bacend = backend
        self.data_thinner = data_thinner
        self.criterion_calculator = criterion_calculator

        raise NotImplementedError

    @abstractmethod
    def compute(self, d, n_repeat):
        """To be overwritten by any sub-class: should calculate the criterion for each
        set of data_element in the lis data

        Notes
        -----
        The data set d is an array-like structures that contain n data
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
        d: Python list
            Contains n data points.


        Returns
        -------
        numpy.ndarray
            The criterion calculated for each data point.
        """

        raise NotImplementedError

    ## Simple_map and Flat_map: Python wrapper for nested parallelization
    def simple_map(self, data, map_function):
        data_pds = self.backend.parallelize(data)
        result_pds = self.backend.map(map_function, data_pds)
        result = self.backend.collect(result_pds)
        main_result, counter = [list(t) for t in zip(*result)]
        return main_result, counter

    def flat_map(self, data, n_repeat, map_function):
        # Create an array of data, with each data repeated n_repeat many times
        repeated_data = np.repeat(data, n_repeat, axis=0)
        # Create an see array
        n_total = n_repeat * data.shape[0]
        seed_arr = self.rng.randint(1, n_total * n_total, size=n_total, dtype=np.int32)
        rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
        # Create data and rng array
        repeated_data_rng = [[repeated_data[ind,:],rng_arr[ind]] for ind in range(n_total)]
        repeated_data_rng_pds = self.backend.parallelize(repeated_data_rng)
        # Map the function on the data using the corresponding rng
        repeated_data_result_pds = self.backend.map(map_function, repeated_data_rng_pds)
        repeated_data_result = self.backend.collect(repeated_data_result_pds)
        repeated_data, result = [list(t) for t in zip(*repeated_data_result)]
        merged_result_data = []
        for ind in range(0, data.shape[0]):
            merged_result_data.append([[[result[np.int(i)][0][0] \
                                         for i in
                                         np.where(np.mean(repeated_data == data[ind, :], axis=1) == 1)[0]]],
                                       data[ind, :]])
        return merged_result_data


class Prototype(Multilevel):
    