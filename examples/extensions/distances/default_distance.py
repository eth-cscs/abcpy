import numpy as np

from abcpy.distances import Distance, Euclidean


class DefaultJointDistance(Distance):
    """
    This class showcases how to implement a distance. It is actually a wrapper of the Euclidean distance, which is
    applied on each component of the provided datasets and summed.

    Parameters
    ----------
    statistics: abcpy.statistics object
        The statistics calculator to be used
    """

    def __init__(self, statistics):
        self.statistics_calc = statistics
        self.distance_calc = Euclidean(self.statistics_calc)

    def distance(self, d1, d2):
        total_distance = 0
        for observed_data, simulated_data in zip(d1, d2):
            total_distance += self.distance_calc.distance([observed_data], [simulated_data])
        total_distance /= len(d2)
        return total_distance

    def dist_max(self):
        return np.inf
