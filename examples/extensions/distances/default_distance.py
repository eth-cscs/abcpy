import numpy as np

from abcpy.distances import Distance, Euclidean


class DefaultJointDistance(Distance):
    """
    This class implements a default distance to be used when multiple root
    models exist. It uses LogReg as the distance calculator for each root model, and
    adds all individual distances.

    Parameters
    ----------
    statistics: abcpy.statistics object
        The statistics calculator to be used
    number_of_models: integer
        The number of root models on which the distance will act.
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
