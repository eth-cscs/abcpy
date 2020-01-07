import unittest
import numpy as np
from abcpy.continuousmodels import Uniform
from abcpy.continuousmodels import Normal
from abcpy.statistics import Identity
from abcpy.backends import BackendDummy as Backend
from abcpy.statisticslearning import Semiautomatic, SemiautomaticNN, TripletDistanceLearning, \
    ContrastiveDistanceLearning

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True


class SemiautomaticTests(unittest.TestCase):
    def setUp(self):
        # define prior and model
        sigma = Uniform([[10], [20]])
        mu = Normal([0, 1])
        Y = Normal([mu, sigma])

        # define backend
        self.backend = Backend()

        # define statistics
        self.statistics_cal = Identity(degree=3, cross=False)

        # Initialize statistics learning
        self.statisticslearning = Semiautomatic([Y], self.statistics_cal, self.backend, n_samples=1000,
                                                n_samples_per_param=1, seed=1)

    def test_transformation(self):
        # Transform statistics extraction
        self.new_statistics_calculator = self.statisticslearning.get_statistics()
        # Simulate observed data
        Obs = Normal([2, 4])
        y_obs = Obs.forward_simulate(Obs.get_input_values(), 1)[0].tolist()

        extracted_statistics = self.new_statistics_calculator.statistics(y_obs)
        self.assertEqual(np.shape(extracted_statistics), (1, 2))

        # NOTE we cannot test this, since the linear regression used uses a random number generator (which we cannot access and is in C). Therefore, our results differ and testing might fail
        # self.assertLess(extracted_statistics[0,0] - 0.00215507052338, 10e-2)
        # self.assertLess(extracted_statistics[0,1] - (-0.0058023274456), 10e-2)


class SemiautomaticNNTests(unittest.TestCase):
    def setUp(self):
        # define prior and model
        sigma = Uniform([[10], [20]])
        mu = Normal([0, 1])
        self.Y = Normal([mu, sigma])

        # define backend
        self.backend = Backend()

        # define statistics
        self.statistics_cal = Identity(degree=3, cross=False)

        if has_torch:
            # Initialize statistics learning
            self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=100,
                                                      n_samples_per_param=1, seed=1, n_epochs=10)

    def test_initialization(self):
        if not has_torch:
            self.assertRaises(ImportError, SemiautomaticNN, [self.Y], self.statistics_cal, self.backend)

    def test_transformation(self):
        if has_torch:
            # Transform statistics extraction
            self.new_statistics_calculator = self.statisticslearning.get_statistics()
            # Simulate observed data
            Obs = Normal([2, 4])
            y_obs = Obs.forward_simulate(Obs.get_input_values(), 1)[0].tolist()

            extracted_statistics = self.new_statistics_calculator.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator.statistics, [np.array([1, 2])])


class ContrastiveDistanceLearningTests(unittest.TestCase):
    def setUp(self):
        # define prior and model
        sigma = Uniform([[10], [20]])
        mu = Normal([0, 1])
        self.Y = Normal([mu, sigma])

        # define backend
        self.backend = Backend()

        # define statistics
        self.statistics_cal = Identity(degree=3, cross=False)

        if has_torch:
            # Initialize statistics learning
            self.statisticslearning = ContrastiveDistanceLearning([self.Y], self.statistics_cal, self.backend,
                                                                  n_samples=100, n_samples_per_param=1, seed=1,
                                                                  n_epochs=10)

    def test_initialization(self):
        if not has_torch:
            self.assertRaises(ImportError, ContrastiveDistanceLearning, [self.Y], self.statistics_cal,
                              self.backend)

    def test_transformation(self):
        if has_torch:
            # Transform statistics extraction
            self.new_statistics_calculator = self.statisticslearning.get_statistics()
            # Simulate observed data
            Obs = Normal([2, 4])
            y_obs = Obs.forward_simulate(Obs.get_input_values(), 1)[0].tolist()

            extracted_statistics = self.new_statistics_calculator.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator.statistics, [np.array([1, 2])])


class TripletDistanceLearningTests(unittest.TestCase):
    def setUp(self):
        # define prior and model
        sigma = Uniform([[10], [20]])
        mu = Normal([0, 1])
        self.Y = Normal([mu, sigma])

        # define backend
        self.backend = Backend()

        # define statistics
        self.statistics_cal = Identity(degree=3, cross=False)

        if has_torch:
            # Initialize statistics learning
            self.statisticslearning = TripletDistanceLearning([self.Y], self.statistics_cal, self.backend,
                                                              n_samples=100, n_samples_per_param=1, seed=1, n_epochs=10)

    def test_initialization(self):
        if not has_torch:
            self.assertRaises(ImportError, TripletDistanceLearning, [self.Y], self.statistics_cal, self.backend)

    def test_transformation(self):
        if has_torch:
            # Transform statistics extraction
            self.new_statistics_calculator = self.statisticslearning.get_statistics()
            # Simulate observed data
            Obs = Normal([2, 4])
            y_obs = Obs.forward_simulate(Obs.get_input_values(), 1)[0].tolist()

            extracted_statistics = self.new_statistics_calculator.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator.statistics, [np.array([1, 2])])


if __name__ == '__main__':
    unittest.main()
