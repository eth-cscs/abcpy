import unittest

import numpy as np

from abcpy.backends import BackendDummy as Backend
from abcpy.continuousmodels import Normal
from abcpy.continuousmodels import Uniform
from abcpy.statistics import Identity
from abcpy.statisticslearning import Semiautomatic, SemiautomaticNN, TripletDistanceLearning, \
    ContrastiveDistanceLearning, ExpFamStatistics

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True
    from abcpy.NN_utilities.networks import createDefaultNN


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
                                                      n_samples_val=100, n_samples_per_param=1, seed=1, n_epochs=2,
                                                      scale_samples=False, use_tqdm=False)
            self.statisticslearning2 = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=10,
                                                       n_samples_val=10, n_samples_per_param=1, seed=1, n_epochs=5,
                                                       scale_samples=False, use_tqdm=False)
            # with sample scaler:
            self.statisticslearning_with_scaler = SemiautomaticNN([self.Y], self.statistics_cal, self.backend,
                                                                  n_samples=100, n_samples_per_param=1, seed=1,
                                                                  n_epochs=2, scale_samples=True, use_tqdm=False)

    def test_initialization(self):
        if not has_torch:
            self.assertRaises(ImportError, SemiautomaticNN, [self.Y], self.statistics_cal, self.backend)

    def test_transformation(self):
        if has_torch:
            # Transform statistics extraction
            self.new_statistics_calculator = self.statisticslearning.get_statistics()
            self.new_statistics_calculator_with_scaler = self.statisticslearning_with_scaler.get_statistics()
            # Simulate observed data
            Obs = Normal([2, 4])
            y_obs = Obs.forward_simulate(Obs.get_input_values(), 1)[0].tolist()

            extracted_statistics = self.new_statistics_calculator.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator.statistics, [np.array([1, 2])])

            extracted_statistics = self.new_statistics_calculator_with_scaler.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator_with_scaler.statistics, [np.array([1, 2])])

    def test_errors(self):
        if has_torch:
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1, parameters=np.ones((100, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          embedding_net=createDefaultNN(1, 2))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1, simulations=np.ones((100, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          simulations=np.ones((100, 1, 3)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          parameters=np.ones((100, 1, 2)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1, simulations=np.ones((100, 1)),
                                                          parameters=np.zeros((99, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          parameters_val=np.ones((100, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          simulations_val=np.ones((100, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          simulations_val=np.ones((100, 1, 3)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          parameters_val=np.ones((100, 1, 2)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          simulations_val=np.ones((100, 1)),
                                                          parameters_val=np.zeros((99, 1)))
            with self.assertRaises(TypeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          parameters=[i for i in range(10)],
                                                          simulations=[i for i in range(10)])
            with self.assertRaises(TypeError):
                self.statisticslearning = SemiautomaticNN([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                          n_samples_per_param=1, seed=1,
                                                          parameters_val=[i for i in range(10)],
                                                          simulations_val=[i for i in range(10)])
            with self.assertRaises(RuntimeError):
                self.statisticslearning2.test_losses = [4, 2, 1]
                self.statisticslearning2.plot_losses()
            with self.assertRaises(NotImplementedError):
                self.statisticslearning.plot_losses(which_losses="foo")

    def test_plots(self):
        if has_torch:
            self.statisticslearning.plot_losses()
            self.statisticslearning.plot_losses(which_losses="train")
            self.statisticslearning.plot_losses(which_losses="test")


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
                                                                  n_samples=100, n_samples_val=100,
                                                                  n_samples_per_param=1, seed=1, n_epochs=2,
                                                                  scale_samples=False, use_tqdm=False)
            # with sample scaler:
            self.statisticslearning_with_scaler = ContrastiveDistanceLearning([self.Y], self.statistics_cal,
                                                                              self.backend, n_samples=100,
                                                                              n_samples_per_param=1, seed=1,
                                                                              n_epochs=2, scale_samples=True,
                                                                              use_tqdm=False)

    def test_initialization(self):
        if not has_torch:
            self.assertRaises(ImportError, ContrastiveDistanceLearning, [self.Y], self.statistics_cal,
                              self.backend)

    def test_transformation(self):
        if has_torch:
            # Transform statistics extraction
            self.new_statistics_calculator = self.statisticslearning.get_statistics()
            self.new_statistics_calculator_with_scaler = self.statisticslearning_with_scaler.get_statistics()
            # Simulate observed data
            Obs = Normal([2, 4])
            y_obs = Obs.forward_simulate(Obs.get_input_values(), 1)[0].tolist()

            extracted_statistics = self.new_statistics_calculator.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator.statistics, [np.array([1, 2])])

            extracted_statistics = self.new_statistics_calculator_with_scaler.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator_with_scaler.statistics, [np.array([1, 2])])

    def test_plots(self):
        if has_torch:
            self.statisticslearning.plot_losses()
            self.statisticslearning.plot_losses(which_losses="train")
            self.statisticslearning.plot_losses(which_losses="test")


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
                                                              n_samples=100, n_samples_val=100, n_samples_per_param=1,
                                                              seed=1, n_epochs=2, scale_samples=False, use_tqdm=False)
            # with sample scaler:
            self.statisticslearning_with_scaler = TripletDistanceLearning([self.Y], self.statistics_cal, self.backend,
                                                                          scale_samples=True, use_tqdm=False,
                                                                          n_samples=100, n_samples_per_param=1, seed=1,
                                                                          n_epochs=2)

    def test_initialization(self):
        if not has_torch:
            self.assertRaises(ImportError, TripletDistanceLearning, [self.Y], self.statistics_cal, self.backend)

    def test_transformation(self):
        if has_torch:
            # Transform statistics extraction
            self.new_statistics_calculator = self.statisticslearning.get_statistics()
            self.new_statistics_calculator_with_scaler = self.statisticslearning_with_scaler.get_statistics()
            # Simulate observed data
            Obs = Normal([2, 4])
            y_obs = Obs.forward_simulate(Obs.get_input_values(), 1)[0].tolist()

            extracted_statistics = self.new_statistics_calculator.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator.statistics, [np.array([1, 2])])

            extracted_statistics = self.new_statistics_calculator_with_scaler.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))

            self.assertRaises(RuntimeError, self.new_statistics_calculator_with_scaler.statistics, [np.array([1, 2])])

    def test_plots(self):
        if has_torch:
            self.statisticslearning.plot_losses()
            self.statisticslearning.plot_losses(which_losses="train")
            self.statisticslearning.plot_losses(which_losses="test")


class ExpFamStatisticsTests(unittest.TestCase):
    def setUp(self):
        # define prior and model
        sigma = Uniform([[1], [2]])
        mu = Normal([0, 1])
        self.Y = Normal([mu, sigma])

        # define backend
        self.backend = Backend()

        # define statistics
        self.statistics_cal = Identity(degree=3, cross=False)

        if has_torch:
            self.statisticslearning_all_defaults = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                                    n_samples=4, n_epochs=2, use_tqdm=False)
            self.statisticslearning_no_sliced = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                                 n_samples=4, n_epochs=2,
                                                                 sliced=False, use_tqdm=False)
            self.statisticslearning_sphere_noise = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                                    n_samples=4, n_epochs=2, use_tqdm=False,
                                                                    noise_type="sphere")
            self.statisticslearning_gaussian_noise = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                                      n_samples=4, n_epochs=2, use_tqdm=False,
                                                                      noise_type="gaussian")
            self.statisticslearning_variance_reduction = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                                          n_samples=4, n_epochs=2, use_tqdm=False,
                                                                          variance_reduction=True)
            self.statisticslearning_no_bn = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=4,
                                                             n_epochs=2, batch_norm=False, use_tqdm=False)
            self.statisticslearning_provide_nets = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                                    n_samples=4, n_epochs=2,
                                                                    statistics_net=createDefaultNN(3, 3)(),
                                                                    parameters_net=createDefaultNN(2, 2)(),
                                                                    use_tqdm=False)
            self.statisticslearning_embedding_dim = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                                     n_samples=4, n_epochs=2,
                                                                     embedding_dimension=4, use_tqdm=False)
            self.statisticslearning_validation_early_stop = ExpFamStatistics([self.Y], self.statistics_cal,
                                                                             self.backend,
                                                                             n_samples=4, n_epochs=20,
                                                                             n_samples_val=20, early_stopping=True,
                                                                             use_tqdm=False)
            self.statisticslearning_scale = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                             n_samples=4, n_epochs=2, scale_samples=False,
                                                             scale_parameters=True, use_tqdm=False)
            self.statisticslearning_bounds = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                              n_samples=4, n_epochs=2,
                                                              lower_bound_simulations=np.array([-1000, -1000, -1000]),
                                                              upper_bound_simulations=np.array([1000, 1000, 1000]),
                                                              use_tqdm=False, seed=1)
            self.statisticslearning_no_schedulers = ExpFamStatistics([self.Y], self.statistics_cal, self.backend,
                                                                     n_samples=4, n_epochs=2,
                                                                     scheduler_parameters=False,
                                                                     scheduler_simulations=False, use_tqdm=False)

    def test_initialization(self):
        if not has_torch:
            self.assertRaises(ImportError, ExpFamStatistics, [self.Y], self.statistics_cal, self.backend)

    def test_transformation(self):
        if has_torch:
            self.new_statistics_calculator = self.statisticslearning_all_defaults.get_statistics()
            # with no scaler on data:
            self.new_statistics_calculator_no_scaler = self.statisticslearning_scale.get_statistics()
            # with no rescaling of the statistics:
            self.new_statistics_calculator_no_rescale = self.statisticslearning_all_defaults.get_statistics(
                rescale_statistics=False)

            # Simulate observed data
            Obs = Normal([2, 4])
            y_obs = Obs.forward_simulate(Obs.get_input_values(), 1)[0].tolist()

            extracted_statistics = self.new_statistics_calculator.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics), (1, 2))
            extracted_statistics_no_rescale = self.new_statistics_calculator_no_rescale.statistics(y_obs)
            self.assertEqual(np.shape(extracted_statistics_no_rescale), (1, 2))
            self.assertFalse(np.allclose(extracted_statistics_no_rescale, extracted_statistics))

            self.assertRaises(RuntimeError, self.new_statistics_calculator.statistics, [np.array([1, 2])])
            self.assertRaises(RuntimeError, self.new_statistics_calculator_no_scaler.statistics, [np.array([1, 2])])

    def test_errors(self):
        if has_torch:
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, parameters=np.ones((100, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, statistics_net=createDefaultNN(1, 3))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, parameters_net=createDefaultNN(1, 3))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, noise_type="ciao", use_tqdm=False)
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, noise_type="sphere", variance_reduction=True,
                                                           use_tqdm=False)
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, simulations=np.ones((100, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           simulations=np.ones((100, 1, 3)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           parameters=np.ones((100, 1, 2)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, simulations=np.ones((100, 1)),
                                                           parameters=np.zeros((99, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           parameters_val=np.ones((100, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           simulations_val=np.ones((100, 1)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           simulations_val=np.ones((100, 1, 3)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           parameters_val=np.ones((100, 1, 2)))
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           simulations_val=np.ones((100, 1)),
                                                           parameters_val=np.zeros((99, 1)))
            with self.assertRaises(TypeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           parameters=[i for i in range(10)],
                                                           simulations=[i for i in range(10)])
            with self.assertRaises(TypeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1,
                                                           parameters_val=[i for i in range(10)],
                                                           simulations_val=[i for i in range(10)])
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, lower_bound_simulations=[1, 2, 3])
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           seed=1, upper_bound_simulations=[1, 2, 3])
            with self.assertRaises(RuntimeError):
                self.statisticslearning = ExpFamStatistics([self.Y], self.statistics_cal, self.backend, n_samples=1000,
                                                           lower_bound_simulations=np.array([-1000, -1000]), seed=1,
                                                           upper_bound_simulations=np.array([1000, 1000, 1000]))

            with self.assertRaises(RuntimeError):
                self.statisticslearning_all_defaults.test_losses = [4, 2, 1]
                self.statisticslearning_all_defaults.plot_losses()
            with self.assertRaises(NotImplementedError):
                self.statisticslearning_all_defaults.plot_losses(which_losses="foo")

    def test_plots(self):
        if has_torch:
            self.statisticslearning_all_defaults.plot_losses()
            self.statisticslearning_all_defaults.plot_losses(which_losses="train")
            self.statisticslearning_all_defaults.plot_losses(which_losses="test")


if __name__ == '__main__':
    unittest.main()
