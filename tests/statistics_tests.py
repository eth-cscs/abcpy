import unittest

import numpy as np

from abcpy.backends import BackendDummy
from abcpy.continuousmodels import Uniform, Normal
from abcpy.inferences import DrawFromPrior
from abcpy.statistics import Identity, LinearTransformation, NeuralEmbedding

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True
    from abcpy.NN_utilities.networks import createDefaultNN, ScalerAndNet, DiscardLastOutputNet


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=1, cross=False)
        self.stat_calc_pipeline = Identity(degree=2, cross=False, previous_statistics=self.stat_calc)

        # try now the statistics rescaling option:
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        sampler = DrawFromPrior([self.model], BackendDummy(), seed=1)
        reference_parameters, reference_simulations = sampler.sample_par_sim_pairs(30, 1)
        reference_simulations = reference_simulations.reshape(reference_simulations.shape[0],
                                                              reference_simulations.shape[2])
        reference_simulations_double = np.concatenate([reference_simulations, reference_simulations], axis=1)

        self.stat_calc_rescaling = Identity(reference_simulations=reference_simulations_double)
        self.stat_calc_rescaling_2 = Identity(reference_simulations=reference_simulations)

    def test_statistics(self):
        self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
        vec1 = np.array([1, 2])
        vec2 = np.array([1])
        self.assertTrue((self.stat_calc.statistics([vec1]) == np.array([vec1])).all())
        self.assertTrue((self.stat_calc.statistics([vec1, vec1]) == np.array([[vec1], [vec1]])).all())
        self.assertTrue((self.stat_calc.statistics([vec2, vec2]) == np.array([[vec2], [vec2]])).all())

        self.assertTrue((self.stat_calc_rescaling.statistics([vec1]) != self.stat_calc.statistics([vec1])).all())
        self.assertTrue((self.stat_calc_rescaling_2.statistics([vec2]) != self.stat_calc.statistics([vec2])).all())

        self.assertRaises(RuntimeError, self.stat_calc_rescaling.statistics, [vec2])

    def test_polynomial_expansion(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.stat_calc._polynomial_expansion, 3.4)

        a = [np.array([0, 2]), np.array([2, 1])]
        # test cross-product part
        self.stat_calc = Identity(degree=2, cross=True)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0, 2, 0, 4, 0], [2, 1, 4, 1, 2]])).all())
        # When a tuple
        a = [np.array([0, 2])]
        self.stat_calc = Identity(degree=2, cross=True)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0, 2, 0, 4, 0]])).all())
        self.stat_calc = Identity(degree=2, cross=False)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0, 2, 0, 4]])).all())
        a = list(np.array([2]))
        self.stat_calc = Identity(degree=2, cross=True)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[2, 4]])).all())

    def test_pipeline(self):
        vec1 = np.array([1, 2])
        self.stat_calc_pipeline.statistics([vec1])


class LinearTransformationTests(unittest.TestCase):
    def setUp(self):
        self.coeff = np.array([[3, 4], [5, 6]])
        self.stat_calc = LinearTransformation(self.coeff, degree=1, cross=False)

        # try now the statistics rescaling option:
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        sampler = DrawFromPrior([self.model], BackendDummy(), seed=1)
        reference_parameters, reference_simulations = sampler.sample_par_sim_pairs(30, 1)
        reference_simulations = reference_simulations.reshape(reference_simulations.shape[0],
                                                              reference_simulations.shape[2])
        reference_simulations_double = np.concatenate([reference_simulations, reference_simulations], axis=1)

        self.stat_calc_rescaling = LinearTransformation(self.coeff, reference_simulations=reference_simulations_double)

    def test_statistics(self):
        self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
        vec1 = np.array([1, 2])
        vec2 = np.array([1])
        self.assertTrue((self.stat_calc.statistics([vec1]) == np.dot(vec1, self.coeff)).all())
        self.assertTrue((self.stat_calc.statistics([vec1, vec1]) == np.array(
            [np.dot(np.array([1, 2]), self.coeff), np.dot(np.array([1, 2]), self.coeff)])).all())
        self.assertRaises(ValueError, self.stat_calc.statistics, [vec2])

        self.assertTrue((self.stat_calc_rescaling.statistics([vec1]) != self.stat_calc.statistics([vec1])).all())

    def test_polynomial_expansion(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.stat_calc._polynomial_expansion, 3.4)

        a = [np.array([0, 2]), np.array([2, 1])]
        # test cross-product part
        self.stat_calc = LinearTransformation(self.coeff, degree=2, cross=True)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[10, 12, 100, 144, 120],
                                                                   [11, 14, 121, 196, 154]])).all())
        # When a tuple
        a = [np.array([0, 2])]
        self.stat_calc = LinearTransformation(self.coeff, degree=2, cross=True)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[10, 12, 100, 144, 120]])).all())
        self.stat_calc = LinearTransformation(self.coeff, degree=2, cross=False)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[10, 12, 100, 144]])).all())
        a = list(np.array([2]))
        self.stat_calc = LinearTransformation(self.coeff, degree=2, cross=True)
        self.assertRaises(ValueError, self.stat_calc.statistics, a)


class NeuralEmbeddingTests(unittest.TestCase):
    def setUp(self):
        if has_torch:
            self.net = createDefaultNN(2, 3)()
            self.net_with_scaler = ScalerAndNet(self.net, None)
            self.net_with_discard_wrapper = DiscardLastOutputNet(self.net)
            self.stat_calc = NeuralEmbedding(self.net)
            self.stat_calc_with_scaler = NeuralEmbedding(self.net_with_scaler)
            self.stat_calc_with_discard_wrapper = NeuralEmbedding(self.net_with_discard_wrapper)
            # reference input and output
            torch.random.manual_seed(1)
            self.tensor = torch.randn(1, 2)
            self.out = self.net(self.tensor)
            self.out_discard = self.net_with_discard_wrapper(self.tensor)

            # try now the statistics rescaling option:
            mu = Uniform([[-5.0], [5.0]], name='mu')
            sigma = Uniform([[0.0], [10.0]], name='sigma')
            # define a Gaussian model
            self.model = Normal([mu, sigma])

            sampler = DrawFromPrior([self.model], BackendDummy(), seed=1)
            reference_parameters, reference_simulations = sampler.sample_par_sim_pairs(30, 1)
            reference_simulations = reference_simulations.reshape(reference_simulations.shape[0],
                                                                  reference_simulations.shape[2])

            self.stat_calc_rescaling = NeuralEmbedding(self.net, reference_simulations=reference_simulations,
                                                       previous_statistics=Identity(degree=2))

        if not has_torch:
            self.assertRaises(ImportError, NeuralEmbedding, None)

    def test_statistics(self):
        if has_torch:
            self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
            vec1 = np.array([1, 2])
            vec2 = np.array([1])
            self.assertTrue((self.stat_calc.statistics([vec1])).all())
            self.assertTrue((self.stat_calc.statistics([vec1, vec1])).all())
            self.assertRaises(RuntimeError, self.stat_calc.statistics, [vec2])

            self.assertTrue((self.stat_calc_rescaling.statistics([vec2])).all())

    def test_save_load(self):
        if has_torch:
            self.stat_calc.save_net("net.pth")
            self.stat_calc_with_scaler.save_net("net.pth", path_to_scaler="scaler.pkl")
            stat_calc_loaded = NeuralEmbedding.fromFile("net.pth", input_size=2, output_size=3)
            stat_calc_loaded = NeuralEmbedding.fromFile("net.pth", network_class=createDefaultNN(2, 3))
            stat_calc_loaded_with_scaler = NeuralEmbedding.fromFile("net.pth", network_class=createDefaultNN(2, 3),
                                                                    path_to_scaler="scaler.pkl")
            # test the network was recovered correctly
            out_new = stat_calc_loaded.net(self.tensor)
            self.assertTrue(torch.allclose(self.out, out_new))

            # now with the DiscardLastOutput wrapper
            self.stat_calc_with_discard_wrapper.save_net("net_with_discard_wrapper.pth")
            stat_calc_with_discard_loaded = NeuralEmbedding.fromFile("net_with_discard_wrapper.pth", input_size=2,
                                                                     output_size=3)
            # test the network was recovered correctly
            out_new_discard = stat_calc_with_discard_loaded.net(self.tensor)
            self.assertTrue(torch.allclose(self.out_discard, out_new_discard))

            # now with both DiscardLastOutput and Scaler wrappers
            stat_calc_with_discard_and_scaler_loaded = NeuralEmbedding.fromFile("net_with_discard_wrapper.pth",
                                                                                input_size=2, output_size=3,
                                                                                path_to_scaler="scaler.pkl")

            with self.assertRaises(RuntimeError):
                self.stat_calc_with_scaler.save_net("net.pth")
                stat_calc_loaded = NeuralEmbedding.fromFile("net.pth")
                stat_calc_loaded = NeuralEmbedding.fromFile("net.pth", network_class=createDefaultNN(2, 3),
                                                            input_size=1)
                stat_calc_loaded = NeuralEmbedding.fromFile("net.pth", network_class=createDefaultNN(2, 3),
                                                            hidden_sizes=[2, 3])


if __name__ == '__main__':
    unittest.main()
