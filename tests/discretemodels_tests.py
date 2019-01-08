from abcpy.discretemodels import *
from tests.probabilisticmodels_tests import AbstractAPIImplementationTests

import unittest

"""Tests whether the methods defined for discrete probabilistic models are working as intended."""


class BernoulliAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Bernoulli]
    model_inputs = [[0.5]]

class BinomialAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Binomial]
    model_inputs = [[3, 0.5]]

class PoissonAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Poisson]
    model_inputs = [[3]]

class DiscreteUniformTests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [DiscreteUniform]
    model_inputs = [[10, 20]]

class CheckParametersAtInitializationTests(unittest.TestCase):
    """Tests that no probabilistic model with invalid parameters can be initialized."""

    # TODO: Test for all distributions the behaviour if input parameters are real distributions and not only
    # hyperparameters

    def test_Bernoulli(self):
        with self.assertRaises(TypeError):
            Bernoulli(np.array([1, 2, 3]))

        with self.assertRaises(ValueError):
            Bernoulli([[1], [4]])

    def test_Binomial(self):
        with self.assertRaises(TypeError):
            Bernoulli(np.array([1, 2, 3]))

        with self.assertRaises(ValueError):
            Bernoulli([1, 2, 3])

    def test_Poisson(self):
        with self.assertRaises(TypeError):
            Poisson(np.array([1, 2, 3]))

        with self.assertRaises(ValueError):
            Poisson([2, 3])

    def test_DiscreteUniform(self):
        with self.assertRaises(TypeError):
            DiscreteUniform(np.array([1, 2, 3]))

        with self.assertRaises(ValueError):
            DiscreteUniform([2, 3, 4])


class DimensionTests(unittest.TestCase):
    """Tests whether the dimensions of all discrete models are defined in the correct way."""

    def test_Bernoulli(self):
        Bn = Bernoulli([0.5])
        self.assertTrue(Bn.get_output_dimension()==1)

    def test_Binomial(self):
        Bi = Binomial([1, 0.5])
        self.assertTrue(Bi.get_output_dimension()==1)

    def test_Poisson(self):
        Po = Poisson([3])
        self.assertTrue(Po.get_output_dimension()==1)

    def test_DiscreteUniform(self):
        Du = DiscreteUniform([10, 20])
        self.assertTrue(Du.get_output_dimension()==1)


class SampleFromDistributionTests(unittest.TestCase):
    """Tests the return value of forward_simulate for all discrete distributions."""
    def test_Bernoulli(self):
        Bn = Bernoulli([0.5])
        samples = Bn.forward_simulate(Bn.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples)==3)

    def test_Binomial(self):
        Bi = Binomial([1, 0.1])
        samples = Bi.forward_simulate(Bi.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)

    def test_Poisson(self):
        Po = Poisson([3])
        samples = Po.forward_simulate(Po.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)

    def test_DiscreteUniform(self):
        Du = DiscreteUniform([10, 20])
        samples = Du.forward_simulate(Du.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)


class CheckParametersBeforeSamplingTests(unittest.TestCase):
    """Tests whether False will be returned if the input parameters of _check_parameters_before_sampling are not
    accepted."""

    def test_Bernoulli(self):
        Bn = Bernoulli([0.5])
        self.assertFalse(Bn._check_input([-.3]))
        self.assertFalse(Bn._check_input([1.2]))

    def test_Binomial(self):
        Bi = Binomial([1, 0.5])
        with self.assertRaises(TypeError):
            self.assertFalse(Bi._check_input([3, .5, 5]))
            self.assertFalse(Bi._check_input([.3, .5]))
            self.assertFalse(Bi._check_input([-2, .5]))
            self.assertFalse(Bi._check_input([3, -.3]))
            self.assertFalse(Bi._check_input([3, 1.2]))

    def test_Poisson(self):
        Po = Poisson([3])
        self.assertFalse(Po._check_input([3, 5]))
        self.assertFalse(Po._check_input([-1]))

    def test_DiscreteUniform(self):
        Du = DiscreteUniform([10, 20])
        self.assertFalse(Du._check_input([3.0, 5]))
        self.assertFalse(Du._check_input([2, 6.0]))
        self.assertFalse(Du._check_input([5, 2]))


if __name__ == '__main__':
    unittest.main()
