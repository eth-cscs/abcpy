from abcpy.discretemodels import *
from tests.probabilisticmodels_tests import AbstractAPIImplementationTests

import unittest

"""Tests whether the methods defined for continuous probabilistic models are working as intended."""


class BernoulliAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Bernoulli]
    model_inputs = [[[0, 1], [1, 2]]]

class BinomialAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Binomial]
    model_inputs = [[0,1]]

class PoissonAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Poisson]
    model_inputs = [[0, 3]]

class CheckParametersAtInitializationTests(unittest.TestCase):
    """Tests that no probabilistic model with invalid parameters can be initialized."""

    # TODO: Test for all distributions the behaviour if input parameters are real distributions and not only
    # hyperparameters

    def test_Bernoulli(self):
        with self.assertRaises(ValueError):
            Bernoulli([[1, 2, 3], [4, 5]])

    def test_Binomial(self):
        with self.assertRaises(ValueError):
            Binomial([1, -0.1])

    def test_Poisson(self):
        with self.assertRaises(ValueError):
            Poisson([1, 0])

class DimensionTests(unittest.TestCase):
    """Tests whether the dimensions of all continuous models are defined in the correct way."""

    def test_Bernoulli(self):
        Bn = Bernoulli([[0, 1], [1, 2]])
        self.assertTrue(Bn.get_output_dimension()==1)

    def test_Binomial(self):
        Bi = Binomial([1, 0.1])
        self.assertTrue(Bi.get_output_dimension()==1)

    def test_Poisson(self):
        Po = Poisson([3, 1])
        self.assertTrue(Po.get_output_dimension()==1)


class SampleFromDistributionTests(unittest.TestCase):
    """Tests the return value of forward_simulate for all continuous distributions."""
    def test_Bernoulli(self):
        Bn = Bernoulli([[0, 1], [1, 2]])
        samples = Bn.forward_simulate(3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples)==3)

    def test_Binomial(self):
        Bi = Binomial([1, 0.1])
        samples = Bi.forward_simulate(3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)

    def test_Poisson(self):
        Po = Poisson([3, 1])
        samples = Po.forward_simulate(3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)

class CheckParametersBeforeSamplingTests(unittest.TestCase):
    """Tests whether False will be returned if the input parameters of _check_parameters_before_sampling are not accepted."""

    def test_Bernoulli(self):
        Bn = Bernoulli([[0, 1], [1, 2]])
        self.assertFalse(Bn._check_input(InputConnector.from_list([1, 1, 0, 2])))
        self.assertFalse(Bn._check_input(InputConnector.from_list([1, 1, 2, 0])))

    def test_Binomial(self):
        Bi = Binomial([1, 0.1])
        self.assertFalse(N._check_input(InputConnector.from_list([1, -0.1])))

    def test_Poisson(self):
        Po = Poisson([3, 1])
        self.assertFalse(S._check_parameters_before_sampling(InputConnector.from_list([3, -1])))

if __name__ == '__main__':
    unittest.main()
