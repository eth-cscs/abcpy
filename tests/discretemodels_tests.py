from abcpy.probabilisticmodels import *
from abcpy.discretemodels import *
import unittest

"""Tests whether the methods defined for discrete probabilistic models are working as intended."""


class CheckParametersAtInitializationTests(unittest.TestCase):
    """Tests that no probabilistic model with invalid parameters can be initialized."""
    def test_binomial(self):
        with self.assertRaises(ValueError):
            Binomial([-1, 0.5])
        with self.assertRaises(ValueError):
            Binomial([1, -0.1])
        with self.assertRaises(ValueError):
            Binomial([1, 3])


class CheckParametersBeforeSamplingTests(unittest.TestCase):
    """Tests whether False will be returned if the input parameters of _check_parameters_before_sampling are not accepted."""
    def test_binomial(self):
        B = Binomial([1, 0.1])
        self.assertFalse(B._check_parameters(InputConnector.from_list([-1, 0.1])))
        self.assertFalse(B._check_parameters(InputConnector.from_list([1, -0.1])))
        self.assertFalse(B._check_parameters(InputConnector.from_list([1, 3])))


class SampleFromDistributionTests(unittest.TestCase):
    """Tests the return value of forward_simulate for all discrete distributions."""
    def test_binomial(self):
        B = Binomial([1, 0.1])
        samples = B.forward_simulate(3)
        self.assertTrue(isinstance(samples, np.ndarray))
        self.assertTrue(len(samples)==3)


if __name__ == '__main__':
    unittest.main()
