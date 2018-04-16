from abcpy.continuousmodels import *
from tests.probabilisticmodels_tests import AbstractAPIImplementationTests

import unittest

"""Tests whether the methods defined for continuous probabilistic models are working as intended."""


class UniformAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Uniform]
    model_inputs = [[[0, 1], [1, 2]]]

class NormalAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Normal]
    model_inputs = [[0,1]]

class StundentTAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [StudentT]
    model_inputs = [[0, 3]]

class MultivariateNormalAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [MultivariateNormal]
    model_inputs = [[[1, 0], [[1, 0], [0, 1]]]]

class MultiStudentTAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [MultiStudentT]
    model_inputs = [[[1, 0], [[1, 0], [0, 1]], 3]]


class CheckParametersAtInitializationTests(unittest.TestCase):
    """Tests that no probabilistic model with invalid parameters can be initialized."""

    # TODO: Test for all distributions the behaviour if input parameters are real distributions and not only
    # hyperparameters

    def test_Uniform(self):
        with self.assertRaises(ValueError):
            Uniform([[1, 2, 3], [4, 5]])

    def test_Normal(self):
        with self.assertRaises(ValueError):
            Normal([1, -0.1])

    def test_StudentT(self):
        with self.assertRaises(ValueError):
            StudentT([1, 0])

    def test_MultivariateNormal(self):
        with self.assertRaises(ValueError):
            MultivariateNormal([[1]])

        with self.assertRaises(ValueError):
            MultivariateNormal([[1, 0, 0], [[1, 0], [0, 1]]])

        with self.assertRaises(ValueError):
            MultivariateNormal([[1, 0], [[1, 1], [0, 1]]])

        with self.assertRaises(ValueError):
            MultivariateNormal([[1, 0], [[-1, 0], [0, -1]]])

    def test_MultiStudentT(self):
        with self.assertRaises(ValueError):
            MultiStudentT([[1, 0], [[1, 1], [0, 1]], 1])

        with self.assertRaises(ValueError):
            MultiStudentT([[1, 0], [[-1, 0], [0, -1]], 1])

        with self.assertRaises(ValueError):
            MultiStudentT([[1, 0], [[1, 0], [0, 1]], -1])



class DimensionTests(unittest.TestCase):
    """Tests whether the dimensions of all continuous models are defined in the correct way."""

    def test_Uniform(self):
        U = Uniform([[0, 1], [1, 2]])
        self.assertTrue(U.get_output_dimension()==2)

    def test_Normal(self):
        N = Normal([1, 0.1])
        self.assertTrue(N.get_output_dimension()==1)

    def test_StudentT(self):
        S = StudentT([3, 1])
        self.assertTrue(S.get_output_dimension()==1)

    def test_MultivariateNormal(self):
        M = MultivariateNormal([[1, 0], [[1, 0], [0, 1]]])
        self.assertTrue(M.get_output_dimension()==2)

    def test_MultiStudentT(self):
        M = MultiStudentT([[1, 0], [[0.1, 0], [0, 0.1]], 1])
        self.assertTrue(M.get_output_dimension()==2)



class SampleFromDistributionTests(unittest.TestCase):
    """Tests the return value of forward_simulate for all continuous distributions."""
    def test_Normal(self):
        N = Normal([1, 0.1])
        samples = N.forward_simulate(N.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples)==3)

    def test_MultivariateNormal(self):
        M = MultivariateNormal([[1, 0], [[0.1, 0], [0, 0.1]]])
        samples = M.forward_simulate(M.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)

    def test_StudentT(self):
        S = StudentT([3, 1])
        samples = S.forward_simulate(S.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)

    def test_MultiStudentT(self):
        S = MultiStudentT([[1, 0], [[0.1, 0], [0, 0.1]], 1])
        samples = S.forward_simulate(S.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)

    def test_Uniform(self):
        U = Uniform([[0, 1], [1, 2]])
        samples = U.forward_simulate(U.get_input_values(), 3)
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(len(samples) == 3)


class CheckParametersBeforeSamplingTests(unittest.TestCase):
    """Tests whether False will be returned if the input parameters of _check_parameters_before_sampling are not accepted."""

    def test_Uniform(self):
        U = Uniform([[0, 1], [1, 2]])
        self.assertFalse(U._check_input([1, 1, 0, 2]))
        self.assertFalse(U._check_input([1, 1, 2, 0]))

    def test_Normal(self):
        N = Normal([1, 0.1])
        self.assertFalse(N._check_input([1, -0.1]))

    def test_StudentT(self):
        S = StudentT([3, 1])
        self.assertFalse(S._check_input([3, -1]))

    def test_MultivariateNormal(self):
        M = MultivariateNormal([[1, 0], [[0.1, 0], [0, 0.1]]])
        self.assertFalse(M._check_input([[1, 0], [[1, 1], [0, 1]]]))

        self.assertFalse(M._check_input([[1, 0], [[-1, 0], [0, -1]]]))

    def test_MultiStudentT(self):
        M = MultiStudentT([[1, 0], [[1, 0], [0, 1]], 1])

        self.assertFalse(M._check_input([[1, 0], [[1, 1], [1, 0]], 1]))

        self.assertFalse(M._check_input([[1, 0], [[-1, 0], [0, -1]], 1]))

        self.assertFalse(M._check_input([[1, 0], [[1, 0], [0, 1]], -1]))


if __name__ == '__main__':
    unittest.main()
