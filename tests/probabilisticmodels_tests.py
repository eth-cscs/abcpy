from abcpy.continuousmodels import *
from abcpy.probabilisticmodels import *
import unittest

"""These test cases implement tests for the probabilistic model class."""


class GetItemTest(unittest.TestCase):
    """Tests whether the get_item operator (i.e. access operator) works as intended."""

    def ErrorTest(self):
        """Tests whether an error is thrown if the operator tries to access an element outside the range of the return values of the probabilistic model."""
        N = Normal([1, 0.1], )
        with self.assertRaises(IndexError):
            N[2]

    def ReturnValueTest(self):
        """Tests whether the return value of the operator is correct"""
        N = Normal([1, 0.1], )
        result = N[0]
        self.assertTrue(isinstance(result,tuple))
        self.assertTrue(result[0]==N)
        self.assertTrue(result[1]==0)


class MappingTest(unittest.TestCase):
    """Tests whether the mapping created during initialization is done correctly."""
    def setUp(self):
        self.U = Uniform([[0, 2], [1, 3]])
        self.M = MultivariateNormal([[self.U[1], self.U[0]], [[1, 0], [0, 1]]])

    def test(self):
        self.assertTrue(self.M.get_input_connector().get_model(0) == self.U)
        self.assertTrue(self.M.get_input_connector().get_model(1) == self.U)


class GetParameterValuesTest(unittest.TestCase):
    """Tests whether get_input_values returns the correct values."""
    def test(self):
        U = Uniform([[0, 2], [1, 3]])
        self.assertTrue(U.get_input_values() == [0, 2, 1, 3])



class SampleParametersTest(unittest.TestCase):
    """Tests whether _forward_simulate_and_store_output returns False if the value of an input parameter is not an allowed value for the
    distribution."""

    def test(self):
        N1 = Normal([0.1, 0.01])
        N2 = Normal([1, N1])
        N1.fixed_values=[-0.1]
        self.assertFalse(N2._check_input(N2.get_input_connector()))


class GetOutputValuesTest(unittest.TestCase):
    """Tests whether get_output_values gives back values that can come from the distribution."""
    def test(self):
        U = Uniform([[0], [1]])
        U._forward_simulate_and_store_output()
        self.assertTrue((U.get_output_values()[0]>=0 and U.get_output_values()[0]<=1))


class ModelResultingFromOperationTests(unittest.TestCase):

    def test_check_parameters_at_initialization(self):
        N1 = Normal([1, 0.1])
        M1 = MultivariateNormal([[1, 1], [[1, 0], [0, 1]]])
        with self.assertRaises(ValueError):
            model = N1+M1

        def test_initialization(self):
            M1 = MultivariateNormal([[1, 1], [[1, 0], [0, 1]]])
            M2 = MultivariateNormal([[1, 1], [[1, 0], [0, 1]]])
            M3 = M1 + M2
            self.assertTrue(M3.get_output_dimension() == 2)


class SummationModelTests(unittest.TestCase):
    """Tests whether all methods associated with the SummationModel are working as intended."""


    def test_forward_simulate(self):
        N1 = Normal([1, 0.1])
        N2 = 10+N1
        rng=np.random.RandomState(1)
        N1._forward_simulate_and_store_output(rng=rng)

        sample = N2.forward_simulate(1, rng)

        self.assertTrue(isinstance(sample, np.ndarray))


class SubtractionModelTests(unittest.TestCase):
    """Tests whether all methods associated with the SubtractionModel are working as intended."""

    def test_forward_simulate(self):
        N1 = Normal([1, 0.1])
        N2 = 10-N1
        rng=np.random.RandomState(1)
        N1._forward_simulate_and_store_output(rng=rng)

        sample = N2.forward_simulate(1, rng)

        self.assertTrue(isinstance(sample, np.ndarray))


class MultiplicationModelTests(unittest.TestCase):
    """Tests whether all methods associated with the MultiplicationModel are working as intended."""

    def test_forward_simulate(self):
        N1 = Normal([1, 0.1])
        N2 = N1*2
        rng=np.random.RandomState(1)
        N1._forward_simulate_and_store_output(rng=rng)

        sample = N2.forward_simulate(1, rng)
        self.assertTrue(isinstance(sample, np.ndarray))

    def test_multiplication_from_right(self):
        N1 = Normal([1, 0.1])
        N2 = 2*N1

        self.assertTrue(N2.get_input_dimension()==2)
        self.assertTrue(isinstance(N2.get_input_connector().get_model(0), Hyperparameter))


class DivisionModelTests(unittest.TestCase):
    """Tests whether all methods associated with the DivisionModel are working as intended."""

    def test_sample_from_distribution(self):
        N1 = Normal([1, 0.1])
        N2 = Normal([2, 0.1])
        N3 = N1/N2
        rng = np.random.RandomState(1)

        N1._forward_simulate_and_store_output(rng=rng)
        N2._forward_simulate_and_store_output(rng=rng)
        sample = N3.forward_simulate(1)

        self.assertTrue(isinstance(sample, np.ndarray))

    def test_division_from_right(self):
        N1 = Normal([1, 0.1])
        N2 = 2/N1

        self.assertEqual(N2.get_input_dimension(), 2)
        self.assertTrue(isinstance(N2.get_input_connector().get_model(0), Hyperparameter))


class ExponentialModelTests(unittest.TestCase):
    """Tests whether all methods associated with ExponentialModel are working as intended."""

    def test_check_parameters_at_initialization(self):
        """Tests whether it is possible to have a multidimensional exponent."""
        M1 = MultivariateNormal([[1, 1], [[1, 0], [0, 1]]])
        N1 = Normal([1, 0.1])
        with self.assertRaises(ValueError):
            N1**M1

    def test_initialization(self):
        """Tests that no errors during initialization are raised."""
        N1 = Normal([1, 0.1])
        N2 = Normal([1, 0.1])

        N3 = N1**N2

        N4 = N1**2

        N5 = 2**N1

    def test_forward_simulate(self):
        """Tests whether forward_simulate gives the desired output."""
        N1 = Normal([1, 0.1])
        N2 = N1**2
        rng = np.random.RandomState(1)
        N1._forward_simulate_and_store_output(rng=rng)
        sample = N2.forward_simulate(1, rng=rng)
        self.assertTrue(isinstance(sample, np.ndarray))


if __name__ == '__main__':
    unittest.main()
