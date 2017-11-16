from continuous import *
from ProbabilisticModel import *
import unittest

"""These test cases implement tests for the probabilistic model class."""


class GetItemTest(unittest.TestCase):
    """Tests whether the get_item operator (i.e. access operator) works as intended."""

    def ErrorTest(self):
        """Tests whether an error is thrown if the operator tries to access an element outside the range of the return values of the probabilistic model."""
        N = Normal([1,0.1])
        with self.assertRaises(IndexError):
            N[2]

    def ReturnValueTest(self):
        """Tests whether the return value of the operator is correct"""
        N = Normal([1,0.1])
        result = N[0]
        self.assertTrue(isinstance(result,tuple))
        self.assertTrue(result[0]==N)
        self.assertTrue(result[1]==0)


class MappingTest(unittest.TestCase):
    """Tests whether the mapping created during initialization is done correctly."""
    def setUp(self):
        self.U = Uniform([[0,2],[1,3]])
        self.M = MultivariateNormal([[self.U[1],self.U[0]],[[1,0],[0,1]]])

    def test(self):
        self.assertTrue(self.M.parents[0]==(self.U, 1))
        self.assertTrue(self.M.parents[1]==(self.U,0))


class GetParameterValuesTest(unittest.TestCase):
    """Tests whether get_parameter_values returns the correct values."""
    def test(self):
        U = Uniform([[0,2],[1,3]])
        self.assertTrue(U.get_parameter_values()==[0,2,1,3])


class SampleParametersTest(unittest.TestCase):
    """Tests whether sample_parameters returns False if the value of an input parameter is not an allowed value for the distribution."""
    def test(self):
        N1=Normal([0.1,0.01])
        N2=Normal([1,N1])
        N1.fixed_values=[-0.1]
        self.assertFalse(N2.sample_parameters())


class GetParametersTest(unittest.TestCase):
    """Tests whether get_parameters gives back values that can come from the distribution."""
    def test(self):
        U = Uniform([[0],[1]])
        U.sample_parameters()
        self.assertTrue((U.fixed_values[0]>=0 and U.fixed_values[0]<=1))


class SummationModelTests(unittest.TestCase):
    """Tests whether all methods associated with the SummationModel are working as intended."""

    def test_check_parameters_at_initialization(self):
        N1 = Normal([1,0.1])
        M1 = MultivariateNormal([[1,1],[[1,0],[0,1]]])
        with self.assertRaises(ValueError):
            model = N1+M1

    def test_initialization(self):
        M1 = MultivariateNormal([[1,1],[[1,0],[0,1]]])
        M2 = MultivariateNormal([[1,1],[[1,0],[0,1]]])
        M3 = M1+M2
        self.assertTrue(M3.dimension==2)

    def test_sample_from_distribution(self):
        N1 = Normal([1,0.1])
        N2 = 10+N1
        rng=np.random.RandomState(1)
        N1.sample_parameters(rng=rng)

        sample = N2.sample_from_distribution(1, rng)

        self.assertTrue(sample[0])
        self.assertTrue(isinstance(sample[1], np.ndarray))

class SubtractionModelTests(unittest.TestCase):
    """Tests whether all methods associated with the SubtractionModel are working as intended."""
    def test_check_parameters_at_initialization(self):
        N1 = Normal([1,0.1])
        M1 = MultivariateNormal([[1,1],[[1,0],[0,1]]])
        with self.assertRaises(ValueError):
            model = N1-M1

    def test_initialization(self):
        M1 = MultivariateNormal([[1,1],[[1,0],[0,1]]])
        M2 = MultivariateNormal([[1,1],[[1,0],[0,1]]])
        M3 = M1-M2
        self.assertTrue(M3.dimension==2)

    def test_sample_from_distribution(self):
        N1 = Normal([1,0.1])
        N2 = 10-N1
        rng=np.random.RandomState(1)
        N1.sample_parameters(rng=rng)

        sample = N2.sample_from_distribution(1, rng)

        self.assertTrue(sample[0])
        self.assertTrue(isinstance(sample[1], np.ndarray))



if __name__ == '__main__':
    unittest.main()
