from abcpy.continuousmodels import *
from abcpy.probabilisticmodels import *
import unittest

"""These test cases implement tests for the probabilistic model class."""

class AbstractAPIImplementationTests():
    def setUp(self):
        self.models = []
        for model_type, model_input in zip(self.model_types, self.model_inputs):
            model = model_type(model_input)
            self.models.append(model)


    def test__getitem__(self):
        for model in self.models:
            item = model[0]
            self.assertTrue(isinstance(item, InputConnector), 'Return value not of type InputConnector for model {}.'.format(type(model)))


    def test_get_input_values(self):
        for model in self.models:
            values = model.get_input_values()
            self.assertTrue(isinstance(values, list), 'Return value not of type list in model {}.'.format(type(model)))
            self.assertEqual(len(values), model.get_input_dimension(), 'Number of parameters not equal to input dimension of model {}.'.format(model))


    def test_get_input_models(self):
        for model in self.models:
            in_models = model.get_input_models()
            self.assertTrue(isinstance(in_models, list), 'Return value not of type list in model {}.'.format(type(model)))
            self.assertEqual(len(in_models), model.get_input_dimension(), 'Number of parameters not equal to input dimension of model {}.'.format(model))


    def test_get_stored_output_values(self):
        for model in self.models:
            rng = np.random.RandomState(1)
            model._forward_simulate_and_store_output(rng)
            out_values = model.get_stored_output_values()
            self.assertTrue(isinstance(out_values, np.ndarray), 'Return value not of type numpy.array in model {}.'.format(type(model)))
            self.assertEqual(len(out_values), model.get_output_dimension(), 'Number of parameters not equal to output dimension of model {}.'.format(model))


    def test_get_input_connector(self):
        for model in self.models:
            in_con = model.get_input_connector()
            self.assertTrue(isinstance(in_con, InputConnector) or in_con == None, 'Return value not of type InputConnector nor None in model {}.'.format(type(model)))


    def test_get_input_dimension(self):
        for model in self.models:
            dim = model.get_input_dimension()
            self.assertTrue(isinstance(dim, Number), 'Return value not of type Number in model {}.'.format(type(model)))
            self.assertGreaterEqual(dim, 0, 'Input dimension must be larger than 0 for model {}.'.format(type(model)))


    def test_set_output_values(self):
        for model in self.models:
            number = 1
            with self.assertRaises(TypeError) as context:
                model.set_output_values(number)
            self.assertTrue(context.exception, 'Model {} should not accept a number as input.'.format(type(model)))

            nparray = np.ones(model.get_output_dimension()+1)
            with self.assertRaises(IndexError) as context:
                model.set_output_values(nparray)
            self.assertTrue(context.exception, 'Model {} should only accept input equal to output dimension.'.format(type(model)))

    def test_pdf(self):
        for model in self.models:
            x = 0
            input = model.get_input_values()
            pdf_at_x = model.pdf(input, x)
            self.assertTrue(isinstance(pdf_at_x, Number), 'Return value not of type Number in model {}.'.format(type(model)))


    def test_check_input(self):
        for model in self.models:
            test_result = model._check_input(model.get_input_values())
            self.assertTrue(test_result, 'The checking method should return True if input is reasonable in model {}.'.format(type(model)))

            with self.assertRaises(Exception) as context:
                model._check_input(0)
            self.assertTrue(context.exception, 'Function should raise an exception in model {} if input not of type InputConnector.'.format(type(model)))


    def test_forward_simulate(self):
        for model in self.models:
            rng = np.random.RandomState(1)
            result_list = model.forward_simulate(model.get_input_values(), 3, rng)
            self.assertTrue(isinstance(result_list, list), 'Return value not of type list in model {}.'.format(type(model)))
            self.assertEqual(len(result_list), 3, 'Model {} did not return the requseted number of formard simulations.'.format(type(model)))

            result = result_list[0]
            self.assertTrue(isinstance(result, np.ndarray), 'A single forward simulation is not of type numpy.array in model {}.'.format(type(model)))


    def test_get_output_dimension(self):
        for model in self.models:
            expected_dim = model.get_output_dimension()
            self.assertTrue(isinstance(expected_dim, Number), 'Return value not of type Number in model {}.'.format(type(model)))
            self.assertGreater(expected_dim, 0, 'Output dimension must be larger than 0 for model {}.'.format(type(model)))

            rng = np.random.RandomState(1)
            result_list = model.forward_simulate(model.get_input_values(), 1, rng)
            result = result_list[0]
            result_dim = result.shape[0]
            self.assertEqual(result_dim, expected_dim, 'Output dimension of forward simulation is not equal to get_output_dimension() for model {}.'.format(type(model)))


class HyperParameterAPITests(AbstractAPIImplementationTests, unittest.TestCase):
    model_types = [Hyperparameter]
    model_inputs = [1]


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
        N1._fixed_values=[-0.1]
        self.assertFalse(N2._check_input(N2.get_input_values()))


class GetOutputValuesTest(unittest.TestCase):
    """Tests whether get_stored_output_values gives back values that can come from the distribution."""
    def test(self):
        U = Uniform([[0], [1]])
        U._forward_simulate_and_store_output()
        self.assertTrue((U.get_stored_output_values()[0] >= 0 and U.get_stored_output_values()[0] <= 1))


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

        sample = N2.forward_simulate(N2.get_input_values(), 1, rng)

        self.assertTrue(isinstance(sample[0], np.ndarray))


class SubtractionModelTests(unittest.TestCase):
    """Tests whether all methods associated with the SubtractionModel are working as intended."""

    def test_forward_simulate(self):
        N1 = Normal([1, 0.1])
        N2 = 10-N1
        rng=np.random.RandomState(1)
        N1._forward_simulate_and_store_output(rng=rng)

        sample = N2.forward_simulate(N2.get_input_values(), 1, rng)

        self.assertTrue(isinstance(sample[0], np.ndarray))


class MultiplicationModelTests(unittest.TestCase):
    """Tests whether all methods associated with the MultiplicationModel are working as intended."""

    def test_forward_simulate(self):
        N1 = Normal([1, 0.1])
        N2 = N1*2
        rng=np.random.RandomState(1)
        N1._forward_simulate_and_store_output(rng=rng)

        sample = N2.forward_simulate(N2.get_input_values(), 1, rng)
        self.assertTrue(isinstance(sample[0], np.ndarray))

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
        sample = N3.forward_simulate(N3.get_input_values(), 1)

        self.assertTrue(isinstance(sample[0], np.ndarray))

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
        sample = N2.forward_simulate(N2.get_input_values(), 1, rng=rng)
        self.assertTrue(isinstance(sample[0], np.ndarray))


if __name__ == '__main__':
    unittest.main()
