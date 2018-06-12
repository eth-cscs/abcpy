import unittest
from abcpy.inferences import *
from abcpy.continuousmodels import *
from abcpy.discretemodels import *
from abcpy.distances import LogReg
from abcpy.statistics import Identity
from abcpy.backends import BackendDummy as Backend
from abcpy.perturbationkernel import *

"""Tests whether the methods defined for operations on the graph work as intended."""


class SampleFromPriorTests(unittest.TestCase):
    """Tests whether sample_from_prior assigns new values to all nodes corresponding to free parameters in the graph."""
    def test(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.03, 0.01])
        N2 = Normal([0.1, N1])
        graph = Normal([B1, N2])

        statistics_calculator = Identity(degree = 2, cross = False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        sampler = RejectionABC([graph], [distance_calculator], backend)

        rng = np.random.RandomState(1)

        sampler.sample_from_prior(rng=rng)
        self.assertIsNotNone(B1._fixed_values)
        self.assertIsNotNone(N1._fixed_values)
        self.assertIsNotNone(N2._fixed_values)


class ResetFlagsTests(unittest.TestCase):
    """Tests whether it is possible to reset all visited flags in the graph."""
    def test(self):
        N1 = Normal([1, 0.1])
        N2 = Normal([N1, 0.1])
        N2.visited = True
        N1.visited = True

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        sampler = RejectionABC([N2], [distance_calculator], backend)

        sampler._reset_flags()

        self.assertFalse(N1.visited)
        self.assertFalse(N2.visited)


class GetParametersTests(unittest.TestCase):
    """Tests whether get_stored_output_values returns only the free parameters of the graph."""
    def setUp(self):
        self.B1 = Binomial([10, 0.2])
        self.N1 = Normal([0.03, 0.01])
        self.N2 = Normal([0.1, self.N1])
        self.graph = Normal([self.B1, self.N2])

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        self.sampler = RejectionABC([self.graph], [distance_calculator], backend)

        self.rng = np.random.RandomState(1)

        self.sampler.sample_from_prior(rng=self.rng)

    def test(self):
        free_parameters = self.sampler.get_parameters()
        self.assertEqual(len(free_parameters),3)


class SetParametersTests(unittest.TestCase):
    """Tests whether it is possible to set values for all free parameters of the graph."""
    def setUp(self):
        self.B1 = Binomial([10, 0.2])
        self.N1 = Normal([0.03, 0.01])
        self.N2 = Normal([0.1, self.N1])
        self.graph = Normal([self.B1, self.N2])

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        self.sampler = RejectionABC([self.graph], [distance_calculator], backend)

        self.rng = np.random.RandomState(1)

        self.sampler.sample_from_prior(rng=self.rng)

    def test(self):
        is_accepted, index = self.sampler.set_parameters([3, 0.12, 0.029])
        self.assertTrue(is_accepted)

        self.assertEqual(self.B1.get_stored_output_values()[0], 3)
        self.assertEqual(self.N1.get_stored_output_values()[0], 0.029)
        self.assertEqual(self.N2.get_stored_output_values()[0], 0.12)


class GetCorrectOrderingTests(unittest.TestCase):
    """Tests whether get_correct_ordering will order the values of free parameters in recursive dfs order."""
    def setUp(self):
        self.B1 = Binomial([10, 0.2])
        self.N1 = Normal([0.03, 0.01])
        self.N2 = Normal([0.1, self.N1])
        self.graph = Normal([self.B1, self.N2])

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        self.sampler = RejectionABC([self.graph], [distance_calculator], backend)

        self.rng = np.random.RandomState(1)

        self.sampler.sample_from_prior(rng=self.rng)

    def test(self):
        parameters_and_models = [(self.N1, [0.029]), (self.B1, [3]), (self.N2, [0.12])]
        ordered_parameters = self.sampler.get_correct_ordering(parameters_and_models)
        self.assertEqual(ordered_parameters, [3,0.12,0.029])


class PerturbTests(unittest.TestCase):
    """Tests whether perturb will change all fixed values for free parameters."""
    def setUp(self):
        self.B1 = Binomial([10, 0.2])
        self.N1 = Normal([0.03, 0.01])
        self.N2 = Normal([0.1, self.N1])
        self.graph = Normal([self.B1, self.N2])

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        self.sampler = PMCABC([self.graph], [distance_calculator], backend)

        self.rng = np.random.RandomState(1)

        self.sampler.sample_from_prior(rng=self.rng)

        kernel = DefaultKernel([self.N1, self.N2, self.B1])
        self.sampler.kernel = kernel

        self.sampler.accepted_parameters_manager.update_broadcast(self.sampler.backend, [[3, 0.11, 0.029],[4,0.098, 0.031]], accepted_cov_mats = [[[1,0],[0,1]]], accepted_weights=np.array([1,1]))


        kernel_parameters = []
        for kernel in self.sampler.kernel.kernels:
            kernel_parameters.append(
                self.sampler.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
        self.sampler.accepted_parameters_manager.update_kernel_values(self.sampler.backend, kernel_parameters)


    def test(self):
        B1_value = self.B1.get_stored_output_values()
        N1_value = self.N1.get_stored_output_values()
        N2_value = self.N2.get_stored_output_values()

        self.sampler.perturb(1, rng=self.rng)

        self.assertNotEqual(B1_value, self.B1._fixed_values)
        self.assertNotEqual(N1_value, self.N1._fixed_values)
        self.assertNotEqual(N2_value, self.N2._fixed_values)


class SimulateTests(unittest.TestCase):
    """Tests whether the simulated data for multiple models has the correct format."""
    def test(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.03, 0.01])
        N2 = Normal([0.1, N1])
        graph1 = Normal([B1, N2])
        graph2 = Normal([1, N2])

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        sampler = RejectionABC([graph1, graph2], [distance_calculator, distance_calculator], backend)

        rng = np.random.RandomState(1)

        sampler.sample_from_prior(rng=rng)

        y_sim = sampler.simulate(1, rng=rng)

        self.assertTrue(isinstance(y_sim, list))

        self.assertTrue(len(y_sim)==2)

        self.assertTrue(isinstance(y_sim[0][0], np.ndarray))


class GetMappingTests(unittest.TestCase):
    """Tests whether the private get_mapping method will return the correct mapping."""
    def test(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.03, 0.01])
        N2 = Normal([0.1, N1])
        graph1 = Normal([B1, N2])
        graph2 = Normal([1, N2])

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        sampler = RejectionABC([graph1, graph2], [distance_calculator, distance_calculator], backend)

        rng = np.random.RandomState(1)

        sampler.sample_from_prior(rng=rng)

        mapping, index = sampler._get_mapping()
        self.assertTrue(mapping==[(B1, 0),(N2, 1),(N1,2)])

from abcpy.continuousmodels import Uniform

class PdfOfPriorTests(unittest.TestCase):
    """Tests the implemetation of pdf_of_prior"""
    def setUp(self):
        class Mockobject(Normal):
            def __init__(self, parameters):
                super(Mockobject, self).__init__(parameters)
            def pdf(self, input_values, x):
                return x

        self.N1 = Uniform([[1.3], [1.55]], name='n1')
        self.N2 = Uniform([[self.N1], [1.60]], name='n2')
        self.N3 = Uniform([[2], [4]], name='n3')
        self.graph1 = Uniform([[self.N1], [self.N2]], name='graph1')
        self.graph2 = Uniform([[.5], [self.N3]])

        self.graph = [self.graph1, self.graph2]

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        self.sampler1 = RejectionABC([self.graph1], [distance_calculator], backend)
        self.sampler2 = RejectionABC([self.graph2], [distance_calculator], backend)
        self.sampler3 = RejectionABC(self.graph, [distance_calculator, distance_calculator], backend)

        self.pdf1 = self.sampler1.pdf_of_prior(self.sampler1.model,  [1.32088846, 1.42945274])
        self.pdf2 = self.sampler2.pdf_of_prior(self.sampler2.model, [3])
        self.pdf3 = self.sampler3.pdf_of_prior(self.sampler3.model, [1.32088846, 1.42945274, 3])

    def test_return_value(self):
        """Tests whether the return value is float."""
        self.assertTrue(isinstance(self.pdf1, float))
        self.assertTrue(isinstance(self.pdf2, float))
        self.assertTrue(isinstance(self.pdf3, float))

    def test_result(self):
        """Test whether pdf calculation works as intended"""
        self.assertTrue(self.pdf1 == 14.331188169432183)
        self.assertTrue(self.pdf2 == 0.5)
        self.assertTrue(self.pdf3 == 7.1655940847160915)


if __name__ == '__main__':
    unittest.main()



