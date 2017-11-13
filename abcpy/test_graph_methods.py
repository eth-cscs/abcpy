import unittest
from inferences import *
from continuous import *
from discrete import *
from abcpy.distances import LogReg
from abcpy.statistics import Identity
from abcpy.backends import BackendDummy as Backend
from perturbationkernel import *

"""Tests whether the methods defined for operations on the graph work as intended."""

class SampleFromPriorTests(unittest.TestCase):
    """Tests whether sample_from_prior assigns new values to all nodes corresponding to free parameters in the graph."""
    def test(self):
        B1 = Binomial([10,0.2])
        N1 = Normal([0.03,0.01])
        N2 = Normal([0.1,N1])
        graph = Normal([B1,N2])

        statistics_calculator = Identity(degree = 2, cross = False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        sampler = RejectionABC([graph], distance_calculator, backend)

        rng = np.random.RandomState(1)

        sampler.sample_from_prior(rng)
        self.assertIsNotNone(B1.fixed_values)
        self.assertIsNotNone(N1.fixed_values)
        self.assertIsNotNone(N2.fixed_values)

class ResetFlagsTests(unittest.TestCase):
    """Tests whether it is possible to reset all visited flags in the graph."""
    def test(self):
        N1 = Normal([1,0.1])
        N2 = Normal([N1,0.1])
        N2.visited = True
        N1.visited = True

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        sampler = RejectionABC([N2], distance_calculator, backend)

        sampler._reset_flags()

        self.assertFalse(N1.visited)
        self.assertFalse(N2.visited)

class GetParametersTests(unittest.TestCase):
    """Tests whether get_parameters returns only the free parameters of the graph."""
    def setUp(self):
        self.B1 = Binomial([10, 0.2])
        self.N1 = Normal([0.03, 0.01])
        self.N2 = Normal([0.1, self.N1])
        self.graph = Normal([self.B1, self.N2])

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        self.sampler = RejectionABC([self.graph], distance_calculator, backend)

        self.rng = np.random.RandomState(1)

        self.sampler.sample_from_prior(self.rng)

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

        self.sampler = RejectionABC([self.graph], distance_calculator, backend)

        self.rng = np.random.RandomState(1)

        self.sampler.sample_from_prior(self.rng)

    def test(self):
        is_accepted, index = self.sampler.set_parameters([3, 0.12, 0.029])
        self.assertTrue(is_accepted)

        self.assertEqual(self.B1.fixed_values[0], 3)
        self.assertEqual(self.N1.fixed_values[0], 0.029)
        self.assertEqual(self.N2.fixed_values[0], 0.12)


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

        self.sampler = RejectionABC([self.graph], distance_calculator, backend)

        self.rng = np.random.RandomState(1)

        self.sampler.sample_from_prior(self.rng)

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

        self.sampler = RejectionABC([self.graph], distance_calculator, backend)

        self.rng = np.random.RandomState(1)

        self.sampler.sample_from_prior(self.rng)

        kernel = StandardKernel([self.N1, self.N2, self.B1])
        self.sampler.kernel = kernel

        self.sampler.accepted_parameters_manager.update_broadcast(self.sampler.backend, [[3,4],[0.11,0.098],[0.029,0.031]], [1,1])

    def test(self):
        B1_value = self.B1.fixed_values
        N1_value = self.N1.fixed_values
        N2_value = self.N2.fixed_values

        self.sampler.perturb(1, rng=self.rng)

        self.assertNotEqual(B1_value, self.B1.fixed_values)
        self.assertNotEqual(N1_value, self.N1.fixed_values)
        self.assertNotEqual(N2_value, self.N2.fixed_values)

class GetProbabilisticModelsCorrespondingToFreeParametersTests(unittest.TestCase):
    """Tests whether get_probabilistic_models_corresponding_to_free_parameters returns all probabilistic models corresponding to free parameters of a given type."""
    def setUp(self):
        self.B1 = Binomial([10, 0.2])
        self.N1 = Normal([0.03, 0.01])
        self.N2 = Normal([0.1, self.N1])
        self.graph = Normal([self.B1, self.N2])

        statistics_calculator = Identity(degree=2, cross=False)
        distance_calculator = LogReg(statistics_calculator)
        backend = Backend()

        self.sampler = RejectionABC([self.graph], distance_calculator, backend)

        self.rng = np.random.RandomState(1)

    def test_continuous(self):
        continuous_models = self.sampler.get_probabilistic_models_corresponding_to_free_parameters(Continuous)
        self.assertEqual(continuous_models, [self.N2,self.N1])

    def test_discrete(self):
        discrete_models = self.sampler.get_probabilistic_models_corresponding_to_free_parameters(Discrete)
        self.assertEqual(discrete_models, [self.B1])




if __name__ == '__main__':
    unittest.main()



