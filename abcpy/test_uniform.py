from uniform import *
import unittest
from copy import deepcopy

class UniformTests(unittest.TestCase):
    def setUp(self):
        seed = 1
        self.rng = np.random.RandomState(seed)
        self.Uniform = Uniform([[-1.,-1.],[1.,1.]])
        self.Uniform_1 = Uniform([[self.Uniform],[1.2,1.2]])
        self.Uniform_1.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        samples = self.Uniform.sample_from_distribution(1000, rng=self.rng)

        samples_avg = samples.mean(axis=0)
        samples_min = samples.min(axis=0)
        samples_max = samples.max(axis=0)
        for (avg, min, max) in zip(samples_avg, samples_min, samples_max):
            self.assertLess(abs(avg), 0.04)
            self.assertLess(abs(min + 1.0), 0.01)
            self.assertLess(abs(max - 1.0), 0.01)

        samples_shape = np.shape(samples)
        self.assertEqual(samples_shape, (1000, 2))

        samples_graph = self.Uniform_1.sample_from_distribution(1000, rng=self.rng)
        samples_graph_avg = samples_graph.mean(axis=0)
        samples_graph_min = samples_graph.min(axis=0)
        samples_graph_max = samples_graph.max(axis=0)

        for (avg, min, max) in zip(samples_graph_avg, samples_graph_min, samples_graph_max):
            self.assertLess(abs(avg), 1.2)
            self.assertLess(abs(min), 1.2)
            self.assertLess(abs(max), 1.2)
        samples_graph_shape = np.shape(samples_graph)
        self.assertEqual(samples_graph_shape, (1000, 2))

    def test_fix_parameters(self):
        self.Uniform_1.set_parameters([[-1.5,-1.5],[]], rng=self.rng)
        self.assertTrue(self.Uniform_1.get_parameters()==[[-1.5,-1.5],[]])
        old_parameter = deepcopy(self.Uniform_1.parameter_values)
        self.Uniform_1.sample_parameters(rng=self.rng)
        self.assertTrue(old_parameter!=self.Uniform_1.parameter_values)

    def test_check_parameters(self):
        with self.assertRaises(TypeError):
            U = Uniform(1,1)
        with self.assertRaises(TypeError):
            U = Uniform([1,1])
        with self.assertRaises(IndexError):
            U = Uniform([[1],[2,3]])
        with self.assertRaises(ValueError):
            U = Uniform([[2,1],[1,1]])

    def test_check_parameters_fixed(self):
        N = Uniform([[0],[1]])
        U = Uniform([[N,1],[2,2]])
        self.assertFalse(U.set_parameters([[1,2],[]]))
        self.assertFalse(U.set_parameters([[1],[3,3]]))
        self.assertFalse(U.set_parameters([[5],[]]))

    def test_pdf(self):
        U = Uniform([[0.],[10.]])
        self.assertEqual(U.pdf(0),0.1)
        self.assertEqual(U.pdf(-1),0)
        self.assertEqual(U.pdf(11),0)

if __name__ == '__main__':
    unittest.main()