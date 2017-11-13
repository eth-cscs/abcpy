import unittest
from continuous import Normal
from discrete import Binomial
from accepted_parameters_manager import AcceptedParametersManager
from abcpy.backends import BackendDummy as Backend
from perturbationkernel import *

"""Tests whether the methods for each perturbation kernel are working as intended"""


class UpdateTests(unittest.TestCase):
    """Tests whether the values returned after perturbation are in the correct format for each perturbation kernel."""
    def test_StandardKernel(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.1, 0.01])
        N2 = Normal([0.3, N1])
        graph = Normal([B1, N2])

        Manager = AcceptedParametersManager([graph])
        backend = Backend()
        kernel = StandardKernel([N1, N2, B1])
        Manager.update_broadcast(backend, [[2, 3], [0.27, 0.32], [0.97, 0.12]], [1,1])

        rng = np.random.RandomState(1)
        perturbed_values_and_models = kernel.update(Manager, 1, rng)

        self.assertEqual(perturbed_values_and_models, [(N1, [-0.85629777838992582]), (N2, [0.37742928108176033]), (B1, [3])])


if __name__ == '__main__':
    unittest.main()