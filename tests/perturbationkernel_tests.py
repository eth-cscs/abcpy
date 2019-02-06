import unittest
from abcpy.continuousmodels import Normal
from abcpy.discretemodels import Binomial
from abcpy.acceptedparametersmanager import AcceptedParametersManager
from abcpy.backends import BackendDummy as Backend
from abcpy.perturbationkernel import *

"""Tests whether the methods for each perturbation kernel are working as intended"""

class JointCheckKernelsTests(unittest.TestCase):
    """Tests whether value errors are raised correctly during initialization."""
    def test_Raises(self):
        N1 = Normal([0.1, 0.01])
        N2 = Normal([0.3, N1])
        kernel = MultivariateNormalKernel([N1,N2,N1])
        with self.assertRaises(ValueError):
            JointPerturbationKernel([kernel])

    def test_doesnt_raise(self):
        N1 = Normal([0.1, 0.01])
        N2 = Normal([0.3, N1])
        kernel = MultivariateNormalKernel([N1, N2])
        try:
            JointPerturbationKernel([kernel])
        except ValueError:
            self.fail("JointPerturbationKernel raises an exception")


class CalculateCovTets(unittest.TestCase):
    """Tests whether the implementation of calculate_cov is working as intended."""
    def test(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.1, 0.01])
        N2 = Normal([0.3, N1])
        graph = Normal([B1, N2])

        Manager = AcceptedParametersManager([graph])
        backend = Backend()
        kernel = DefaultKernel([N1, N2, B1])
        Manager.update_broadcast(backend, [[2, 0.27, 0.097], [3, 0.32, 0.012]], np.array([1, 1]))

        kernel_parameters = []
        for krnl in kernel.kernels:
            kernel_parameters.append(Manager.get_accepted_parameters_bds_values(krnl.models))
        Manager.update_kernel_values(backend, kernel_parameters)

        covs = kernel.calculate_cov(Manager)
        self.assertTrue(len(covs)==2)

        self.assertTrue(len(covs[0])==2)

        self.assertTrue(not(covs[1]))

class UpdateTests(unittest.TestCase):
    """Tests whether the values returned after perturbation are in the correct format for each perturbation kernel."""
    def test_DefaultKernel(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.1, 0.01])
        N2 = Normal([0.3, N1])
        graph = Normal([B1, N2])

        Manager = AcceptedParametersManager([graph])
        backend = Backend()
        kernel = DefaultKernel([N1, N2, B1])
        Manager.update_broadcast(backend, [[2, 0.27, 0.097], [3, 0.32, 0.012]], np.array([1,1]), accepted_cov_mats=[[[0.01,0],[0,0.01]],[]])

        kernel_parameters = []
        for krnl in kernel.kernels:
            kernel_parameters.append(
                Manager.get_accepted_parameters_bds_values(krnl.models))

        Manager.update_kernel_values(backend, kernel_parameters=kernel_parameters)

        rng = np.random.RandomState(1)
        perturbed_values_and_models = kernel.update(Manager, 1, rng)
        self.assertEqual(perturbed_values_and_models, [(N1, [0.17443453636632419]), (N2, [0.25882435863499248]), (B1, [3])])


class PdfTests(unittest.TestCase):
    """Tests whether the pdf returns the correct results."""
    def test_return_value(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.1, 0.01])
        N2 = Normal([0.3, N1])
        graph = Normal([B1, N2])

        Manager = AcceptedParametersManager([graph])
        backend = Backend()
        kernel = DefaultKernel([N1, N2, B1])
        Manager.update_broadcast(backend, [[2, 0.4, 0.09], [3, 0.2, 0.008]], np.array([0.5, 0.2]))
        kernel_parameters = []
        for krnl in kernel.kernels:
            kernel_parameters.append(Manager.get_accepted_parameters_bds_values(krnl.models))
        Manager.update_kernel_values(backend, kernel_parameters)
        mapping, mapping_index = Manager.get_mapping(Manager.model)
        covs = [[[1,0],[0,1]],[]]
        Manager.update_broadcast(backend, accepted_cov_mats=covs)
        pdf = kernel.pdf(mapping, Manager, Manager.accepted_parameters_bds.value()[1], [2,0.3,0.1])
        self.assertTrue(isinstance(pdf, float))


if __name__ == '__main__':
    unittest.main()