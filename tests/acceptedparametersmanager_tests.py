import unittest
from abcpy.continuousmodels import Normal
from abcpy.discretemodels import Binomial
from abcpy.acceptedparametersmanager import *
from abcpy.backends import BackendDummy as Backend

"""Tests whether the methods defined for AcceptedParametersManager work as intended."""


class BroadcastTests(unittest.TestCase):
    """Tests whether observations can be broadcasted using broadcast."""
    def test(self):
        model = Normal([1, 0.1])
        Manager = AcceptedParametersManager([model])
        backend = Backend()
        Manager.broadcast(backend, [1,2,3])
        self.assertEqual(Manager.observations_bds.value(), [1,2,3])


class UpdateKernelValuesTests(unittest.TestCase):
    """Tests whether kernel_parameters_bds can be updated."""
    def test(self):
        model = Normal([1, 0.1])
        Manager = AcceptedParametersManager([model])
        backend = Backend()
        Manager.update_kernel_values(backend, [1])
        self.assertEqual(Manager.kernel_parameters_bds.value(),[1])


class UpdateBroadcastTests(unittest.TestCase):
    """Tests whether it is possible to update accepted_parameters_bds, accepted_weights_bds and accepted_cov_mats_bds through update_broadcast."""
    def setUp(self):
        self.model = Normal([1, 0.1])
        self.backend = Backend()
        self.Manager = AcceptedParametersManager([self.model])

    def test_accepted_parameters(self):
        self.Manager.update_broadcast(self.backend, [1,2,3])
        self.assertEqual(self.Manager.accepted_parameters_bds.value(),[1,2,3])

    def test_accepted_weights(self):
        self.Manager.update_broadcast(self.backend, accepted_weights=[1,2,3])
        self.assertEqual(self.Manager.accepted_weights_bds.value(),[1,2,3])

    def test_accepted_cov_matsrix(self):
        self.Manager.update_broadcast(self.backend, accepted_cov_mats=[[1,0],[0,1]])
        self.assertEqual(self.Manager.accepted_cov_mats_bds.value(), [[1,0],[0,1]])


class GetMappingTests(unittest.TestCase):
    """Tests whether the dfs mapping returned from get_mapping is in the correct order."""
    def test(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.1, 0.01])
        N2 = Normal([0.3, N1])
        graph = Normal([B1, N2])

        Manager = AcceptedParametersManager([graph])

        mapping, mapping_index = Manager.get_mapping([graph])
        self.assertEqual(mapping, [(B1,0),(N2,1),(N1,2)])


class GetAcceptedParametersBdsValuesTests(unittest.TestCase):
    """Tests whether get_accepted_parameters_bds_values returns the correct values."""
    def test(self):
        B1 = Binomial([10, 0.2])
        N1 = Normal([0.1, 0.01])
        N2 = Normal([0.3, N1])
        graph = Normal([B1, N2])

        Manager = AcceptedParametersManager([graph])
        backend = Backend()
        Manager.update_broadcast(backend, [[2,3,4],[0.27,0.32,0.28],[0.97,0.12,0.99]])

        values = Manager.get_accepted_parameters_bds_values([B1,N2,N1])
        values_expected = [np.array(x).reshape(-1,) for x in [[2,3,4],[0.27,0.32,0.28],[0.97,0.12,0.99]]]

        self.assertTrue(all([all(a == b) for a, b in zip(values, values_expected)]))


if __name__ == '__main__':
    unittest.main()