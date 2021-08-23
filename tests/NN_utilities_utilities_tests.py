import unittest

import numpy as np

from abcpy.statistics import Identity, LinearTransformation, NeuralEmbedding

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True
    from abcpy.NN_utilities.utilities import jacobian_second_order, jacobian, jacobian_hessian
    from abcpy.NN_utilities.networks import createDefaultNN


class test_jacobian_functions(unittest.TestCase):
    # it tests that this gives the correct errors and that the result is same if you put diffable=True or False.
    # it does not test that the derivatives numerical errors are correct but they are.

    def setUp(self):
        net = createDefaultNN(5, 2, nonlinearity=torch.nn.Softplus(), batch_norm_last_layer=False)()
        net_bn = createDefaultNN(5, 2, nonlinearity=torch.nn.Softplus(), batch_norm_last_layer=True)()
        self.tensor = torch.randn((10, 5), requires_grad=True)
        self.y = net(self.tensor)
        self.y_bn = net_bn(self.tensor)

        self.y_with_infinities = self.y.detach().clone()
        self.y_with_infinities[0, 0] = np.inf

        self.f, self.s = jacobian_second_order(self.tensor, self.y)  # reference derivatives
        self.f_bn, self.s_bn = jacobian_second_order(self.tensor, self.y_bn)  # reference derivatives

    def test_first_der(self):
        # compute derivative with forward pass
        f2 = jacobian(self.tensor, self.y, diffable=False)

        assert torch.allclose(self.f, f2)

    def test_first_and_second_der(self):
        # compute derivative with forward pass
        f2, s2 = jacobian_second_order(self.tensor, self.y, diffable=False)

        assert torch.allclose(self.f, f2)
        assert torch.allclose(self.s, s2)

    def test_first_der_and_hessian(self):
        # compute derivative with forward pass
        f1, H1 = jacobian_hessian(self.tensor, self.y)
        f2, H2 = jacobian_hessian(self.tensor, self.y, diffable=False)
        s2 = torch.einsum('biik->bik', H2)  # obtain the second order jacobian from Hessian matrix

        assert torch.allclose(self.f, f2)
        assert torch.allclose(f1, f2)
        assert torch.allclose(H1, H2)
        assert torch.allclose(self.s, s2)

    def test_first_der_bn(self):
        # compute derivative with forward pass
        f2 = jacobian(self.tensor, self.y_bn, diffable=False)

        assert torch.allclose(self.f_bn, f2)

    def test_first_and_second_der_bn(self):
        # compute derivative with forward pass
        f2, s2 = jacobian_second_order(self.tensor, self.y_bn, diffable=False)

        assert torch.allclose(self.f_bn, f2)
        assert torch.allclose(self.s_bn, s2)

    def test_first_der_and_hessian_bn(self):
        # compute derivative with forward pass
        f1, H1 = jacobian_hessian(self.tensor, self.y_bn)
        f2, H2 = jacobian_hessian(self.tensor, self.y_bn, diffable=False)
        s2 = torch.einsum('biik->bik', H2)  # obtain the second order jacobian from Hessian matrix

        assert torch.allclose(self.f_bn, f2)
        assert torch.allclose(f1, f2)
        assert torch.allclose(H1, H2)
        assert torch.allclose(self.s_bn, s2)

    def test_errors(self):
        with self.assertRaises(ValueError):
            f1 = jacobian(self.tensor, self.y_with_infinities)
        with self.assertRaises(ValueError):
            f1, s1 = jacobian_second_order(self.tensor, self.y_with_infinities)
        with self.assertRaises(ValueError):
            f1, H1 = jacobian_hessian(self.tensor, self.y_with_infinities)


if __name__ == '__main__':
    unittest.main()
