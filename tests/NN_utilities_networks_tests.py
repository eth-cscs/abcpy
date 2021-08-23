import unittest

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True
    from abcpy.NN_utilities.networks import createDefaultNNWithDerivatives, createDefaultNN, DiscardLastOutputNet
    from abcpy.NN_utilities.utilities import jacobian_second_order, jacobian, jacobian_hessian


class test_default_NN_with_derivatives(unittest.TestCase):

    def setUp(self):
        self.net = createDefaultNNWithDerivatives(5, 2, nonlinearity=torch.nn.Softplus)()
        self.net_first_der_only = createDefaultNNWithDerivatives(5, 2, nonlinearity=torch.nn.Softplus,
                                                                 first_derivative_only=True)()
        self.tensor = torch.randn((10, 5), requires_grad=True)

    def test_first_der(self):
        # compute derivative with forward pass
        y, f1 = self.net_first_der_only.forward_and_derivatives(self.tensor)
        f2 = jacobian(self.tensor, y)

        assert torch.allclose(f1, f2)

    def test_first_and_second_der(self):
        # compute derivative with forward pass
        y, f1, s1 = self.net.forward_and_derivatives(self.tensor)
        f2, s2 = jacobian_second_order(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(s1, s2)

    def test_first_der_and_hessian(self):
        # compute derivative with forward pass
        y, f1, H1 = self.net.forward_and_full_derivatives(self.tensor)
        f2, H2 = jacobian_hessian(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(H1, H2)

    def test_error(self):
        with self.assertRaises(RuntimeError):
            self.net = createDefaultNNWithDerivatives(5, 2, nonlinearity=torch.nn.Softsign)()


class test_discard_last_output_wrapper(unittest.TestCase):

    def setUp(self):
        self.net = createDefaultNN(2, 3)()
        self.net_with_discard_wrapper = DiscardLastOutputNet(self.net)
        # reference input and output
        torch.random.manual_seed(1)
        self.tensor_1 = torch.randn(2)
        self.tensor_2 = torch.randn(1, 2)
        self.tensor_3 = torch.randn(1, 3, 2)

    def test_output(self):
        out = self.net(self.tensor_1)
        out_discard = self.net_with_discard_wrapper(self.tensor_1)
        self.assertTrue(torch.allclose(out[:-1], out_discard))

        out = self.net(self.tensor_2)
        out_discard = self.net_with_discard_wrapper(self.tensor_2)
        self.assertTrue(torch.allclose(out[:, :-1], out_discard))

        out = self.net(self.tensor_3)
        out_discard = self.net_with_discard_wrapper(self.tensor_3)
        self.assertTrue(torch.allclose(out[:, :, :-1], out_discard))


if __name__ == '__main__':
    unittest.main()
