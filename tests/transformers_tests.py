import unittest

import numpy as np

from abcpy.transformers import DummyTransformer, BoundedVarTransformer, BoundedVarScaler

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True


class DummyTransformerTests(unittest.TestCase):
    def test(self):
        transformer = DummyTransformer()
        x = [np.array([3.2]), np.array([2.4])]
        self.assertEqual(x, transformer.transform(x))
        self.assertEqual(x, transformer.inverse_transform(x))
        self.assertEqual(0, transformer.jac_log_det_inverse_transform(x))
        self.assertEqual(0, transformer.jac_log_det(x))


class BoundedVarTransformerTests(unittest.TestCase):
    def setUp(self):
        self.transformer_lower_bounded = BoundedVarTransformer(lower_bound=np.array([0, 0]),
                                                               upper_bound=np.array([None, None]))
        self.transformer_two_sided = BoundedVarTransformer(lower_bound=np.array([0, 0]), upper_bound=np.array([10, 10]))
        self.transformer_mixed = BoundedVarTransformer(lower_bound=np.array([0, 0]), upper_bound=np.array([10, None]))
        self.transformer_dummy = BoundedVarTransformer(lower_bound=np.array([None, None]),
                                                       upper_bound=np.array([None, None]))
        self.list_transformers = [self.transformer_dummy, self.transformer_mixed,
                                  self.transformer_two_sided, self.transformer_lower_bounded]

    def test(self):
        x = [np.array([3.2]), np.array([2.4])]
        for transformer in self.list_transformers:
            self.assertEqual(len(x), len(transformer.inverse_transform(transformer.transform(x))))
            self.assertTrue(np.allclose(np.array(x), np.array(transformer.inverse_transform(transformer.transform(x)))))
            self.assertAlmostEqual(transformer.jac_log_det(x),
                                   transformer.jac_log_det_inverse_transform(transformer.transform(x)), delta=1e-7)

        # test dummy transformer actually does nothing:
        self.assertEqual(x, self.transformer_dummy.transform(x))
        self.assertEqual(x, self.transformer_dummy.inverse_transform(x))
        self.assertEqual(0, self.transformer_dummy.jac_log_det_inverse_transform(x))
        self.assertEqual(0, self.transformer_dummy.jac_log_det(x))

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            transformer = BoundedVarTransformer(lower_bound=[0, 0], upper_bound=[10, 10])
        with self.assertRaises(RuntimeError):
            transformer = BoundedVarTransformer(lower_bound=np.array([0, 0]), upper_bound=np.array([10]))
        with self.assertRaises(NotImplementedError):
            transformer = BoundedVarTransformer(lower_bound=np.array([None, 0]), upper_bound=np.array([10, 10]))
        with self.assertRaises(RuntimeError):
            self.transformer_lower_bounded.transform(x=[np.array([3.2]), np.array([-2.4])])
        with self.assertRaises(RuntimeError):
            self.transformer_two_sided.transform(x=[np.array([13.2]), np.array([2.4])])


class test_BoundedVarScaler(unittest.TestCase):

    def setUp(self):
        self.scaler_lower_bounded = BoundedVarScaler(lower_bound=np.array([0, 0]),
                                                     upper_bound=np.array([None, None]))
        self.scaler_two_sided = BoundedVarScaler(lower_bound=np.array([0, 0]), upper_bound=np.array([10, 10]))
        self.scaler_mixed = BoundedVarScaler(lower_bound=np.array([0, 0]), upper_bound=np.array([10, None]))
        self.scaler_dummy = BoundedVarScaler(lower_bound=np.array([None, None]),
                                             upper_bound=np.array([None, None]))
        # without minmax
        self.scaler_lower_bounded_no_minmax = BoundedVarScaler(lower_bound=np.array([0, 0]),
                                                               upper_bound=np.array([None, None]),
                                                               rescale_transformed_vars=False)
        self.scaler_two_sided_no_minmax = BoundedVarScaler(lower_bound=np.array([0, 0]), upper_bound=np.array([10, 10]),
                                                           rescale_transformed_vars=False)
        self.scaler_mixed_no_minmax = BoundedVarScaler(lower_bound=np.array([0, 0]), upper_bound=np.array([10, None]),
                                                       rescale_transformed_vars=False)
        self.scaler_dummy_no_minmax = BoundedVarScaler(lower_bound=np.array([None, None]),
                                                       upper_bound=np.array([None, None]),
                                                       rescale_transformed_vars=False)

        self.list_scalers_minmax = [self.scaler_dummy, self.scaler_mixed,
                                    self.scaler_two_sided, self.scaler_lower_bounded]
        self.list_scalers_no_minmax = [self.scaler_dummy_no_minmax, self.scaler_mixed_no_minmax,
                                       self.scaler_two_sided_no_minmax, self.scaler_lower_bounded_no_minmax]

        self.list_scalers = self.list_scalers_minmax + self.list_scalers_no_minmax

        # data
        self.x = np.array([[3.2, 4.5]])
        self.x2 = np.array([[4.2, 3.5]])

    def test(self):
        for scaler in self.list_scalers:
            scaler.fit(self.x)
            self.assertEqual(self.x.shape, scaler.inverse_transform(scaler.transform(self.x)).shape)
            self.assertTrue(np.allclose(np.array(self.x), np.array(scaler.inverse_transform(scaler.transform(self.x)))))
            self.assertAlmostEqual(scaler.jac_log_det(self.x),
                                   scaler.jac_log_det_inverse_transform(scaler.transform(self.x)), delta=1e-7)

        # test dummy scaler actually does nothing:
        self.assertTrue(np.allclose(self.x, self.scaler_dummy_no_minmax.transform(self.x)))
        self.assertTrue(np.allclose(self.x, self.scaler_dummy_no_minmax.inverse_transform(self.x)))
        self.assertEqual(0, self.scaler_dummy.jac_log_det_inverse_transform(self.x))
        self.assertEqual(0, self.scaler_dummy.jac_log_det(self.x))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det_inverse_transform(self.x))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det(self.x))

        # test that the jacobian works on 1d things as well:
        self.assertEqual(0, self.scaler_dummy.jac_log_det_inverse_transform(self.x.squeeze()))
        self.assertEqual(0, self.scaler_dummy.jac_log_det(self.x.squeeze()))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det_inverse_transform(self.x.squeeze()))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det(self.x.squeeze()))

    def test_torch(self):
        # same as test but using torch input
        if has_torch:
            x_torch = torch.from_numpy(self.x)
            for scaler in self.list_scalers:
                scaler.fit(x_torch)
                self.assertEqual(x_torch.shape, scaler.inverse_transform(scaler.transform(x_torch)).shape)
                self.assertTrue(np.allclose(self.x, np.array(scaler.inverse_transform(scaler.transform(x_torch)))))
                self.assertAlmostEqual(scaler.jac_log_det(x_torch),
                                       scaler.jac_log_det_inverse_transform(scaler.transform(x_torch)), delta=1e-7)

            # test dummy scaler actually does nothing:
            self.assertTrue(np.allclose(x_torch, self.scaler_dummy_no_minmax.transform(x_torch)))
            self.assertTrue(np.allclose(x_torch, self.scaler_dummy_no_minmax.inverse_transform(x_torch)))
            self.assertEqual(0, self.scaler_dummy.jac_log_det_inverse_transform(x_torch))
            self.assertEqual(0, self.scaler_dummy.jac_log_det(x_torch))
            self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det_inverse_transform(x_torch))
            self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det(x_torch))

            # test that the jacobian works on 1d things as well:
            self.assertEqual(0, self.scaler_dummy.jac_log_det_inverse_transform(x_torch.squeeze()))
            self.assertEqual(0, self.scaler_dummy.jac_log_det(x_torch.squeeze()))
            self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det_inverse_transform(x_torch.squeeze()))
            self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det(x_torch.squeeze()))

    def test_jacobian_difference(self):
        # the values of the jacobian log det do not take into account the linear transformation as what
        # really matters are the difference between them for two x values (in an MCMC acceptance rate).
        # Then the difference of the jacobian for the same two points in original and transformed space should be
        # the same.
        for scaler_minmax, scaler_no_minmax in zip(self.list_scalers_minmax, self.list_scalers_no_minmax):
            scaler_minmax.fit(self.x)
            scaler_no_minmax.fit(self.x)

            # the difference of the log det of jacobian between two points in the original space should be the same
            self.assertAlmostEqual(
                scaler_minmax.jac_log_det(self.x) - scaler_minmax.jac_log_det(self.x2),
                scaler_no_minmax.jac_log_det(self.x) - scaler_no_minmax.jac_log_det(self.x2),
                delta=1e-7)

            # the difference of the log det of jacobian between two points corresponding to the same two points in the
            # original space (either if the linear rescaling is applied or not) should be the same
            self.assertAlmostEqual(
                scaler_minmax.jac_log_det_inverse_transform(scaler_minmax.transform(self.x)) -
                scaler_minmax.jac_log_det_inverse_transform(scaler_minmax.transform(self.x2)),
                scaler_no_minmax.jac_log_det_inverse_transform(scaler_no_minmax.transform(self.x)) -
                scaler_no_minmax.jac_log_det_inverse_transform(scaler_no_minmax.transform(self.x2)),
                delta=1e-7)

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            self.scaler_mixed.jac_log_det(np.array([[1.1, 2.2], [3.3, 4.4]]))


if __name__ == '__main__':
    unittest.main()
