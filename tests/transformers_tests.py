import unittest

import numpy as np

from abcpy.transformers import DummyTransformer, BoundedVarTransformer


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


if __name__ == '__main__':
    unittest.main()
