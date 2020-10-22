import unittest

import numpy as np

from abcpy.statistics import Identity, LinearTransformation, NeuralEmbedding

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True
    from abcpy.NN_utilities.networks import createDefaultNN


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=1, cross=0)

    def test_statistics(self):
        self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
        vec1 = np.array([1, 2])
        vec2 = np.array([1])
        self.assertTrue((self.stat_calc.statistics([vec1]) == np.array([vec1])).all())
        self.assertTrue((self.stat_calc.statistics([vec1, vec1]) == np.array([[vec1], [vec1]])).all())
        self.assertTrue((self.stat_calc.statistics([vec2, vec2]) == np.array([[vec2], [vec2]])).all())

    def test_polynomial_expansion(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.stat_calc._polynomial_expansion, 3.4)

        a = [np.array([0, 2]), np.array([2, 1])]
        # test cross-product part
        self.stat_calc = Identity(degree=2, cross=1)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0, 2, 0, 4, 0], [2, 1, 4, 1, 2]])).all())
        # When a tuple
        a = [np.array([0, 2])]
        self.stat_calc = Identity(degree=2, cross=1)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0, 2, 0, 4, 0]])).all())
        self.stat_calc = Identity(degree=2, cross=0)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0, 2, 0, 4]])).all())
        a = list(np.array([2]))
        self.stat_calc = Identity(degree=2, cross=1)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[2, 4]])).all())


class LinearTransformationTests(unittest.TestCase):
    def setUp(self):
        self.coeff = np.array([[3, 4], [5, 6]])
        self.stat_calc = LinearTransformation(self.coeff, degree=1, cross=0)

    def test_statistics(self):
        self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
        vec1 = np.array([1, 2])
        vec2 = np.array([1])
        self.assertTrue((self.stat_calc.statistics([vec1]) == np.dot(vec1, self.coeff)).all())
        self.assertTrue((self.stat_calc.statistics([vec1, vec1]) == np.array(
            [np.dot(np.array([1, 2]), self.coeff), np.dot(np.array([1, 2]), self.coeff)])).all())
        self.assertRaises(ValueError, self.stat_calc.statistics, [vec2])

    def test_polynomial_expansion(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.stat_calc._polynomial_expansion, 3.4)

        a = [np.array([0, 2]), np.array([2, 1])]
        # test cross-product part
        self.stat_calc = LinearTransformation(self.coeff, degree=2, cross=1)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[10, 12, 100, 144, 120],
                                                                   [11, 14, 121, 196, 154]])).all())
        # When a tuple
        a = [np.array([0, 2])]
        self.stat_calc = LinearTransformation(self.coeff, degree=2, cross=1)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[10, 12, 100, 144, 120]])).all())
        self.stat_calc = LinearTransformation(self.coeff, degree=2, cross=0)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[10, 12, 100, 144]])).all())
        a = list(np.array([2]))
        self.stat_calc = LinearTransformation(self.coeff, degree=2, cross=1)
        self.assertRaises(ValueError, self.stat_calc.statistics, a)


class NeuralEmbeddingTests(unittest.TestCase):
    def setUp(self):
        if has_torch:
            self.net = createDefaultNN(2, 3)()

    def test_statistics(self):
        if not has_torch:
            self.assertRaises(ImportError, NeuralEmbedding, None)
        else:
            self.stat_calc = NeuralEmbedding(self.net)

            self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
            vec1 = np.array([1, 2])
            vec2 = np.array([1])
            self.assertTrue((self.stat_calc.statistics([vec1])).all())
            self.assertTrue((self.stat_calc.statistics([vec1, vec1])).all())
            self.assertRaises(RuntimeError, self.stat_calc.statistics, [vec2])


if __name__ == '__main__':
    unittest.main()
