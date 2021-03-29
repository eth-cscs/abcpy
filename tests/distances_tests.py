import numpy as np
import unittest
import warnings

from abcpy.distances import Euclidean, PenLogReg, LogReg, Wasserstein, SlicedWasserstein, GammaDivergence, MMD, \
    EnergyDistance, KLDivergence, SquaredHellingerDistance
from abcpy.statistics import Identity


class EuclideanTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=1, cross=0)
        self.distancefunc = Euclidean(self.stat_calc)

    def test_distance(self):
        # test simple distance computation
        a = [[0, 0, 0], [0, 0, 0]]
        b = [[0, 0, 0], [0, 0, 0]]
        c = [[1, 1, 1], [1, 1, 1]]
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, b)
        self.assertRaises(TypeError, self.distancefunc.distance, a, 3.4)

        # test input has different dimensionality
        self.assertRaises(BaseException, self.distancefunc.distance, a, np.array([[0, 0], [1, 2]]))
        self.assertRaises(BaseException, self.distancefunc.distance, a, np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]]))

        # test whether they compute correct values
        self.assertTrue(self.distancefunc.distance(a, b) == np.array([0]))
        self.assertTrue(self.distancefunc.distance(a, c) == np.array([1.7320508075688772]))

    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == np.inf)


class PenLogRegTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=1, cross=False)
        self.distancefunc = PenLogReg(self.stat_calc)
        self.rng = np.random.RandomState(1)

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10
        d3 = 0.5 * self.rng.randn(95, 2) + 10

        d1 = d1.tolist()
        d2 = d2.tolist()
        d3 = d3.tolist()
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc.distance, d1, 3.4)

        # completely separable datasets should have a distance of 1.0
        self.assertEqual(self.distancefunc.distance(d1, d2), 1.0)

        # equal data sets should have a distance of 0.0
        self.assertEqual(self.distancefunc.distance(d1, d1), 0.0)

        # equal data sets should have a distance of 0.0; check that in case where n_samples is not a multiple of n_folds
        # in cross validation (10)
        self.assertEqual(self.distancefunc.distance(d3, d3), 0.0)

        # check if it returns the correct error when the number of datasets:
        self.assertRaises(RuntimeError, self.distancefunc.distance, d1, d3)

    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == 1.0)


class LogRegTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=2, cross=False)
        self.distancefunc = LogReg(self.stat_calc, seed=1)
        self.rng = np.random.RandomState(1)

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10

        d1 = d1.tolist()
        d2 = d2.tolist()

        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc.distance, d1, 3.4)

        # completely separable datasets should have a distance of 1.0
        self.assertEqual(self.distancefunc.distance(d1, d2), 1.0)

        # equal data sets should have a distance of 0.0
        self.assertEqual(self.distancefunc.distance(d1, d1), 0.0)

    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == 1.0)


class WassersteinTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=2, cross=False)
        self.distancefunc = Wasserstein(self.stat_calc)
        self.rng = np.random.RandomState(1)

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10

        d1 = d1.tolist()
        d2 = d2.tolist()

        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc.distance, d1, 3.4)

        self.assertEqual(self.distancefunc.distance(d1, d2), 28.623685155319652)

        # equal data sets should have a distance of approximately 0.0; it won't be exactly 0 due to numerical rounding
        self.assertAlmostEqual(self.distancefunc.distance(d1, d1), 0.0, delta=1e-5)

    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == np.inf)


class SlicedWassersteinTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=2, cross=False)
        self.rng = np.random.RandomState(1)
        self.distancefunc = SlicedWasserstein(self.stat_calc, rng=self.rng)

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10

        d1 = d1.tolist()
        d2 = d2.tolist()

        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc.distance, d1, 3.4)

        self.assertAlmostEqual(self.distancefunc.distance(d1, d2), 12.604402810464576)

        # equal data sets should have a distance of approximately 0.0; it won't be exactly 0 due to numerical rounding
        self.assertAlmostEqual(self.distancefunc.distance(d1, d1), 0.0, delta=1e-5)

    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == np.inf)


class GammaDivergenceTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=2, cross=False)
        self.rng = np.random.RandomState(1)
        # notice: the GammaDivergence estimator becomes infinity when one element in the s1 dataset is equal to one in
        # the s2 dataset and k=1 (as the distance between those two would be 0, which gives infinity when dividing)
        self.distancefunc_k1 = GammaDivergence(self.stat_calc, gam=0.1, k=1)
        self.distancefunc_k2 = GammaDivergence(self.stat_calc, gam=0.1, k=2)

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10
        d3 = 0.5 * self.rng.randn(100, 2) + 10
        d3[0] = d1[0]  # set one single element to be equal
        d4 = 0.5 * self.rng.randn(100, 2) + 10
        d4[0] = d4[1] = d1[0]  # set two elements to be equal
        d5 = 0.5 * self.rng.randn(100, 2) + 10
        d5[0] = d5[1] = d5[2] = d1[0]  # set three elements to be equal

        d1 = d1.tolist()
        d2 = d2.tolist()
        d3 = d3.tolist()
        d4 = d4.tolist()
        d5 = d5.tolist()

        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc_k1.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc_k1.distance, d1, 3.4)

        self.assertRaises(ValueError, self.distancefunc_k2.distance, [d1[i] for i in range(2)],
                          [d2[i] for i in range(2)])

        # print(self.distancefunc.distance(d1, d2))
        self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d2), 11.781045117610248)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d2), 9.898652219975974)

        # check identical dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d1), - np.inf)
            self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d3), - np.inf)
            self.assertAlmostEqual(self.distancefunc_k1.distance(d2, d4), + np.inf)
            self.assertAlmostEqual(self.distancefunc_k2.distance(d2, d5), + np.inf)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d1), -1.8398859674447485)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d3), 9.904082614090905)

    def test_dist_max(self):
        self.assertTrue(self.distancefunc_k1.dist_max() == np.inf)


class KLDivergenceTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=2, cross=False)
        self.rng = np.random.RandomState(1)
        # notice: the KLDivergenceTests estimator becomes infinity when one element in the s1 dataset is equal to one
        # in the s2 dataset and k=1 (as the distance between those two would be 0, which gives infinity when dividing)
        self.distancefunc_k1 = KLDivergence(self.stat_calc, k=1)
        self.distancefunc_k2 = KLDivergence(self.stat_calc, k=2)

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10
        d3 = 0.5 * self.rng.randn(100, 2) + 10
        d3[0] = d1[0]  # set one single element to be equal

        d1 = d1.tolist()
        d2 = d2.tolist()
        d3 = d3.tolist()

        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc_k1.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc_k1.distance, d1, 3.4)

        self.assertRaises(ValueError, self.distancefunc_k2.distance, [d1[i] for i in range(2)],
                          [d2[i] for i in range(2)])

        # print(self.distancefunc.distance(d1, d2))
        self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d2), 11.268656821978732)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d2), 9.632589300292643)

        # check identical dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d1), - np.inf)
            self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d3), - np.inf)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d1), -1.6431603119265736)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d3), 9.629190258363401)

    def test_dist_max(self):
        self.assertTrue(self.distancefunc_k1.dist_max() == np.inf)


class MMDTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=2, cross=False)
        self.rng = np.random.RandomState(1)
        self.distancefunc = MMD(self.stat_calc)
        self.distancefunc_biased = MMD(self.stat_calc, biased_estimator=True)

    def test_initialization(self):
        self.assertRaises(RuntimeError, MMD, self.stat_calc, 5)
        self.assertRaises(NotImplementedError, MMD, self.stat_calc, "ciao")

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10

        d1 = d1.tolist()
        d2 = d2.tolist()

        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc.distance, d1, 3.4)

        # print(self.distancefunc.distance(d1, d2))
        self.assertAlmostEqual(self.distancefunc.distance(d1, d2), 0.01078569482366823)
        self.assertAlmostEqual(self.distancefunc_biased.distance(d1, d2), 0.030677837875431546)

        # check identical dataset
        self.assertAlmostEqual(self.distancefunc.distance(d1, d1), - 0.019872124487359623)
        self.assertAlmostEqual(self.distancefunc_biased.distance(d1, d1), 0.0)

        # check if it is symmetric:
        self.assertAlmostEqual(self.distancefunc.distance(d1, d2), self.distancefunc.distance(d2, d1))
        self.assertAlmostEqual(self.distancefunc_biased.distance(d1, d2), self.distancefunc_biased.distance(d2, d1))

    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == np.inf)


class EnergyDistanceTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=2, cross=False)
        self.rng = np.random.RandomState(1)
        self.distancefunc = EnergyDistance(self.stat_calc)
        self.distancefunc_unbiased = EnergyDistance(self.stat_calc, biased_estimator=False)

    def test_initialization(self):
        self.assertRaises(RuntimeError, MMD, self.stat_calc, 5)
        self.assertRaises(NotImplementedError, MMD, self.stat_calc, "ciao")

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10

        d1 = d1.tolist()
        d2 = d2.tolist()

        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc.distance, d1, 3.4)

        # print(self.distancefunc.distance(d1, d2))
        self.assertAlmostEqual(self.distancefunc.distance(d1, d2), 33.76256466781305)
        self.assertAlmostEqual(self.distancefunc_unbiased.distance(d1, d2), 33.41913579594933)

        # check identical dataset
        self.assertAlmostEqual(self.distancefunc.distance(d1, d1), 0.0)
        self.assertAlmostEqual(self.distancefunc_unbiased.distance(d1, d1), -0.31969528899748667)

        # check if it is symmetric:
        self.assertAlmostEqual(self.distancefunc.distance(d1, d2), self.distancefunc.distance(d2, d1))
        self.assertAlmostEqual(self.distancefunc_unbiased.distance(d1, d2),
                               self.distancefunc_unbiased.distance(d2, d1), )

    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == np.inf)


class SquaredHellingerDistanceTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree=1, cross=False)
        self.rng = np.random.RandomState(1)
        # notice: the SquaredHellingerDistance estimator becomes infinity when one element in the s1 dataset is equal to one in
        # the s2 dataset and k=1 (as the distance between those two would be 0, which gives infinity when dividing)
        self.distancefunc_k1 = SquaredHellingerDistance(self.stat_calc, k=1)
        self.distancefunc_k2 = SquaredHellingerDistance(self.stat_calc, k=2)

    # def test_different_distances(self):
    #     d1 = 0.5 * self.rng.randn(100, 2) - 10
    #     d2 = 0.5 * self.rng.randn(100, 2) - 10
    #     d1 = d1.tolist()
    #     for i in range(-20, 40):
    #         d2_list = (d2 + i).tolist()
    #         print(i, self.distancefunc_k1.distance(d1, d2_list))
    #
    #     # the estimator seems to be OK, even if it reaches super small values when the two datasets are from same
    #     # parameters

    def test_distance(self):
        d1 = 0.5 * self.rng.randn(100, 2) - 10
        d2 = 0.5 * self.rng.randn(100, 2) + 10
        d3 = 0.5 * self.rng.randn(100, 2) + 10
        d3[0] = d1[0]  # set one single element to be equal
        d1 = d1.tolist()
        d2 = d2.tolist()
        d3 = d3.tolist()

        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc_k1.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc_k1.distance, d1, 3.4)

        self.assertRaises(ValueError, self.distancefunc_k2.distance, [d1[i] for i in range(2)],
                          [d2[i] for i in range(2)])

        self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d2), 1.123969688201248)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d2), 0.7593257032591699)

        # check infinities when there are same elements in the two datasets
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d1), - np.inf)
            self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d3), - np.inf)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d1), -324.97681257084014)
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d3), -274.4911947796949)

        # check if it is symmetric:
        self.assertAlmostEqual(self.distancefunc_k1.distance(d1, d2), self.distancefunc_k1.distance(d2, d1))
        self.assertAlmostEqual(self.distancefunc_k2.distance(d1, d2), self.distancefunc_k2.distance(d2, d1))

    def test_dist_max(self):
        self.assertTrue(self.distancefunc_k1.dist_max() == 2)


if __name__ == '__main__':
    unittest.main()
