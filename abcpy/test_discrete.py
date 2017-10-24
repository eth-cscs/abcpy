from discrete import *
import unittest
import numpy as np
from scipy.stats import binom

class BinomialTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(1)
        self.binomial = Binomial([10,0.3])

    def test_sample_from_distribution(self):
        samples = self.binomial.sample_from_distribution(100, rng=self.rng)
        expected_samples = binom.rvs(10, 0.3, size=100, random_state = np.random.RandomState(1))
        self.assertLess((samples-expected_samples).all(),1e-5)

    def test_pmf(self):
        computed_pmf = self.binomial.pmf(2)
        expected_pmf = binom.pmf(2, 10, 0.3)
        self.assertLess(abs(computed_pmf-expected_pmf), 1e-5)


if __name__ == '__main__':
    unittest.main()