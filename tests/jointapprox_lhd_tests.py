import unittest
import numpy as np

from abcpy.approx_lhd import SynLiklihood
from abcpy.statistics import Identity
from abcpy.continuousmodels import Normal, Uniform
from abcpy.jointapprox_lhd import ProductCombination

class ProductCombinationTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc1 = Identity(degree = 1, cross = 0)
        self.stat_calc2 = Identity(degree= 1, cross = 0)
        self.likfun1 = SynLiklihood(self.stat_calc1)
        self.likfun2 = SynLiklihood(self.stat_calc2)
        ## Define Models
        # define a uniform prior distribution
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model1 = Normal([self.mu,self.sigma])
        self.model2 = Normal([self.mu,self.sigma])

        #Check whether wrong sized distnacefuncs gives an error
        self.assertRaises(ValueError, ProductCombination, [self.model1,self.model2], [self.likfun1])

        self.jointapprox_lhd = ProductCombination([self.model1, self.model2], [self.likfun1, self.likfun2])

    def test_likelihood(self):
        # test simple distance computation
        a = [[0, 0, 0],[0, 0, 0]]
        b = [[0, 0, 0],[0, 0, 0]]
        c = [[1, 1, 1],[1, 1, 1]]

        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.jointapprox_lhd.likelihood, 3.4, [[2,1]])
        self.assertRaises(TypeError, self.jointapprox_lhd.likelihood, [[2,4]], 3.4)

        # test input has different dimensionality
        self.assertRaises(BaseException, self.jointapprox_lhd.likelihood, [a], [b,c])
        self.assertRaises(BaseException, self.jointapprox_lhd.likelihood, [b,c], [a])

        # test whether they compute correct values
        # create observed data
        y_obs = [[9.8], [9.8]]
        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        y_sim_1 = self.model1.forward_simulate(self.model1.get_input_values(), 100, rng=np.random.RandomState(1))
        y_sim_2 = self.model2.forward_simulate(self.model2.get_input_values(), 100, rng=np.random.RandomState(1))
        # calculate the statistics of the observed data
        comp_likelihood = self.jointapprox_lhd.likelihood(y_obs, [y_sim_1, y_sim_2])
        expected_likelihood = 8.612491843767518e-43
        # This checks whether it computes a correct value and dimension is right
        self.assertLess(comp_likelihood - expected_likelihood, 10e-2)


if __name__ == '__main__':
    unittest.main()
