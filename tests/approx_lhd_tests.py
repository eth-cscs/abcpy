import unittest
import numpy as np

from abcpy.continuousmodels import Normal
from abcpy.continuousmodels import Uniform
from abcpy.statistics import Identity
from abcpy.approx_lhd import PenLogReg, SynLiklihood
# NOTE SOME THINGS NOT FIXED

class PenLogRegTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform([[-5.0, 5.0], [5.0, 10.0]])
        self.prior.sample_parameters()
        self.model = Normal([self.prior])
        self.stat_calc = Identity(degree = 2, cross = 0)
        self.likfun = PenLogReg(self.stat_calc, self.model, n_simulate = 100, n_folds = 10, max_iter = 100000, seed = 1)

    def test_likelihood(self):
        
        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.likelihood, 3.4, [2,1])
        self.assertRaises(TypeError, self.likfun.likelihood, [2,4], 3.4)

        # create observed data
        y_obs = self.model.sample_from_distribution(1, rng=np.random.RandomState(1))[1]
        # create fake simulated data
        self.prior.fixed_values = [1.1,1.0]
        y_sim = self.model.simulate(100)
        comp_likelihood = self.likfun.likelihood(y_obs, y_sim)
        expected_likelihood = 4.3996556327224594
        # This checks whether it computes a correct value and dimension is right
        self.assertLess(comp_likelihood - expected_likelihood, 10e-2)
        
class SynLiklihoodTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform([[-5.0, 5.0], [5.0, 10.0]])
        self.model = Normal([self.prior])
        self.stat_calc = Identity(degree = 2, cross = 0)
        self.likfun = SynLiklihood(self.stat_calc) 


    def test_likelihood(self):
        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.likelihood, 3.4, [2,1])
        self.assertRaises(TypeError, self.likfun.likelihood, [2,4], 3.4)
               
        # create observed data
        y_obs = self.model.sample_from_distribution(1,rng=np.random.RandomState(1))[1]
        # create fake simulated data
        self.prior.fixed_values = [1.1,1.0]
        y_sim = self.model.simulate(100)
        # calculate the statistics of the observed data
        comp_likelihood = self.likfun.likelihood(y_obs, y_sim)
        expected_likelihood = 0.00924953470649
        # This checks whether it computes a correct value and dimension is right
        self.assertLess(comp_likelihood - expected_likelihood, 10e-2)

if __name__ == '__main__':
    unittest.main()
        
