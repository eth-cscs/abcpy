import unittest
import numpy as np

from abcpy.models import Gaussian
from abcpy.distributions import Uniform
from abcpy.statistics import Identity
from abcpy.approx_lhd import PenLogReg, SynLiklihood


class PenLogRegTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform([-5.0, 5.0], [5.0, 10.0], seed=1)
        self.model = Gaussian(self.prior, mu=1.1, sigma=1.0, seed=1)
        self.stat_calc = Identity(degree = 2, cross = 0)
        self.likfun = PenLogReg(self.stat_calc, self.model, n_simulate = 100, n_folds = 10, max_iter = 100000, seed = 1)

    def test_likelihood(self):
        
        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.likelihood, 3.4, [2,1])
        self.assertRaises(TypeError, self.likfun.likelihood, [2,4], 3.4)

        # create observed data
        y_obs = self.model.simulate(1)
        # create fake simulated data
        self.model.set_parameters(np.array([1.1,1.0]))
        y_sim = self.model.simulate(100)
        comp_likelihood = self.likfun.likelihood(y_obs, y_sim)
        expected_likelihood = 4.3996556327224594
        # This checks whether it computes a correct value and dimension is right
        self.assertLess(comp_likelihood - expected_likelihood, 10e-2)
        
class SynLiklihoodTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform([-5.0, 5.0], [5.0, 10.0], seed=1)
        self.model = Gaussian(self.prior, mu=1.1, sigma=1.0, seed = 1)
        self.stat_calc = Identity(degree = 2, cross = 0)
        self.likfun = SynLiklihood(self.stat_calc) 


    def test_likelihood(self):
        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.likfun.likelihood, 3.4, [2,1])
        self.assertRaises(TypeError, self.likfun.likelihood, [2,4], 3.4)
               
        # create observed data
        y_obs = self.model.simulate(1)
        # create fake simulated data
        self.model.set_parameters(np.array([1.1,1.0]))
        y_sim = self.model.simulate(100)
        # calculate the statistics of the observed data
        comp_likelihood = self.likfun.likelihood(y_obs, y_sim)
        expected_likelihood = 0.00924953470649
        # This checks whether it computes a correct value and dimension is right
        self.assertLess(comp_likelihood - expected_likelihood, 10e-2)
        
