import unittest
import numpy as np

from abcpy.distances import Euclidean
from abcpy.statistics import Identity
from abcpy.continuousmodels import Normal, Uniform
from abcpy.jointdistances import LinearCombination

class LinearCombinationTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc1 = Identity(degree = 1, cross = 0)
        self.stat_calc2 = Identity(degree= 1, cross = 0)
        self.distancefunc1 = Euclidean(self.stat_calc1)
        self.distancefunc2 = Euclidean(self.stat_calc2)
        ## Define Models
        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model1 = Normal([mu,sigma])
        self.model2 = Normal([mu,sigma])

        #Check whether wrong sized distnacefuncs gives an error
        self.assertRaises(ValueError, LinearCombination, [self.model1,self.model2], [self.distancefunc1], [1.0, 1.0])

        #Check whether wrong sized weights gives an error
        self.assertRaises(ValueError, LinearCombination, [self.model1,self.model2], [self.distancefunc1, self.distancefunc2], [1.0, 1.0, 1.0])

        self.jointdistancefunc = LinearCombination([self.model1,self.model2], [self.distancefunc1, self.distancefunc2], [1.0, 1.0])

    def test_distance(self):
        # test simple distance computation
        a = [[0, 0, 0],[0, 0, 0]]
        b = [[0, 0, 0],[0, 0, 0]]
        c = [[1, 1, 1],[1, 1, 1]]

        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.jointdistancefunc.distance, 3.4, [b])
        self.assertRaises(TypeError, self.jointdistancefunc.distance, [a], 3.4)

        # test input has different dimensionality
        self.assertRaises(BaseException, self.jointdistancefunc.distance, [a], [b,c])
        self.assertRaises(BaseException, self.jointdistancefunc.distance, [b,c], [a])

        # test whether they compute correct values
        self.assertTrue(self.jointdistancefunc.distance([a,b],[a,b]) == np.array([0]))
        self.assertTrue(self.jointdistancefunc.distance([a,c],[c,b]) == np.array([1.7320508075688772]))
        
    def test_dist_max(self):
        self.assertTrue(self.jointdistancefunc.dist_max() == np.inf)


if __name__ == '__main__':
    unittest.main()
