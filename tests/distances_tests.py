import unittest
import numpy as np

from abcpy.distances import Euclidean, PenLogReg, LogReg
from abcpy.statistics import Identity

class EuclideanTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree = 1, cross = 0)
        self.distancefunc = Euclidean(self.stat_calc)
    
    def test_distance(self):
        # test simple distance computation
        a = np.array([[0, 0, 0],[0, 0, 0]])
        b = np.array([[0, 0, 0],[0, 0, 0]])
        c = np.array([[1, 1, 1],[1, 1, 1]])
        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, b)
        self.assertRaises(TypeError, self.distancefunc.distance, a, 3.4)        

        # test input has different dimensionality
        self.assertRaises(BaseException, self.distancefunc.distance, a, np.array([[0, 0], [1, 2]]))  
        self.assertRaises(BaseException, self.distancefunc.distance, a, np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]]))        

        # test whether they compute correct values        
        self.assertTrue(all(self.distancefunc.distance(list(a),list(b)) == np.array([0, 0])))
        self.assertTrue(all(self.distancefunc.distance(list(a),list(c)) == np.array([1.7320508075688772, 1.7320508075688772])))
        
    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == np.inf)        


class PenLogRegTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree = 1, cross = 0)
        self.distancefunc = PenLogReg(self.stat_calc)
    
    def test_distance(self):
        d1 = 0.5 * np.random.randn(100,2) - 10
        d2 = 0.5 * np.random.randn(100,2) + 10
        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc.distance, d1, 3.4)

        # completely separable datasets should have a distance of 1.0
        self.assertEqual(self.distancefunc.distance(list(d1),list(d2)), 1.0)

        # equal data sets should have a distance of 0.0
        self.assertEqual(self.distancefunc.distance(list(d1),list(d1)), 0.0)
        
    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == 1.0)        



class LogRegTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree = 1, cross = 0)
        self.distancefunc = LogReg(self.stat_calc)
        
    def test_distance(self):
        d1 = 0.5 * np.random.randn(100,2) - 10
        d2 = 0.5 * np.random.randn(100,2) + 10
        
        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.distancefunc.distance, 3.4, d2)
        self.assertRaises(TypeError, self.distancefunc.distance, d1, 3.4)
        
        # completely separable datasets should have a distance of 1.0
        self.assertEqual(self.distancefunc.distance(list(d1),list(d2)), 1.0)

        # equal data sets should have a distance of 0.0
        self.assertEqual(self.distancefunc.distance(list(d1),list(d1)), 0.0)
        
    def test_dist_max(self):
        self.assertTrue(self.distancefunc.dist_max() == 1.0)        
        

if __name__ == '__main__':
    unittest.main()
