import unittest
import numpy as np
from abcpy.statistics import Identity

class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree = 1, cross = 0)

    def test_statistics(self):
        self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
        vec1 = np.array([1,2])
        vec2 = np.array([1])
        self.assertTrue((self.stat_calc.statistics([vec1]) == np.array([vec1])).all())
        self.assertTrue((self.stat_calc.statistics([vec1,vec1]) == np.array([[vec1],[vec1]])).all())
        self.assertTrue((self.stat_calc.statistics([vec2,vec2]) == np.array([[vec2],[vec2]])).all())
    
    def test_polynomial_expansion(self):
        #Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.stat_calc._polynomial_expansion, 3.4)

        a = [np.array([0, 2]),np.array([2,1])]             
        # test cross-product part
        self.stat_calc = Identity(degree = 2, cross = 1)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0,2,0,4,0],[2,1,4,1,2]])).all())
        # When a tuple
        a = [np.array([0, 2])] 
        self.stat_calc = Identity(degree = 2, cross = 1)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0,2,0,4,0]])).all())
        self.stat_calc = Identity(degree = 2, cross = 0)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[0,2,0,4]])).all())
        a = list(np.array([2])) 
        self.stat_calc = Identity(degree = 2, cross = 1)
        self.assertTrue((self.stat_calc.statistics(a) == np.array([[2,4]])).all())
        
if __name__ == '__main__':
    unittest.main()
