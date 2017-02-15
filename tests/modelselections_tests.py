import unittest
import numpy as np
from abcpy.distributions import Uniform
from abcpy.models import Gaussian
from abcpy.models import Student_t
from abcpy.statistics import Identity
from abcpy.distances import Euclidean
from abcpy.backends import BackendDummy as Backend
from abcpy.modelselections import Knn, RandomForest



class KnnTests(unittest.TestCase):
    def setUp(self):
        # define observation for true parameters mean=170, std=15
        self.y_obs = [160.82499176, 167.24266737, 185.71695756, 153.7045709, 163.40568812, 140.70658699, 169.59102084, 172.81041696, 187.38782738, 179.66358934, 176.63417241, 189.16082803, 181.98288443, 170.18565017, 183.78493886, 166.58387299, 161.9521899, 155.69213073, 156.17867343, 144.51580379, 170.29847515, 197.96767899, 153.36646527, 162.22710198, 158.70012047, 178.53470703, 170.77697743, 164.31392633, 165.88595994, 177.38083686, 146.67058471763457, 179.41946565658628, 238.02751620619537, 206.22458790620766, 220.89530574344568, 221.04082532837026, 142.25301427453394, 261.37656571434275, 171.63761180867033, 210.28121820385866, 237.29130237612236, 175.75558340169619, 224.54340549862235, 197.42448680731226, 165.88273684581381, 166.55094082844519, 229.54308602661584, 222.99844054358519, 185.30223966014586, 152.69149367593846, 206.94372818527413, 256.35498655339154, 165.43140916577741, 250.19273595481803, 148.87781549665536, 223.05547559193792, 230.03418198709608, 146.13611923127021, 138.24716809523139, 179.26755740864527, 141.21704876815426, 170.89587081800852, 222.96391329259626, 188.27229523693822, 202.67075179617672, 211.75963110985992, 217.45423324370509]
        self.model_array = [None]*2
        #Model 1: Gaussian
        # define prior
        prior = Uniform([150, 5],[200, 25])
        # define the model
        self.model_array[0] = Gaussian(prior)

        #Model 2: Student t
        # define prior
        prior = Uniform([150, 1],[200, 30])
        # define the model
        self.model_array[1] = Student_t(prior)
        # define statistics
        self.statistics_calc = Identity(degree = 2, cross = False)
        # define distance
        self.distance_calc = Euclidean(self.statistics_calc)
        # define backend
        self.backend = Backend()


    def test_modelchoice(self):
        modelselection = Knn(model_array, statistics_calc, distance_calc, backend)
        model = modelselection.modelchoice(y_obs)        
        
        self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
        vec1 = np.array([1,2])
        vec2 = np.array([1])
        self.assertTrue((self.stat_calc.statistics([vec1]) == np.array([vec1])).all())
        self.assertTrue((self.stat_calc.statistics([vec1,vec1]) == np.array([[vec1],[vec1]])).all())
        self.assertTrue((self.stat_calc.statistics([vec2,vec2]) == np.array([[vec2],[vec2]])).all())
    
    def test_posteriorprobability(self):
        model_prob = modelselection.posteriorprobability(y_obs)
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
     
class RandomForestTests(unittest.TestCase):
    def setUp(self):
        # define observation for true parameters mean=170, std=15
        self.y_obs = [160.82499176, 167.24266737, 185.71695756, 153.7045709, 163.40568812, 140.70658699, 169.59102084, 172.81041696, 187.38782738, 179.66358934, 176.63417241, 189.16082803, 181.98288443, 170.18565017, 183.78493886, 166.58387299, 161.9521899, 155.69213073, 156.17867343, 144.51580379, 170.29847515, 197.96767899, 153.36646527, 162.22710198, 158.70012047, 178.53470703, 170.77697743, 164.31392633, 165.88595994, 177.38083686, 146.67058471763457, 179.41946565658628, 238.02751620619537, 206.22458790620766, 220.89530574344568, 221.04082532837026, 142.25301427453394, 261.37656571434275, 171.63761180867033, 210.28121820385866, 237.29130237612236, 175.75558340169619, 224.54340549862235, 197.42448680731226, 165.88273684581381, 166.55094082844519, 229.54308602661584, 222.99844054358519, 185.30223966014586, 152.69149367593846, 206.94372818527413, 256.35498655339154, 165.43140916577741, 250.19273595481803, 148.87781549665536, 223.05547559193792, 230.03418198709608, 146.13611923127021, 138.24716809523139, 179.26755740864527, 141.21704876815426, 170.89587081800852, 222.96391329259626, 188.27229523693822, 202.67075179617672, 211.75963110985992, 217.45423324370509]
        self.model_array = [None]*2
        #Model 1: Gaussian
        # define prior
        prior = Uniform([150, 5],[200, 25])
        # define the model
        self.model_array[0] = Gaussian(prior)

        #Model 2: Student t
        # define prior
        prior = Uniform([150, 1],[200, 30])
        # define the model
        self.model_array[1] = Student_t(prior)
        # define statistics
        self.statistics_calc = Identity(degree = 2, cross = False)
        # define distance
        self.distance_calc = Euclidean(self.statistics_calc)
        # define backend
        self.backend = Backend()


    def test_modelchoice(self):
        modelselection = RandomForest(model_array, statistics_calc, distance_calc, backend)
        model = modelselection.modelchoice([y_obs[10]],n_samples = 100, n_samples_per_param = 1)

        
        self.assertRaises(TypeError, self.stat_calc.statistics, 3.4)
        vec1 = np.array([1,2])
        vec2 = np.array([1])
        self.assertTrue((self.stat_calc.statistics([vec1]) == np.array([vec1])).all())
        self.assertTrue((self.stat_calc.statistics([vec1,vec1]) == np.array([[vec1],[vec1]])).all())
        self.assertTrue((self.stat_calc.statistics([vec2,vec2]) == np.array([[vec2],[vec2]])).all())
    
    def test_posteriorprobability(self):
        modelselection = RandomForest(model_array, statistics_calc, distance_calc, backend)
        model_prob = modelselection.posteriorprobability([y_obs[10]])

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