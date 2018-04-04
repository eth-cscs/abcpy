import unittest
from abcpy.continuousmodels import Uniform
from abcpy.continuousmodels import Normal
from abcpy.continuousmodels import StudentT
from abcpy.statistics import Identity
from abcpy.backends import BackendDummy as Backend
from abcpy.modelselections import RandomForest
     
class RandomForestTests(unittest.TestCase):
    def setUp(self):
        # define observation for true parameters mean=170, std=15
        self.y_obs = [160.82499176]
        self.model_array = [None]*2
        #Model 1: Gaussian
        # define prior
        self.mu1 = Uniform([[150], [200]], name='mu1')
        self.sigma1 = Uniform([[5.0], [25.0]], name='sigma1')
        # define the model
        self.model_array[0] = Normal([self.mu1, self.sigma1])
        #Model 2: Student t
        # define prior
        self.mu2 = Uniform([[150], [200]], name='mu2')
        self.sigma2 = Uniform([[1], [30.0]], name='sigma2')
        # define the model
        self.model_array[1] = StudentT([self.mu2, self.sigma2])

        # define statistics
        self.statistics_calc = Identity(degree = 2, cross = False)
        # define backend
        self.backend = Backend()


    def test_select_model(self):
        modelselection = RandomForest(self.model_array, self.statistics_calc, self.backend, seed = 1)
        model = modelselection.select_model(self.y_obs,n_samples = 100, n_samples_per_param = 1)

        self.assertTrue(self.model_array[0] == model)
    
    def test_posterior_probability(self):
        modelselection = RandomForest(self.model_array, self.statistics_calc, self.backend, seed = 1)
        model_prob = modelselection.posterior_probability(self.y_obs)

        self.assertTrue(model_prob > 0.7)
 
if __name__ == '__main__':
    unittest.main()