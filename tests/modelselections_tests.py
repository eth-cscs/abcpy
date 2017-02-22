import unittest
from abcpy.distributions import Uniform
from abcpy.models import Gaussian
from abcpy.models import Student_t
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
        prior = Uniform([150, 5],[200, 25])
        # define the model
        self.model_array[0] = Gaussian(prior, seed = 1)
        #Model 2: Student t
        # define prior
        prior = Uniform([150, 1],[200, 30])
        # define the model
        self.model_array[1] = Student_t(prior, seed = 1)

        # define statistics
        self.statistics_calc = Identity(degree = 2, cross = False)
        # define backend
        self.backend = Backend()


    def test_select_model(self):
        modelselection = RandomForest(self.model_array, self.statistics_calc, self.backend, seed = 1)
        model = modelselection.select_model(self.y_obs,n_samples = 100, n_samples_per_param = 1)

        self.assertTrue(self.model_array[1] == model)
    
    def test_posterior_probability(self):
        modelselection = RandomForest(self.model_array, self.statistics_calc, self.backend, seed = 1)
        model_prob = modelselection.posterior_probability(self.y_obs)

        self.assertTrue(model_prob == 0.8238)
 
if __name__ == '__main__':
    unittest.main()