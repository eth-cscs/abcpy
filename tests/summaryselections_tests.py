import unittest
import numpy as np
from abcpy.distributions import Uniform
from abcpy.models import Gaussian
from abcpy.statistics import Identity
from abcpy.backends import BackendDummy as Backend
from abcpy.summaryselections import Semiautomatic



class SemiautomaticTests(unittest.TestCase):
    def setUp(self):
        self.stat_calc = Identity(degree = 1, cross = 0)
        
        # define prior and model
        prior = Uniform([150, 5],[200, 25])
        self.model = Gaussian(prior, seed = 1)

        # define backend
        self.backend = Backend()

        # define statistics
        self.statistics_cal = Identity(degree = 2, cross = False)
        
        #Initialize summaryselection
        self.summaryselection = Semiautomatic(self.model, self.statistics_cal, self.backend, n_samples = 1000, seed = 1)

        
    def test_transformation(self):
        #Transform statistics extraction
        self.statistics_cal.statistics = lambda x, f2=self.summaryselection.transformation, f1=self.statistics_cal.statistics: f2(f1(x))
        y_obs = self.model.simulate(10)
        extracted_statistics_10 = self.statistics_cal.statistics(y_obs)
        self.assertEqual(np.shape(extracted_statistics_10), (10,2))
        y_obs = self.model.simulate(1)
        extracted_statistics_1 = self.statistics_cal.statistics(y_obs)
        self.assertLess(extracted_statistics_1[0,0] - 111.012664458, 10e-2)
        self.assertLess(extracted_statistics_1[0,1] - (-63.224510811), 10e-2)
        
if __name__ == '__main__':
    unittest.main()
