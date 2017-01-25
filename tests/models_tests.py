import unittest

import numpy as np

from abcpy.distributions import Uniform
from abcpy.models import Gaussian, Student_t, MixtureNormal, StochLorenz95, Ricker


class GaussianTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform([-1.0, 0.0], [1.0, 1.0], seed=1)
        self.model = Gaussian(self.prior, 0, 1, seed=1)

    def test_simulate(self):
        samples = self.model.simulate(10)
        self.assertIsInstance(samples, list)
        expected_output = [1.6243453636632417, -0.61175641365007538, -0.5281717522634557, \
                            -1.0729686221561705, 0.86540762932467852, -2.3015386968802827, \
                            1.74481176421648, -0.76120690089510279, 0.31903909605709857, \
                            -0.24937037547741009]    
        self.assertEqual(samples, expected_output)
        
    def test_get_parameters(self):
        self.model.sample_from_prior()
        params = self.model.get_parameters()

        # test shape of parameters
        param_len = len(params)
        self.assertEqual(param_len, 2)
        
    def test_set_parameters(self):
        self.assertRaises(TypeError, self.model.set_parameters, 3.4)
        self.assertFalse(self.model.set_parameters([1,3,2]))
        self.assertFalse(self.model.set_parameters([2,-1]))    
        


class Student_tTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform([-1.0, 0.0], [1.0, 1.0], seed=1)
        self.model = Student_t(self.prior, 2.0, 2.0, seed = 1)


        
    def test_simulate(self):
        samples = self.model.simulate(10)
        self.assertIsInstance(samples, list)
        expected_output = [153.88005569908395, 0.98042403895439678, 1.2570920974156827, \
                           0.78037076267288352, 5.6483149691097685, 1.4753038279725934, \
                           5.7592295012149668, -2.3845364422426973, 2.7841289920756855, \
                           1.2667482581389569]
        self.assertEqual(samples, expected_output)


        
    def test_get_parameters(self):
        self.model.sample_from_prior()
        params = self.model.get_parameters()

        # test shape of parameters
        param_shape = np.shape(params)
        self.assertEqual(param_shape, (2,))


        
    def test_set_parameters(self):
        self.assertRaises(TypeError, self.model.set_parameters, 3.4)
        self.assertFalse(self.model.set_parameters([1,3,2]))
        self.assertFalse(self.model.set_parameters([2,-1])) 
    
class MixtureNormalTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform(np.array(-10*np.ones(shape=(2,))), np.array(10*np.ones(shape=(2,))), seed=1)
        self.model = MixtureNormal(self.prior, np.array([3,5]), seed = 1)
        
    def test_simulate(self):
        samples = self.model.simulate(10)
        self.assertIsInstance(samples, list)
        #expected_output = 
        #np.testing.assert_equal()
        
    def test_get_parameters(self):
        self.model.sample_from_prior()
        params = self.model.get_parameters()

        # test shape of parameters
        param_shape = np.shape(params)
        self.assertEqual(param_shape, (2,))


        
    def test_set_parameters(self):
        self.assertRaises(TypeError, self.model.set_parameters, 3.4)

class StochLorenz95Tests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform(np.array([1,.1]), np.array([3,.3]), seed=1)
        self.model = StochLorenz95(self.prior, np.array([2.1,.1]), initial_state = None, n_timestep = 160, seed = 1)

    def test_simulate(self):
        samples = self.model.simulate(1)
        self.assertIsInstance(samples, list)
        #expected_output = 
        #np.testing.assert_equal()


        
    def test_get_parameters(self):
        self.model.sample_from_prior()
        params = self.model.get_parameters()

        # test shape of parameters
        param_shape = np.shape(params)
        self.assertEqual(param_shape, (2,))


        
    def test_set_parameters(self):
        self.assertRaises(TypeError, self.model.set_parameters, 3.4)

class RickerTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform([3., 0., 1.], [5., 1., 20.], seed = 1)
        self.model = Ricker(self.prior, np.array([3.8,.3,10]),  n_timestep = 100, seed = 1)

    def test_simulate(self):
        samples = self.model.simulate(1)
        self.assertIsInstance(samples, list)
        #expected_output = 
        #np.testing.assert_equal()


        
    def test_get_parameters(self):
        self.model.sample_from_prior()
        params = self.model.get_parameters()

        # test shape of parameters
        param_shape = np.shape(params)
        self.assertEqual(param_shape, (3,))


        
    def test_set_parameters(self):
        self.assertRaises(TypeError, self.model.set_parameters, 3.4)



if __name__ == '__main__':
    unittest.main()
