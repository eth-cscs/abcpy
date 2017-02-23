import unittest
import numpy as np

from abcpy.distributions import MultiNormal
from abcpy.distributions import Uniform
from abcpy.distributions import MultiStudentT
from abcpy.distributions import Normal

class MultiNormalTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.mean = np.array([-13.0, .0, 7.0])
        self.cov = np.eye(3)
        self.distribution = MultiNormal(self.mean, self.cov, seed=1)
        

        
    def test_sample(self):
        samples = self.distribution.sample(100)
        computed_means = samples.mean(axis=0)
        computed_vars = samples.var(axis=0)
        expected_means = np.array([-12.9820723, 0.08671813, 7.11855369])
        expected_vars = np.array([0.99725084, 0.8610233, 0.8089557])
        self.assertTrue((computed_means - expected_means < 1e-5).all())
        self.assertTrue((computed_vars - expected_vars < 1e-5).all())


        
    def test_set_parameters(self):
        new_mean = np.array([130.0, 10.0, .0, .0])
        new_cov = np.eye(4) * 1e-2
        self.distribution.set_parameters([new_mean, new_cov])
        pdf_value = self.distribution.pdf(new_mean)
        self.assertLess(abs(pdf_value - 253.302959106), 1e-6)

        samples = self.distribution.sample(100)
        computed_means = samples.mean(axis=0)
        computed_vars = samples.var(axis=0)
        expected_means = np.array([1.30004201e+02, 1.00043990e+01, 1.08618430e-02, 8.21679910e-04])
        expected_vars = np.array([0.01023298, 0.00919317, 0.00876968, 0.00987364])
        self.assertTrue((computed_means - expected_means < 1e-5).all())
        self.assertTrue((computed_vars - expected_vars < 1e-5).all())
        


        
class UniformTests(unittest.TestCase):
    def setUp(self):
        self.prior = Uniform([-1.0, -1.0],[1.0, 1.0], seed=1)

        
    def test_init(self):
        self.assertRaises(TypeError, Uniform, 3.14, [1.0, 1.0])
        self.assertRaises(TypeError, Uniform, [-1.0, -1.0], 3.14)
        self.assertRaises(BaseException, Uniform, [-1.0, -1.0], [.0, 1.0, 1.0])

        
    def test_sample(self):
        samples = self.prior.sample(1000)
        samples_avg = samples.mean(axis=0)
        samples_min = samples.min(axis=0)
        samples_max = samples.max(axis=0)
        for (avg, min, max) in zip(samples_avg, samples_min, samples_max):
            self.assertLess(abs(avg), 0.05)
            self.assertLess(abs(min + 1.0), 0.05)
            self.assertLess(abs(max - 1.0), 0.05)

        samples_shape = np.shape(samples)
        self.assertEqual(samples_shape, (1000,2))

        
    def test_set_parameters(self):
        distribution = Uniform([-101],[-100], seed=1)
        sample = distribution.sample(1)[0,0]
        self.assertLessEqual(sample, -100)
        self.assertGreaterEqual(sample, -101)

        distribution.set_parameters([[100],[101]])
        sample = distribution.sample(1)[0,0]
        self.assertLessEqual(sample, 101)
        self.assertGreaterEqual(sample, 100)
        

    def test_pdf(self):
        new_prior = Uniform(np.array([0.0]), np.array([10.0]), seed=1)
        self.assertEqual(new_prior.pdf(0), 0.1)
        self.assertEqual(new_prior.pdf(-1), 0)
        self.assertEqual(new_prior.pdf(11), 0)



class MultiStudentTTests(unittest.TestCase):
    def test_pdf(self):
        m = np.array([0,0])
        cov = np.eye(2)
        distribution = MultiStudentT(m, cov, 1)
        self.assertLess(abs(distribution.pdf([0., 0.]) - 0.15915), 1e-5)

        cov = np.array([[2,0],[0,2]])
        distribution = MultiStudentT(m, cov, 1)
        self.assertLess(abs(distribution.pdf([0., 0.]) - 0.079577), 1e-5)
        self.assertLess(abs(distribution.pdf([1., 1.]) - 0.028135), 1e-5)



    def test_sample(self):
        m = np.array([0,0])
        cov = np.eye(2)
        distribution = MultiStudentT(m, cov, 4)
        samples = distribution.sample(10000000)
        expected_mean = np.array([0., 0.])
        expected_var = np.array([1.9978, 2.0005])
        diff_mean = np.abs(samples.mean(axis=0) - expected_mean)
        diff_var = np.abs(samples.var(axis=0) - expected_var)
        self.assertLess(diff_mean.sum(), 2e-2)
        self.assertLess(diff_var.sum(), 2e-2)

class NormalTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.mean = np.array([-13.0])
        self.var = 1
        self.distribution = Normal(self.mean, self.var, seed=1)
        

        
    def test_sample(self):
        samples = self.distribution.sample(100)
        computed_means = samples.mean(axis=0)
        computed_vars = samples.var(axis=0)
        expected_means = np.array([-12.93941715])
        expected_vars = np.array([0.78350152])
        self.assertTrue((computed_means - expected_means < 1e-5).all())
        self.assertTrue((computed_vars - expected_vars < 1e-5).all())


        
    def test_set_parameters(self):
        new_mean = np.array([130.0])
        new_var = 4*1e-2
        self.distribution.set_parameters([new_mean, new_var])
        pdf_value = self.distribution.pdf(new_mean)
        self.assertLess(abs(pdf_value - 9.97355701), 1e-6)

        samples = self.distribution.sample(100)
        computed_means = samples.mean(axis=0)
        computed_vars = samples.var(axis=0)
        expected_means = np.array([130.00242331])
        expected_vars = np.array([0.0012536])
        self.assertTrue((computed_means - expected_means < 1e-5).all())
        self.assertTrue((computed_vars - expected_vars < 1e-5).all())
        
        
if __name__ == '__main__':
    unittest.main()
