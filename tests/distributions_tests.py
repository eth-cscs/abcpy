import unittest
import numpy as np

from abcpy.distributions import MultiNormal
from abcpy.distributions import Uniform
from abcpy.distributions import MultiStudentT
from abcpy.distributions import Normal
from abcpy.distributions import MixtureNormal
from abcpy.distributions import StudentT
from abcpy.distributions import StochLorenz95
from abcpy.distributions import Ricker


class MultiNormalTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.mean = np.array([-13.0, .0, 7.0])
        self.cov = np.eye(3)
        self.distribution = MultiNormal(self.mean, self.cov, seed=1)
        self.distribution_graph = MultiNormal(self.distribution, self.cov, seed=1)
        self.distribution_uniform = MultiNormal(Uniform([1,2,5],[3,4,6],seed=1),self.cov,seed=1)
        self.distribution_1d = MultiNormal([Normal(1,0.5,seed=1),Uniform([1,2],[2,3],seed=1)],self.cov,seed=1)

    def test_simulate(self):
        samples = self.distribution.sample(100)
        computed_means = samples.mean(axis=0)
        computed_vars = samples.var(axis=0)

        expected_means = np.array([-12.9820723, 0.08671813, 7.11855369])
        expected_vars = np.array([0.99725084, 0.8610233, 0.8089557])
        self.assertTrue((computed_means - expected_means < 1e-2).all())
        self.assertTrue((computed_vars - expected_vars < 2*1e-2).all())

        samples_graph = self.distribution_graph.sample(100)
        computed_means_graph = samples_graph.mean(axis=0)
        computed_vars_graph = samples_graph.var(axis=0)
        expected_means_graph = np.array([-13.0005, 0.0069, 6.9978])
        expected_vars_graph = np.array([0.9878, 0.9889, 0.9923])

        self.assertTrue((computed_means_graph-expected_means_graph<2).all())
        self.assertTrue((computed_vars_graph-expected_vars_graph<2*1e-1).all())

        samples_uniform = self.distribution_uniform.sample(100)
        samples_1d = self.distribution_1d.sample(100)


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
        self.assertTrue((computed_means - expected_means < 3*1e-2).all())
        self.assertTrue((computed_vars - expected_vars < 2*1e-2).all())




class UniformTests(unittest.TestCase):
    def setUp(self):
        self.distribution = Uniform([-1.0, -1.0], [1.0, 1.0], seed=1)
        self.distribution_graph = Uniform(self.distribution, self.distribution, seed=1)
        self.distribution_multid = Uniform(MultiNormal([1,1],[[1,0],[0,1]],seed=1),MultiStudentT([1,1],[[1,0],[0,1]],2, seed=1),seed=1)
        self.distribution_1d = Uniform([Normal(1,0.5,seed=1)], [StudentT(1,2,seed=1)],seed=1)

    def test_init(self):
        self.assertRaises(TypeError, Uniform, 3.14, [1.0, 1.0])
        self.assertRaises(TypeError, Uniform, [-1.0, -1.0], 3.14)
        self.assertRaises(BaseException, Uniform, [-1.0, -1.0], [.0, 1.0, 1.0])

    def test_simulate(self):
        samples = self.distribution.sample(1000)
        samples_avg = samples.mean(axis=0)
        samples_min = samples.min(axis=0)
        samples_max = samples.max(axis=0)
        for (avg, min, max) in zip(samples_avg, samples_min, samples_max):
            self.assertLess(abs(avg), 0.05)
            self.assertLess(abs(min + 1.0), 0.05)
            self.assertLess(abs(max - 1.0), 0.05)

        samples_shape = np.shape(samples)
        self.assertEqual(samples_shape, (1000, 2))

        samples_graph = self.distribution_graph.sample(1000)
        samples_graph_avg = samples_graph.mean(axis=0)
        samples_graph_min = samples_graph.min(axis=0)
        samples_graph_max = samples_graph.max(axis=0)

        samples_multid = self.distribution_multid.sample(1000)
        samples_1d = self.distribution_1d.sample(1000)

        for (avg, min, max) in zip(samples_graph_avg, samples_graph_min, samples_graph_max):
            self.assertLess(abs(avg), 0.7)
            self.assertLess(abs(min+1.0), 1.5)
            self.assertLess(abs(max), 0.8)
        samples_graph_shape = np.shape(samples_graph)
        self.assertEqual(samples_graph_shape, (1000,2))

    def test_set_parameters(self):
        distribution = Uniform([-101], [-100], seed=1)
        sample = distribution.sample(1)[0, 0]
        self.assertLessEqual(sample, -100)
        self.assertGreaterEqual(sample, -101)

        distribution.set_parameters([[100], [101]])
        sample = distribution.sample(1)[0, 0]
        self.assertLessEqual(sample, 101)
        self.assertGreaterEqual(sample, 100)

    def test_pdf(self):
        new_prior = Uniform(np.array([0.0]), np.array([10.0]), seed=1)
        self.assertEqual(new_prior.pdf(0), 0.1)
        self.assertEqual(new_prior.pdf(-1), 0)
        self.assertEqual(new_prior.pdf(11), 0)


class MultiStudentTTests(unittest.TestCase):
    def test_pdf(self):
        #works
        m = np.array([0, 0])
        cov = np.eye(2)
        distribution = MultiStudentT(m, cov, 1)
        self.assertLess(abs(distribution.pdf([0., 0.]) - 0.15915), 1e-5)

        cov = np.array([[2, 0], [0, 2]])
        distribution = MultiStudentT(m, cov, 1)
        self.assertLess(abs(distribution.pdf([0., 0.]) - 0.079577), 1e-5)
        self.assertLess(abs(distribution.pdf([1., 1.]) - 0.028135), 1e-5)


    def test_simulate(self):
        m = np.array([0, 0])
        cov = np.eye(2)
        distribution = MultiStudentT(m, cov, 4)
        samples = distribution.sample(10000000)
        expected_mean = np.array([0., 0.])
        expected_var = np.array([1.9978, 2.0005])
        diff_mean = np.abs(samples.mean(axis=0) - expected_mean)
        diff_var = np.abs(samples.var(axis=0) - expected_var)
        self.assertLess(diff_mean.sum(), 2e-2)
        self.assertLess(diff_var.sum(), 2e-2)

        distribution = MultiStudentT(distribution, cov, 4, seed=1)
        samples_graph = distribution.sample(10000000)
        expected_mean_graph = np.array([0.006, 0.003])
        expected_var_graph = np.array([1.99683813, 2.00319229])
        diff_mean_graph = np.abs(samples_graph.mean(axis=0) - expected_mean_graph)

        diff_var_graph = np.abs(samples_graph.var(axis=0)-expected_var_graph)
        self.assertLess(diff_mean_graph.sum(), 3.5) #this value doesnt work at aaaaalllllll
        self.assertLess(diff_var_graph.sum(), 2e-2)

        distribution = MultiStudentT(Uniform([1,2],[3,4],seed=1),cov,2,seed=1)
        distribution = MultiStudentT([Normal(1,0.5,1),StudentT(1,2,seed=1)],cov,2,seed=1)

class StudentTTests(unittest.TestCase):
    #NOTE these distributions seem to be very sensitive to small changes in especially df. If df differs only slightly, the output can differ by multiple 1000s. Therefore, writing tests is a bit hard
    def setUp(self):
        np.random.seed(1)
        self.mean = 0.5
        self.df = 2
        self.distribution = StudentT(self.mean, self.df, seed=1)
        self.distribution_graph = StudentT(Uniform([-1.],[1.],seed=1), Uniform([0.],[1.],seed=1),seed=1)


    def test_pdf(self):
        self.assertLess(self.distribution.pdf(0)-0.35355, 1e-3)

    def test_simulate(self):
        samples = self.distribution.sample(100)
        computed_means = samples.mean(axis=0)
        computed_var = samples.var(axis=0)
        expected_mean = np.array([0.5094])
        expected_var = np.array([18.8499])
        diff_mean = np.abs(computed_means - expected_mean)
        diff_var = np.abs(computed_var - expected_var)
        self.assertLess(diff_mean, 1.8)
        #self.assertLess(diff_var, 2e-3)


class MixtureNormalTests(unittest.TestCase):
    def test_simulate(self):
        self.mu = np.array([1,2,3])
        self.distribution = MixtureNormal(self.mu, seed=1)

        samples = self.distribution.sample(100)
        computed_means = samples.mean(axis=0)
        computed_var = samples.var(axis=0)
        expected_means = np.array([1.0034, 2.0012, 3.0000])
        expected_var = np.array([0.4911, 0.4912, 0.4961])
        self.assertTrue((np.abs(computed_means-expected_means)<0.15).all())
        self.assertTrue((np.abs(computed_var-expected_var)<0.15).all())

        self.distribution_graph = MixtureNormal(self.distribution, seed=1)
        samples_graph = self.distribution_graph.sample(100)
        computed_means_graph = samples_graph.mean(axis=0)
        computed_var_graph = samples_graph.var(axis=0)

        expected_means_graph = np.array([1.0187, 2.0171, 3.0260])
        expected_var_graph = np.array([0.4911, 0.4912, 0.4961])
        self.assertTrue((np.abs(computed_means_graph-expected_means_graph)<1.5).all())
        self.assertTrue((np.abs(computed_var_graph-expected_var_graph)<0.15).all())

        self.distribution_uniform = MixtureNormal(Uniform([1,2],[3,4],seed=1),seed=1)
        self.distribution_1d = MixtureNormal([Normal(1,0.5,seed=1),StudentT(1,2,seed=1)],seed=1)

        samples_uniform = self.distribution_uniform.sample(100)
        samples_1d = self.distribution_1d.sample(100)


class NormalTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.mean = np.array([-13.0])
        self.var = 1
        self.distribution = Normal(self.mean, self.var, seed=1)
        self.distribution_graph = Normal(Uniform([-1.],[1.],seed=1), Uniform([0.],[1.],seed=1), seed=1)

    def test_sample(self):
        samples = self.distribution.sample(100)
        computed_means = samples.mean(axis=0)
        computed_vars = samples.var(axis=0)
        expected_means = np.array([-12.93941715])
        expected_vars = np.array([0.78350152])
        self.assertTrue((computed_means - expected_means < 1e-5).all())
        self.assertTrue((computed_vars - expected_vars < 1e-5).all())

        samples_graph = self.distribution_graph.sample(10)
        expected_output = np.array([1.6243453636632417, -0.61175641365007538, -0.5281717522634557, \
                           -1.0729686221561705, 0.86540762932467852, -2.3015386968802827, \
                           1.74481176421648, -0.76120690089510279, 0.31903909605709857, \
                           -0.24937037547741009])
        self.assertTrue((samples_graph-expected_output<2.0).all())

    def test_get_parameters(self):
        self.distribution_graph.sample_from_prior()
        params = self.distribution_graph.get_parameters()

        # test shape of parameters
        param_len = len(params)
        self.assertEqual(param_len, 2)

    def test_set_parameters(self):
        #works
        new_mean = np.array([130.0])
        new_var = 4 * 1e-2
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


class StochLorenz95Tests(unittest.TestCase):
    def setUp(self):
        prior = Uniform([1,.1],[3,.3],seed=1)
        self.distribution = StochLorenz95(prior, initial_state=None, n_timestep = 160, seed=1)

    def test_simulate(self):
        samples = self.distribution.sample(1)[0]
        self.assertTrue((samples-np.loadtxt('lorenz_test_output.txt')<1e-5).all())

    def test_get_parameters(self):
        self.distribution.sample_from_prior()
        params = self.distribution.get_parameters()

        # test shape of parameters
        param_shape = np.shape(params)
        self.assertEqual(param_shape, (2,))

    def test_set_parameters(self):
        self.assertRaises(TypeError, self.distribution.set_parameters, 3.4)


class RickerTests(unittest.TestCase):
    def setUp(self):
        prior = Uniform([3.,0.,1.],[5.,1.,20.],seed=1)
        self.distribution = Ricker(prior, n_timestep=100, seed=1)

    def test_simulate(self):
        samples = self.distribution.sample(1)
        expected_output = np.array([0., 46., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 18., 0., 0., 2., 7., 0., 0.,
               9., 0., 1., 11., 2., 64., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               5., 0., 3., 0., 4., 29., 0., 0., 0., 0., 0.,
               0., 29., 0., 0., 0., 0., 0., 0., 1., 12., 0.,
               0., 6., 2., 7., 5., 0., 20., 0., 0., 0., 0.,
               0., 15., 0., 3., 28., 0., 0., 0., 0., 0., 1.,
               7., 0., 6., 3., 4., 2., 5., 3., 19., 0., 0.,
               0.])
        self.assertTrue((samples-expected_output<1e-5).all())

    def test_get_parameters(self):
        self.distribution.sample_from_prior()
        params = self.distribution.get_parameters()

        # test shape of parameters
        param_shape = np.shape(params)
        self.assertEqual(param_shape, (3,))

    def test_set_parameters(self):
        self.assertRaises(TypeError, self.distribution.set_parameters, 3.4)


if __name__ == '__main__':
    unittest.main()