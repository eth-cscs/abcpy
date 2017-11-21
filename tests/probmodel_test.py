from continuous import *
import unittest
from copy import deepcopy

#TODO if possible, test the distribution of cov somehow?

class NormalTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.Normal_fixed = Normal([-13.0,1])
        self.Normal_mu = Normal([self.Normal_fixed, 0.5])
        self.Normal_mu.sample_parameters(rng=self.rng)
        helper = Normal([0.3,0.01])
        self.Normal_sigma = Normal([1, helper])
        self.Normal_sigma.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        samples = self.Normal_fixed.sample_from_distribution(100, self.rng)
        computed_means = samples.mean(axis=0)
        computed_vars = samples.var(axis=0)
        expected_means = np.array([-12.93941715])
        expected_vars = np.array([0.78350152])

        self.assertTrue((computed_means - expected_means < 1e-5).all())
        self.assertTrue((computed_vars - expected_vars < 1e-5).all())

        samples_mu = self.Normal_mu.sample_from_distribution(100, self.rng)
        computed_means = samples_mu.mean(axis=0)
        expected_means = np.array(-13.3060988017)
        computed_vars = samples_mu.var(axis=0)
        expected_vars = np.array(0.249393290388)
        self.assertTrue((computed_means-expected_means<3).all())
        self.assertTrue((computed_vars-expected_vars<1e-1).all())

        samples_sigma = self.Normal_sigma.sample_from_distribution(100, self.rng)
        computed_means = samples_sigma.mean(axis=0)
        expected_means = np.array(1.00291966007)
        self.assertTrue((computed_means-expected_means<1e-2).all())
        computed_vars = samples_sigma.var(axis=0)
        expected_vars = np.array(0.0872874950351)
        self.assertTrue((computed_vars-expected_vars<1e-1).all())

    def test_fix_parameters(self):
        self.Normal_mu.set_parameters([-12.])
        self.assertTrue(self.Normal_mu.get_parameters()==[-12.])
        old_parameter = [-12, 0.5]
        self.Normal_mu.sample_parameters(rng=self.rng)
        self.assertTrue(isinstance(self.Normal_mu.parameter_values,list))
        self.assertNotEqual(self.Normal_mu.parameter_values, old_parameter)

        old_parameter = self.Normal_fixed.parameter_values
        self.Normal_fixed.sample_parameters(rng=self.rng)
        self.assertEqual(self.Normal_fixed.parameter_values, old_parameter)

    def test_check_parameters(self):
        with self.assertRaises(TypeError):
            N = Normal(1,0.5)
        with self.assertRaises(ValueError):
            N = Normal([1,-0.5])

    def test_check_parameters_fixed(self):
        N1 = Normal([1,0.5])
        N2 = Normal([2,0.5])
        N3 = Normal([N1,N2])
        self.assertFalse(N3.set_parameters([1]))
        self.assertFalse(N3.set_parameters([1,-0.3]))

    def test_pdf(self):
        N = Normal([1,0.5])
        expected_pdf = 0.797884560803
        self.assertLess(expected_pdf-N.pdf(1),1e-5)


class MultivariateNormalTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.MultivariateNormal = MultivariateNormal([-13.0, 0, 7.0, [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        self.helper = Normal([-13.0,0.1])
        self.MultivariateNormal_mean_1 = MultivariateNormal([self.helper,0,7.0,[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        self.MultivariateNormal_mean_1.sample_parameters(rng=self.rng)
        self.MultivariateNormal_mean_2 = MultivariateNormal([self.MultivariateNormal,[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        self.MultivariateNormal_mean_2.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        samples = self.MultivariateNormal.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        computed_vars = samples.var(axis=0)

        expected_means = np.array([-12.9820723, 0.08671813, 7.11855369])
        expected_vars = np.array([0.99725084, 0.8610233, 0.8089557])
        self.assertTrue((computed_means - expected_means < 2*1e-1).all())
        self.assertTrue((computed_vars - expected_vars < 3*1e-1).all())

        samples = self.MultivariateNormal_mean_1.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        expected_means = np.array([ -1.28545429e+01, -4.56076950e-03, 7.03029377e+00])
        self.assertTrue((computed_means-expected_means<1.1).all())
        computed_vars = samples.var(axis=0)
        expected_vars = np.array([ 1.03959334, 0.96300538, 1.02001915])
        self.assertTrue((computed_vars-expected_vars<4*1e-1).all())

        samples = self.MultivariateNormal_mean_2.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        expected_means = np.array([-11.35109217, -0.60033941,  6.46293576])
        self.assertTrue((computed_means-expected_means<2).all())
        computed_vars = samples.var(axis=0)
        expected_vars = np.array([ 1.01881388,1.05377349,0.9703697 ])
        self.assertTrue((computed_vars-expected_vars<3*1e-1).all())

    def test_fix_parameters(self):
        self.MultivariateNormal_mean_1.set_parameters([-12.])
        self.assertTrue(self.MultivariateNormal_mean_1.get_parameters()==[-12.])
        old_parameter = self.MultivariateNormal_mean_1.parameter_values
        self.MultivariateNormal_mean_1.sample_parameters(rng=self.rng)
        self.assertTrue((self.MultivariateNormal.parameter_values!=old_parameter))

        self.helper.updated=False #needs to be reset if we dont go through the whole graph!
        self.MultivariateNormal_mean_1.set_parameters([-12.6])
        self.assertTrue(self.MultivariateNormal_mean_1.get_parameters()==[-12.6])

        self.MultivariateNormal_mean_2.set_parameters([3,4,5])
        self.assertTrue(self.MultivariateNormal_mean_2.get_parameters()==[3,4,5])

    def test_check_parameters(self):
        with self.assertRaises(TypeError):
            M = MultivariateNormal(1,1,[[1,0],[0,1]])
        with self.assertRaises(IndexError):
            M = MultivariateNormal([1])
        with self.assertRaises(IndexError):
            M = MultivariateNormal([1,2,3,[[1,0],[0,1]]])
        with self.assertRaises(ValueError):
            M = MultivariateNormal([1,1,[[-1,0],[0,-1]]])

    def test_check_parameters_fixed(self):
        N = Normal([1,0.5])
        M = MultivariateNormal([N,1,[[1,0],[0,1]]])
        self.assertFalse(M.set_parameters([1,2]))

    def test_pdf(self):
        M = MultivariateNormal([1,1,[[1,0],[0,1]]])
        expected_pdf = 0.12394999431
        self.assertLess(expected_pdf-M.pdf(1.5),1e-5)

class MixtureNormalTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.MixtureNormal = MixtureNormal([1,2,3])
        self.MixtureNormal_1 = MixtureNormal([self.MixtureNormal])
        self.MixtureNormal_1.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        samples = self.MixtureNormal.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        computed_var = samples.var(axis=0)
        expected_means = np.array([1.0034, 2.0012, 3.0000])
        expected_var = np.array([0.4911, 0.4912, 0.4961])
        self.assertTrue((np.abs(computed_means - expected_means) < 0.15).all())
        self.assertTrue((np.abs(computed_var - expected_var) < 0.15).all())

        samples = self.MixtureNormal_1.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        computed_var = samples.var(axis=0)

        expected_means = np.array([1.0187, 2.0171, 3.0260])
        expected_var = np.array([0.4911, 0.4912, 0.4961])
        self.assertTrue((np.abs(computed_means - expected_means) < 1.5).all())
        self.assertTrue((np.abs(computed_var - expected_var) < 0.15).all())

    def test_check_parameters(self):
        with self.assertRaises(TypeError):
            M = MixtureNormal(1,1)


class StudentTTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.StudentT = StudentT([0.5,7])
        self.StudentT_mean = StudentT([self.StudentT,7])
        self.StudentT_mean.sample_parameters(rng=self.rng)
        self.helper = Uniform([[50],[60]])
        self.StudentT_df = StudentT([1.,self.helper])
        self.StudentT_df.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        samples = self.StudentT.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        computed_var = samples.var(axis=0)
        expected_mean = np.array([0.0307729611237])
        expected_var = np.array([1.32285700983])
        diff_mean = np.abs(computed_means - expected_mean)
        diff_var = np.abs(computed_var - expected_var)
        self.assertLess(diff_mean, 1.4)
        self.assertLess(diff_var, 0.4)

        #we fix this value since otherwise the value of self.StudentT is about 2.5
        #this could be circumvented if we averaged over samples at initialization
        self.StudentT_mean.set_parameters([0.0307729611237])
        samples = self.StudentT_mean.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        computed_var = samples.var(axis=0)
        expected_mean = np.array([0.0497426237691])
        expected_var = np.array([1.36042518713])
        diff_mean = np.abs(computed_means-expected_mean)
        diff_var = np.abs(computed_var-expected_var)
        self.assertLess(diff_mean, 1.4)
        self.assertLess(diff_var, 0.4)

        samples = self.StudentT_df.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        computed_var = samples.var(axis=0)
        expected_mean = np.array([1.00112140746])
        expected_var = np.array([2.04012170739])
        self.assertLess(np.abs(computed_means-expected_mean),3)
        self.assertLess(np.abs(computed_var-expected_var),2)

    def test_check_parameters(self):
        with self.assertRaises(TypeError):
            S = StudentT(0.5,7)
        with self.assertRaises(IndexError):
            S = StudentT([1,2,3])
        with self.assertRaises(ValueError):
            S = StudentT([1,-1])

    def test_check_parameters_fixed(self):
        S1 = StudentT([1,1])
        S2 = StudentT([2,2])
        S3 = StudentT([S1,S2])
        self.assertFalse(S3.set_parameters([1]))
        self.assertFalse(S3.set_parameters([1,-1]))

    def test_pdf(self):
        S = StudentT([1,3])
        expected_pdf = 0.346453574275
        self.assertLess(expected_pdf-S.pdf(0.7),1e-5)


class MultiStudentTTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.MultiStudentT = MultiStudentT([0,0,[[1,0],[0,1]],4])
        self.MultiStudentT_1 = MultiStudentT([self.MultiStudentT, [[1,0],[0,1]],8])
        self.MultiStudentT_1.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        samples = self.MultiStudentT.sample_from_distribution(10000000, rng=self.rng)
        expected_mean = np.array([0.00246609, -0.0074305])
        expected_var = np.array([1.9817695, 1.97633865])
        diff_mean = np.abs(samples.mean(axis=0) - expected_mean)
        diff_var = np.abs(samples.var(axis=0) - expected_var)
        self.assertLess(diff_mean.sum(), 2e-2)
        self.assertLess(diff_var.sum(), 5e-2)

        samples = self.MultiStudentT_1.sample_from_distribution(100000, rng=self.rng)
        expected_mean = np.array([-0.0588958, -0.02922973])
        expected_var = np.array([ 3.22611737, 3.19224511])
        diff_mean = np.abs(samples.mean(axis=0)-expected_mean)
        diff_var = np.abs(samples.var(axis=0)-expected_var)
        self.assertLess(diff_mean.sum(), 3.5)
        self.assertLess(diff_var.sum(), 4.)

    def test_check_parameters(self):
        with self.assertRaises(IndexError):
            M = MultiStudentT([1,1,1,[[1,0],[0,1]],1])
        with self.assertRaises(ValueError):
            M = MultiStudentT([1,1,[[1,0],[0,1]],-1])
        with self.assertRaises(ValueError):
            M = MultiStudentT([1,1,[[-1,0],[0,-1]],1])

    def test_check_parameters_fixed(self):
        N = Normal([1,0.5])
        M = MultiStudentT([N,1,[[1,0],[0,1]],1])
        self.assertFalse(M.set_parameters([1,2]))

    def test_pdf(self):
        M = MultiStudentT([0,0,[[1,0],[0,1]],1])
        self.assertLess(abs(M.pdf([0,0])-0.15915),1e-5)
        M2 = MultiStudentT([0,0,[[2,0],[0,2]],1])
        self.assertLess(abs(M2.pdf([0,0])-0.079577),1e-5)
        self.assertLess(abs(M2.pdf([1.,1.])-0.028135), 1e-5)


class UniformTests(unittest.TestCase):
    def setUp(self):
        seed = 1
        self.rng = np.random.RandomState(seed)
        self.Uniform = Uniform([[-1.,-1.],[1.,1.]])
        self.Uniform_1 = Uniform([[self.Uniform],[1.2,1.2]])
        self.Uniform_1.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        samples = self.Uniform.sample_from_distribution(1000, rng=self.rng)

        samples_avg = samples.mean(axis=0)
        samples_min = samples.min(axis=0)
        samples_max = samples.max(axis=0)
        for (avg, min, max) in zip(samples_avg, samples_min, samples_max):
            self.assertLess(abs(avg), 0.04)
            self.assertLess(abs(min + 1.0), 0.01)
            self.assertLess(abs(max - 1.0), 0.01)

        samples_shape = np.shape(samples)
        self.assertEqual(samples_shape, (1000, 2))

        samples_graph = self.Uniform_1.sample_from_distribution(1000, rng=self.rng)
        samples_graph_avg = samples_graph.mean(axis=0)
        samples_graph_min = samples_graph.min(axis=0)
        samples_graph_max = samples_graph.max(axis=0)

        for (avg, min, max) in zip(samples_graph_avg, samples_graph_min, samples_graph_max):
            self.assertLess(abs(avg), 1.2)
            self.assertLess(abs(min), 1.2)
            self.assertLess(abs(max), 1.2)
        samples_graph_shape = np.shape(samples_graph)
        self.assertEqual(samples_graph_shape, (1000, 2))

    def test_fix_parameters(self):
        self.Uniform_1.set_parameters([-1.5,-1.5], rng=self.rng)
        self.assertTrue(self.Uniform_1.get_parameters()==[-1.5,-1.5])
        old_parameter = deepcopy(self.Uniform_1.parameter_values)
        self.Uniform_1.sample_parameters(rng=self.rng)
        self.assertTrue(old_parameter!=self.Uniform_1.parameter_values)

    def test_check_parameters(self):
        with self.assertRaises(TypeError):
            U = Uniform(1,1)
        with self.assertRaises(TypeError):
            U = Uniform([1,1])
        with self.assertRaises(IndexError):
            U = Uniform([[1],[2,3]])
        with self.assertRaises(ValueError):
            U = Uniform([[2,1],[1,1]])

    def test_check_parameters_fixed(self):
        N = Normal([1,0.5])
        U = Uniform([[N,1],[2,2]])
        self.assertFalse(U.set_parameters([1,2]))
        self.assertFalse(U.set_parameters([1,3,3]))
        self.assertFalse(U.set_parameters([5]))

    def test_pdf(self):
        U = Uniform([[0.],[10.]])
        self.assertEqual(U.pdf(0),0.1)
        self.assertEqual(U.pdf(-1),0)
        self.assertEqual(U.pdf(11),0)


class StochLorenz95Tests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        prior = Uniform([[1,0.1],[3,0.3]])
        self.model = StochLorenz95([prior], initial_state=None, n_timestep=160)
        self.model.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        sample = self.model.sample_from_distribution(1, rng=self.rng)[0]
        self.assertTrue((sample - np.loadtxt('lorenz_test_output.txt') < 1e-5).all())

class RickerTest(unittest.TestCase):
    def setUp(self):
        prior = Uniform([[3.,0.,1.],[5.,1.,20.]])
        seed=1
        self.rng = np.random.RandomState(seed)
        self.distribution = Ricker([prior], n_timestep=100)
        self.distribution.sample_parameters(rng=self.rng)

    def test_sample_from_distribution(self):
        samples = self.distribution.sample_from_distribution(1,rng=self.rng)
        expected_output = np.array([0.,   8.,   0.,   0.,  44.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,  13.,   0.,   0.,  67.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   1.,  32.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   2.,   8.,   0.,   1.,  29.,   0.,   0.,   0.,   0.,   0.,
         0.,   3.,   5.,   4.,   5.,   0.,   9.,   0.,   1.,  11.,   0.,
         0.,  16.,   0.,   0.,  10.,   0.,   0.,  17.,   0.,   0.,  20.,
         0.,   0.,   0.,   2.,  37.,   0.,   0.,   0.,   0.,   0.,   3.,
        21.,   0.,   0.,   0.,   0.,   6.,   4.,  43.,   0.,   0.,   0.,
         0.])
        self.assertTrue((samples-expected_output<1e-5).all())

if __name__ == '__main__':
    unittest.main()


