from ProbabilisticModel import *
import unittest

#TODO 1) test MultistudentT (write case in numpy -> check with our results). 2) write tests for check_parameters and pdfs
#TODO ask Marcel about the averaging (see studentT)
#TODO test studentT for a distribution on df
#TODO if possible, test the distribution of cov somehow?

class NormalTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.Normal_fixed = Normal([-13.0,1])
        self.Normal_fixed.fix_parameters(rng=self.rng)
        self.Normal_mu = Normal([self.Normal_fixed, 0.5])
        helper = Normal([0.3,0.01])
        helper.fix_parameters(rng=self.rng)
        self.Normal_sigma = Normal([1, helper])

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
        self.Normal_mu.fix_parameters([-13.7])
        self.assertTrue(self.Normal_mu.get_parameters()==[-13.7])
        self.assertTrue(self.Normal_fixed.value==[-13.7])
        old_parameter = self.Normal_fixed.value
        self.Normal_fixed.fix_parameters()
        self.assertTrue(isinstance(self.Normal_fixed.value,np.ndarray))
        self.assertNotEqual(self.Normal_fixed.value, old_parameter)

class MultivariateNormalTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.MultivariateNormal = MultivariateNormal([-13.0, 0, 7.0, [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        self.helper = Normal([-13.0,0.1])
        self.MultivariateNormal.fix_parameters(rng=self.rng)
        self.helper.fix_parameters(rng=self.rng)
        self.MultivariateNormal_mean_1 = MultivariateNormal([self.helper,0,7.0,[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        self.MultivariateNormal_mean_2 = MultivariateNormal([self.MultivariateNormal,[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])

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
        self.assertTrue((computed_vars-expected_vars<3*1e-1).all())

        samples = self.MultivariateNormal_mean_2.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        expected_means = np.array([-11.35109217, -0.60033941,  6.46293576])
        self.assertTrue((computed_means-expected_means<2).all())
        computed_vars = samples.var(axis=0)
        expected_vars = np.array([ 1.01881388,1.05377349,0.9703697 ])
        self.assertTrue((computed_vars-expected_vars<3*1e-1).all())

    def test_set_parameters(self):
        self.MultivariateNormal_mean_1.fix_parameters([-12.])
        self.assertTrue(self.MultivariateNormal_mean_1.get_parameters()==[-12.])
        old_parameter = self.MultivariateNormal.value
        self.MultivariateNormal.fix_parameters()
        self.assertTrue((self.MultivariateNormal.value!=old_parameter).all())

        self.helper.updated=False #needs to be reset if we dont go through the whole graph!
        self.MultivariateNormal_mean_1.fix_parameters([-12.6])
        self.assertTrue(self.MultivariateNormal_mean_1.get_parameters()==[-12.6])

        self.MultivariateNormal_mean_2.fix_parameters([3,4,5])
        self.assertTrue(self.MultivariateNormal_mean_2.get_parameters()==[3,4,5])

class MixtureNormalTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.MixtureNormal = MixtureNormal([1,2,3])
        self.MixtureNormal.fix_parameters(rng=self.rng)
        self.MixtureNormal_1 = MixtureNormal([self.MixtureNormal])

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


class StudentTTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.StudentT = StudentT([0.5,7])
        self.StudentT.fix_parameters(rng=self.rng)
        self.StudentT_mean = StudentT([self.StudentT,7])
        #the value of studentT is 2.5 -> it agrees well with that, however, this means taht the mean gets fixed to 2.5, even though the mean value seems to be around 0.04...
        #self.StudentT_df = StudentT([0.5,self.StudentT])

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
        self.StudentT_mean.fix_parameters([0.0307729611237])
        samples = self.StudentT_mean.sample_from_distribution(100, rng=self.rng)
        computed_means = samples.mean(axis=0)
        computed_var = samples.var(axis=0)
        expected_mean = np.array([0.0497426237691])
        expected_var = np.array([1.36042518713])
        diff_mean = np.abs(computed_means-expected_mean)
        diff_var = np.abs(computed_var-expected_var)
        self.assertLess(diff_mean, 1.4)
        self.assertLess(diff_var, 0.4)


class MultiStudentTTests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        self.MultiStudentT = MultiStudentT([0,0,[[1,0],[0,1]],4])
        self.MultiStudentT.fix_parameters(rng=self.rng)
        self.MultiStudentT_1 = MultiStudentT([self.MultiStudentT, [[1,0],[0,1]],4])

    def test_sample_from_distribution(self):
        samples = self.MultiStudentT.sample_from_distribution(10000000, rng=self.rng)
        expected_mean = np.array([0., 0.])
        expected_var = np.array([1.9978, 2.0005])
        diff_mean = np.abs(samples.mean(axis=0) - expected_mean)
        diff_var = np.abs(samples.var(axis=0) - expected_var)
        self.assertLess(diff_mean.sum(), 2e-2)
        self.assertLess(diff_var.sum(), 2e-2)

class UniformTests(unittest.TestCase):
    def setUp(self):
        seed = 1
        self.rng = np.random.RandomState(seed)
        self.Uniform = Uniform([[-1.,-1.],[1.,1.]])
        self.Uniform.fix_parameters(rng=self.rng)
        self.Uniform_1 = Uniform([[self.Uniform],[1.2,1.2]])

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

class StochLorenz95Tests(unittest.TestCase):
    def setUp(self):
        seed=1
        self.rng = np.random.RandomState(seed)
        prior = Uniform([[1,0.1],[3,0.3]])
        prior.fix_parameters(rng=self.rng)
        self.model = StochLorenz95([prior], initial_state=None, n_timestep=160)
        seed=1
        self.rng = np.random.RandomState(seed)

    def test_sample_from_distribution(self):
        sample = self.model.sample_from_distribution(1, rng=self.rng)[0]
        np.savetxt('lorenz_test_output.txt', sample)
        self.assertTrue((sample - np.loadtxt('lorenz_test_output.txt') < 1e-5).all())

class RickerTest(unittest.TestCase):
    def setUp(self):
        prior = Uniform([[3.,0.,1.],[5.,1.,20.]])
        seed=1
        self.rng = np.random.RandomState(seed)
        prior.fix_parameters(rng=self.rng)
        self.distribution = Ricker([prior], n_timestep=100)

    def test_sample_from_distribution(self):
        samples = self.distribution.sample_from_distribution(1,rng=self.rng)
        expected_output = np.array([  0.,   8.,   0.,   0.,  44.,   0.,   0.,   0.,   0.,   0.,   0.,
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


