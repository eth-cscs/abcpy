import unittest
import numpy as np

from abcpy.backends import BackendDummy
from abcpy.models import Gaussian

from abcpy.distances import Euclidean

from abcpy.approx_lhd import SynLiklihood

from abcpy.distributions import MultiNormal
from abcpy.distributions import Uniform

from abcpy.statistics import Identity

from abcpy.inferences import RejectionABC, PMC, PMCABC, _RemoteContextPMCABC, SABC, ABCsubsim, SMCABC, APMCABC, RSMCABC

class RejectionABCTest(unittest.TestCase):
    def test_sample(self):
        # setup backend
        dummy = BackendDummy()

        # define a uniform prior distribution
        lb = np.array([-5, 0])
        ub = np.array([5,10])
        prior = Uniform(lb, ub, seed=1)

        # define a Gaussian model
        model = Gaussian(prior, mu=2.1, sigma=5.0, seed=1)

        # define sufficient statistics for the model
        stat_calc = Identity(degree=2, cross=0)
        
        # define a distance function
        dist_calc = Euclidean(stat_calc)

        # create fake observed data
        y_obs = model.simulate(1)

        # use the rejection sampling scheme
        sampler = RejectionABC(model, dist_calc, dummy, seed = 1)
        journal = sampler.sample(y_obs, 10, 1, 0.1)
        samples = journal.get_parameters()

        # test shape of samples
        samples_shape = np.shape(samples)
        self.assertEqual(samples_shape, (10,2))

        # Compute posterior mean
        self.assertEqual((np.average(np.asarray(samples[:,0])), np.average(np.asarray(samples[:,1]))), (1.6818856447333246, 8.4384177826766518))



class PMCTests(unittest.TestCase):
        
    def test_sample(self):
        # setup backend
        backend = BackendDummy()
        
        # define a uniform prior distribution
        lb = np.array([-5, 0])
        ub = np.array([5,10])
        prior = Uniform(lb, ub, seed=1)

        # define a Gaussian model
        model = Gaussian(prior, mu=2.1, sigma=5.0, seed=1)

        # define sufficient statistics for the model
        stat_calc = Identity(degree = 2, cross = 0)

        # create fake observed data
        y_obs = model.simulate(1)
      
        # Define the likelihood function
        likfun = SynLiklihood(stat_calc)

        # use the PMC scheme for T = 1
        mean = np.array([-13.0, .0, 7.0])
        cov = np.eye(3)
        kernel = MultiNormal(mean, cov, seed=1)

        T, n_sample, n_samples_per_param = 1, 10, 100
        sampler = PMC(model, likfun, kernel, backend, seed = 1)
        journal = sampler.sample(y_obs, T, n_sample, n_samples_per_param, covFactor =  np.array([.1,.1]), iniPoints = None)
        samples = (journal.get_parameters(), journal.get_weights())

        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.array(samples[0][:,0]), np.array(samples[0][:,1]), np.array(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(abs(mu_post_mean - (-1.48953333102)), 1e-10)
        self.assertLess(abs(sigma_post_mean - 6.50695612708), 1e-10)


        # use the PMC scheme for T = 2
        T, n_sample, n_samples_per_param = 2, 10, 100
        sampler = PMC(model, likfun, kernel, backend, seed = 1)
        journal = sampler.sample(y_obs, T, n_sample, n_samples_per_param, covFactor = np.array([.1,.1]), iniPoints = None)
        samples = (journal.get_parameters(), journal.get_weights())
        
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(abs(mu_post_mean - (-1.4033145848) ), 1e-10)
        self.assertLess(abs(sigma_post_mean - 7.05175546876), 1e-10)

        


class PMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        lb = np.array([-5, 0])
        ub = np.array([5,10])
        prior = Uniform(lb, ub, seed=1)

        # define a Gaussian model
        self.model = Gaussian(prior, mu=2.1, sigma=5.0, seed=1)

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.simulate(1)
        
        # define kernel
        mean = np.array([-13.0, .0, 7.0])
        cov = np.eye(3)
        self.kernel = MultiNormal(mean, cov, seed=1)


        
    def test_calculate_weight(self):
        n_samples = 2
        rc = _RemoteContextPMCABC(self.backend, self.model, self.dist_calc, self.kernel, self.observation, n_samples, 1)
        theta = np.array([1.0])


        weight = rc._calculate_weight(theta)
        self.assertEqual(weight, 0.5)
        
        accepted_parameters = np.array([[1.0], [1.0 + np.sqrt(2)]])
        accepted_weights = np.array([[.5], [.5]])
        accepted_cov_mat = np.array([[1.0]])
        rc._update_broadcasts(self.backend, accepted_parameters, accepted_weights, accepted_cov_mat)
        weight = rc._calculate_weight(theta)
        expected_weight = (2.0 * np.sqrt(2.0 * np.pi)) /(( 1 + np.exp(-1))*100)
        self.assertEqual(weight, expected_weight)
        

        
    def test_sample(self):
        # use the PMCABC scheme for T = 1
        T, n_sample, n_simulate, eps_arr, eps_percentile = 1, 10, 1, [.1], 10
        sampler = PMCABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, T, eps_arr, n_sample, n_simulate, eps_percentile)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))

        #self.assertEqual((mu_post_mean, sigma_post_mean), (,))
        
        # use the PMCABC scheme for T = 2
        T, n_sample, n_simulate, eps_arr, eps_percentile = 2, 10, 1, [.1,.05], 10
        sampler = PMCABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, T, eps_arr, n_sample, n_simulate, eps_percentile)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - 3.80593164247, 10e-2)
        self.assertLess(sigma_post_mean - 7.21421951262, 10e-2)     


class SABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        lb = np.array([-5, 0])
        ub = np.array([5,10])
        prior = Uniform(lb, ub, seed=1)

        # define a Gaussian model
        self.model = Gaussian(prior, mu=2.1, sigma=5.0, seed=1)

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.simulate(1)
        
        # define kernel
        mean = np.array([-13.0, .0, 7.0])
        cov = np.eye(3)
        self.kernel = MultiNormal(mean, cov, seed=1)

       
    def test_sample(self):
        # use the SABC scheme for T = 1
        steps, epsilon, n_samples, n_samples_per_param = 1, .1, 10, 1
        sampler = SABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, steps, epsilon, n_samples, n_samples_per_param)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))


        # use the SABC scheme for T = 2
        steps, epsilon, n_samples, n_samples_per_param = 2, .1, 10, 1
        sampler = SABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, steps, epsilon, n_samples, n_samples_per_param)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)
        
        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - 1.51315443746, 10e-2)
        self.assertLess(sigma_post_mean - 6.85230360302, 10e-2)     

class ABCsubsimTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        lb = np.array([-5, 0])
        ub = np.array([5,10])
        prior = Uniform(lb, ub, seed=1)

        # define a Gaussian model
        self.model = Gaussian(prior, mu=2.1, sigma=5.0, seed=1)

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.simulate(1)
        
        # define kernel
        mean = np.array([-13.0, .0, 7.0])
        cov = np.eye(3)
        self.kernel = MultiNormal(mean, cov, seed=1)

       
    def test_sample(self):

        # use the ABCsubsim scheme for T = 1
        steps, n_samples, n_samples_per_param = 1, 10, 1
        sampler = ABCsubsim(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, steps, n_samples, n_samples_per_param)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))


        # use the ABCsubsim scheme for T = 2
        steps, n_samples, n_samples_per_param = 2, 10, 1
        sampler = ABCsubsim(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, steps, n_samples, n_samples_per_param)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)
        
        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - (-2.98633946126), 10e-2)
        self.assertLess(sigma_post_mean - 6.40146881524, 10e-2)     

class SMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        lb = np.array([-5, 0])
        ub = np.array([5,10])
        prior = Uniform(lb, ub, seed=1)

        # define a Gaussian model
        self.model = Gaussian(prior, mu=2.1, sigma=5.0, seed=1)

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.simulate(1)
        
        # define kernel
        mean = np.array([-13.0, .0, 7.0])
        cov = np.eye(3)
        self.kernel = MultiNormal(mean, cov, seed=1)
      
    def test_sample(self):
        # use the SMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = SMCABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, steps, n_sample, n_simulate)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))

        #self.assertEqual((mu_post_mean, sigma_post_mean), (,))
        
        # use the SMCABC scheme for T = 2
        T, n_sample, n_simulate = 2, 10, 1
        sampler = SMCABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, T, n_sample, n_simulate)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - (-1.12595029091), 10e-2)
        self.assertLess(sigma_post_mean - 4.62512055437, 10e-2)     

class APMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        lb = np.array([-5, 0])
        ub = np.array([5,10])
        prior = Uniform(lb, ub, seed=1)

        # define a Gaussian model
        self.model = Gaussian(prior, mu=2.1, sigma=5.0, seed=1)

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.simulate(1)
        
        # define kernel
        mean = np.array([-13.0, .0, 7.0])
        cov = np.eye(3)
        self.kernel = MultiNormal(mean, cov, seed=1)
      
    def test_sample(self):
        # use the APMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = APMCABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, steps, n_sample, n_simulate)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))

        #self.assertEqual((mu_post_mean, sigma_post_mean), (,))
        
        # use the APMCABC scheme for T = 2
        T, n_sample, n_simulate = 2, 10, 1
        sampler = APMCABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, T, n_sample, n_simulate)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - (2.19137364411), 10e-2)
        self.assertLess(sigma_post_mean - 5.66226403628, 10e-2)     

class RSMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        lb = np.array([-5, 0])
        ub = np.array([5,10])
        prior = Uniform(lb, ub, seed=1)

        # define a Gaussian model
        self.model = Gaussian(prior, mu=2.1, sigma=5.0, seed=1)

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.simulate(1)
        
        # define kernel
        mean = np.array([-13.0, .0, 7.0])
        cov = np.eye(3)
        self.kernel = MultiNormal(mean, cov, seed=1)
      
    def test_sample(self):
        # use the RSMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = RSMCABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, steps, n_sample, n_simulate)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))

        #self.assertEqual((mu_post_mean, sigma_post_mean), (,))
        
        # use the RSMCABC scheme for T = 2
        steps, n_sample, n_simulate = 2, 10, 1
        sampler = RSMCABC(self.model, self.dist_calc, self.kernel, self.backend, seed = 1)
        journal = sampler.sample(self.observation, steps, n_sample, n_simulate)
        samples = (journal.get_parameters(), journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - (-0.349310337252), 10e-2)
        self.assertLess(sigma_post_mean - 6.30221177368, 10e-2)     

if __name__ == '__main__':
    unittest.main()
