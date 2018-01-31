import unittest
import numpy as np
import warnings

from abcpy.backends import BackendDummy
from abcpy.continuousmodels import Normal

from abcpy.distances import Euclidean

from abcpy.approx_lhd import SynLiklihood

from abcpy.continuousmodels import Uniform

from abcpy.statistics import Identity

from abcpy.inferences import RejectionABC, PMC, PMCABC, SABC, ABCsubsim, SMCABC, APMCABC, RSMCABC

class RejectionABCTest(unittest.TestCase):
    def test_sample(self):
        # setup backend
        dummy = BackendDummy()

        # define a uniform prior distribution
        prior = Uniform([[-5,0],[5,10]])
        prior.sample_parameters(np.random.RandomState(1))
        # define a Gaussian model
        model = Normal([prior])

        # define sufficient statistics for the model
        stat_calc = Identity(degree=2, cross=0)
        
        # define a distance function
        dist_calc = Euclidean(stat_calc)

        # create fake observed data
        y_obs = model.sample_from_distribution(1, np.random.RandomState(1))[1].tolist()

        # use the rejection sampling scheme
        sampler = RejectionABC([model], dist_calc, dummy, seed = 1)
        journal = sampler.sample([y_obs], 10, 1, 0.1)
        samples = journal.parameters[len(journal.parameters)-1]

        # test shape of samples
        samples_shape = np.shape(samples)
        self.assertEqual(samples_shape, (10,2))

        # Compute posterior mean
        self.assertAlmostEqual(np.average(np.asarray(samples[:,0])),1.981031422)
        self.assertAlmostEqual(np.average(np.asarray(samples[:,1])),7.029206462)

        self.assertFalse(journal.number_of_simulations==0)




class PMCTests(unittest.TestCase):
        
    def test_sample(self):
        # setup backend
        backend = BackendDummy()
        
        # define a uniform prior distribution
        prior = Uniform([[-5,0],[5,10]])
        prior.sample_parameters(np.random.RandomState(1))

        # define a Gaussian model
        model = Normal([prior])

        # define sufficient statistics for the model
        stat_calc = Identity(degree = 2, cross = 0)

        # create fake observed data
        y_obs = model.sample_from_distribution(1, np.random.RandomState(1))[1].tolist()
      
        # Define the likelihood function
        likfun = SynLiklihood(stat_calc)


        T, n_sample, n_samples_per_param = 1, 10, 100
        sampler = PMC([model], likfun, backend, seed = 1)
        journal = sampler.sample([y_obs], T, n_sample, n_samples_per_param, covFactors =  np.array([.1,.1]), iniPoints = None)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())

        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.array(samples[0][:,0]), np.array(samples[0][:,1]), np.array(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(abs(mu_post_mean - (-3.57893895906)), 1e-10)
        self.assertLess(abs(sigma_post_mean - 3.91547596824), 1e-10)

        self.assertFalse(journal.number_of_simulations == 0)


        # use the PMC scheme for T = 2
        T, n_sample, n_samples_per_param = 2, 10, 100
        sampler = PMC([model], likfun, backend, seed = 1)
        journal = sampler.sample([y_obs], T, n_sample, n_samples_per_param, covFactors = np.array([.1,.1]), iniPoints = None)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
        
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(abs(mu_post_mean - (-3.02112130335) ), 1e-10)
        self.assertLess(abs(sigma_post_mean - 5.91683779367), 1e-10)

        self.assertFalse(journal.number_of_simulations == 0)


class PMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        prior = Uniform([[-5,0],[5,10]])
        prior.sample_parameters(np.random.RandomState(1))

        # define a Gaussian model
        self.model = Normal([prior])

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.sample_from_distribution(1, np.random.RandomState(1))[1].tolist()

        
    def test_calculate_weight(self):
        n_samples = 2
        rc = PMCABC([self.model], self.dist_calc, self.backend, seed=1)
        theta = np.array([1.0,1.0])


        weight = rc._calculate_weight(theta)
        self.assertEqual(weight, 0.5)
        
        accepted_parameters = [[1.0, 1.0 + np.sqrt(2)],[0,0]]
        accepted_weights = np.array([[.5], [.5]])
        accepted_cov_mat = [np.array([[1.0,0],[0,1]])]
        rc.accepted_parameters_manager.update_broadcast(rc.backend, accepted_parameters, accepted_weights, accepted_cov_mat)
        kernel_parameters = []
        for kernel in rc.kernel.kernels:
            kernel_parameters.append(
                rc.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

        rc.accepted_parameters_manager.update_kernel_values(rc.backend, kernel_parameters=kernel_parameters)
        weight = rc._calculate_weight(theta)
        expected_weight = 0.170794684453
        self.assertAlmostEqual(weight, expected_weight)
        

        
    def test_sample(self):
        # use the PMCABC scheme for T = 1
        T, n_sample, n_simulate, eps_arr, eps_percentile = 1, 10, 1, [.1], 10
        sampler = PMCABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], T, eps_arr, n_sample, n_simulate, eps_percentile)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
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
        sampler = PMCABC([self.model], self.dist_calc, self.backend, seed = 1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal = sampler.sample([self.observation], T, eps_arr, n_sample, n_simulate, eps_percentile)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - 2.764805681, 10e-2)
        self.assertLess(sigma_post_mean - 8.0092032071, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)


class SABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        prior = Uniform([[-5,0],[5,10]])
        prior.sample_parameters(np.random.RandomState(1))
        # define a Gaussian model
        self.model = Normal([prior])

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.sample_from_distribution(1, np.random.RandomState(1))[1].tolist()

       
    def test_sample(self):
        # use the SABC scheme for T = 1
        steps, epsilon, n_samples, n_samples_per_param = 1, .1, 10, 1
        sampler = SABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], steps, epsilon, n_samples, n_samples_per_param)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
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
        sampler = SABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], steps, epsilon, n_samples, n_samples_per_param)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)
        
        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - 1.23885607112, 10e-2)
        self.assertLess(sigma_post_mean - 7.60598318182, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)

class ABCsubsimTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        prior = Uniform([[-5,0],[5,10]])
        prior.sample_parameters(np.random.RandomState(1))

        # define a Gaussian model
        self.model = Normal([prior])

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.sample_from_distribution(1, np.random.RandomState(1))[1].tolist()

       
    def test_sample(self):

        # use the ABCsubsim scheme for T = 1
        steps, n_samples, n_samples_per_param = 1, 10, 1
        sampler = ABCsubsim([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], steps, n_samples, n_samples_per_param)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
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
        sampler = ABCsubsim([self.model], self.dist_calc, self.backend, seed = 1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal = sampler.sample([self.observation], steps, n_samples, n_samples_per_param)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)
        
        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - (-1.78001349646), 10e-2)
        self.assertLess(sigma_post_mean - 8.78354551702, 0.5)

        self.assertFalse(journal.number_of_simulations == 0)


class SMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        prior = Uniform([[-5,0],[5,10]])
        prior.sample_parameters(np.random.RandomState(1))
        # define a Gaussian model
        self.model = Normal([prior])

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.sample_from_distribution(1, np.random.RandomState(1))[1].tolist()

      
    def test_sample(self):
        # use the SMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = SMCABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], steps, n_sample, n_simulate)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
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
        sampler = SMCABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], T, n_sample, n_simulate)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - (-0.786118677019), 10e-2)
        self.assertLess(sigma_post_mean - 4.63324738665, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)

class APMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        prior = Uniform([[-5,0],[5,10]])
        prior.sample_parameters(np.random.RandomState(1))

        # define a Gaussian model
        self.model = Normal([prior])

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.sample_from_distribution(1, np.random.RandomState(1))[1].tolist()

      
    def test_sample(self):
        # use the APMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = APMCABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], steps, n_sample, n_simulate)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))

        self.assertFalse(journal.number_of_simulations == 0)

        #self.assertEqual((mu_post_mean, sigma_post_mean), (,))
        
        # use the APMCABC scheme for T = 2
        T, n_sample, n_simulate = 2, 10, 1
        sampler = APMCABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], T, n_sample, n_simulate)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (10,))
        self.assertEqual(sigma_sample_shape, (10,))
        self.assertEqual(weights_sample_shape, (10,))
        self.assertLess(mu_post_mean - (-0.382408628178), 10e-2)
        self.assertLess(sigma_post_mean - 3.47804653162, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)

class RSMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        prior = Uniform([[-5,0],[5,10]])
        prior.sample_parameters(np.random.RandomState(1))

        # define a Gaussian model
        self.model = Normal([prior])

        # define a distance function
        stat_calc = Identity(degree=2, cross=0)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = self.model.sample_from_distribution(1, np.random.RandomState(1))[1].tolist()

      
    def test_sample(self):
        # use the RSMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = RSMCABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], steps, n_sample, n_simulate)
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (9,))
        self.assertEqual(sigma_sample_shape, (9,))
        self.assertEqual(weights_sample_shape, (9,))

        self.assertFalse(journal.number_of_simulations == 0)

        #self.assertEqual((mu_post_mean, sigma_post_mean), (,))
        
        # use the RSMCABC scheme for T = 2
        steps, n_sample, n_simulate = 2, 10, 1
        sampler = RSMCABC([self.model], self.dist_calc, self.backend, seed = 1)
        journal = sampler.sample([self.observation], steps, n_sample, n_simulate)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        samples = (journal.parameters[len(journal.parameters)-1], journal.get_weights())
          
        # Compute posterior mean
        mu_post_sample, sigma_post_sample, post_weights = np.asarray(samples[0][:,0]), np.asarray(samples[0][:,1]), np.asarray(samples[1][:,0])
        mu_post_mean, sigma_post_mean = np.average(mu_post_sample, weights = post_weights), np.average(sigma_post_sample, weights = post_weights)

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = np.shape(mu_post_sample), np.shape(mu_post_sample), np.shape(post_weights)
        self.assertEqual(mu_sample_shape, (9,))
        self.assertEqual(sigma_sample_shape, (9,))
        self.assertEqual(weights_sample_shape, (9,))
        self.assertLess(mu_post_mean - (1.52651600439), 10e-2)
        self.assertLess(sigma_post_mean - 6.49994754262, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)

if __name__ == '__main__':
    unittest.main()
