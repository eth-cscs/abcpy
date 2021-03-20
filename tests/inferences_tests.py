import unittest

import numpy as np

from abcpy.approx_lhd import SynLikelihood
from abcpy.backends import BackendDummy
from abcpy.continuousmodels import Normal
from abcpy.continuousmodels import Uniform
from abcpy.distances import Euclidean, MMD
from abcpy.inferences import RejectionABC, PMC, PMCABC, SABC, ABCsubsim, SMCABC, APMCABC, RSMCABC, \
    MCMCMetropoliHastings
from abcpy.statistics import Identity


class RejectionABCTest(unittest.TestCase):
    def test_sample(self):
        # setup backend
        dummy = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define sufficient statistics for the model
        stat_calc = Identity(degree=2, cross=False)

        # define a distance function
        dist_calc = Euclidean(stat_calc)

        # create fake observed data
        y_obs = [np.array(9.8)]

        # use the rejection sampling scheme
        sampler = RejectionABC([self.model], [dist_calc], dummy, seed=1)
        journal = sampler.sample([y_obs], 10, 1, 10)
        mu_sample = np.array(journal.get_parameters()['mu'])
        sigma_sample = np.array(journal.get_parameters()['sigma'])

        # test shape of samples
        mu_shape, sigma_shape = (len(mu_sample), mu_sample[0].shape[1]), \
                                (len(sigma_sample),
                                 sigma_sample[0].shape[1])
        self.assertEqual(mu_shape, (10, 1))
        self.assertEqual(sigma_shape, (10, 1))

        # Compute posterior mean
        # self.assertAlmostEqual(np.average(np.asarray(samples[:,0])),1.22301,10e-2)
        self.assertLess(np.average(mu_sample) - 1.22301, 1e-2)
        self.assertLess(np.average(sigma_sample) - 6.992218, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)


class PMCTests(unittest.TestCase):

    def test_sample(self):
        # setup backend
        backend = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define sufficient statistics for the model
        stat_calc = Identity(degree=2, cross=False)

        # create fake observed data
        # y_obs = self.model.forward_simulate(1, np.random.RandomState(1))[0].tolist()
        y_obs = [np.array(9.8)]

        # Define the likelihood function
        likfun = SynLikelihood(stat_calc)

        T, n_sample, n_samples_per_param = 1, 10, 100
        sampler = PMC([self.model], [likfun], backend, seed=1)
        journal = sampler.sample([y_obs], T, n_sample, n_samples_per_param, covFactors=np.array([.1, .1]),
                                 iniPoints=None)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertAlmostEqual(mu_post_mean, -3.3711206204663773, delta=1e-3)
        self.assertAlmostEqual(sigma_post_mean, 6.519325027532673, delta=1e-3)

        self.assertFalse(journal.number_of_simulations == 0)

        # use the PMC scheme for T = 2
        T, n_sample, n_samples_per_param = 2, 10, 100
        sampler = PMC([self.model], [likfun], backend, seed=1)
        journal = sampler.sample([y_obs], T, n_sample, n_samples_per_param, covFactors=np.array([.1, .1]),
                                 iniPoints=None)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertAlmostEqual(mu_post_mean, -3.2517600952705257, delta=1e-3)
        self.assertAlmostEqual(sigma_post_mean, 6.9214661382633365, delta=1e-3)
        self.assertFalse(journal.number_of_simulations == 0)


class MCMCMetropoliHastingsTests(unittest.TestCase):
    # test if MCMCMetropoliHastings works

    def setUp(self):
        # setup backend
        self.backend = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])
        self.model2 = Normal([mu, sigma])

        # define sufficient statistics for the model
        stat_calc = Identity(degree=2, cross=False)

        # create fake observed data
        # y_obs = self.model.forward_simulate(1, np.random.RandomState(1))[0].tolist()
        self.y_obs = [np.array(9.8)]
        self.y_obs2 = [np.array(3.4)]

        # Define the likelihood function
        self.likfun = SynLikelihood(stat_calc)
        self.likfun2 = SynLikelihood(stat_calc)

        self.bounds = {"mu": [-5, 5], "sigma": (0, 10)}

    def test_sample(self):
        n_sample, n_samples_per_param = 50, 20

        sampler = MCMCMetropoliHastings([self.model], [self.likfun], self.backend, seed=1)
        journal = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None,
                                 iniPoint=None, burnin=10, adapt_proposal_cov_interval=5, use_tqdm=False)
        # without speedup_dummy
        sampler = MCMCMetropoliHastings([self.model], [self.likfun], self.backend, seed=1)  # to reset seed
        journal_repr = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None, use_tqdm=False,
                                      iniPoint=None, speedup_dummy=False, burnin=10, adapt_proposal_cov_interval=5)
        # Compute posterior mean
        mu_post_mean_1, sigma_post_mean_1 = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']
        mu_post_mean_2, sigma_post_mean_2 = journal_repr.posterior_mean()['mu'], journal_repr.posterior_mean()['sigma']

        self.assertAlmostEqual(mu_post_mean_1, 3.735372456561986)
        self.assertAlmostEqual(mu_post_mean_2, -0.6946660151693353)
        self.assertAlmostEqual(sigma_post_mean_1, 5.751158868437219)
        self.assertAlmostEqual(sigma_post_mean_2, 8.103358539327967)

    def test_sample_with_transformer(self):
        n_sample, n_samples_per_param = 50, 20

        sampler = MCMCMetropoliHastings([self.model], [self.likfun], self.backend, seed=1)
        journal = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None,
                                 iniPoint=None, burnin=10, adapt_proposal_cov_interval=5, use_tqdm=False,
                                 bounds=self.bounds)
        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        self.assertAlmostEqual(mu_post_mean, 1.3797371606192235)
        self.assertAlmostEqual(sigma_post_mean, 8.097776586316062)

        # test raises correct errors:
        with self.assertRaises(TypeError):
            journal = sampler.sample([self.y_obs], n_sample, bounds=[0, 1])
        with self.assertRaises(KeyError):
            journal = sampler.sample([self.y_obs], n_sample, bounds={"hello": (0, 1)})
        with self.assertRaises(RuntimeError):
            journal = sampler.sample([self.y_obs], n_sample, bounds={"mu": (0)})
        with self.assertRaises(RuntimeError):
            journal = sampler.sample([self.y_obs], n_sample, bounds={"mu": (0, 1, 2)})

    def test_sample_two_models(self):
        n_sample, n_samples_per_param = 50, 20

        sampler = MCMCMetropoliHastings([self.model, self.model2], [self.likfun, self.likfun2], self.backend,
                                        seed=1)
        journal = sampler.sample([self.y_obs, self.y_obs2], n_sample, n_samples_per_param, cov_matrices=None,
                                 iniPoint=None, burnin=10, adapt_proposal_cov_interval=5, use_tqdm=False)
        # without speedup_dummy
        sampler = MCMCMetropoliHastings([self.model, self.model2], [self.likfun, self.likfun2], self.backend,
                                        seed=1)  # to reset seed
        journal_repr = sampler.sample([self.y_obs, self.y_obs2], n_sample, n_samples_per_param, cov_matrices=None,
                                      iniPoint=None, speedup_dummy=False, burnin=10, adapt_proposal_cov_interval=5,
                                      use_tqdm=False)
        # Compute posterior mean
        mu_post_mean_1, sigma_post_mean_1 = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']
        mu_post_mean_2, sigma_post_mean_2 = journal_repr.posterior_mean()['mu'], journal_repr.posterior_mean()['sigma']

        self.assertAlmostEqual(mu_post_mean_1, 0.1920594166217264)
        self.assertAlmostEqual(mu_post_mean_2, -1.0095854412936525)
        self.assertAlmostEqual(sigma_post_mean_1, 9.143353645946233)
        self.assertAlmostEqual(sigma_post_mean_2, 7.539268611159257)

    def test_restart_from_journal(self):
        for speedup_dummy in [True, False]:
            # do at once:
            n_sample, n_samples_per_param = 40, 20
            sampler = MCMCMetropoliHastings([self.model], [self.likfun], self.backend, seed=1)
            journal_at_once = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None,
                                             iniPoint=None, speedup_dummy=speedup_dummy, burnin=20,
                                             adapt_proposal_cov_interval=10, use_tqdm=False)
            # do separate:
            n_sample, n_samples_per_param = 20, 20
            sampler = MCMCMetropoliHastings([self.model], [self.likfun], self.backend, seed=1)
            journal = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None, iniPoint=None,
                                     speedup_dummy=speedup_dummy, burnin=20, adapt_proposal_cov_interval=10,
                                     use_tqdm=False)
            journal.save("tmp.jnl")
            journal_separate = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None,
                                              iniPoint=None, journal_file="tmp.jnl", burnin=0,
                                              speedup_dummy=speedup_dummy, use_tqdm=False)  # restart from this journal

            self.assertEqual(journal_separate.configuration['n_samples'], journal_at_once.configuration['n_samples'])
            self.assertEqual(journal_separate.number_of_simulations[-1], journal_at_once.number_of_simulations[-1])
            self.assertEqual(journal_separate.configuration["acceptance_rates"][-1],
                             journal_at_once.configuration["acceptance_rates"][-1])
            self.assertEqual(len(journal_separate.get_parameters()), len(journal_at_once.get_parameters()))
            self.assertEqual(len(journal_separate.get_parameters()['mu']), len(journal_at_once.get_parameters()['mu']))
            self.assertEqual(len(journal_separate.get_accepted_parameters()),
                             len(journal_at_once.get_accepted_parameters()))
            self.assertEqual(journal_separate.get_weights().shape, journal_at_once.get_weights().shape)
            self.assertEqual(journal_separate.posterior_mean()['mu'], journal_at_once.posterior_mean()['mu'])

    def test_restart_from_journal_with_transformer(self):
        for speedup_dummy in [True, False]:
            # do at once:
            n_sample, n_samples_per_param = 40, 20
            sampler = MCMCMetropoliHastings([self.model], [self.likfun], self.backend, seed=1)
            journal_at_once = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None,
                                             iniPoint=None, speedup_dummy=speedup_dummy, burnin=20,
                                             adapt_proposal_cov_interval=10, use_tqdm=False, bounds=self.bounds)
            # do separate:
            n_sample, n_samples_per_param = 20, 20
            sampler = MCMCMetropoliHastings([self.model], [self.likfun], self.backend, seed=1)
            journal = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None, iniPoint=None,
                                     speedup_dummy=speedup_dummy, burnin=20, adapt_proposal_cov_interval=10,
                                     use_tqdm=False, bounds=self.bounds)

            journal.save("tmp.jnl")
            journal_separate = sampler.sample([self.y_obs], n_sample, n_samples_per_param, cov_matrices=None,
                                              iniPoint=None, journal_file="tmp.jnl", burnin=0,
                                              speedup_dummy=speedup_dummy, use_tqdm=False,
                                              bounds=self.bounds)  # restart from this journal

            self.assertEqual(journal_separate.configuration['n_samples'], journal_at_once.configuration['n_samples'])
            self.assertEqual(journal_separate.number_of_simulations[-1], journal_at_once.number_of_simulations[-1])
            self.assertEqual(journal_separate.configuration["acceptance_rates"][-1],
                             journal_at_once.configuration["acceptance_rates"][-1])
            self.assertEqual(len(journal_separate.get_parameters()), len(journal_at_once.get_parameters()))
            self.assertEqual(len(journal_separate.get_parameters()['mu']), len(journal_at_once.get_parameters()['mu']))
            self.assertEqual(len(journal_separate.get_accepted_parameters()),
                             len(journal_at_once.get_accepted_parameters()))
            self.assertEqual(journal_separate.get_weights().shape, journal_at_once.get_weights().shape)
            self.assertAlmostEqual(journal_separate.posterior_mean()['mu'], journal_at_once.posterior_mean()['mu'])


class PMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define a distance function
        stat_calc = Identity(degree=2, cross=False)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        # self.observation = self.model.forward_simulate(1, np.random.RandomState(1))[0].tolist()
        self.observation = [np.array(9.8)]

    def test_calculate_weight(self):
        n_samples = 2
        rc = PMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        theta = np.array([1.0, 1.0])

        weight = rc._calculate_weight(theta)
        self.assertEqual(weight, 0.5)

        accepted_parameters = [[1.0, 1.0 + np.sqrt(2)], [0, 0]]
        accepted_weights = np.array([[.5], [.5]])
        accepted_cov_mat = [np.array([[1.0, 0], [0, 1]])]
        rc.accepted_parameters_manager.update_broadcast(rc.backend, accepted_parameters, accepted_weights,
                                                        accepted_cov_mat)
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
        T, n_sample, n_simulate, eps_arr, eps_percentile = 1, 10, 1, [10], 10
        sampler = PMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], T, eps_arr, n_sample, n_simulate, eps_percentile)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape

        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertLess(mu_post_mean - 0.03713, 10e-2)
        self.assertLess(sigma_post_mean - 7.727, 10e-2)

        # self.assertEqual((mu_post_mean, sigma_post_mean), (,))

        # use the PMCABC scheme for T = 2
        T, n_sample, n_simulate, eps_arr, eps_percentile = 2, 10, 1, [10, 5], 10
        sampler = PMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal = sampler.sample([self.observation], T, eps_arr, n_sample, n_simulate, eps_percentile)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape

        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertLess(mu_post_mean - 0.9356, 10e-2)
        self.assertLess(sigma_post_mean - 7.819, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)

        # use the PMCABC scheme for T = 2 providing only first value for eps_arr
        T, n_sample, n_simulate, eps_arr, eps_percentile = 2, 10, 1, [10], 10
        sampler = PMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal = sampler.sample([self.observation], T, eps_arr, n_sample, n_simulate, eps_percentile)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape

        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertLess(mu_post_mean - 0.9356, 10e-2)
        self.assertLess(sigma_post_mean - 7.819, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)

    def test_restart_from_journal(self):
        # test with value of eps_arr_2 > percentile of distances
        n_sample, n_simulate, eps_arr, eps_percentile = 10, 1, [10, 5], 10
        # 2 steps with intermediate journal:
        sampler = PMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal_intermediate = sampler.sample([self.observation], 1, [eps_arr[0]], n_sample, n_simulate, eps_percentile)
        journal_intermediate.save("tmp.jnl")
        journal_final_1 = sampler.sample([self.observation], 1, [eps_arr[1]], n_sample, n_simulate, eps_percentile,
                                         journal_file="tmp.jnl")
        # 2 steps directly
        sampler = PMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal_final_2 = sampler.sample([self.observation], 2, eps_arr, n_sample, n_simulate, eps_percentile)

        self.assertEqual(journal_final_1.configuration["epsilon_arr"], journal_final_2.configuration["epsilon_arr"])
        self.assertEqual(journal_final_1.posterior_mean()['mu'], journal_final_2.posterior_mean()['mu'])

        # test with value of eps_arr_2 < percentile of distances
        n_sample, n_simulate, eps_arr, eps_percentile = 10, 1, [10, 1], 10
        # 2 steps with intermediate journal:
        sampler = PMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal_intermediate = sampler.sample([self.observation], 1, [eps_arr[0]], n_sample, n_simulate, eps_percentile)
        journal_intermediate.save("tmp.jnl")
        journal_final_1 = sampler.sample([self.observation], 1, [eps_arr[1]], n_sample, n_simulate, eps_percentile,
                                         journal_file="tmp.jnl")
        # 2 steps directly
        sampler = PMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal_final_2 = sampler.sample([self.observation], 2, eps_arr, n_sample, n_simulate, eps_percentile)

        self.assertEqual(journal_final_1.configuration["epsilon_arr"], journal_final_2.configuration["epsilon_arr"])
        self.assertEqual(journal_final_1.posterior_mean()['mu'], journal_final_2.posterior_mean()['mu'])


class SABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define a distance function
        stat_calc = Identity(degree=2, cross=False)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        # self.observation = self.model.forward_simulate(1, np.random.RandomState(1))[0].tolist()
        self.observation = [np.array(9.8)]

    def test_sample(self):
        # use the SABC scheme for T = 1
        steps, epsilon, n_samples, n_samples_per_param = 1, 10, 10, 1
        sampler = SABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], steps, epsilon, n_samples, n_samples_per_param)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[0]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape

        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))

        # use the SABC scheme for T = 2
        steps, epsilon, n_samples, n_samples_per_param = 2, 10, 10, 1
        sampler = SABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], steps, epsilon, n_samples, n_samples_per_param)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape

        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertLess(mu_post_mean - 0.55859197, 10e-2)
        self.assertLess(sigma_post_mean - 7.03987723, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)


class ABCsubsimTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define a distance function
        stat_calc = Identity(degree=2, cross=False)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        # self.observation = self.model.forward_simulate(1, np.random.RandomState(1))[0].tolist()
        self.observation = [np.array(9.8)]

    def test_sample(self):
        # use the ABCsubsim scheme for T = 1
        steps, n_samples, n_samples_per_param = 1, 10, 1
        sampler = ABCsubsim([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], steps, n_samples, n_samples_per_param)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape

        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))

        # use the ABCsubsim scheme for T = 2
        steps, n_samples, n_samples_per_param = 2, 10, 1
        sampler = ABCsubsim([self.model], [self.dist_calc], self.backend, seed=1)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        journal = sampler.sample([self.observation], steps, n_samples, n_samples_per_param)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape

        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertLess(mu_post_mean - (-0.81410299), 10e-2)
        self.assertLess(sigma_post_mean - 9.25442675, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)


class SMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define a distance function
        stat_calc = Identity(degree=2, cross=False)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        self.observation = [np.array(9.8)]

        # set up for the Bernton et al. implementation:
        # define a distance function
        stat_calc_2 = Identity(degree=1, cross=False)
        self.dist_calc_2 = MMD(stat_calc_2)
        # create fake observed data
        seed = 12
        self.rng = np.random.RandomState(seed)
        self.observation_2 = self.rng.normal(loc=0, scale=1, size=10)
        self.observation_2 = [x for x in self.observation_2]

    def test_sample_delmoral(self):
        # use the SMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = SMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], steps, n_sample, n_simulate)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape

        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))

        # self.assertEqual((mu_post_mean, sigma_post_mean), (,))

        # use the SMCABC scheme for T = 2
        T, n_sample, n_simulate = 2, 10, 1
        sampler = SMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], T, n_sample, n_simulate)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertAlmostEqual(mu_post_mean, - 0.8888295384029634, delta=10e-3)
        self.assertAlmostEqual(sigma_post_mean, 4.299346466029422, delta=10e-3)

        self.assertEqual(journal.number_of_simulations[-1], 19)

        # try now with the r-hit kernel version 1:
        T, n_sample, n_simulate = 2, 10, 1
        sampler = SMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], T, n_sample, n_simulate, which_mcmc_kernel=1)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertAlmostEqual(mu_post_mean, -0.6507386970288184, delta=10e-3)
        self.assertAlmostEqual(sigma_post_mean, 6.253446572247367, delta=10e-3)

        self.assertEqual(journal.number_of_simulations[-1], 56)

        # try now with the r-hit kernel version 2:
        T, n_sample, n_simulate = 2, 10, 1
        sampler = SMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], T, n_sample, n_simulate, which_mcmc_kernel=2)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertAlmostEqual(mu_post_mean, -0.5486451602421536, delta=10e-3)
        self.assertAlmostEqual(sigma_post_mean, 3.633148439032683, delta=10e-3)

        self.assertFalse(journal.number_of_simulations == 0)

    def test_sample_bernton(self):
        # check using the standard MCMC kernel raises error:
        sampler = SMCABC([self.model], [self.dist_calc_2], self.backend, seed=1, version="Bernton")
        with self.assertRaises(RuntimeError):
            journal = sampler.sample([self.observation_2], 1, 10, 10, which_mcmc_kernel=0, alpha=0.5,
                                     epsilon_final=0)

        # try now with the r-hit kernel version 1:
        T, n_sample, n_simulate = 3, 10, 10
        sampler = SMCABC([self.model], [self.dist_calc_2], self.backend, seed=1, version="Bernton")
        journal = sampler.sample([self.observation_2], T, n_sample, n_simulate, which_mcmc_kernel=1, alpha=0.5,
                                 epsilon_final=0)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertAlmostEqual(mu_post_mean, -0.7294075767448996, delta=10e-3)
        self.assertAlmostEqual(sigma_post_mean, 3.406347345226374, delta=10e-3)

        self.assertEqual(journal.number_of_simulations[-1], 2286)

        # try now with the r-hit kernel version 2:
        T, n_sample, n_simulate = 3, 10, 10
        sampler = SMCABC([self.model], [self.dist_calc_2], self.backend, seed=1, version="Bernton")
        journal = sampler.sample([self.observation_2], T, n_sample, n_simulate, which_mcmc_kernel=2,
                                 epsilon_final=0.1)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertAlmostEqual(mu_post_mean, -2.1412732987491303, delta=10e-3)
        self.assertAlmostEqual(sigma_post_mean, 5.146988585331478, delta=10e-3)

        self.assertEqual(journal.number_of_simulations[-1], 127)

    def test_restart_from_journal_delmoral(self):
        n_sample, n_simulate = 10, 1
        # loop over standard MCMC kernel, r-hit kernel version 1 and r-hit kernel version 2
        for which_mcmc_kernel in [0, 1, 2]:
            # 2 steps with intermediate journal:
            sampler = SMCABC([self.model], [self.dist_calc], self.backend, seed=1)
            journal_intermediate = sampler.sample([self.observation], 2, n_sample, n_simulate,
                                                  which_mcmc_kernel=which_mcmc_kernel)
            journal_intermediate.save("tmp.jnl")
            journal_final_1 = sampler.sample([self.observation], 1, n_sample, n_simulate,
                                             which_mcmc_kernel=which_mcmc_kernel,
                                             journal_file="tmp.jnl")

            # 2 steps directly
            sampler = SMCABC([self.model], [self.dist_calc], self.backend, seed=1)
            journal_final_2 = sampler.sample([self.observation], 3, n_sample, n_simulate, full_output=1,
                                             which_mcmc_kernel=which_mcmc_kernel)
            self.assertEqual(journal_final_1.configuration["epsilon_arr"], journal_final_2.configuration["epsilon_arr"])
            self.assertEqual(journal_final_1.posterior_mean()['mu'], journal_final_2.posterior_mean()['mu'])

    def test_restart_from_journal_bernton(self):
        n_sample, n_simulate = 10, 10
        # loop over standard MCMC kernel, r-hit kernel version 1 and r-hit kernel version 2
        for which_mcmc_kernel in [1, 2]:
            # 2 steps with intermediate journal:
            sampler = SMCABC([self.model], [self.dist_calc_2], self.backend, seed=1, version="Bernton")
            journal_intermediate = sampler.sample([self.observation_2], 1, n_sample, n_simulate,
                                                  which_mcmc_kernel=which_mcmc_kernel)
            journal_intermediate.save("tmp.jnl")
            journal_final_1 = sampler.sample([self.observation_2], 1, n_sample, n_simulate,
                                             which_mcmc_kernel=which_mcmc_kernel,
                                             journal_file="tmp.jnl")

            # 2 steps directly
            sampler = SMCABC([self.model], [self.dist_calc_2], self.backend, seed=1, version="Bernton")
            journal_final_2 = sampler.sample([self.observation_2], 2, n_sample, n_simulate, full_output=1,
                                             which_mcmc_kernel=which_mcmc_kernel)
            self.assertEqual(journal_final_1.configuration["epsilon_arr"], journal_final_2.configuration["epsilon_arr"])
            self.assertEqual(journal_final_1.posterior_mean()['mu'], journal_final_2.posterior_mean()['mu'])

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            sampler = SMCABC([self.model], [self.dist_calc_2], self.backend, seed=1, version="DelMoral")
        with self.assertRaises(RuntimeError):
            sampler = SMCABC([self.model], [self.dist_calc], self.backend, seed=1, version="Ciao")

        sampler = SMCABC([self.model], [self.dist_calc], self.backend, seed=1, version="Bernton")
        with self.assertRaises(RuntimeError):
            journal = sampler.sample([self.observation], 1, 10, 10, which_mcmc_kernel=4, alpha=0.5,
                                     epsilon_final=0)


class APMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define a distance function
        stat_calc = Identity(degree=2, cross=False)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        # self.observation = self.model.forward_simulate(1, np.random.RandomState(1))[0].tolist()
        self.observation = [np.array(9.8)]

    def test_sample(self):
        # use the APMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = APMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], steps, n_sample, n_simulate, alpha=.9)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))

        self.assertFalse(journal.number_of_simulations == 0)

        # use the APMCABC scheme for T = 2
        T, n_sample, n_simulate = 2, 10, 1
        sampler = APMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], T, n_sample, n_simulate, alpha=.9)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertLess(mu_post_mean - (-3.397848324005792), 10e-2)
        self.assertLess(sigma_post_mean - 6.451434816944525, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)


class RSMCABCTests(unittest.TestCase):
    def setUp(self):
        # find spark and initialize it
        self.backend = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define a distance function
        stat_calc = Identity(degree=2, cross=False)
        self.dist_calc = Euclidean(stat_calc)

        # create fake observed data
        # self.observation = self.model.forward_simulate(1, np.random.RandomState(1))[0].tolist()
        self.observation = [np.array(9.8)]

    def test_sample(self):
        # use the RSMCABC scheme for T = 1
        steps, n_sample, n_simulate = 1, 10, 1
        sampler = RSMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], steps, n_sample, n_simulate)
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))

        self.assertFalse(journal.number_of_simulations == 0)

        # self.assertEqual((mu_post_mean, sigma_post_mean), (,))

        # use the RSMCABC scheme for T = 2
        steps, n_sample, n_simulate = 2, 10, 1
        sampler = RSMCABC([self.model], [self.dist_calc], self.backend, seed=1)
        journal = sampler.sample([self.observation], steps, n_sample, n_simulate)
        sampler.sample_from_prior(rng=np.random.RandomState(1))
        mu_post_sample, sigma_post_sample, post_weights = np.array(journal.get_parameters()['mu']), np.array(
            journal.get_parameters()['sigma']), np.array(journal.get_weights())

        # Compute posterior mean
        mu_post_mean, sigma_post_mean = journal.posterior_mean()['mu'], journal.posterior_mean()['sigma']

        # test shape of sample
        mu_sample_shape, sigma_sample_shape, weights_sample_shape = (len(mu_post_sample), mu_post_sample[0].shape[1]), \
                                                                    (len(sigma_post_sample),
                                                                     sigma_post_sample[0].shape[1]), post_weights.shape
        self.assertEqual(mu_sample_shape, (10, 1))
        self.assertEqual(sigma_sample_shape, (10, 1))
        self.assertEqual(weights_sample_shape, (10, 1))
        self.assertLess(mu_post_mean - 1.52651600439, 10e-2)
        self.assertLess(sigma_post_mean - 6.49994754262, 10e-2)

        self.assertFalse(journal.number_of_simulations == 0)


if __name__ == '__main__':
    unittest.main()
