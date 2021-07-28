import unittest

import numpy as np

from abcpy.backends import BackendDummy
from abcpy.continuousmodels import Normal
from abcpy.continuousmodels import Uniform
from abcpy.inferences import DrawFromPrior
from abcpy.output import Journal, GenerateFromJournal


class JournalTests(unittest.TestCase):
    # def test_add_parameters(self):
    #     params1 = np.zeros((2,4))
    #     params2 = np.ones((2,4))
    #
    #     # test whether production mode only stores the last set of parameters
    #     journal_prod = Journal(0)
    #     journal_prod.add_parameters(params1)
    #     journal_prod.add_parameters(params2)
    #     self.assertEqual(len(journal_prod.parameters), 1)
    #     np.testing.assert_equal(journal_prod.parameters[0], params2)
    #
    #     # test whether reconstruction mode stores all parameter sets
    #     journal_recon = Journal(1)
    #     journal_recon.add_parameters(params1)
    #     journal_recon.add_parameters(params2)
    #     self.assertEqual(len(journal_recon.parameters), 2)
    #     np.testing.assert_equal(journal_recon.parameters[0], params1)
    #     np.testing.assert_equal(journal_recon.parameters[1], params2)

    def test_add_weights(self):
        weights1 = np.zeros((2, 4))
        weights2 = np.ones((2, 4))

        # test whether production mode only stores the last set of parameters
        journal_prod = Journal(0)
        journal_prod.add_weights(weights1)
        journal_prod.add_weights(weights2)
        self.assertEqual(len(journal_prod.weights), 1)
        np.testing.assert_equal(journal_prod.weights[0], weights2)

        # test whether reconstruction mode stores all parameter sets
        journal_recon = Journal(1)
        journal_recon.add_weights(weights1)
        journal_recon.add_weights(weights2)
        self.assertEqual(len(journal_recon.weights), 2)
        np.testing.assert_equal(journal_recon.weights[0], weights1)
        np.testing.assert_equal(journal_recon.weights[1], weights2)

    def test_add_simulations(self):
        simulations1 = np.zeros((2, 4))
        simulations2 = np.ones((2, 4))

        # test whether production mode only stores the last set of parameters
        journal_prod = Journal(0)
        journal_prod.add_accepted_simulations(simulations1)
        journal_prod.add_accepted_simulations(simulations2)
        self.assertEqual(len(journal_prod.get_accepted_simulations()), 2)
        np.testing.assert_equal(journal_prod.get_accepted_simulations(), simulations2)

        # test whether reconstruction mode stores all parameter sets
        journal_recon = Journal(1)
        journal_recon.add_accepted_simulations(simulations1)
        journal_recon.add_accepted_simulations(simulations2)
        self.assertEqual(len(journal_recon.get_accepted_simulations()), 2)
        np.testing.assert_equal(journal_recon.get_accepted_simulations(0), simulations1)
        np.testing.assert_equal(journal_recon.get_accepted_simulations(1), simulations2)

        # test whether not storing it returns the correct value
        journal_empty = Journal(0)
        self.assertIsNone(journal_empty.get_accepted_simulations())

    def test_add_cov_mats(self):
        cov_mats1 = np.zeros((2, 4))
        cov_mats2 = np.ones((2, 4))

        # test whether production mode only stores the last set of parameters
        journal_prod = Journal(0)
        journal_prod.add_accepted_cov_mats(cov_mats1)
        journal_prod.add_accepted_cov_mats(cov_mats2)
        self.assertEqual(len(journal_prod.get_accepted_cov_mats()), 2)
        np.testing.assert_equal(journal_prod.get_accepted_cov_mats(), cov_mats2)

        # test whether reconstruction mode stores all parameter sets
        journal_recon = Journal(1)
        journal_recon.add_accepted_cov_mats(cov_mats1)
        journal_recon.add_accepted_cov_mats(cov_mats2)
        self.assertEqual(len(journal_recon.get_accepted_cov_mats()), 2)
        np.testing.assert_equal(journal_recon.get_accepted_cov_mats(0), cov_mats1)
        np.testing.assert_equal(journal_recon.get_accepted_cov_mats(1), cov_mats2)

        # test whether not storing it returns the correct value
        journal_empty = Journal(0)
        self.assertIsNone(journal_empty.get_accepted_cov_mats())


    def test_load_and_save(self):
        params1 = np.zeros((2, 4))
        weights1 = np.zeros((2, 4))

        journal = Journal(0)
        # journal.add_parameters(params1)
        journal.add_weights(weights1)
        journal.save('journal_tests_testfile.pkl')

        new_journal = Journal.fromFile('journal_tests_testfile.pkl')
        # np.testing.assert_equal(journal.parameters, new_journal.parameters)
        np.testing.assert_equal(journal.weights, new_journal.weights)

    def test_ESS(self):
        weights_identical = np.ones((100, 1))
        weights = np.arange(100).reshape(-1, 1)
        journal = Journal(1)
        journal.add_weights(weights_identical)
        journal.add_weights(weights)
        journal.add_ESS_estimate(weights=weights_identical)
        journal.add_ESS_estimate(weights=weights)
        self.assertEqual(len(journal.ESS), 2)
        self.assertAlmostEqual(journal.get_ESS_estimates(), 74.62311557788945)
        self.assertAlmostEqual(journal.get_ESS_estimates(0), 100)

    def test_plot_ESS(self):
        weights_identical = np.ones((100, 1))
        weights_1 = np.arange(100).reshape(-1, 1)
        weights_2 = np.arange(100, 200).reshape(-1, 1)
        journal = Journal(1)
        journal.add_weights(weights_identical)
        journal.add_ESS_estimate(weights=weights_identical)
        journal.add_weights(weights_1)
        journal.add_ESS_estimate(weights=weights_1)
        journal.add_weights(weights_2)
        journal.add_ESS_estimate(weights=weights_2)
        journal.plot_ESS()
        journal_2 = Journal(0)
        self.assertRaises(RuntimeError, journal_2.plot_ESS)

    def test_plot_wass_dist(self):
        rng = np.random.RandomState(1)
        weights_identical = np.ones((100, 1))
        params_0 = rng.randn(100).reshape(-1, 1)
        weights_1 = np.arange(100)
        params_1 = rng.randn(100).reshape(-1, 1, 1)
        weights_2 = np.arange(100, 200)
        params_2 = rng.randn(100).reshape(-1, 1)
        weights_3 = np.arange(200, 300)
        params_3 = rng.randn(100).reshape(-1, 1)
        weights_4 = np.arange(300, 400)
        params_4 = rng.randn(100).reshape(-1, 1)
        journal = Journal(1)
        journal.add_weights(weights_identical)
        journal.add_accepted_parameters(params_0)
        journal.add_weights(weights_1)
        journal.add_accepted_parameters(params_1)
        journal.add_weights(weights_2)
        journal.add_accepted_parameters(params_2)
        journal.add_weights(weights_3)
        journal.add_accepted_parameters(params_3)
        journal.add_weights(weights_4)
        journal.add_accepted_parameters(params_4)
        fig, ax, wass_dist_lists = journal.Wass_convergence_plot()
        self.assertAlmostEqual(wass_dist_lists[0], 0.22829193592175878)
        # check the Errors
        journal_2 = Journal(0)
        self.assertRaises(RuntimeError, journal_2.Wass_convergence_plot)
        journal_3 = Journal(1)
        journal_3.add_weights(weights_identical)
        self.assertRaises(RuntimeError, journal_3.Wass_convergence_plot)
        journal_4 = Journal(1)
        journal_4.add_accepted_parameters(np.array([np.array([1]), np.array([1, 2])], dtype="object"))
        print(len(journal_4.accepted_parameters))
        self.assertRaises(RuntimeError, journal_4.Wass_convergence_plot)

    def test_plot_post_distr(self):
        rng = np.random.RandomState(1)
        weights_identical = np.ones((100, 1))
        params = rng.randn(100, 2, 1, 1)
        weights = np.arange(100).reshape(-1, 1)
        journal = Journal(1)
        journal.add_user_parameters([("par1", params[:, 0]), ("par2", params[:, 1])])
        journal.add_user_parameters([("par1", params[:, 0]), ("par2", params[:, 1])])
        journal.add_weights(weights=weights_identical)
        journal.add_weights(weights=weights)
        journal.plot_posterior_distr(single_marginals_only=True, iteration=0)
        journal.plot_posterior_distr(true_parameter_values=[0.5, 0.3], show_samples=True)
        journal.plot_posterior_distr(double_marginals_only=True, show_samples=True,
                                     true_parameter_values=[0.5, 0.3])
        journal.plot_posterior_distr(contour_levels=10, ranges_parameters={"par1": [-1, 1]},
                                     parameters_to_show=["par1"])

        with self.assertRaises(KeyError):
            journal.plot_posterior_distr(parameters_to_show=["par3"])
        with self.assertRaises(RuntimeError):
            journal.plot_posterior_distr(single_marginals_only=True, double_marginals_only=True)
        with self.assertRaises(RuntimeError):
            journal.plot_posterior_distr(parameters_to_show=["par1"], double_marginals_only=True)
        with self.assertRaises(RuntimeError):
            journal.plot_posterior_distr(parameters_to_show=["par1"], true_parameter_values=[0.5, 0.3])
        with self.assertRaises(TypeError):
            journal.plot_posterior_distr(ranges_parameters={"par1": [-1]})
        with self.assertRaises(TypeError):
            journal.plot_posterior_distr(ranges_parameters={"par1": np.zeros(1)})

    def test_traceplot(self):
        rng = np.random.RandomState(1)
        weights_identical = np.ones((100, 1))
        params = rng.randn(100).reshape(-1, 1)
        journal = Journal(1)
        journal.add_weights(weights_identical)
        journal.add_accepted_parameters(params)
        journal.add_user_parameters([("mu", params[:, 0])])
        self.assertRaises(RuntimeError, journal.traceplot)  # as it does not have "acceptance_rates" in configuration
        journal.configuration["acceptance_rates"] = [0.3]
        with self.assertRaises(KeyError):
            journal.traceplot(parameters_to_show=["sigma"])
        # now try correctly:
        fig, ax = journal.traceplot()

    def test_resample(self):
        # -- setup --
        # setup backend
        dummy = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        model = Normal([mu, sigma])

        sampler = DrawFromPrior([model], dummy, seed=1)
        original_journal = sampler.sample(100)

        # expected mean values from bootstrapped samples:
        mu_mean = -0.5631214403709973
        sigma_mean = 5.2341427118053705
        # expected mean values from subsampled samples:
        mu_mean_2 = -0.6414897172489
        sigma_mean_2 = 6.217381777130734

        # -- bootstrap --
        new_j = original_journal.resample(path_to_save_journal="tmp.jnl", seed=42)
        mu_sample = np.array(new_j.get_parameters()['mu'])
        sigma_sample = np.array(new_j.get_parameters()['sigma'])

        accepted_parameters = new_j.get_accepted_parameters()
        self.assertEqual(len(accepted_parameters), 100)
        self.assertEqual(len(accepted_parameters[0]), 2)

        # test shape of samples
        mu_shape, sigma_shape = (len(mu_sample), mu_sample[0].shape[1]), \
                                (len(sigma_sample), sigma_sample[0].shape[1])
        self.assertEqual(mu_shape, (100, 1))
        self.assertEqual(sigma_shape, (100, 1))

        # Compute posterior mean
        self.assertAlmostEqual(np.average(mu_sample), mu_mean)
        self.assertAlmostEqual(np.average(sigma_sample), sigma_mean)

        self.assertTrue(new_j.number_of_simulations[0] == 0)

        # check whether the dictionary or parameter list contain same data:
        self.assertEqual(new_j.get_parameters()["mu"][9], new_j.get_accepted_parameters()[9][0])
        self.assertEqual(new_j.get_parameters()["sigma"][7], new_j.get_accepted_parameters()[7][1])

        # -- subsample (replace=False, smaller number than the full sample) --
        new_j_2 = original_journal.resample(replace=False, n_samples=10, seed=42)
        mu_sample = np.array(new_j_2.get_parameters()['mu'])
        sigma_sample = np.array(new_j_2.get_parameters()['sigma'])

        accepted_parameters = new_j_2.get_accepted_parameters()
        self.assertEqual(len(accepted_parameters), 10)
        self.assertEqual(len(accepted_parameters[0]), 2)

        # test shape of samples
        mu_shape, sigma_shape = (len(mu_sample), mu_sample[0].shape[1]), \
                                (len(sigma_sample), sigma_sample[0].shape[1])
        self.assertEqual(mu_shape, (10, 1))
        self.assertEqual(sigma_shape, (10, 1))

        # Compute posterior mean
        self.assertAlmostEqual(np.average(mu_sample), mu_mean_2)
        self.assertAlmostEqual(np.average(sigma_sample), sigma_mean_2)

        self.assertTrue(new_j_2.number_of_simulations[0] == 0)

        # check whether the dictionary or parameter list contain same data:
        self.assertEqual(new_j_2.get_parameters()["mu"][9], new_j_2.get_accepted_parameters()[9][0])
        self.assertEqual(new_j_2.get_parameters()["sigma"][7], new_j_2.get_accepted_parameters()[7][1])

        # -- check that resampling the full samples with replace=False gives the exact same posterior mean and std --
        new_j_3 = original_journal.resample(replace=False, n_samples=100)
        mu_sample = np.array(new_j_3.get_parameters()['mu'])
        sigma_sample = np.array(new_j_3.get_parameters()['sigma'])

        # original journal
        mu_sample_original = np.array(original_journal.get_parameters()['mu'])
        sigma_sample_original = np.array(original_journal.get_parameters()['sigma'])

        # Compute posterior mean and std
        self.assertAlmostEqual(np.average(mu_sample), np.average(mu_sample_original))
        self.assertAlmostEqual(np.average(sigma_sample), np.average(sigma_sample_original))
        self.assertAlmostEqual(np.std(mu_sample), np.std(mu_sample_original))
        self.assertAlmostEqual(np.std(sigma_sample), np.std(sigma_sample_original))

        # check whether the dictionary or parameter list contain same data:
        self.assertEqual(new_j_3.get_parameters()["mu"][9], new_j_3.get_accepted_parameters()[9][0])
        self.assertEqual(new_j_3.get_parameters()["sigma"][7], new_j_3.get_accepted_parameters()[7][1])

        # -- test the error --
        with self.assertRaises(RuntimeError):
            original_journal.resample(replace=False, n_samples=200)


class GenerateFromJournalTests(unittest.TestCase):
    def setUp(self):
        # setup backend
        dummy = BackendDummy()

        # define a uniform prior distribution
        mu = Uniform([[-5.0], [5.0]], name='mu')
        sigma = Uniform([[0.0], [10.0]], name='sigma')
        # define a Gaussian model
        self.model = Normal([mu, sigma])

        # define a stupid uniform model now
        self.model2 = Uniform([[0], [10]])

        self.sampler = DrawFromPrior([self.model], dummy, seed=1)
        self.original_journal = self.sampler.sample(100)

        self.generate_from_journal = GenerateFromJournal([self.model], dummy, seed=2)
        self.generate_from_journal_2 = GenerateFromJournal([self.model2], dummy, seed=2)

        # expected mean values from bootstrapped samples:
        self.mu_mean = -0.2050921750330999
        self.sigma_mean = 5.178647189918053
        # expected mean values from subsampled samples:
        self.mu_mean_2 = -0.021275259024241676
        self.sigma_mean_2 = 5.672004487129107

    def test_generate(self):
        # sample single simulation for each par value
        parameters, simulations, normalized_weights = self.generate_from_journal.generate(journal=self.original_journal)
        self.assertEqual(parameters.shape, (100, 2))
        self.assertEqual(simulations.shape, (100, 1, 1))
        self.assertEqual(normalized_weights.shape, (100,))

        # sample multiple simulations for each par value
        parameters, simulations, normalized_weights = self.generate_from_journal.generate(self.original_journal,
                                                                                          n_samples_per_param=3,
                                                                                          iteration=-1)
        self.assertEqual(parameters.shape, (100, 2))
        self.assertEqual(simulations.shape, (100, 3, 1))
        self.assertEqual(normalized_weights.shape, (100,))

    def test_errors(self):
        # check whether using a different model leads to errors:
        with self.assertRaises(RuntimeError):
            self.generate_from_journal_2.generate(self.original_journal)


if __name__ == '__main__':
    unittest.main()
