import unittest

import numpy as np

from abcpy.output import Journal


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

    def test_add_opt_values(self):
        opt_values1 = np.zeros((2, 4))
        opt_values2 = np.ones((2, 4))

        # test whether production mode only stores the last set of parameters
        journal_prod = Journal(0)
        journal_prod.add_opt_values(opt_values1)
        journal_prod.add_opt_values(opt_values2)
        self.assertEqual(len(journal_prod.opt_values), 1)
        np.testing.assert_equal(journal_prod.opt_values[0], opt_values2)

        # test whether reconstruction mode stores all parameter sets
        journal_recon = Journal(1)
        journal_recon.add_opt_values(opt_values1)
        journal_recon.add_opt_values(opt_values2)
        self.assertEqual(len(journal_recon.opt_values), 2)
        np.testing.assert_equal(journal_recon.opt_values[0], opt_values1)
        np.testing.assert_equal(journal_recon.opt_values[1], opt_values2)

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
        weights_identical = np.ones(100)
        weights = np.arange(100)
        journal = Journal(1)
        journal.add_weights(weights_identical)
        journal.add_weights(weights)
        journal.add_ESS_estimate(weights=weights_identical)
        journal.add_ESS_estimate(weights=weights)
        self.assertEqual(len(journal.ESS), 2)
        self.assertAlmostEqual(journal.get_ESS_estimates(), 74.62311557788945)
        self.assertAlmostEqual(journal.get_ESS_estimates(0), 100)

    def test_plot_ESS(self):
        weights_identical = np.ones(100)
        weights_1 = np.arange(100)
        weights_2 = np.arange(100, 200)
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
        weights_identical = np.ones(100)
        params_0 = rng.randn(100)
        weights_1 = np.arange(100)
        params_1 = rng.randn(100)
        weights_2 = np.arange(100, 200)
        params_2 = rng.randn(100)
        weights_3 = np.arange(200, 300)
        params_3 = rng.randn(100)
        weights_4 = np.arange(300, 400)
        params_4 = rng.randn(100)
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
        self.assertAlmostEqual(wass_dist_lists[0], 0.05211720800690442)
        # check the Errors
        journal_2 = Journal(0)
        self.assertRaises(RuntimeError, journal_2.Wass_convergence_plot)
        journal_3 = Journal(1)
        journal_3.add_weights(weights_identical)
        self.assertRaises(RuntimeError, journal_2.Wass_convergence_plot)

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
        journal.plot_posterior_distr(double_marginals_only=True, show_samples=True,
                                     true_parameter_values=[0.5, 0.3])
        journal.plot_posterior_distr(contour_levels=10, ranges_parameters={"par1": [-1, 1]},
                                     parameters_to_show=["par1"])

        with self.assertRaises(KeyError):
            journal.plot_posterior_distr(parameters_to_show=["par3"])
        with self.assertRaises(RuntimeError):
            journal.plot_posterior_distr(single_marginals_only=True, double_marginals_only=True)
            journal.plot_posterior_distr(parameters_to_show=["par1"], double_marginals_only=True)
            journal.plot_posterior_distr(parameters_to_show=["par1"], true_parameter_values=[0.5, 0.3])
        with self.assertRaises(TypeError):
            journal.plot_posterior_distr(ranges_parameters={"par1": [-1]})
            journal.plot_posterior_distr(ranges_parameters={"par1": np.zeros(1)})


if __name__ == '__main__':
    unittest.main()
