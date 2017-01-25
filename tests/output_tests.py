import unittest
import numpy as np

from abcpy.output import Journal

class JournalTests(unittest.TestCase):
    def test_add_parameters(self):
        params1 = np.zeros((2,4))
        params2 = np.ones((2,4))

        # test whether production mode only stores the last set of parameters
        journal_prod = Journal(0)
        journal_prod.add_parameters(params1)
        journal_prod.add_parameters(params2)
        self.assertEqual(len(journal_prod.parameters), 1)
        np.testing.assert_equal(journal_prod.parameters[0], params2)

        # test whether reconstruction mode stores all parameter sets
        journal_recon = Journal(1)
        journal_recon.add_parameters(params1)
        journal_recon.add_parameters(params2)
        self.assertEqual(len(journal_recon.parameters), 2)
        np.testing.assert_equal(journal_recon.parameters[0], params1)
        np.testing.assert_equal(journal_recon.parameters[1], params2)



    def test_add_weights(self):
        weights1 = np.zeros((2,4))
        weights2 = np.ones((2,4))

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
        opt_values1 = np.zeros((2,4))
        opt_values2 = np.ones((2,4))

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
        params1 = np.zeros((2,4))
        weights1 = np.zeros((2,4))

        journal = Journal(0)
        journal.add_parameters(params1)
        journal.add_weights(weights1)
        journal.save('journal_tests_testfile.pkl')

        new_journal = Journal.fromFile('journal_tests_testfile.pkl')
        np.testing.assert_equal(journal.parameters, new_journal.parameters)
        np.testing.assert_equal(journal.weights, new_journal.weights)
        




if __name__ == '__main__':
    unittest.main()
