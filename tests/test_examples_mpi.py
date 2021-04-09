"""Tests here example with MPI."""

import unittest

from abcpy.backends import BackendMPI


def setUpModule():
    '''
    If an exception is raised in a setUpModule then none of
    the tests in the module will be run.

    This is useful because the teams run in a while loop on initialization
    only responding to the scheduler's commands and will never execute anything else.

    On termination of scheduler, the teams call quit() that raises a SystemExit().
    Because of the behaviour of setUpModule, it will not run any unit tests
    for the team and we now only need to write unit-tests from the scheduler's
    point of view.
    '''
    global backend_mpi
    backend_mpi = BackendMPI()


class ExampleGaussianMPITest(unittest.TestCase):
    def test_example(self):
        from examples.backends.mpi.pmcabc_gaussian import infer_parameters
        journal = infer_parameters(backend_mpi, steps=3, n_sample=50)
        test_result = journal.posterior_mean()['mu']
        expected_result = 174.94717012502286
        self.assertAlmostEqual(test_result, expected_result)


if __name__ == '__main__':
    unittest.main()
