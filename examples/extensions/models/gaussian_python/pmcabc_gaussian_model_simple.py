import numpy as np

from numbers import Number

from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector

class Gaussian(ProbabilisticModel, Continuous):
    """
    This class is an re-implementation of the `abcpy.continousmodels.Normal` for documentation purposes.
    """

    def __init__(self, parameters, name='Gaussian'):
        # We expect input of type parameters = [mu, sigma]
        if not isinstance(parameters, list):
            raise TypeError('Input of Normal model is of type list')

        if len(parameters) != 2:
            raise RuntimeError('Input list must be of length 2, containing [mu, sigma].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)


    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 2:
            raise ValueError('Number of parameters of Normal model must be 2.')

        # Check whether input is from correct domain
        mu = input_values[0]
        sigma = input_values[1]
        if sigma < 0:
            return False

        return True


    def _check_output(self, values):
        if not isinstance(values, Number):
            raise ValueError('Output of the normal distribution is always a number.')

        # At this point values is a number (int, float); full domain for Normal is allowed
        return True


    def get_output_dimension(self):
        return 1


    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        # Extract the input parameters
        mu = input_values[0]
        sigma = input_values[1]

        # Do the actual forward simulation
        vector_of_k_samples = np.array(rng.normal(mu, sigma, k))

        # Format the output to obey API
        result = [np.array([x]) for x in vector_of_k_samples]
        return result


    def pdf(self, input_values, x):
        mu = input_values[0]
        sigma = input_values[1]
        pdf = np.norm(mu,sigma).pdf(x)
        return pdf


def infer_parameters():
    # define observation for true parameters mean=170, std=15
    height_obs = [160.82499176, 167.24266737, 185.71695756, 153.7045709, 163.40568812, 140.70658699, 169.59102084, 172.81041696, 187.38782738, 179.66358934, 176.63417241, 189.16082803, 181.98288443, 170.18565017, 183.78493886, 166.58387299, 161.9521899, 155.69213073, 156.17867343, 144.51580379, 170.29847515, 197.96767899, 153.36646527, 162.22710198, 158.70012047, 178.53470703, 170.77697743, 164.31392633, 165.88595994, 177.38083686, 146.67058471763457, 179.41946565658628, 238.02751620619537, 206.22458790620766, 220.89530574344568, 221.04082532837026, 142.25301427453394, 261.37656571434275, 171.63761180867033, 210.28121820385866, 237.29130237612236, 175.75558340169619, 224.54340549862235, 197.42448680731226, 165.88273684581381, 166.55094082844519, 229.54308602661584, 222.99844054358519, 185.30223966014586, 152.69149367593846, 206.94372818527413, 256.35498655339154, 165.43140916577741, 250.19273595481803, 148.87781549665536, 223.05547559193792, 230.03418198709608, 146.13611923127021, 138.24716809523139, 179.26755740864527, 141.21704876815426, 170.89587081800852, 222.96391329259626, 188.27229523693822, 202.67075179617672, 211.75963110985992, 217.45423324370509]
    # define prior
    from abcpy.continuousmodels import Uniform
    mu = Uniform([[150], [200]], )
    sigma = Uniform([[5], [25]], )
    # define the model
    from abcpy.continuousmodels import Normal as Gaussian
    height = Gaussian([mu, sigma], name='height')

    # define statistics
    from abcpy.statistics import Identity
    statistics_calculator = Identity(degree = 2, cross = False)

    # define distance
    from abcpy.distances import LogReg
    distance_calculator = LogReg(statistics_calculator)

    # define kernel
    from abcpy.perturbationkernel import DefaultKernel
    kernel = DefaultKernel([mu, sigma])

    # define backend
    # Note, the dummy backend does not parallelize the code!
    from abcpy.backends import BackendDummy as Backend
    backend = Backend()

    # define sampling scheme
    from abcpy.inferences import PMCABC
    sampler = PMCABC([height], [distance_calculator], backend, kernel, seed=1)

    # sample from scheme
    T, n_sample, n_samples_per_param = 3, 250, 10
    eps_arr = np.array([.75])
    epsilon_percentile = 10
    journal = sampler.sample([height_obs],  T, eps_arr, n_sample, n_samples_per_param, epsilon_percentile)

    return journal


def analyse_journal(journal):
    # output parameters and weights
    print(journal.parameters)
    print(journal.weights)

    # do post analysis
    print(journal.posterior_mean())
    print(journal.posterior_cov())
    print(journal.posterior_histogram())

    # print configuration
    print(journal.configuration)

    # save and load journal
    journal.save("experiments.jnl")

    from abcpy.output import Journal
    new_journal = Journal.fromFile('experiments.jnl')


# this code is for testing purposes only and not relevant to run the example
import unittest
class ExampleExtendModelGaussianPython(unittest.TestCase):
    def test_example(self):
        journal = infer_parameters()
        test_result = journal.posterior_mean()[0]
        expected_result = 177.02
        self.assertLess(abs(test_result - expected_result), 2.)


if __name__  == "__main__":
    journal = infer_parameters()
    analyse_journal(journal)
    
