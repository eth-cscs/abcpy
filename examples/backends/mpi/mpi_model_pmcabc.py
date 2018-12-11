import numpy as np
from mpi4py import MPI
from abcpy.probabilisticmodels import ProbabilisticModel, InputConnector


def setup_backend():
    global backend

    from abcpy.backends import BackendMPI as Backend
    backend = Backend(process_per_model=2)
    # backend = Backend()


def run_model():
    def square_mpi(model_comm, x):
        local_res = np.array([x ** 2], 'i')
        global_res = np.array([0], 'i')
        model_comm.Reduce([local_res, MPI.INT], [global_res, MPI.INT], op=MPI.SUM, root=0)
        return global_res[0]

    data = [1, 2, 3, 4, 5]
    pds = backend.parallelize(data)
    pds_map = backend.map(square_mpi, pds)
    res = backend.collect(pds_map)
    return res


class NestedBivariateGaussian(ProbabilisticModel):
    """
    This is a show case model of bi-variate Gaussian distribution where we assume
    the standard deviation to be unit.
    """

    def __init__(self, parameters, name='Gaussian'):
        # We expect input of type parameters = [mu, sigma]
        if not isinstance(parameters, list):
            raise TypeError('Input of Normal model is of type list')

        if len(parameters) != 2:
            raise RuntimeError('Input list must be of length 2, containing [mu1, mu1].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 2:
            raise ValueError('Number of parameters are 2 (two means).')
        return True

    def _check_output(self, values):
        if not isinstance(values, np.ndarray):
            raise ValueError('Output of the normal distribution is always a numpy array.')

        if values.shape[0] != 2:
            raise ValueError('Output shape should be of dimension 2.')

        return True

    def get_output_dimension(self):
        return 2

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):

        rank = mpi_comm.Get_rank()
        # Extract the input parameters
        mu = input_values[rank]
        sigma = 1
        # print(mu)
        # Do the actual forward simulation
        vector_of_k_samples = np.array(rng.normal(mu, sigma, k))

        # Send everything back to rank 0
        # print("Hello from forward_simulate before gather, rank = ", rank)
        data = mpi_comm.gather(vector_of_k_samples)
        # print("Hello from forward_simulate after gather, rank = ", rank)

        # Format the output to obey API but only on rank 0
        if rank == 0:
            result = [None] * k
            for i in range(k):
                element0 = data[0][i]
                element1 = data[1][i]
                point = np.array([element0, element1])
                result[i] = point
            # print("Process 0 will return : ", result)
            return [np.array([result[i]]).reshape(-1, ) for i in range(k)]
        else:
            return

def infer_parameters():
    # define observation for true parameters mean=170, 65
    rng = np.random.RandomState()
    y_obs = [np.array(rng.multivariate_normal([170, 65], np.eye(2), 1).reshape(2,))]

    # define prior
    from abcpy.continuousmodels import Uniform
    mu0 = Uniform([[150], [200]], )
    mu1 = Uniform([[25], [100]], )

    # define the model
    height_weight_model = NestedBivariateGaussian([mu0, mu1])

    # define statistics
    from abcpy.statistics import Identity
    statistics_calculator = Identity(degree = 2, cross = False)

    # define distance
    from abcpy.distances import Euclidean
    distance_calculator = Euclidean(statistics_calculator)

    # define sampling scheme
    from abcpy.inferences import PMCABC
    sampler = PMCABC([height_weight_model], [distance_calculator], backend, seed=1)
    print('sampling')
    # sample from scheme
    T, n_sample, n_samples_per_param = 2, 10, 1
    eps_arr = np.array([10000])
    epsilon_percentile = 90

    journal = sampler.sample([y_obs],  T, eps_arr, n_sample, n_samples_per_param, epsilon_percentile)

    return journal

import unittest
from mpi4py import MPI

def setUpModule():
    setup_backend()

class ExampleMPIModelTest(unittest.TestCase):
    def test_example(self):
        result = run_model()
        data = [1,2,3,4,5]
        expected_result = list(map(lambda x:2*(x**2),data))
        assert result==expected_result

if __name__ == "__main__":
    setup_backend()
    print('Posterior Mean: ' + str(infer_parameters().posterior_mean()))
    print('True Value was: ' + str([170, 65]))