import numpy as np
from mpi4py import MPI
from abcpy.probabilisticmodels import ProbabilisticModel, InputConnector

def setup_backend():
    global backend
    
    from abcpy.backends import BackendMPI as Backend
    backend = Backend(process_per_model=2)

def run_model():
    def square_mpi(model_comm, x):
        local_res = np.array([x**2], 'i')
        global_res = np.array([0], 'i')
        model_comm.Reduce([local_res,MPI.INT], [global_res,MPI.INT], op=MPI.SUM, root=0)
        return global_res[0]
        
    data = [1,2,3,4,5]
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
            raise RuntimeError('Input list must be of length 2, containing [mu, sigma].')

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

        if value.shape[0] != 2:
            raise ValueError('Output shape should be of dimension 2.')

        return True


    def get_output_dimension(self):
        return 2


    #def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
    #def forward_simulate(self, mpi_comm, input_values, k, rng=np.random.RandomState()): #, mpi_comm=None):
    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):

        rank = mpi_comm.Get_rank()

        # Extract the input parameters
        mu = input_values[rank]
        sigma = 1

        # Do the actual forward simulation
        vector_of_k_samples = np.array(rng.normal(mu, sigma, k))

        # Send everything back to rank 0
        # print("Hello from forward_simulate before gather, rank = ", rank)
        data = mpi_comm.gather(vector_of_k_samples)
        # print("Hello from forward_simulate after gather, rank = ", rank)

        # Format the output to obey API but only on rank 0
        if rank == 0:
            result = [None]*k
            for i in range(k):
                element0 = data[0][i]
                element1 = data[1][i]
                point = np.array([element0, element1])
                result[i] = point
            print("Process 0 will return : ", result)
            return result
        else:
            return


    def pdf(self, input_values, x):
        mu = input_values[0]
        sigma = input_values[1]
        pdf = np.norm(mu,sigma).pdf(x)
        return pdf


def infer_parameters():
    # define observation for true parameters mean=170, 65
    rng = np.random.RandomState()
    y_obs = rng.multivariate_normal([170, 65], np.eye(2), 100)

    # define prior
    from abcpy.continuousmodels import Uniform
    mu0 = Uniform([[150], [200]], )
    mu1 = Uniform([[25], [100]], )

    # define the model
    from abcpy.continuousmodels import Normal
    height_weight_model = NestedBivariateGaussian([mu0, mu1])

    # define statistics
    from abcpy.statistics import Identity
    statistics_calculator = Identity(degree = 2, cross = False)

    # define distance
    from abcpy.distances import LogReg
    distance_calculator = LogReg(statistics_calculator)

    from abcpy.approx_lhd import SynLiklihood
    approx_lhd = SynLiklihood(statistics_calculator)

    # define sampling scheme    
    from abcpy.inferences import PMC
    sampler = PMC([height_weight_model], [approx_lhd], backend, seed=1)

    # sample from scheme
    #T, n_sample, n_samples_per_param = 3, 250, 10
    T, n_sample, n_samples_per_param = 1, 1, 1

    journal = sampler.sample([y_obs],  T, n_sample, n_samples_per_param)

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
    #print(run_mod#print(run_model())
    model = NestedBivariateGaussian([100,200])
    print(infer_parameters())
