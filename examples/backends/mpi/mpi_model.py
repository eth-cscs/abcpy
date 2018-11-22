import numpy as np

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
    print(run_model())
