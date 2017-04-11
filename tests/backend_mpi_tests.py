import unittest
from abcpy.backend_mpi import BackendMPI

from mpi4py import MPI



class MPIBackendTests(unittest.TestCase):

    def setUp(self):
        self.backend = BackendMPI()
        

    def map_test(self):
        data = [1,2,3,4,5]
        pds = self.backend.parallelize(data)
        pds_map = self.backend.map(lambda x:x**2,pds)
        res = self.backend.collect(pds_map)
        assert res==list(map(lambda x:x**2,data))

    def function_pickle_map_test(self):

        def square(x):
            return x**2

        class staticfunctest:
            @staticmethod 
            def cube(x):
                return x**3

if __name__ == '__main__':
    print("Inside Main")
    comm = MPI.COMM_WORLD
    if comm.Get_rank()==0:
        unittest.main()
