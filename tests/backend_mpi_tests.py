
import unittest
from mpi4py import MPI
from abcpy.backend_mpi import BackendMPI


class remoteContext:
    def __init__(self):
        self.bds = backend.broadcast(1)

    def func(self,x):
        print("Real Rank:",MPI.COMM_WORLD.Get_rank(),"self.bds's backend rank:",self.bds.backend.rank)
        return self.bds.value()+x

def setUpModule():
    '''
    If an exception is raised in a setUpModule then none of 
    the tests in the module will be run. 
    
    This is useful because the slaves run in a while loop on initialization
    only responding to the master's commands and will never execute anything else.

    On termination of master, the slaves call quit() that raises a SystemExit(). 
    Because of the behaviour of setUpModule, it will not run any unit tests
    for the slave and we now only need to write unit-tests from the master's 
    point of view. 
    '''
    global rank,backend
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    backend = BackendMPI()

class MPIBackendTests(unittest.TestCase):
    def test_parallelize(self):
        return
        data = [0]*backend.size
        pds = backend.parallelize(data)
        pds_map = backend.map(lambda x: x + MPI.COMM_WORLD.Get_rank(), pds)
        res = backend.collect(pds_map)

        print(">>>",res)
        for master_index in backend.master_node_ranks:
            self.assertTrue(master_index not in res,"Node in master_node_ranks performed map.")

    def test_map(self):
        return
        data = [1,2,3,4,5]
        pds = backend.parallelize(data)
        pds_map = backend.map(lambda x:x**2,pds)
        res = backend.collect(pds_map)
        assert res==list(map(lambda x:x**2,data))


    def test_broadcast(self):
        data = [1,2,3,4,5]
        pds = backend.parallelize(data)
        rc = remoteContext()
        pds_map = backend.map(rc.func, pds)
        res = backend.collect(pds_map)
        print(res)

    def test_function_pickle(self):
        return
        def square(x):
            return x**2

        class staticfunctest:
            @staticmethod 
            def square(x):
                return x**2

        class nonstaticfunctest:
            def square(self,x):
                return x**2

        data = [1,2,3,4,5]
        expected_result = [1,4,9,16,25]
        pds = backend.parallelize(data)


        pds_map1 = backend.map(square,pds)
        pds_res1 = backend.collect(pds_map1)
        self.assertTrue(pds_res1==expected_result,"Failed pickle test for general function")


        pds_map2 = backend.map(lambda x:x**2,pds)
        pds_res2 = backend.collect(pds_map2)
        self.assertTrue(pds_res2==expected_result,"Failed pickle test for lambda function")


        pds_map3 = backend.map(staticfunctest.square,pds)
        pds_res3 = backend.collect(pds_map3)
        self.assertTrue(pds_res3==expected_result,"Failed pickle test for static function")

        obj = nonstaticfunctest()
        pds_map4 = backend.map(obj.square ,pds)
        pds_res4 = backend.collect(pds_map4)
        self.assertTrue(pds_res4==expected_result,"Failed pickle test for non-static function")
