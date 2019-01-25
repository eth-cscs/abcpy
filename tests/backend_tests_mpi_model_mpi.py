import unittest
from mpi4py import MPI
from abcpy.backends import BackendMPI,BackendMPITestHelper
import numpy

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
    global rank,backend_mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    backend_mpi = BackendMPI(process_per_model=2)

class MPIBackendTests(unittest.TestCase):

    def test_parallelize(self):
        data = [0]*backend_mpi.size()
        pds = backend_mpi.parallelize(data)
        pds_map = backend_mpi.map(lambda x, npc=None: x + MPI.COMM_WORLD.Get_rank(), pds)
        res = backend_mpi.collect(pds_map)

        for scheduler_index in backend_mpi.scheduler_node_ranks():
            self.assertTrue(scheduler_index not in res,"Node in scheduler_node_ranks performed map.")

    def test_map(self):
        def square_mpi(x, npc=None):
            local_res = numpy.array([2*(x**2)], 'i')
            #global_res = numpy.array([0], 'i')
            #MPI.COMM_WORLD.Reduce([local_res,MPI.INT], [global_res,MPI.INT], op=MPI.SUM, root=0)
            return local_res[0]
        
        data = [1,2,3,4,5]
        pds = backend_mpi.parallelize(data)
        pds_map = backend_mpi.map(square_mpi, pds)
        res = backend_mpi.collect(pds_map)
        assert res==list(map(lambda x:2*(x**2),data))


    def test_broadcast(self):
        data = [1,2,3,4,5]
        pds = backend_mpi.parallelize(data)

        bds = backend_mpi.broadcast(100)

        #Pollute the BDS values of the scheduler to confirm teams
        # use their broadcasted value
        for k,v in  backend_mpi.bds_store.items():
             backend_mpi.bds_store[k] = 99999

        def test_map(x, npc=None):
            return x + bds.value()

        pds_m = backend_mpi.map(test_map, pds)
        self.assertTrue(backend_mpi.collect(pds_m)==[101,102,103,104,105])

    def test_pds_delete(self):

        def check_if_exists(x, npc):
            obj = BackendMPITestHelper()
            if npc.communicator().Get_rank() == 0:
                return obj.check_pds(x)
            return None

        data = [1,2,3,4,5]
        pds = backend_mpi.parallelize(data)

        #Check if the pds we just created exists in all the teams(+scheduler)

        id_check_pds = backend_mpi.parallelize([pds.pds_id]*5)
        pds_check_result = backend_mpi.map(check_if_exists, id_check_pds)
        self.assertTrue(False not in backend_mpi.collect(pds_check_result),"PDS was not created")

        #Delete the PDS on scheduler and try again
        del pds
        pds_check_result = backend_mpi.map(check_if_exists,id_check_pds)

        self.assertTrue(True not in backend_mpi.collect(pds_check_result),"PDS was not deleted")


    def test_bds_delete(self):
        
        def check_if_exists(x, npc=None):
            obj = BackendMPITestHelper()
            return obj.check_bds(x)

        data = [1,2,3,4,5]
        bds = backend_mpi.broadcast(data)

        #Check if the pds we just created exists in all the teams(+scheduler)
        id_check_bds = backend_mpi.parallelize([bds.bds_id]*5)
        bds_check_result = backend_mpi.map(check_if_exists, id_check_bds)
        self.assertTrue(False not in backend_mpi.collect(bds_check_result),"BDS was not created")

        #Delete the PDS on scheduler and try again
        del bds
        bds_check_result = backend_mpi.map(check_if_exists,id_check_bds)
        self.assertTrue(True not in backend_mpi.collect(bds_check_result),"BDS was not deleted")


    def test_function_pickle(self):

        def square_mpi(x, npc=None):
            local_res = numpy.array([2*(x**2)], 'i')
            #global_res = numpy.array([0], 'i')
            #model_comm.Reduce([local_res,MPI.INT], [global_res,MPI.INT], op=MPI.SUM, root=0)
            return local_res[0]

        class staticfunctest_mpi:
            @staticmethod 
            def square_mpi(x, npc=None):
                local_res = numpy.array([2*(x**2)], 'i')
                #global_res = numpy.array([0], 'i')
                #model_comm.Reduce([local_res,MPI.INT], [global_res,MPI.INT], op=MPI.SUM, root=0)
                return local_res[0]

        class nonstaticfunctest_mpi:
            def square_mpi(self, x, npc=None):
                local_res = numpy.array([2*(x**2)], 'i')
                #global_res = numpy.array([0], 'i')
                #model_comm.Reduce([local_res,MPI.INT], [global_res,MPI.INT], op=MPI.SUM, root=0)
                return local_res[0]

        data = [1,2,3,4,5]
        expected_result = [2,8,18,32,50]

        pds = backend_mpi.parallelize(data)
        pds_map1 = backend_mpi.map(square_mpi,pds)
        pds_res1 = backend_mpi.collect(pds_map1)
        
        self.assertTrue(pds_res1==expected_result,"Failed pickle test for general function")

        pds_map3 = backend_mpi.map(staticfunctest_mpi.square_mpi,pds)
        pds_res3 = backend_mpi.collect(pds_map3)
        self.assertTrue(pds_res3==expected_result,"Failed pickle test for static function")

        obj = nonstaticfunctest_mpi()
        pds_map4 = backend_mpi.map(obj.square_mpi ,pds)
        pds_res4 = backend_mpi.collect(pds_map4)
        self.assertTrue(pds_res4==expected_result,"Failed pickle test for non-static function")
