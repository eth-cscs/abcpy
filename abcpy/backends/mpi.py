# noinspection PyInterpreter
import cloudpickle
import numpy as np
import pickle
import time
import logging

from mpi4py import MPI

from abcpy.backends import BDS, PDS, Backend, NestedParallelizationController


import abcpy.backends.mpimanager
from mpi4py import MPI


class NestedParallelizationControllerMPI(NestedParallelizationController):
    def __init__(self, mpi_comm):
        self.logger = logging.getLogger(__name__)
        self.logger.info("#### Initialize NPC ####")
        self.loop_workers = True
        self.mpi_comm = mpi_comm
        self.nested_func = "NoFunction"
        self.func_args = ()
        self.func_kwargs = {}
        self.result = None
        if self.mpi_comm.Get_rank() != 0:
            self.nested_execution()


    def communicator(self):
        return self.mpi_comm


    def nested_execution(self):
        rank = self.mpi_comm.Get_rank()
        self.logger.debug("Starting nested loop on rank {}".format(rank))
        while self.loop_workers:
            self.mpi_comm.barrier()
            self.loop_workers = self.mpi_comm.bcast(self.loop_workers, root=0)
            if self.loop_workers == False:
                return
            func_p = None
            func_args_p = None
            func_kwargs_p = None
            if self.mpi_comm.Get_rank() == 0:
                self.logger.debug("Start pickling func on rank {}".format(rank))
                func_p = cloudpickle.dumps(self.nested_func, pickle.HIGHEST_PROTOCOL)
                func_args_p = cloudpickle.dumps(self.func_args, pickle.HIGHEST_PROTOCOL)
                func_kwargs_p = cloudpickle.dumps(self.func_kwargs, pickle.HIGHEST_PROTOCOL)

            self.logger.debug("Broadcasting function {} on rank {}".format(self.nested_func, rank))
            func_p = self.mpi_comm.bcast(func_p, root=0)
            func_args_p = self.mpi_comm.bcast(func_args_p, root=0)
            func_kwargs_p = self.mpi_comm.bcast(func_kwargs_p, root=0)
            self.nested_func = cloudpickle.loads(func_p)
            self.func_args = cloudpickle.loads(func_args_p)
            self.func_kwargs = cloudpickle.loads(func_kwargs_p)

            func = self.nested_func
            self.logger.debug("Starting map function {} on rank {}".format(func.__name__, self.mpi_comm.Get_rank()))
            self.func_kwargs['mpi_comm'] = self.mpi_comm
            self.mpi_comm.barrier()
            self.result = func(*(self.func_args), **(self.func_kwargs))
            self.logger.debug("Ending map function on rank {}".format(self.mpi_comm.Get_rank()))
            self.mpi_comm.barrier()
            if self.mpi_comm.Get_rank() == 0:
                return
        self.loop_workers = True
        self.logger.debug("Ending nested loop on rank {}".format(self.mpi_comm.Get_rank()))

    def run_nested(self, func, *args, **kwargs):
        self.logger.debug("Executing nested function {}.".format(func.__name__))
        self.nested_func = func
        self.func_args = args
        self.func_kwargs = kwargs
        self.nested_execution()
        self.logger.debug("Return from nested execution of master rank")
        self.nested_func = None
        self.func_args = ()
        self.func_kwargs = {}
        self.logger.info(self.result)
        return self.result

    def __del__(self):
        rank = self.mpi_comm.Get_rank()
        self.logger.debug("Stopping npc on rank {}".format(rank))
        self.loop_workers = False
        if rank == 0:
            self.mpi_comm.barrier()
            self.loop_workers = self.mpi_comm.bcast(self.loop_workers, root=0)
        self.logger.debug(">>>>>>>> NPC stopped on rank {}".format(rank))

class BackendMPIScheduler(Backend):
    """Defines the behavior of the scheduler process

    This class defines the behavior of the scheduler process (The one
    with rank==0) in MPI.

    """

    #Define some operation codes to make it more readable
    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]
    finalized = False

    def __init__(self, chunk_size=1):
        """
        Parameters
        ----------
        chunk_size: Integer
            size of one block of data to be sent to free
            execution teams
       """

        #Initialize the current_pds_id and bds_id
        self.__current_pds_id = 0
        self.__current_bds_id = 0

        #Initialize a BDS store for both scheduler & team.
        self.bds_store = {}
        self.pds_store = {}

        #Initialize a store for the pds data that 
        #.. hasn't been sent to the teams yet
        self.pds_pending_store = {}

        self.chunk_size = chunk_size


    def __command_teams(self, command, data):
        """Tell teams to enter relevant execution block
        This method handles the sending of the command to the teams
        telling them what operation to perform next.

        Parameters
        ----------
        command: operation code of OP_xxx
            One of the operation codes defined in the class definition as OP_xxx
            which tell the teams what operation they're performing.
        data:  tuple
            Any of the data required for the operation which needs to be bundled
            in the data packet sent.
        """

        if command == self.OP_PARALLELIZE:
            #In parallelize we receive data as (pds_id)
            data_packet = (command, data[0])

        elif command == self.OP_MAP:
            #In map we receive data as (pds_id,pds_id_new,func)
            #Use cloudpickle to dump the function into a string.
            function_packed = cloudpickle.dumps(data[2],pickle.HIGHEST_PROTOCOL)
            data_packet = (command, data[0], data[1], function_packed)

        elif command == self.OP_BROADCAST:
            data_packet = (command, data[0])

        elif command == self.OP_COLLECT:
            #In collect we receive data as (pds_id)
            data_packet = (command, data[0])

        elif command == self.OP_DELETEPDS or command == self.OP_DELETEBDS:
            #In deletepds we receive data as (pds_id) or bds_id
            data_packet = (command, data[0])

        elif command == self.OP_FINISH:
            data_packet = (command,)

        _ = self.mpimanager.get_scheduler_communicator().bcast(data_packet, root=0)



    def __generate_new_pds_id(self):
        """
        This method generates a new pds_id to associate a PDS with it's remote counterpart
        that teams use to store & index data based on the pds_id they receive

        Returns
        -------
        Returns a unique integer id.

        """

        self.__current_pds_id += 1
        return self.__current_pds_id


    def __generate_new_bds_id(self):
        """
        This method generates a new bds_id to associate a BDS with it's remote counterpart
        that teams use to store & index data based on the bds_id they receive

        Returns
        -------
        Returns a unique integer id.

        """

        self.__current_bds_id += 1
        return self.__current_bds_id


    def parallelize(self, python_list):
        """
        This method distributes the list on the available teams and returns a
        reference object.

        The list is split into number of teams many parts as a numpy array.
        Each part is sent to a separate team node using the MPI scatter.

        scheduler: python_list is the real data that is to be split up

        Parameters
        ----------
        list: Python list
            the list that should get distributed on the leader nodes of the teams

        Returns
        -------
        PDSMPI class (parallel data set)
            A reference object that represents the parallelized list
        """

        # Tell the teams to enter parallelize()
        pds_id = self.__generate_new_pds_id()
        self.__command_teams(self.OP_PARALLELIZE, (pds_id,))

        #Don't send any data. Just keep it as a queue we're going to pop.
        self.pds_store[pds_id] = list(python_list)

        pds = PDSMPI([], pds_id, self)

        return pds

    def orchestrate_map(self,pds_id):
        """Orchestrates the teams to perform a map function
        
        This works by keeping track of the teams who haven't finished executing,
        waiting for them to request the next chunk of data when they are free,
        responding to them with the data and then sending them a Sentinel
        signalling that they can exit.
        """
        is_map_done = [True if i in self.mpimanager.get_scheduler_node_ranks() else False for i in range(self.mpimanager.get_scheduler_size())]
        status = MPI.Status()

        #Copy it to the pending. This is so when scheduler accesses
        #the PDS data it's not empty.
        self.pds_pending_store[pds_id] = list(self.pds_store[pds_id])

        #While we have some ranks that haven't finished
        while sum(is_map_done)<self.mpimanager.get_scheduler_size():
            #Wait for a reqest from anyone
            data_request = self.mpimanager.get_scheduler_communicator().recv(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status,
            )
            request_from_rank = status.source

            if data_request!=pds_id:
                print("Ignoring stale PDS data request from",
                    request_from_rank,":",data_request,"/",pds_id)
                continue

            #Pointer so we don't have to keep doing dict lookups
            current_pds_items = self.pds_pending_store[pds_id]
            num_current_pds_items = len(current_pds_items)

            #Everyone's already exhausted all the data.
            # Send a sentinel and mark the node as finished
            if num_current_pds_items == 0:
                self.mpimanager.get_scheduler_communicator().send(None, dest=request_from_rank, tag=pds_id)
                is_map_done[request_from_rank] = True
            else:
                #Create the chunk of data to send. Pop off items and tag them with an id.
                # so we can sort them later
                chunk_to_send = []
                for i in range(self.chunk_size):
                    chunk_to_send+=[(num_current_pds_items-i,current_pds_items.pop())]
                    self.mpimanager.get_scheduler_communicator().send(chunk_to_send, dest=request_from_rank, tag=pds_id)

    def map(self, func, pds):
        """
        A distributed implementation of map that works on parallel data sets (PDS).

        On every element of pds the function func is called.

        Parameters
        ----------
        func: Python func
            A function that can be applied to every element of the pds
        pds: PDS class
            A parallel data set to which func should be applied

        Returns
        -------
        PDSMPI class
            a new parallel data set that contains the result of the map
        """

        # Tell the teams to enter the map() with the current pds_id & func.
        #Get pds_id of dataset we want to operate on
        pds_id = pds.pds_id

        #Generate a new pds_id to be used by the teams for the resultant PDS
        pds_id_new = self.__generate_new_pds_id()

        data = (pds_id, pds_id_new, func)
        self.__command_teams(self.OP_MAP, data)
        self.orchestrate_map(pds_id)

        pds_res = PDSMPI([], pds_id_new, self)

        return pds_res


    def collect(self, pds):
        """
        Gather the pds from all the teams,
            send it to the scheduler and return it as a standard Python list.

        Parameters
        ----------
        pds: PDS class
            a parallel data set

        Returns
        -------
        Python list
            all elements of pds as a list
        """

        # Tell the teams to enter collect with the pds's pds_id
        self.__command_teams(self.OP_COLLECT, (pds.pds_id,))

        #all_data = self.world_communicator.gather(pds.python_list, root=0)
        all_data = self.mpimanager.get_scheduler_communicator().gather(pds.python_list, root=0)

        #Initialize lists to accumulate results
        all_data_indices,all_data_items = [],[]

        for node_data in all_data:
            for index, item in node_data:
                if isinstance(item, Exception):
                    raise item
                all_data_indices.append(index)
                all_data_items.append(item)

        #Sort the accumulated data according to the indices we tagged
        #them with when distributing 
        rdd_sorted = [all_data_items[i] for i in np.argsort(all_data_indices)]
        return rdd_sorted


    def broadcast(self, value):
        """
        Sends a data to all leaders and workers
        First instruction is sent to leaders which then send it to their workers
        Then every process enters a broadcast to receive data from scheduler
        """
        # Tell the teams to enter broadcast()
        bds_id = self.__generate_new_bds_id()
        self.__command_teams(self.OP_BROADCAST, (bds_id,))

        _ = self.mpimanager.get_world_communicator().bcast(value, root=0)

        bds = BDSMPI(value, bds_id, self)
        return bds


    def delete_remote_pds(self, pds_id):
        """
        A public function for the PDS objects on the scheduler to call when they go out of
        scope or are deleted in order to ensure the same happens on the teams.

        Parameters
        ----------
        pds_id: int
            A pds_id identifying the remote PDS on the teams to delete.
        """

        if  not self.finalized:
            self.__command_teams(self.OP_DELETEPDS, (pds_id,))


    def delete_remote_bds(self, bds_id):
        """
        Public function for the BDS objects on the scheduler to call when they go
        out of score or are deleted in order to ensure they are deleted
        ont he teams as well.

        Parameters
        ----------
        bds_id: int
            A bds_id identifying the remote BDS on the teams to delete.
        """

        if  not self.finalized:
            #The scheduler deallocates it's BDS data. Explicit because
            #.. bds_store and BDSMPI object are disconnected.
            del backend.bds_store[bds_id]
            self.__command_teams(self.OP_DELETEBDS, (bds_id,))


    def __del__(self):
        """
        Overriding the delete function to explicitly call MPI.finalize().
        This is also required so we can tell the teams to get out of the
        while loop they are in and exit gracefully and they themselves call
        finalize when they die.
        """

        #Tell the teams they can exit gracefully.
        self.__command_teams(self.OP_FINISH, None)

        #Finalize the connection because the teams should have finished.
        MPI.Finalize()
        self.finalized = True


class BackendMPIWorker(Backend):
    """
    Workers are processes that are used to execute (maybe MPI) models
    There is one communicator per model to execute, compounded of one leader and workers
    Leaders receives instructions from the scheduler which then transmit them to workers
    Leaders are themselves workers 
    """

    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]

    def __init__(self):
        """ No parameter, just call worker_run """
        self.logger = logging.getLogger(__name__)
        self.__worker_run()

    def run_function(self, function_packed, data_item):
        """
        Receives a serialized function unpack it and run it
        Passes the model communicator if ther is more than one process per model
        """
        func = cloudpickle.loads(function_packed)
        res = None
        try:
            if(self.mpimanager.get_model_size() > 1):
                npc = NestedParallelizationControllerMPI(self.mpimanager.get_model_communicator())
                if self.mpimanager.get_model_communicator().Get_rank() == 0:
                    self.logger.debug("Executing map function on master rank 0.")
                    res = func(data_item, npc=npc)
                del(npc)
            else:
                res = func(data_item)
        except Exception as e:
            msg = "Exception occured while calling the map function {}: ".format(func.__name__)
            res = type(e)(msg + str(e))
        return res


    def __worker_run(self):
        """
        Workers enter an infinite loop and waits for instructions from their leader
        """
        while True:
            data = self.mpimanager.get_model_communicator().bcast(None, root=0)
            op = data[0]
            if op == self.OP_MAP:
                #Receive data from scheduler of the model
                function_packed = self.mpimanager.get_model_communicator().bcast(None, root=0)[0]
                data_item = self.mpimanager.get_model_communicator().bcast(None, root=0)[0]
                self.run_function(function_packed, data_item)
            elif op == self.OP_BROADCAST:
                self._bds_id = data[1]
                self.broadcast(None)
            elif op == self.OP_FINISH:  
                quit()
            else:
                raise Exception("worker model received unknown command code")

    def collect(self):
        pass

    def map(self):
        pass

    def parallelize():
        pass

    def broadcast(self, value):
        """
        Receives data from scheduler
        """
        value = self.mpimanager.get_world_communicator().bcast(None, root=0)
        self.bds_store[self._bds_id] = value


class BackendMPILeader(BackendMPIWorker):
    """Defines the behavior of the leader processes

    This class defines how the leaders should behave during operation.
    leaders are those processes(not nodes like Spark) that have rank==0 in the model communicator
    """

    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]


    def __init__(self):
        """ No parameter, just call leader_run """
        self.logger = logging.getLogger(__name__)
        self.__leader_run()



    def __leader_run(self):
        """
        This method is the infinite loop a leader enters directly from init.
        It makes the leader wait for a command to perform from the scheduler and
        then calls the appropriate function.

        This method also takes care of the synchronization of data between the
        scheduler and the leaders by matching PDSs based on the pds_ids sent by the scheduler
        with the command.

        Commands received from the scheduler are of the form of a tuple.
        The first component of the tuple is always the operation to be performed
        and the rest are conditional on the operation.

        (op,pds_id) where op == OP_PARALLELIZE for parallelize
        (op,pds_id, pds_id_result,func) where op == OP_MAP for map.
        (op,pds_id) where op == OP_COLLECT for a collect operation
        (op,pds_id) where op == OP_DELETEPDS for a delete of the remote PDS on leaders
        (op,) where op==OP_FINISH for the leader to break out of the loop and terminate
        """

        # Initialize PDS data store here because only teams need to do it.
        self.pds_store = {}

        while True:
            data = self.mpimanager.get_scheduler_communicator().bcast(None, root=0)

            op = data[0]
            if op == self.OP_PARALLELIZE:
                pds_id = data[1]
                self._rec_pds_id = pds_id
                pds_id, pds_id_new = self.__get_received_pds_id()
                self.pds_store[pds_id] = None


            elif op == self.OP_MAP:
                pds_id, pds_id_result, function_packed = data[1:]
                self._rec_pds_id, self._rec_pds_id_result = pds_id, pds_id_result

                #Enter the map so we can grab data and perform the func.
                #Func sent before and not during for performance reasons
                pds_res = self.map(function_packed)

                # Store the result in a newly gnerated PDS pds_id
                self.pds_store[pds_res.pds_id] = pds_res

            elif op == self.OP_BROADCAST:
                self._bds_id = data[1]
                #relay command and data into model communicator
                self.mpimanager.get_model_communicator().bcast(data, root=0)
                self.broadcast(None)

            elif op == self.OP_COLLECT:
                pds_id = data[1]

                # Access an existing PDS from data store
                pds = self.pds_store[pds_id]

                self.collect(pds)

            elif op == self.OP_DELETEPDS:
                pds_id = data[1]
                del self.pds_store[pds_id]

            elif op == self.OP_DELETEBDS:
                bds_id = data[1]
                del self.bds_store[bds_id]

            elif op == self.OP_FINISH:
                # tells other processes of the worker to finish
                self.mpimanager.get_model_communicator().bcast([self.OP_FINISH], root=0)
                quit()
            else:
                raise Exception("team received unknown command code")


    def __get_received_pds_id(self):
        """
        Function to retrieve the pds_id(s) we received from the scheduler to associate
        our team's created PDS with the scheduler's.
        """

        return self._rec_pds_id, self._rec_pds_id_result

    def __leader_run_function(self, function_packed, data_item):
        """
        This function sends data and serialized function to workers and executes it
        """
        self.mpimanager.get_model_communicator().bcast([self.OP_MAP], root=0)
        self.mpimanager.get_model_communicator().bcast([function_packed], root=0)
        self.mpimanager.get_model_communicator().bcast([data_item], root=0)
        return self.run_function(function_packed, data_item)


    def parallelize(self):
        pass

    def map(self, function_packed):
        """
        A distributed implementation of map that works on parallel data sets (PDS).
        On every element of pds the function func is called.
        We consider that process 0 of each MPI model should return the final result.

        Parameters
        ----------
        func: Python function_packed
            A serialized function that can be applied to every element of the pds

        Returns
        -------
        PDSMPI class
            a new parallel data set that contains the result of the map
        """

        map_start = time.time()

        #Get the PDS id we operate on and the new one to store the result in
        pds_id, pds_id_new = self.__get_received_pds_id()

        rdd = []
        while True:
            #Ask for a chunk of data since it's free
            data_chunks = self.mpimanager.get_scheduler_communicator().sendrecv(pds_id, 0, pds_id)
            
            #If it receives a sentinel, it's done and it can exit
            if data_chunks is None:
                break

            #Accumulate the indicess and *processed* chunks
            for chunk in data_chunks:
                data_index,data_item = chunk
                res = self.__leader_run_function(function_packed, data_item)
                rdd+=[(data_index,res)]

        pds_res = PDSMPI(rdd, pds_id_new, self)

        return pds_res


    def collect(self, pds):
        """
        Gather the pds from all the leaders,
        send it to the scheduler and return it as a standard Python list.

        Parameters
        ----------
        pds: PDS class
            a parallel data set

        Returns
        -------
        Python list
            all elements of pds as a list
        """

        #Send the data we have back to the scheduler
        _ = self.mpimanager.get_scheduler_communicator().gather(pds.python_list, root=0)



class BackendMPITeam(BackendMPILeader if abcpy.backends.mpimanager.get_mpi_manager().is_leader() else  BackendMPIWorker):
    """
    A team is compounded by workers and a leader. One process per team is a leader, others are workers
    """

    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]

    def __init__(self):
        #Define the vars that will hold the pds ids received from scheduler to operate on
        self._rec_pds_id = None
        self._rec_pds_id_result = None

        #Initialize a BDS store for both scheduler & team.
        self.bds_store = {}

        #print("In BackendMPITeam, rank : ", self.rank, ", model_rank_global : ", globals()['model_rank_global'])

        self.logger = logging.getLogger(__name__)
        super().__init__()



class BackendMPI(BackendMPIScheduler if abcpy.backends.mpimanager.get_mpi_manager().is_scheduler() else BackendMPITeam):
    """A backend parallelized by using MPI

    The backend conditionally inherits either the BackendMPIScheduler class
    or the BackendMPIteam class depending on it's rank. This lets
    BackendMPI have a uniform interface for the user but allows for a
    logical split between functions performed by the scheduler
    and the teams.
    """

    def __init__(self, scheduler_node_ranks=[0], process_per_model=1):
        """
        Parameters
        ----------
        scheduler_node_ranks: Python list
            list of scheduler nodes
        
        process_per_model: Integer
            number of MPI processes to allocate to each model
        """
        # get mpimanager instance from the mpimanager module (which has to be setup before calling the constructor)
        self.logger = logging.getLogger(__name__)
        self.mpimanager = abcpy.backends.mpimanager.get_mpi_manager()

        if self.mpimanager.get_world_size() < 2:
            raise ValueError('A minimum of 2 ranks are required for the MPI backend')

        #Set the global backend
        globals()['backend'] = self

        #Call the appropriate constructors and pass the required data
        super().__init__()


    def size(self):
        """ Returns world size """
        return self.mpimanager.get_world_size()

    def scheduler_node_ranks(self):
        """ Returns scheduler node ranks """
        return self.mpimanager.get_scheduler_node_ranks()


    @staticmethod
    def disable_nested(mpi_comm):
        if mpi_comm.Get_rank() != 0:
            mpi_comm.Barrier()


    @staticmethod
    def enable_nested(mpi_comm):
        if mpi_comm.Get_rank() == 0:
            mpi_comm.Barrier()



class PDSMPI(PDS):
    """
    This is an MPI wrapper for a Python parallel data set.
    """

    def __init__(self, python_list, pds_id, backend_obj):
        self.python_list = python_list
        self.pds_id = pds_id
        self.backend_obj = backend_obj

    def __del__(self):
        """
        Destructor to be called when a PDS falls out of scope and/or is being deleted.
        Uses the backend to send a message to destroy the teams' copy of the pds.
        """
        try:
            self.backend_obj.delete_remote_pds(self.pds_id)
        except AttributeError:
            #Catch "delete_remote_pds not defined" for teams and ignore.
            pass


class BDSMPI(BDS):
    """
    This is a wrapper for MPI's BDS class.
    """

    def __init__(self, object, bds_id, backend_obj):
        #The BDS data is no longer saved in the BDS object.
        #It will access & store the data only from the current backend
        self.bds_id = bds_id
        backend.bds_store[self.bds_id] = object

    def value(self):
        """
        This method returns the actual object that the broadcast data set represents.
        """
        return backend.bds_store[self.bds_id]

    def __del__(self):
        """
        Destructor to be called when a BDS falls out of scope and/or is being deleted.
        Uses the backend to send a message to destroy the teams' copy of the bds.
        """

        try:
            backend.delete_remote_bds(self.bds_id)
        except AttributeError:
            #Catch "delete_remote_pds not defined" for teams and ignore.
            pass

class BackendMPITestHelper:
    """
    Helper function for some of the test cases to be able to access and verify class members.
    """
    def check_pds(self, k):
        """Checks if a PDS exists in the pds data store. Used to verify deletion and creation
        """
        return k in backend.pds_store.keys()

    def check_bds(self, k):
        """Checks if a BDS exists in the BDS data store. Used to verify deletion and creation
        """
        return k in backend.bds_store.keys()
