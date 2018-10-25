import pickle
import time

import cloudpickle
import numpy as np
from mpi4py import MPI

from abcpy.backends import BDS, PDS, Backend


class BackendMPIMaster(Backend):
    """Defines the behavior of the master process

    This class defines the behavior of the master process (The one
    with rank==0) in MPI.

    """

    #Define some operation codes to make it more readable
    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]
    finalized = False

    def __init__(self, master_node_ranks=[0],chunk_size=1):
        """
        Parameters
        ----------
        master_node_ranks: Python list
            list of ranks computation should not happen on.
            Should include the master so it doesn't get 
            overwhelmed with work.

        chunk_size: Integer
            size of one block of data to be sent to free
            executors
       """
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.master_node_ranks = master_node_ranks

        #Initialize the current_pds_id and bds_id
        self.__current_pds_id = 0
        self.__current_bds_id = 0

        #Initialize a BDS store for both master & slave.
        self.bds_store = {}
        self.pds_store = {}

        #Initialize a store for the pds data that 
        #.. hasn't been sent to the workers yet
        self.pds_pending_store = {}

        self.chunk_size = chunk_size


    def __command_slaves(self, command, data):
        """Tell slaves to enter relevant execution block
        This method handles the sending of the command to the slaves
        telling them what operation to perform next.

        Parameters
        ----------
        command: operation code of OP_xxx
            One of the operation codes defined in the class definition as OP_xxx
            which tell the slaves what operation they're performing.
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
            # function_packed = self.__sanitize_and_pack_func()
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

        _ = self.comm.bcast(data_packet, root=0)



    def __generate_new_pds_id(self):
        """
        This method generates a new pds_id to associate a PDS with it's remote counterpart
        that slaves use to store & index data based on the pds_id they receive

        Returns
        -------
        Returns a unique integer id.

        """

        self.__current_pds_id += 1
        return self.__current_pds_id


    def __generate_new_bds_id(self):
        """
        This method generates a new bds_id to associate a BDS with it's remote counterpart
        that slaves use to store & index data based on the bds_id they receive

        Returns
        -------
        Returns a unique integer id.

        """

        self.__current_bds_id += 1
        return self.__current_bds_id


    def parallelize(self, python_list):
        """
        This method distributes the list on the available workers and returns a
        reference object.

        The list is split into number of workers many parts as a numpy array.
        Each part is sent to a separate worker node using the MPI scatter.

        MASTER: python_list is the real data that is to be split up

        Parameters
        ----------
        list: Python list
            the list that should get distributed on the worker nodes

        Returns
        -------
        PDSMPI class (parallel data set)
            A reference object that represents the parallelized list
        """

        # Tell the slaves to enter parallelize()
        pds_id = self.__generate_new_pds_id()
        self.__command_slaves(self.OP_PARALLELIZE, (pds_id,))

        #Don't send any data. Just keep it as a queue we're going to pop.
        self.pds_store[pds_id] = list(python_list)


        pds = PDSMPI([], pds_id, self)

        return pds

    def orchestrate_map(self,pds_id):
        """Orchestrates the slaves/workers to perform a map function
        
        This works by keeping track of the workers who haven't finished executing,
        waiting for them to request the next chunk of data when they are free,
        responding to them with the data and then sending them a Sentinel
        signalling that they can exit.
        """
        is_map_done = [True if i in self.master_node_ranks else False for i in range(self.size)]
        status = MPI.Status()

        #Copy it to the pending. This is so when master accesses
        #the PDS data it's not empty.
        self.pds_pending_store[pds_id] = list(self.pds_store[pds_id])

        #While we have some ranks that haven't finished
        while sum(is_map_done)<self.size:
            #Wait for a reqest from anyone
            data_request = self.comm.recv(
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
                self.comm.send(None, dest=request_from_rank, tag=pds_id)
                is_map_done[request_from_rank] = True
            else:
                #Create the chunk of data to send. Pop off items and tag them with an id.
                # so we can sort them later
                chunk_to_send = []
                for i in range(self.chunk_size):
                    chunk_to_send+=[(num_current_pds_items-i,current_pds_items.pop())]

                    self.comm.send(chunk_to_send, dest=request_from_rank, tag=pds_id)

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

        # Tell the slaves to enter the map() with the current pds_id & func.
        #Get pds_id of dataset we want to operate on
        pds_id = pds.pds_id

        #Generate a new pds_id to be used by the slaves for the resultant PDS
        pds_id_new = self.__generate_new_pds_id()

        data = (pds_id, pds_id_new, func)
        self.__command_slaves(self.OP_MAP, data)

        self.orchestrate_map(pds_id)

        pds_res = PDSMPI([], pds_id_new, self)

        return pds_res


    def collect(self, pds):
        """
        Gather the pds from all the workers,
            send it to the master and return it as a standard Python list.

        Parameters
        ----------
        pds: PDS class
            a parallel data set

        Returns
        -------
        Python list
            all elements of pds as a list
        """

        # Tell the slaves to enter collect with the pds's pds_id
        self.__command_slaves(self.OP_COLLECT, (pds.pds_id,))

        all_data = self.comm.gather(pds.python_list, root=0)

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
        # Tell the slaves to enter broadcast()
        bds_id = self.__generate_new_bds_id()
        self.__command_slaves(self.OP_BROADCAST, (bds_id,))

        _ = self.comm.bcast(value, root=0)

        bds = BDSMPI(value, bds_id, self)
        return bds


    def delete_remote_pds(self, pds_id):
        """
        A public function for the PDS objects on the master to call when they go out of
        scope or are deleted in order to ensure the same happens on the slaves.

        Parameters
        ----------
        pds_id: int
            A pds_id identifying the remote PDS on the slaves to delete.
        """

        if  not self.finalized:
            self.__command_slaves(self.OP_DELETEPDS, (pds_id,))


    def delete_remote_bds(self, bds_id):
        """
        Public function for the BDS objects on the master to call when they go
        out of score or are deleted in order to ensure they are deleted
        ont he slaves as well.

        Parameters
        ----------
        bds_id: int
            A bds_id identifying the remote BDS on the slaves to delete.
        """

        if  not self.finalized:
            #The master deallocates it's BDS data. Explicit because
            #.. bds_store and BDSMPI object are disconnected.
            del backend.bds_store[bds_id]
            self.__command_slaves(self.OP_DELETEBDS, (bds_id,))


    def __del__(self):
        """
        Overriding the delete function to explicitly call MPI.finalize().
        This is also required so we can tell the slaves to get out of the
        while loop they are in and exit gracefully and they themselves call
        finalize when they die.
        """

        #Tell the slaves they can exit gracefully.
        self.__command_slaves(self.OP_FINISH, None)

        #Finalize the connection because the slaves should have finished.
        MPI.Finalize()
        self.finalized = True


class BackendMPISlave(Backend):
    """Defines the behavior of the slaves/worker processes

    This class defines how the slaves should behave during operation.
    Slaves are those processes(not nodes like Spark) that have rank!=0
    and whose ids are not present in the list of non workers.
    """

    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]


    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        #Define the vars that will hold the pds ids received from master to operate on
        self.__rec_pds_id = None
        self.__rec_pds_id_result = None

        #Initialize a BDS store for both master & slave.
        self.bds_store = {}

        #Go into an infinite loop waiting for commands from the user.
        self.slave_run()


    def slave_run(self):
        """
        This method is the infinite loop a slave enters directly from init.
        It makes the slave wait for a command to perform from the master and
        then calls the appropriate function.

        This method also takes care of the synchronization of data between the
        master and the slaves by matching PDSs based on the pds_ids sent by the master
        with the command.

        Commands received from the master are of the form of a tuple.
        The first component of the tuple is always the operation to be performed
        and the rest are conditional on the operation.

        (op,pds_id) where op == OP_PARALLELIZE for parallelize
        (op,pds_id, pds_id_result,func) where op == OP_MAP for map.
        (op,pds_id) where op == OP_COLLECT for a collect operation
        (op,pds_id) where op == OP_DELETEPDS for a delete of the remote PDS on slaves
        (op,) where op==OP_FINISH for the slave to break out of the loop and terminate
        """

        # Initialize PDS data store here because only slaves need to do it.
        self.pds_store = {}

        while True:
            data = self.comm.bcast(None, root=0)

            op = data[0]
            if op == self.OP_PARALLELIZE:
                pds_id = data[1]
                self.__rec_pds_id = pds_id
                pds_id, pds_id_new = self.__get_received_pds_id()
                self.pds_store[pds_id] = None


            elif op == self.OP_MAP:
                pds_id, pds_id_result, function_packed = data[1:]
                self.__rec_pds_id, self.__rec_pds_id_result = pds_id, pds_id_result

                #Use cloudpickle to convert back function string to a function
                func = cloudpickle.loads(function_packed)

                #Enter the map so we can grab data and perform the func.
                #Func sent before and not during for performance reasons
                pds_res = self.map(func)

                # Store the result in a newly gnerated PDS pds_id
                self.pds_store[pds_res.pds_id] = pds_res

            elif op == self.OP_BROADCAST:
                self.__bds_id = data[1]
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
                quit()
            else:
                raise Exception("Slave recieved unknown command code")


    def __get_received_pds_id(self):
        """
        Function to retrieve the pds_id(s) we received from the master to associate
        our slave's created PDS with the master's.
        """

        return self.__rec_pds_id, self.__rec_pds_id_result



    def parallelize(self):
        pass

    def map(self, func):
        """
        A distributed implementation of map that works on parallel data sets (PDS).

        On every element of pds the function func is called.

        Parameters
        ----------
        func: Python func
            A function that can be applied to every element of the pds

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
            data_chunks = self.comm.sendrecv(pds_id, 0, pds_id)

            #If it receives a sentinel, it's done and it can exit
            if data_chunks is None:
                break

            #Accumulate the indicess and *processed* chunks
            for chunk in data_chunks:
                data_index,data_item = chunk
                try:
                    result = func(data_item)
                except Exception as e:
                    result = e
                rdd.append((data_index, result))

        pds_res = PDSMPI(rdd, pds_id_new, self)

        return pds_res


    def collect(self, pds):
        """
        Gather the pds from all the workers,
        send it to the master and return it as a standard Python list.

        Parameters
        ----------
        pds: PDS class
            a parallel data set

        Returns
        -------
        Python list
            all elements of pds as a list
        """

        #Send the data we have back to the master
        _ = self.comm.gather(pds.python_list, root=0)


    def broadcast(self, value):
        """
        Value is ignored for the slaves. We get data from master
        """
        value = self.comm.bcast(None, root=0)
        self.bds_store[self.__bds_id] = value


class BackendMPI(BackendMPIMaster if MPI.COMM_WORLD.Get_rank() == 0 else BackendMPISlave):
    """A backend parallelized by using MPI

    The backend conditionally inherits either the BackendMPIMaster class
    or the BackendMPISlave class depending on it's rank. This lets
    BackendMPI have a uniform interface for the user but allows for a
    logical split between functions performed by the master
    and the slaves.
    """

    def __init__(self, master_node_ranks=[0]):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        if self.size < 2:
            raise ValueError('A minimum of 2 ranks are required for the MPI backend')


        #Set the global backend
        globals()['backend'] = self


        #Call the appropriate constructors and pass the required data
        if self.rank == 0:
            super().__init__(master_node_ranks)
        else:
            super().__init__()
            raise Exception("Slaves exitted main loop.")



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
        Uses the backend to send a message to destroy the slaves' copy of the pds.
        """
        try:
            self.backend_obj.delete_remote_pds(self.pds_id)
        except AttributeError:
            #Catch "delete_remote_pds not defined" for slaves and ignore.
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
        # self.backend_obj = backend_obj

    def value(self):
        """
        This method returns the actual object that the broadcast data set represents.
        """
        return backend.bds_store[self.bds_id]

    def __del__(self):
        """
        Destructor to be called when a BDS falls out of scope and/or is being deleted.
        Uses the backend to send a message to destroy the slaves' copy of the bds.
        """

        try:
            backend.delete_remote_bds(self.bds_id)
        except AttributeError:
            #Catch "delete_remote_pds not defined" for slaves and ignore.
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
