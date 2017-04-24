from abc import ABCMeta, abstractmethod
from abcpy.backends import Backend,PDS,BDS
from mpi4py import MPI
import numpy as np
import cloudpickle
import sys

class BackendMPI(Backend):
    """
    A parallelization backend for MPI.

    """

    comm = None
    size = None
    rank = None
    finalized = False

    #Define some operation codes to make it more readable
    OP_PARALLELIZE = 1
    OP_MAP = 2
    OP_COLLECT = 3
    OP_BROADCAST = 4
    OP_DELETEPDS = 5
    OP_DELETEBDS = 6
    OP_FINISH = 7



    def __init__(self,master_node_ranks = [0,]):
        """
        Initialize the backend identifying all the ranks.

        """


        # Define a list of processes on the master node which should *not* perform
        # .. any computation
        self.master_node_ranks = master_node_ranks

        #Initialize some private variables for pds_ids we need for communication 
        #.. between Master and slaves
        self.__current_pds_id = 0
        self.__current_bds_id = 0
        self.__rec_pds_id = None
        self.__rec_pds_id_result = None


        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.bds_ids = {}

        self.is_master = (self.rank == 0)
        if self.size < 2:
            raise ValueError('Please, use at least 2 ranks.')



        if (self.is_master):
            print("Hello World, I am the master.")
        else:
            print("Hello World, I am worker number %s." % (self.rank))
            self.slave_run()
            raise Exception("Slaves exitted main loop.")


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
        (op,pds_id,pds_id_result,func) where op == OP_MAP for map.
        (op,pds_id) where op == OP_COLLECT for a collect operation
        (op,pds_id) where op == OP_DELETEPDS for a delete of the remote PDS on slaves
        (op,) where op==OP_FINISH for the slave to break out of the loop and terminate
        """

        # Initialized data store here because only slaves need to do it.
        self.data_store = {}

        while True:
            data = self.comm.bcast(None, root=0)
            op = data[0]
            
            if op == self.OP_PARALLELIZE:
                pds_id = data[1]
                self.__rec_pds_id = pds_id
                pds = self.parallelize([])
                self.data_store[pds.pds_id] = pds


            elif op == self.OP_MAP:
                pds_id,pds_id_result,function_packed = data[1:]
                self.__rec_pds_id, self.__rec_pds_id_result = pds_id,pds_id_result

                #Use cloudpickle to convert back function string to a function
                func = cloudpickle.loads(function_packed)
                func.__globals__['backend'] = self

                # Access an existing PDS
                pds = self.data_store[pds_id]
                pds_res = self.map(func, pds)

                # Store the result in a newly gnerated PDS pds_id
                self.data_store[pds_res.pds_id] = pds_res

            elif op == self.OP_COLLECT:
                pds_id = data[1]

                # Access an existing PDS from data store
                pds = self.data_store[pds_id]

                self.collect(pds)

            elif op == self.OP_BROADCAST:
                bds_id = data[1]
                value = data[2]
                self.broadcast(value, id=bds_id)

            elif op == self.OP_DELETEPDS:
                pds_id = data[1]

                del self.data_store[pds_id]

            elif op == self.OP_FINISH:
                quit()

    def __get_received_pds_id(self):
        """
        Function to retrieve the pds_id(s) we received from the master to associate
        our slave's created PDS with the master's.
        """
        return self.__rec_pds_id,self.__rec_pds_id_result

    def __command_slaves(self,command,data):
        """ 
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

        assert self.is_master,"Slaves are not allowed to call this function"

        if command == self.OP_PARALLELIZE:
            #In parallelize we receive data as (pds_id)
            data_packet = (command , data[0])

        elif command == self.OP_MAP:
            #In map we receive data as (pds_id,pds_id_new,func)
            #Use cloudpickle to dump the function into a string.
            function_packed = cloudpickle.dumps(data[2])
            data_packet = (command,data[0],data[1],function_packed)

        elif command == self.OP_COLLECT:
            #In collect we receive data as (pds_id)
            data_packet = (command,data[0])

        elif command == self.OP_BROADCAST:
            #In collect we receive data as (pds_id)
            data_packet = (command,data[0],data[1])

        elif command == self.OP_DELETEPDS:
            #In deletepds we receive data as (pds_id)
            data_packet = (command,data[0])

        elif command == self.OP_FINISH:
            data_packet = (command,)

        print(data_packet)
        _ = self.comm.bcast(data_packet, root=0)

    def __generate_new_pds_id(self):
        """
        This method generates a new pds_id to associate a PDS with it's remote counterpart
        that slaves use to store & index data based on the pds_id they receive

        Returns
        -------
        Returns a unique integer.

        """

        self.__current_pds_id += 1
        return self.__current_pds_id


    def parallelize(self, python_list):
        """
        This method distributes the list on the available workers and returns a
        reference object.

        The list is split into number of workers many parts as a numpy array.
        Each part is sent to a separate worker node using the MPI scatter.

        MASTER: python_list is the real data that is to be split up
        SLAVE: python_list should be [] and is ignored by the scatter()

        Parameters
        ----------
        list: Python list
            the list that should get distributed on the worker nodes

        Returns
        -------
        PDSMPI class (parallel data set)
            A reference object that represents the parallelized list
        """

        if self.is_master:
            # Tell the slaves to enter parallelize()
            pds_id = self.__generate_new_pds_id()
            self.__command_slaves(self.OP_PARALLELIZE,(pds_id,))
        else:
            pds_id,pds_id_new = self.__get_received_pds_id()

        #Initialize empty data lists for the processes on the master node
        rdd_masters = [[] for i in range(len(self.master_node_ranks))]

        #Split the data only amongst the number of workers
        rdd_slaves = np.array_split(python_list, self.size - len(self.master_node_ranks), axis=0)

        #Combine the lists into the final rdd before we split it across all ranks.
        rdd = rdd_masters + rdd_slaves

        data_chunk = self.comm.scatter(rdd, root=0)

        pds = PDSMPI(data_chunk, pds_id, self)

        return pds


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

        if self.is_master:
            # Tell the slaves to enter the map() with the current pds_id & func.

            #Get pds_id of dataset we want to operate on
            pds_id = pds.pds_id

            #Generate a new pds_id to be used by the slaves for the resultant PDS
            pds_id_new = self.__generate_new_pds_id()
            
            data = (pds_id,pds_id_new,func)
            self.__command_slaves(self.OP_MAP,data)

        else:
            pds_id,pds_id_new = self.__get_received_pds_id()

        rdd = list(map(func, pds.python_list))

        pds_res = PDSMPI(rdd, pds_id_new, self)
        return pds_res


    def collect(self, pds):
        """
        Gather the pds from all the workers, send it to the master and return it as a standard Python list.

        Parameters
        ----------
        pds: PDS class
            a parallel data set

        Returns
        -------
        Python list
            all elements of pds as a list
        """

        if self.is_master:
            # Tell the slaves to enter collect with the pds's pds_id
            self.__command_slaves(self.OP_COLLECT,(pds.pds_id,))

        python_list = self.comm.gather(pds.python_list, root=0)


        if self.is_master:
            # When we gather, the results are a list of lists one
            # .. per rank. Undo that by one level and still maintain multi
            # .. dimensional output (which is why we cannot use np.flatten)
            combined_result = []
            list(map(combined_result.extend, python_list))
            return combined_result

    def delete_remote_pds(self,pds_id):
        """
        A public function for the PDS objects on the master to call when they go out of 
        scope or are deleted in order to ensure the same happens on the slaves. 

        Parameters
        ----------
        pds_id: int
            A pds_id identifying the remote PDS on the slaves to delete.
        """
        if self.is_master and not self.finalized:
            self.__command_slaves(self.OP_DELETEPDS,(pds_id,))

    def __del__(self):
        """
        Overriding the delete function to explicitly call MPI.finalize().
        This is also required so we can tell the slaves to get out of the
        while loop they are in and exit gracefully and they themselves call
        finalize when they die.
        """

        if self.is_master:
            self.__command_slaves(self.OP_FINISH,None)

        MPI.Finalize()
        self.finalized = True


    def broadcast(self, value, id=None):
        """
        Send object to all worker nodes without splitting it up.

        Parameters
        ----------
        object: Python object
            An arbitrary object that should be available on all workers

        pds_id: Int (Default: None)
            the pds_id identifier of the parallelize. The master will overwrite
            but the slaves will use it.

        Returns
        -------
        BDS class (broadcast data set)
            A reference to the broadcasted object
        """
        
        if self.is_master:
            id = self.__current_bds_id
            self.__current_bds_id += 1
            self.__command_slaves(self.OP_BROADCAST, (id, value,))

        self.bds_ids[id] = value
        globals()['backend']  = self
        
        if self.is_master:
            bds = BDSMPI(id)
            return bds


class PDSMPI(PDS):
    """
    This is a wrapper for a Python parallel data set.
    """

    def __init__(self, python_list, pds_id , backend_obj):
        self.python_list = python_list
        self.pds_id = pds_id
        self.backend_obj = backend_obj

    def __del__(self):
        """
        Destructor to be called when a PDS falls out of scope and\or is being deleted.
        Uses the backend to send a message to destroy the slaves' copy of the pds.
        """
        self.backend_obj.delete_remote_pds(self.pds_id)


class BDSMPI(BDS):
    """
    The reference class for broadcast data set (BDS).
    """

    def __init__(self, id):
        self.id = id
        
    def value(self):
        """
        This method returns the actual object that the broadcast data set represents.
        """
        return backend.bds_ids[self.id]

        
