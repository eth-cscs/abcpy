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

    OP_PARALLELIZE = 1
    OP_MAP = 2
    OP_COLLECT = 3
    OP_BROADCAST = 4
    OP_DELETEPDS = 5
    OP_DELETEBDS = 6
    OP_FINISH = 7

    ATTR_TAG = 11
    ATTR_RESTAG = 12
    ATTR_FUNC = 13


    def __init__(self):
        """
        Initialize the backend identifying all the ranks.

        """

        #Initialize some private variables for tags we need for communication 
        #.. between Master and slaves
        self.__current_tag = 0
        self.__rec_tag = None
        self.__rec_tag_new = None


        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()



        self.is_master = (self.rank == 0)
        if self.size < 2:
            raise ValueError('Please, use at least 2 ranks.')

        # List of available workes, check on Master node
        avail_workers = list(range(1, self.size))

        if (self.is_master):
            print("Hello World, I am the master.")
        else:
            print("Hello World, I am worker number %s." % (self.rank))
            self.slave_run()



    def slave_run(self):
        """
        This method is the infinite loop a slave enters directly from init.
        It makes the slave wait for a command to perform from the master and
        then calls the appropriate function.

        This method also takes care of the synchronization of data between the 
        master and the slaves by matching PDSs based on the tags sent by the master 
        with the command.

        Commands received from the master are of the form of a tuple. 
        The first component of the tuple is always the operation to be performed
        and the rest are conditional on the operation.

        (op) where op=="par" for parallelize 
        (op,tag,func) where op=="map" for map.
        (op,tag) where op=="col" for a collect operation
        (op,) where op=="die" for the slave to break and die
        """

        self.data_store = {}


        while True:
            data = self.comm.bcast(None, root=0)

            op = data["op"]

            if op == self.OP_PARALLELIZE:
                tag = data[self.ATTR_TAG]
                self.__rec_tag = tag
                pds = self.parallelize([])
                self.data_store[pds.tag] = pds


            elif op == self.OP_MAP:
                tag,tag_new,func_dump = data[self.ATTR_TAG],data[self.ATTR_RESTAG],data[self.ATTR_FUNC]
                self.__rec_tag, self.__rec_tag_new = tag,tag_new

                #Use cloudpickle to convert back our string into a function
                func = cloudpickle.loads(func_dump)

                # Access an existing PDS
                pds = self.data_store[tag]
                pds_new = self.map(func, pds)

                # Store the result in a newly gnerated PDS tag
                self.data_store[pds_new.tag] = pds_new

            elif op == self.OP_COLLECT:
                tag = data[self.ATTR_TAG]

                # Access an existing PDS
                pds = self.data_store[tag]

                self.collect(pds)

            elif op == self.OP_DELETEPDS:
                #Delete the remote PDS when the master tells it to.
                tag = data[self.ATTR_TAG]
                del self.data_store[tag]

            elif op == self.OP_FINISH:
                quit()

    def __get_received_tag(self):
        """
        Function to retrieve the tag(s) we received from the master to associate
        our slave's created PDS with the master's.
        """
        return self.__rec_tag,self.__rec_tag_new

    def command_slaves(self,command,data):
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
        data_packet = {}
        data_packet["op"] = command

        if command == self.OP_PARALLELIZE:
            #In parallelize, we get only one entry of tuple data
            # which is the tag of the data we are going to receive.
            tag = data[0]
            data_packet[self.ATTR_TAG] = tag

        elif command == self.OP_MAP:
            #In map we recieve data as (tag,tag_new,func)
            tag,tag_new,func = data

            #Use cloudpickle to dump the function into a string.
            func_dump = cloudpickle.dumps(func)
            data_packet[self.ATTR_TAG] = tag
            data_packet[self.ATTR_RESTAG] = tag_new
            data_packet[self.ATTR_FUNC] = func_dump


        elif command == self.OP_COLLECT:
            #In collect we receive data as (tag)
            tag = data[0]
            data_packet[self.ATTR_TAG] = tag


        _ = self.comm.bcast(data_packet, root=0)

    def __generate_new_tag(self):
        """
        This method generates a new tag to associate a PDS with it's remote counterpart
        that slaves use to store & index data based on the tag they receive

        Returns
        -------
        Returns a unique integer.

        """

        self.__current_tag += 1
        return self.__current_tag


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
            tag = self.__generate_new_tag()
            self.command_slaves(self.OP_PARALLELIZE,(tag,))
        else:
            tag,tag_new = self.__get_received_tag()

        rdd = np.array_split(python_list, self.size, axis=0)

        data_chunk = self.comm.scatter(rdd, root=0)

        pds = PDSMPI(data_chunk, tag)

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
            # Tell the slaves to enter the map() with the current tag & func.

            #Get tag of dataset we want to operate on
            tag = pds.tag

            #Generate a new tag to be used by the slaves for the resultant PDS
            tag_new = self.__generate_new_tag()
            
            data = (tag,tag_new,func)
            self.command_slaves(self.OP_MAP,data)

        else:
            tag,tag_new = self.__get_received_tag()

        rdd = list(map(func, pds.python_list))

        pds_res = PDSMPI(rdd, tag_new)
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
            # Tell the slaves to enter collect with the pds's tag
            self.command_slaves(self.OP_COLLECT,(pds.tag,))

        python_list = self.comm.gather(pds.python_list, root=0)

        return python_list


    def __del__(self):
        """
        Overriding the delete function to explicitly call MPI.finalize().
        This is also required so we can tell the slaves to get out of the
        while loop they are in and exit gracefully and they themselves call
        finalize when they die.
        """

        if self.is_master:
            self.command_slaves(self.OP_FINISH,None)

        MPI.Finalize()


    def broadcast(self, object, tag = None):
        """
        Send object to all worker nodes without splitting it up.

        Parameters
        ----------
        object: Python object
            An arbitrary object that should be available on all workers

        tag: Int (Default: None)
            the tag identifier of the parallelize. The master will overwrite
            but the slaves will use it.

        Returns
        -------
        BDS class (broadcast data set)
            A reference to the broadcasted object
        """

        raise NotImplementedError

        bcv = self.comm.bcast(object, root=0)
        bds = BDSMPI(bcv)

        return bds


class PDSMPI(PDS):
    """
    This is a wrapper for a Python parallel data set.
    """

    def __init__(self, python_list, tag):
        self.python_list = python_list
        self.tag = tag

    def __del__(self):
        """
        Destructor to be called when a PDS falls out of scope and\or is being deleted.
        Tells the slaves to delete their copy of the PDS.
        """
        if MPI.Is_finalized()==False:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank ==0:
                data_packet = {"op":BackendMPI.OP_DELETEPDS,BackendMPI.ATTR_TAG:self.tag}
                _ = comm.bcast(data_packet, root=0)



class BDSMPI(BDS):
    """
    The reference class for broadcast data set (BDS).
    """

    def __init__(self, object, tag):

        self.object = object
        self.tag = tag

    def value(self):
        """
        This method returns the actual object that the broadcast data set represents.
        """

        return self.object
