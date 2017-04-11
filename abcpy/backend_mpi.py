from abc import ABCMeta, abstractmethod

from mpi4py import MPI
import numpy as np
import marshal
import sys

class Backend(metaclass = ABCMeta):
    """
    This is the base class for every parallelization backend. It essentially
    resembles the map/reduce API from Spark.

    An idea for the future is to implement a MPI version of the backend with the
    hope to be more complient with standard HPC infrastructure and a potential
    speed-up.

    """

    @abstractmethod
    def parallelize(self, list):
        """
        This method distributes the list on the available workers and returns a
        reference object.

        The list should be split into number of workers many parts. Each
        part should then be sent to a separate worker node.

        Parameters
        ----------
        list: Python list
            the list that should get distributed on the worker nodes
        Returns
        -------
        PDS class (parallel data set)
            A reference object that represents the parallelized list
        """
        
        raise NotImplemented


    @abstractmethod
    def broadcast(self, object):
        """
        Send object to all worker nodes without splitting it up.

        Parameters
        ----------
        object: Python object
            An abitrary object that should be available on all workers

        Returns
        -------
        BDS class (broadcast data set)
            A reference to the broadcasted object
        """
        
        raise NotImplemented


    @abstractmethod
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
        PDS class
            a new parallel data set that contains the result of the map
        """
        
        raise NotImplemented


    @abstractmethod
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
        
        raise NotImplemented

    
class PDS:
    """
    The reference class for parallel data sets (PDS).
    """

    @abstractmethod
    def __init__(self):
        raise NotImplemented


class BDS:
    """
    The reference class for broadcast data set (BDS).
    """
    
    @abstractmethod
    def __init__(self):
        raise NotImplemented


    @abstractmethod
    def value(self):
        """
        This method should return the actual object that the broadcast data set represents. 
        """
        raise NotImplemented



class BackendMPI(Backend):
    """
    A parallelization backend for MPI.

    """

    comm = None
    size = None
    rank = None
    MPI = None


    def __init__(self):
        """
        Initialize the backend identifying all the ranks.

        """

        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.current_tag = 0

        self.is_master = (self.rank == 0)
        if self.size < 2:
            raise ValueError('Please, use at least 2 ranks.')

        # List of available workes, check on Master node
        avail_workers = list(range(1, size))

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
            # Get the next broadcasted operation from the root
            data = self.comm.bcast(None, root=0)

            if data[0] == "par":
                pds = self.parallelize([])
                self.data_store[pds.tag] = pds

            elif data[0] == "map":
                tag, func = data[1:]
                # Access an existing PDS
                pds = self.data_store[tag]
                pds_new = self.map(func, pds)

                # Store the result in a newly gnerated PDS tag
                self.data_store[pds_new.tag] = pds_new

            elif data[0] == "col":
                tag = data[1]
                # Access an existing PDS
                pds = self.data_store[tag]
                self.collect(pds)

            elif data[0] == "die":
                quit()


    def master_run(self, command, tag = None, remote_function = None):
        """
        This method handles the sending of the command to the slaves 
        telling them what operation to perform next.


        Parameters
        ----------
        command: str
            A string telling the slave what the next operation is.
            valid options are (par,map,col,dir)
        tag: int (Default: None)
            A "tag" telling the slave which pds it should operate on
        remote_function: Python Function (Default:None)
            A python function passed for the "map". Is None otherwise

        """

        _ = self.comm.bcast((command, tag, remote_function), root=0)


    def generate_new_tag(self):
        """
        This method generates a new tag to associate a PDS with it's remote counterpart
        that slaves use to store & index data based on the tag they receive

        Returns
        -------
        Returns a unique integer.

        """

        self.current_tag += 1
        return self.current_tag


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
            self.master_run("par", None)

        rdd = np.array_split(python_list, self.size, axis=0)

        data_chunk = self.comm.scatter(rdd, root=0)

        # Generate a new tag to associate the data to.
        # Assumption: It's in sync with master
        tag = self.generate_new_tag()
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
            self.master_run("map", pds.tag, remote_function = func)

        rdd = list(map(func, pds.python_list))

        tag_res = self.generate_new_tag()
        pds_res = PDSMPI(rdd, tag_res)

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
            self.master_run("col", pds.tag)

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
            self.master_run("die")

        self.MPI.Finalize()


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
        Destructor
        """
        print self.tag, 'Died'


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
