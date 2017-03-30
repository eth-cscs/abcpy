from abc import ABCMeta, abstractmethod

import numpy as np
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


class BackendDummy(Backend):
    """
    This is a dummy parallelization backend, meaning it doesn't parallelize
    anything. It is mainly implemented for testing purpose.

    """
    
    def __init__(self):
        pass

    
    def parallelize(self, python_list):
        """
        This actually does nothing: it just wraps the Python list into dummy pds (PDSDummy).

        Parameters
        ----------
        python_list: Python list
        Returns
        -------        
        PDSDummy (parallel data set)
        """
        
        return PDSDummy(python_list)

    
    def broadcast(self, object):
        """
        This actually does nothing: it just wraps the object into BDSDummy.

        Parameters
        ----------
        object: Python object
        
        Returns
        -------        
        BDSDummy class
        """
        
        return BDSDummy(object)

    
    def map(self, func, pds):
        """
        This is a wrapper for the Python internal map function.

        Parameters
        ----------
        func: Python func
            A function that can be applied to every element of the pds
        pds: PDSDummy class
            A pseudo-parallel data set to which func should be applied
        
        Returns
        -------
        PDSDummy class
            a new pseudo-parallel data set that contains the result of the map
        """
        
        result_map = map(func, pds.python_list)
        result_pds = PDSDummy(list(result_map))
        return result_pds

    
    def collect(self, pds):
        """
        Returns the Python list stored in PDSDummy

        Parameters
        ----------
        pds: PDSDummy class
            a pseudo-parallel data set
        Returns
        -------
        Python list
            all elements of pds as a list
        """
        
        return pds.python_list

    

class PDSDummy(PDS):
    """
    This is a wrapper for a Python list to fake parallelization.
    """
    
    def __init__(self, python_list):
        self.python_list = python_list

        

class BDSDummy(BDS):
    """
    This is a wrapper for a Python object to fake parallelization.
    """
    
    def __init__(self, object):
        self.object = object

        
    def value(self):
        return self.object

    


class BackendSpark(Backend):
    """
    A parallelization backend for Apache Spark. It is essetially a wrapper for
    the required Spark functionality.
    """
    
    def __init__(self, sparkContext, parallelism=4):
        """
        Initialize the backend with an existing and configured SparkContext.

        Parameters
        ----------
        sparkContext: pyspark.SparkContext
            an existing and fully configured PySpark context
        parallelism: int
            defines on how many workers a distributed dataset can be distributed
        """
        self.sc = sparkContext
        self.parallelism = parallelism


    def parallelize(self, python_list):
        """
        This is a wrapper of pyspark.SparkContext.parallelize().

        Parameters
        ----------
        list: Python list
            list that is distributed on the workers
        
        Returns
        -------
        PDSSpark class (parallel data set)
            A reference object that represents the parallelized list
        """
        
        rdd = self.sc.parallelize(python_list, self.parallelism)
        pds = PDSSpark(rdd)
        return pds


    def broadcast(self, object):
        """
        This is a wrapper for pyspark.SparkContext.broadcast().

        Parameters
        ----------
        object: Python object
            An abitrary object that should be available on all workers
        Returns
        -------
        BDSSpark class (broadcast data set)
            A reference to the broadcasted object
        """
        
        bcv = self.sc.broadcast(object)
        bds = BDSSpark(bcv)
        return bds


    def map(self, func, pds):
        """
        This is a wrapper for pyspark.rdd.map()

        Parameters
        ----------
        func: Python func
            A function that can be applied to every element of the pds
        pds: PDSSpark class
            A parallel data set to which func should be applied
        Returns
        -------
        PDSSpark class
            a new parallel data set that contains the result of the map
        """
        
        rdd = pds.rdd.map(func)
        new_pds = PDSSpark(rdd)
        return new_pds


    def collect(self, pds):
        """
        A wrapper for pyspark.rdd.collect()

        Parameters
        ----------
        pds: PDSSpark class
            a parallel data set
        Returns
        -------
        Python list
            all elements of pds as a list
        """
        
        python_list = pds.rdd.collect()
        return python_list

    
    
class PDSSpark(PDS):
    """
    This is a wrapper for Apache Spark RDDs.
    """
    
    def __init__(self, rdd):
        """
        Returns
        -------
        rdd: pyspark.rdd
            initialize with an Spark RDD
        """
        
        self.rdd = rdd



class BDSSpark(BDS):
    """
    This is a wrapper for Apache Spark Broadcast variables.
    """
    
    def __init__(self, bcv):
        """
        Parameters
        ----------
        bcv: pyspark.broadcast.Broadcast
            Initialize with a Spark broadcast variable
        """
        
        self.bcv = bcv


    def value(self):
        """
        Returns
        -------
        object
            returns the referenced object that was broadcasted.
        """
        
        return self.bcv.value


class BackendMPI(Backend):
    """
    A parallelization backend for MPI.

    """

    comm = None
    size = None
    rank = None

    def __init__(self):
        """
        Initialize the backend identifying all the ranks.

        """
        # Extremely unpythonic. 
        #Find a cleaner way to check this and import conditionally on the backend.
        global MPI
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        if (self.rank == 0):
            print("Hello World, I am the master.")
        else:
            print("Hello World, I am worker number %s." % (self.rank))


    def parallelize(self, python_list):
        """
        This method distributes the list on the available workers and returns a
        reference object.

        The list is split into number of workers many parts as a numpy array.
        Each part is sent to a separate worker node using the MPI scatter.

        Parameters
        ----------
        list: Python list
            the list that should get distributed on the worker nodes
        Returns
        -------
        PDSMPI class (parallel data set)
            A reference object that represents the parallelized list
        """

        rdd = np.array_split(python_list, self.size, axis=0)
        data_chunk = self.comm.scatter(rdd, root=0)
        return PDSMPI(data_chunk)


    def broadcast(self, object):
        """
        Send object to all worker nodes without splitting it up.

        Parameters
        ----------
        object: Python object
            An arbitrary object that should be available on all workers

        Returns
        -------
        BDS class (broadcast data set)
            A reference to the broadcasted object
        """

        bcv = self.comm.bcast(object, root=0)
        bds = BDSMPI(bcv)

        return bds


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

        rdd = list(map(func, pds.python_list))
        pds_res = PDSMPI(rdd)

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

        python_list = self.comm.gather(pds.python_list, root=0)

        return python_list



class PDSMPI(PDS):
    """
    This is a wrapper for a Python parallel data set.
    """

    def __init__(self, python_list):
        """
        Returns
        -------
        python_list
            a Python list
        """

        self.python_list = python_list



class BDSMPI(BDS):
    """
    The reference class for broadcast data set (BDS).
    """

    def __init__(self, object):

        self.object = object

    def value(self):
        """
        This method returns the actual object that the broadcast data set represents.
        """

        return self.object