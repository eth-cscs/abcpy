
from abcpy.backends import Backend, PDS, BDS

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
