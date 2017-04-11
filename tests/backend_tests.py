import unittest

from mpi4py import MPI
import numpy as np
import sys

from abcpy import backend_mpi

try:
	import marshal
except ImportError:
	marhsal = None



class MPIBackendTests(unittest.TestCase):

    def setUp(self):
    	assert 1==0, "Die"


    def test_parallelize(self):

    	data = list(range(100))
    	data_pds = self.backend_mpi.parallelize(data)
    	# Assert sum lenght of chunks sums up to length data
    	self.assertTrue()
    	# Assert type returned object is list

    def test_map(self):

    	# Assert with simple function each element in each chunk is correct

    	# Assert type returned object is list

    def test_collect(self):

    	# Assert length of returned object matches length of original dataset

    	# Assert type returned object is list


   if __name__ = '__main__':

   	# Test only on Master node
   	if self.rank == 0:
   		try:
   			unittest.main()
   		except SystemExit:
   			pass
