import unittest
import cloudpickle
import numpy as np
import pickle

'''We use pickle in our MPI backend to send a method from the master to the workers. The object with which this method is associated cotains the backend as an attribute, while the backend itself contains the data on which the workers should work. Pickling the method results in pickling the backend, which results in the whole data being pickled and sent, which is undersirable.

In pickle, the "getstate" method can be specified. When an object cotaining a "getstate" method is pickled, only the attributes specified within that method are pickled.

This test checks whether everything is working correctly with cloudpickle.
'''

class ToBePickled:
    def __init__(self):
        self.included = 5
        self.notIncluded = np.zeros(10**5)

    def __getstate__(self):
        """Method that tells cloudpickle which attributes should be pickled
        Returns
        -------
        state
            all the attributes that should be pickled
        """
        state = self.__dict__.copy()
        del state['notIncluded']
        return state

class PickleTests(unittest.TestCase):
    def test_exclusion(self):
        """Tests whether after pickling and unpickling the object, the attribute which should not be included exists"""
        pickled_object = cloudpickle.dumps(ToBePickled(), pickle.HIGHEST_PROTOCOL)
        unpickled_object = cloudpickle.loads(pickled_object)
        self.assertTrue(not(hasattr(pickled_object,'notIncluded')))

if __name__ == '__main__':
    unittest.main()
