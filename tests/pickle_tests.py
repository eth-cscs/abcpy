import unittest
import cloudpickle
import numpy as np

class ToBePickled:
    def __init__(self):
        self.included = 5
        self.notIncluded = np.zeros(10**5)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['notIncluded']
        return state

class PickleTests(unittest.TestCase):
    def test_exclusion(self,A,string):
        pickled_object = cloudpickle.dumps(A())
        unpickled_object = cloudpickle.loads(pickled_object)
        assert(not(hasattr(pickled_object,string)))

A=PickleTests()
A.test_exclusion(ToBePickled,'notIncluded')