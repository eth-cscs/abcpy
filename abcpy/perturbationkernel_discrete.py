from abc import ABCMeta, abstractmethod

from perturbationkernel import PerturbationKernel

class discreteKernel(PerturbationKernel):
    """
    This kernel is an implementation of a perturbation kernel that can be used for discrete parameters.
    """
    #TODO should return discrete parameter values
    def perturb(self, parameters):
        return parameters
    #TODO should return a pmf of the kernel (?)
    def pdf(self, x):
        return 1.