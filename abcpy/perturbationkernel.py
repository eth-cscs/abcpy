from abc import ABCMeta, abstractmethod
from ProbabilisticModel import Continuous
import numpy as np
from scipy.stats import multivariate_normal


# TODO ask rito how the pdf should be calculated here as well

#TODO overall naming and docstrings, nicer unction writing
class PerturbationKernel(metaclass = ABCMeta):
    """This abstract base class represents all perturbation kernels"""
    @abstractmethod
    def __init__(self, models):
        raise NotImplementedError

    @abstractmethod
    def update(self, accepted_parameters_manager, index, rng):
        raise NotImplementedError

    def pdf(self, accepted_parameters_manager, index, x):
        if(isinstance(self, DiscreteKernel)):
            return self.pmf(x)
        else:
            raise NotImplementedError

class ContinuousKernel(metaclass = ABCMeta):
    """This abstract base class represents all perturbation kernels acting on continuous parameters."""

    @abstractmethod
    def pdf(self, accepted_parameters_manager, index, x):
        raise NotImplementedError

class DiscreteKernel(metaclass = ABCMeta):
    """This abstract base class represents all perturbation kernels acting on discrete parameters."""

    @abstractmethod
    def pmf(self, accepted_parameters_manager, index, x):
        raise NotImplementedError

class JointPerturbationKernel(PerturbationKernel):
    """This class joins different kernels to make up the overall perturbation kernel. Any user-implemented perturbation kernel should derive from this class.

    Parameters
    ----------
    kernels: list
        List of abcpy.PerturbationKernels
    """

    def __init__(self, kernels):
        self._check_kernels(kernels)
        self.kernels = kernels

    def _check_kernels(self, kernels):
        """Checks whether each model is only used in one perturbation kernel.
        Commonly called from the constructor.

        Parameters
        ----------
        kernels: list
            List of abcpy.PertubationKernels
        """
        models = []
        for kernel in kernels:
            for model in kernel.models:
                for already_contained_model in models:
                    if(already_contained_model==model):
                        raise ValueError("No two kernels can perturb the same probabilistic model.")
                models.append(model)


    def update(self, accepted_parameters_manager, row_index, rng=np.random.RandomState()):
        """Perturbes the parameter values contained in accepted_parameters_manager. Commonly used while perturbing.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            Defines the AcceptedParametersManager to be used.
        row_index: integer
            The index of the row that should be considered from the accepted_parameters_bds matrix.
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list
            The list contains tupels. Each tupel contains as the first entry a probabilistic model and as the second entry the perturbed parameter values corresponding to this model.
        """

        perturbed_values = []

        # Perturb values according to each kernel defined
        for kernel in self.kernels:
            perturbed_values.append(kernel.update(accepted_parameters_manager, row_index, rng=rng))

        perturbed_values_including_models = []

        # Match the results from the perturbations and their models
        for i, kernel in enumerate(self.kernels):
            index=0
            for model in kernel.models:
                model_values = []
                for j in range(model.dimension):
                    model_values.append(perturbed_values[i][index])
                    index+=1
                perturbed_values_including_models.append((model, model_values))

        return perturbed_values_including_models

    def pdf(self, accepted_parameters_manager, index, x):
        """Calculates the overall pdf of the kernel.
        Commonly used to calculate weights.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            The AcceptedParametersManager to be used.
        index: integer
            The row to be considered in the accepted_parameters_bds matrix.
        x: The point at which the pdf should be evaluated.

        Returns
        -------
        float
            The pdf evaluated at point x.
        """
        # Obtain a mapping between the models of the graph and the index of the parameter in each row of the accepted_parameters_bds matrix
        mapping = accepted_parameters_manager.get_mapping(accepted_parameters_manager.model)

        result = 1.

        for kernel in self.kernels:
            # Define a list containing the parameter values relevant to the current kernel
            theta = []
            for kernel_model in kernel.models:
                for model, model_output_index in mapping:
                    if(kernel_model==model):
                        theta.append(x[model_output_index])
            theta = np.array(theta)
            result*=kernel.pdf(accepted_parameters_manager, index, theta)

        return result





class MultivariateNormalKernel(PerturbationKernel, ContinuousKernel):
    """This class defines a kernel perturbing the parameters using a multivariate normal distribution."""
    def __init__(self, models):
        self.models = models

    def update(self, accepted_parameters_manager, row_index, rng=np.random.RandomState()):
        """Updates the parameter values contained in the accepted_paramters_manager using a multivariate normal distribution.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            Defines the AcceptedParametersManager to be used.
        row_index: integer
            The index of the row that should be considered from the accepted_parameters_bds matrix.
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        np.ndarray
            The perturbed parameter values.

        """
        # Get all current parameter values relevant for this model
        continuous_model_values = accepted_parameters_manager.get_accepted_parameters_bds_values(self.models)

        correctly_ordered_parameters = [[] for i in range(len(continuous_model_values))]

        index=0

        # Order the parameters in the order required by the kernel
        for model in self.models:
            for i in range(model.dimension):
                for j in range(len(continuous_model_values)):
                    correctly_ordered_parameters[j].append(continuous_model_values[j][index])
                index+=1

        # Perturb
        weights = accepted_parameters_manager.accepted_weights_bds.value()
        correctly_ordered_parameters = np.array(correctly_ordered_parameters)
        cov = np.cov(correctly_ordered_parameters, aweights=weights.reshape(-1), rowvar=False)
        perturbed_continuous_values = rng.multivariate_normal(correctly_ordered_parameters[row_index], cov)

        return perturbed_continuous_values

    def pdf(self, accepted_parameters_manager, index, x):
        """Calculates the pdf of the kernel.
        Commonly used to calculate weights.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            The AcceptedParametersManager to be used.
        index: integer
            The row to be considered in the accepted_parameters_bds matrix.
        x: The point at which the pdf should be evaluated.

        Returns
        -------
        float
            The pdf evaluated at point x.
                """
        # Get the parameters relevant to this kernel
        accepted_parameters = accepted_parameters_manager.get_accepted_parameters_bds_values(self.models)

        # Gets the relevant accepted parameters from the manager in order to calculate the pdf
        mean = []
        for accepted_parameter in accepted_parameters:
            mean.append(accepted_parameter[index])
        mean = np.array(mean)
        weights = accepted_parameters_manager.accepted_weights_bds.value()
        cov = np.cov(mean, aweights=weights.reshape(-1), rowvar=False)

        return multivariate_normal(mean, cov).pdf(x)


class RandomWalkKernel(PerturbationKernel, DiscreteKernel):
    """This class defines a kernel perturbing discrete parameters using a naive random walk.

    Parameters
    ----------
    models: list
        List of abcpy.ProbabilisticModel objects
    """
    def __init__(self, models):
        self.models = models

    def update(self, accepted_parameters_manager, row_index, rng=np.random.RandomState()):
        """Updates the parameter values contained in the accepted_paramters_manager using a random walk.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            Defines the AcceptedParametersManager to be used.
        row_index: integer
            The index of the row that should be considered from the accepted_parameters_bds matrix.
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        np.ndarray
            The perturbed parameter values.

                """
        # Get parameter values relevant to this kernel
        discrete_model_values = accepted_parameters_manager.get_accepted_parameters_bds_values(self.models)
        correctly_ordered_parameters = [[] for i in range(len(discrete_model_values))]

        index = 0

        # Order the obtained parameters in the order required by the kernel
        for model in self.models:
            for i in range(model.dimension):
                for j in range(len(discrete_model_values)):
                    correctly_ordered_parameters[j].append(discrete_model_values[j][index])
                index += 1

        perturbed_discrete_values = []
        correctly_ordered_parameters = np.array(correctly_ordered_parameters)[row_index]

        # Implement a random walk for the discrete parameter values
        for discrete_value in correctly_ordered_parameters:
            perturbed_discrete_values.append(rng.randint(discrete_value - 1, discrete_value + 2))

        return perturbed_discrete_values

    def pmf(self, accepted_parameters_manager, index, x):
        """Calculates the pmf of the kernel.
        Commonly used to calculate weights.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            The AcceptedParametersManager to be used.
        index: integer
            The row to be considered in the accepted_parameters_bds matrix.
        x: The point at which the pdf should be evaluated.

        Returns
        -------
        float
            The pmf evaluated at point x.
                """
        return 1./3


class StandardKernel(JointPerturbationKernel):
    """This class implements a kernel that perturbs all continuous parameters using a multivariate normal, and all discrete parameters using a random walk.
    To be used as an example for user defined kernels.

    Parameters
    ----------
    models: list
        List of abcpy.ProbabilisticModel objects, the models for which the kernel should be defined.
    """
    def __init__(self, models):
        continuous_models = []
        discrete_models = []
        for model in models:
            if(isinstance(model, Continuous)):
                continuous_models.append(model)
            else:
                discrete_models.append(model)
        continuous_kernel = MultivariateNormalKernel(continuous_models)
        discrete_kernel = RandomWalkKernel(discrete_models)
        super(StandardKernel, self).__init__([continuous_kernel, discrete_kernel])


