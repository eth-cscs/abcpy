from abc import ABCMeta, abstractmethod
from ProbabilisticModel import Continuous
import numpy as np
from scipy.stats import multivariate_normal

# TODO check docstrings again

# NOTE c[0] gets perturbed differently than c[1] is not supported! we could add it if we want, but it sounds a bit weird?


class PerturbationKernel(metaclass = ABCMeta):
    """This abstract base class represents all perturbation kernels"""
    @abstractmethod
    def __init__(self, models):
        raise NotImplementedError

    @abstractmethod
    def calculate_cov(self, accepted_parameters_manager, kernel_index):
        raise NotImplementedError

    @abstractmethod
    def update(self, accepted_parameters_manager, index, rng):
        raise NotImplementedError

    def pdf(self, accepted_parameters_manager, kernel_index, index, x):
        if(isinstance(self, DiscreteKernel)):
            return self.pmf(accepted_parameters_manager, kernel_index, index, x)
        else:
            raise NotImplementedError


class ContinuousKernel(metaclass = ABCMeta):
    """This abstract base class represents all perturbation kernels acting on continuous parameters."""

    @abstractmethod
    def pdf(self, accepted_parameters_manager, kernel_index, index, x):
        raise NotImplementedError


class DiscreteKernel(metaclass = ABCMeta):
    """This abstract base class represents all perturbation kernels acting on discrete parameters."""

    @abstractmethod
    def pmf(self, accepted_parameters_manager, kernel_index, index, x):
        raise NotImplementedError


class JointPerturbationKernel(PerturbationKernel):
    """This class joins different kernels to make up the overall perturbation kernel. Any user-implemented perturbation kernel should derive from this class. Any kernels defined on their own should be joined in the end using this class.

    Parameters
    ----------
    kernels: list
        List of abcpy.PerturbationKernels
    """

    def __init__(self, kernels):
        self._check_kernels(kernels)
        self.kernels = kernels

    def calculate_cov(self, accepted_parameters_manager):
        """
        Calculates the covariance matrix corresponding to each kernel. Commonly used before calculating weights to avoid repeated calculation.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
             The AcceptedParametersManager to be uesd.

        Returns
        -------
        list
            Each entry corresponds to the covariance matrix of the corresponding kernel.
        """
        all_covs = []
        for kernel_index, kernel in enumerate(self.kernels):
            all_covs.append(kernel.calculate_cov(accepted_parameters_manager, kernel_index))
        return all_covs


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
        """Perturbs the parameter values contained in accepted_parameters_manager. Commonly used while perturbing.

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
        for kernel_index, kernel in enumerate(self.kernels):
            index=0
            for model in kernel.models:
                model_values = []
                for j in range(model.dimension):
                    model_values.append(perturbed_values[kernel_index][index])
                    index+=1
                perturbed_values_including_models.append((model, model_values))

        return perturbed_values_including_models

    def pdf(self, mapping, accepted_parameters_manager, index, x):
        """Calculates the overall pdf of the kernel.
        Commonly used to calculate weights.

        Parameters
        ----------
        mapping: list
            Each entry is a tupel of which the first entry is a abcpy.ProbabilisticModel object, the second entry is the index in the accepted_parameters_bds list corresponding to an output of this model.
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

        result = 1.

        for kernel_index, kernel in enumerate(self.kernels):
            # Define a list containing the parameter values relevant to the current kernel
            theta = []
            for kernel_model in kernel.models:
                for model, model_output_index in mapping:
                    if(kernel_model==model):
                        theta.append(x[model_output_index])
            theta = np.array(theta)
            result*=kernel.pdf(accepted_parameters_manager, kernel_index, index, theta)

        return result


class MultivariateNormalKernel(PerturbationKernel, ContinuousKernel):
    """This class defines a kernel perturbing the parameters using a multivariate normal distribution."""
    def __init__(self, models):
        self.models = models

    def calculate_cov(self, accepted_parameters_manager, kernel_index):
        """Calculates the covariance matrix relevant to this kernel.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            AcceptedParametersManager to be used.
        kernel_index: integer
            The index of the kernel in the list of kernels of the joint kernel.

        Returns
        -------
        list
            The covariance matrix corresponding to this kernel.
        """
        weights = accepted_parameters_manager.accepted_weights_bds.value()
        cov = np.cov(
            accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index], aweights=weights.reshape(-1),
            rowvar=False)
        return cov

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

        # NOTE I think this is not required?
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

    def pdf(self, accepted_parameters_manager, kernel_index, index, x):
        """Calculates the pdf of the kernel.
        Commonly used to calculate weights.

        Parameters
        ----------
        cov: list
            The covariance matrix to be used to calculate the pdf.
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            The AcceptedParametersManager to be used.
        kernel_index: integer
            The index of the kernel in the list of kernels in the joint kernel.
        index: integer
            The row to be considered in the accepted_parameters_bds matrix.
        x: The point at which the pdf should be evaluated.

        Returns
        -------
        float
            The pdf evaluated at point x.
                """
        # Get the parameters relevant to this kernel
        #accepted_parameters = accepted_parameters_manager.get_accepted_parameters_bds_values(self.models)

        # Gets the relevant accepted parameters from the manager in order to calculate the pdf
        mean = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][index]

        cov = accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]

        #weights = accepted_parameters_manager.accepted_weights_bds.value()
        #cov = accepted_parameters_manager.covFactor*np.cov(accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index], aweights=weights.reshape(-1), rowvar=False)

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
        # NOTE see above
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

    def calculate_cov(self, accepted_parameters_manager, kernel_index):
        """Calculates the covariance matrix of this kernel. Since there is no covariance matrix associated with this random walk, it returns an empty list."""
        return []

    # NOTE is this correct?
    def pmf(self, accepted_parameters_manager, kernel_index, index, x):
        """Calculates the pmf of the kernel.
        Commonly used to calculate weights.

        Parameters
        ----------
        cov: list
            The covariance matrix used for this kernel. This is a dummy input.
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            The AcceptedParametersManager to be used.
        kernel_index: integer
            The index of the kernel in the list of kernels of the joint kernel.
        index: integer
            The row to be considered in the accepted_parameters_bds matrix.
        x: The point at which the pdf should be evaluated.

        Returns
        -------
        float
            The pmf evaluated at point x.
                """
        return 1./3


class DefaultKernel(JointPerturbationKernel):
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
        if(not(continuous_models)):
            super(DefaultKernel, self).__init__([discrete_kernel])
        elif(not(discrete_models)):
            super(DefaultKernel, self).__init__([continuous_kernel])
        else:
            super(DefaultKernel, self).__init__([continuous_kernel, discrete_kernel])


