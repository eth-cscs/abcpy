import numpy as np
from scipy.stats import multivariate_normal

from abcpy.perturbationkernel import PerturbationKernel, ContinuousKernel


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
        if accepted_parameters_manager.accepted_weights_bds is not None:
            weights = accepted_parameters_manager.accepted_weights_bds.value()
            cov = np.cov(accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index],
                         aweights=weights.reshape(-1), rowvar=False)
        else:
            cov = np.cov(accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index], rowvar=False)
        return cov

    def update(self, accepted_parameters_manager, kernel_index, row_index, rng=np.random.RandomState()):
        """Updates the parameter values contained in the accepted_paramters_manager using a multivariate normal distribution.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            Defines the AcceptedParametersManager to be used.
        kernel_index: integer
            The index of the kernel in the list of kernels in the joint kernel.
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
        continuous_model_values = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index]

        # Perturb
        continuous_model_values = np.array(continuous_model_values)
        cov = accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]
        perturbed_continuous_values = rng.multivariate_normal(correctly_ordered_parameters[row_index], cov)

        return perturbed_continuous_values

    def pdf(self, accepted_parameters_manager, kernel_index, row_index, x):
        """Calculates the pdf of the kernel.
        Commonly used to calculate weights.

        Parameters
        ----------
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

        # Gets the relevant accepted parameters from the manager in order to calculate the pdf
        mean = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][row_index]

        cov = accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]

        return multivariate_normal(mean, cov).pdf(x)
