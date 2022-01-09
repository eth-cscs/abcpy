import numpy as np
from scipy.stats import multivariate_normal

from abcpy.perturbationkernel import PerturbationKernel, ContinuousKernel


class MultivariateNormalKernel(PerturbationKernel, ContinuousKernel):
    """This class defines a kernel perturbing the parameters using a multivariate normal distribution."""

    def __init__(self, models):
        self.models = models

    def calculate_cov(self, accepted_parameters_manager, kernel_index):
        """
        Calculates the covariance matrix relevant to this kernel.

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
        continuous_model = [[] for i in
                            range(len(accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index]))]
        for i in range(len(accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index])):
            if isinstance(accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][i][0],
                          (float, np.float32, np.float64, int, np.int32, np.int64)):
                continuous_model[i] = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][i]
            else:
                continuous_model[i] = np.concatenate(
                    accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][i])
        continuous_model = np.array(continuous_model).astype(float)

        if accepted_parameters_manager.accepted_weights_bds is not None:
            weights = accepted_parameters_manager.accepted_weights_bds.value()
            cov = np.cov(continuous_model, aweights=weights.reshape(-1).astype(float), rowvar=False)
        else:
            cov = np.cov(continuous_model, rowvar=False)
        return cov

    def update(self, accepted_parameters_manager, kernel_index, row_index, rng=np.random.RandomState()):
        """
        Updates the parameter values contained in the accepted_paramters_manager using a multivariate normal distribution.

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

        # Get all current parameter values relevant for this model and the structure
        continuous_model_values = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index]

        if isinstance(continuous_model_values[row_index][0],
                      (float, np.float32, np.float64, int, np.int32, np.int64)):
            # Perturb
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            continuous_model_values = np.array(continuous_model_values).astype(float)

            # Perturbed values anc split according to the structure
            perturbed_continuous_values = rng.multivariate_normal(continuous_model_values[row_index], cov)
        else:
            # print('Hello')
            # Learn the structure
            struct = [[] for i in range(len(continuous_model_values[row_index]))]
            for i in range(len(continuous_model_values[row_index])):
                struct[i] = continuous_model_values[row_index][i].shape[0]
            struct = np.array(struct).cumsum()
            continuous_model_values = np.concatenate(continuous_model_values[row_index])

            # Perturb
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            continuous_model_values = np.array(continuous_model_values).astype(float)

            # Perturbed values anc split according to the structure
            perturbed_continuous_values = np.split(rng.multivariate_normal(continuous_model_values, cov), struct)[:-1]

        return perturbed_continuous_values

    def pdf(self, accepted_parameters_manager, kernel_index, mean, x):
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

        if isinstance(mean[0], (float, np.float32, np.float64, int, np.int32, np.int64)):
            mean = np.array(mean).astype(float)
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            return multivariate_normal(mean, cov, allow_singular=True).pdf(np.array(x).astype(float))
        else:
            mean = np.array(np.concatenate(mean)).astype(float)
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            return multivariate_normal(mean, cov, allow_singular=True).pdf(np.concatenate(x))
