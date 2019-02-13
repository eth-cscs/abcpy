from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import gamma
from scipy.stats import multivariate_normal

from abcpy.probabilisticmodels import Continuous


class PerturbationKernel(metaclass = ABCMeta):
    """This abstract base class represents all perturbation kernels"""

    @abstractmethod
    def __init__(self, models):
        """
        Parameters
        ----------
        models: list
            The list of abcpy.probabilisticmodel objects that should be perturbed by this kernel.
        """

        raise NotImplementedError


    @abstractmethod
    def calculate_cov(self, accepted_parameters_manager, kernel_index):
        """
        Calculates the covariance matrix for the kernel.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.acceptedparametersmanager object
            The accepted parameters manager that manages all bds objects.
        kernel_index: integer
            The index of the kernel in the list of kernels of the joint perturbation kernel.

        Returns
        -------
        numpy.ndarray:
            The covariance matrix for the kernel.
        """

        raise NotImplementedError


    @abstractmethod
    def update(self, accepted_parameters_manager, row_index, rng):
        """
        Perturbs the parameters for this kernel.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.acceptedparametersmanager object
            The accepted parameters manager that manages all bds objects.
        row_index: integer
            The index of the accepted parameters bds that should be perturbed.
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        numpy.ndarray:
            The perturbed parameters.
        """

        raise NotImplementedError


    def pdf(self, accepted_parameters_manager, kernel_index, row_index, x):
        """
        Calculates the pdf of the kernel at point x.

        Parameters
        ----------
        accepted_parameters_manager: abcpy.acceptedparametersmanager object
            The accepted parameters manager that manages all bds objects.
        kernel_index: integer
            The index of the kernel in the list of kernels of the joint perturbation kernel.
        row_index: integer
            The index of the accepted parameters bds for which the pdf should be evaluated.
        x: list or float
            The point at which the pdf should be evaluated.

        Returns
        -------
        float:
            The pdf evaluated at point x.

        """

        if(isinstance(self, DiscreteKernel)):
            return self.pmf(accepted_parameters_manager, kernel_index, row_index, x)
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
    def __init__(self, kernels):
        """
        This class joins different kernels to make up the overall perturbation kernel. Any user-implemented
        perturbation kernel should derive from this class. Any kernels defined on their own should be joined in the end
        using this class.

        Parameters
        ----------
        kernels: list
            List of abcpy.PerturbationKernels
        """

        self._check_kernels(kernels)
        self.kernels = kernels


    def calculate_cov(self, accepted_parameters_manager):
        """
        Calculates the covariance matrix corresponding to each kernel. Commonly used before calculating weights to avoid
        repeated calculation.

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
        """
        Checks whether each model is only used in one perturbation kernel. Commonly called from the constructor.

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
        """
        Perturbs the parameter values contained in accepted_parameters_manager. Commonly used while perturbing.

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
            The list contains tupels. Each tupel contains as the first entry a probabilistic model and as the second
            entry the perturbed parameter values corresponding to this model.
        """

        perturbed_values = []

        # Perturb values according to each kernel defined
        for kernel_index, kernel in enumerate(self.kernels):
            perturbed_values.append(kernel.update(accepted_parameters_manager, kernel_index, row_index, rng=rng))

        perturbed_values_including_models = []

        # Match the results from the perturbations and their models
        for kernel_index, kernel in enumerate(self.kernels):
            index=0
            for model in kernel.models:
                model_values = []
                #for j in range(model.get_output_dimension()):
                model_values.append(perturbed_values[kernel_index][index])
                index+=1
                perturbed_values_including_models.append((model, model_values))

        return perturbed_values_including_models


    def pdf(self, mapping, accepted_parameters_manager, mean, x):
        """
        Calculates the overall pdf of the kernel. Commonly used to calculate weights.

        Parameters
        ----------
        mapping: list
            Each entry is a tupel of which the first entry is a abcpy.ProbabilisticModel object, the second entry is the
            index in the accepted_parameters_bds list corresponding to an output of this model.
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
            mean_kernel, theta = [], []
            for kernel_model in kernel.models:
                for model, model_output_index in mapping:
                    if(kernel_model==model):
                        theta.append(x[model_output_index])
                        mean_kernel.append(mean[model_output_index])
            result*=kernel.pdf(accepted_parameters_manager, kernel_index, mean_kernel, theta)

        return result


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
                          (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
                continuous_model[i] = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][i]
            else:
                continuous_model[i] = np.concatenate(
                    accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][i])
        continuous_model = np.array(continuous_model).astype(float)

        if(accepted_parameters_manager.accepted_weights_bds is not None):
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

        if isinstance(continuous_model_values[row_index][0], (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
            # Perturb
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            continuous_model_values = np.array(continuous_model_values).astype(float)

            # Perturbed values anc split according to the structure
            perturbed_continuous_values = rng.multivariate_normal(continuous_model_values[row_index], cov)
        else:
            #print('Hello')
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

        if isinstance(mean[0], (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
            mean = np.array(mean).astype(float)
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            return multivariate_normal(mean, cov, allow_singular=True).pdf(np.array(x).astype(float))
        else:
            mean = np.array(np.concatenate(mean)).astype(float)
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            return multivariate_normal(mean, cov, allow_singular=True).pdf(np.concatenate(x))


class MultivariateStudentTKernel(PerturbationKernel, ContinuousKernel):
    def __init__(self, models, df):
        """This class defines a kernel perturbing the parameters using a multivariate normal distribution.

        Parameters
        ----------
        models: list of abcpy.probabilisticmodel objects
            The models that should be perturbed using this kernel
        df: integer
            The degrees of freedom to be used.
        """

        self.models = models
        self.df = df


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
                          (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
                continuous_model[i] = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][i]
            else:
                continuous_model[i] = np.concatenate(
                    accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][i])
        continuous_model = np.array(continuous_model).astype(float)

        if(accepted_parameters_manager.accepted_weights_bds is not None):
            weights = np.array(accepted_parameters_manager.accepted_weights_bds.value())
            cov = np.cov(continuous_model, aweights=weights.reshape(-1).astype(float),rowvar=False)
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

        # Get all parameters relevant to this kernel
        continuous_model_values = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][row_index]

        if isinstance(continuous_model_values[0],
                      (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
            # Perturb
            continuous_model_values = np.array(continuous_model_values)
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            p = len(continuous_model_values)

            if (self.df == np.inf):
                chisq = 1.0
            else:
                chisq = rng.chisquare(self.df, 1) / self.df
                chisq = chisq.reshape(-1, 1).repeat(p, axis=1)

            mvn = rng.multivariate_normal(np.zeros(p), cov.astype(float), 1)
            perturbed_continuous_values = continuous_model_values + np.divide(mvn, np.sqrt(chisq))[0]
        else:
            # Learn the structure
            struct = [[] for i in range(len(continuous_model_values))]
            for i in range(len(continuous_model_values)):
                struct[i] = continuous_model_values[i].shape[0]
            struct = np.array(struct).cumsum()
            continuous_model_values = np.concatenate(continuous_model_values)

            # Perturb
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)
            p = len(continuous_model_values)

            if (self.df == np.inf):
                chisq = 1.0
            else:
                chisq = rng.chisquare(self.df, 1) / self.df
                chisq = chisq.reshape(-1, 1).repeat(p, axis=1)

            mvn = rng.multivariate_normal(np.zeros(p), cov.astype(float), 1)
            perturbed_continuous_values = continuous_model_values + np.divide(mvn, np.sqrt(chisq))[0]

            # Perturbed values anc split according to the structure
            perturbed_continuous_values = np.split(perturbed_continuous_values, struct)[:-1]

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

        if isinstance(mean[0],
                      (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
            mean = np.array(mean).astype(float)
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)

            v = self.df
            p = len(mean)

            numerator = gamma((v + p) / 2)
            denominator = gamma(v / 2) * pow(v * np.pi, p / 2.) * np.sqrt(abs(np.linalg.det(cov)))
            normalizing_const = numerator / denominator
            tmp = 1 + 1 / v * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))
            density = normalizing_const * pow(tmp, -((v + p) / 2.))

            return density
        else:
            mean = np.array(np.concatenate(mean)).astype(float)
            cov = np.array(accepted_parameters_manager.accepted_cov_mats_bds.value()[kernel_index]).astype(float)

            v = self.df
            p = len(mean)

            numerator = gamma((v + p) / 2)
            denominator = gamma(v / 2) * pow(v * np.pi, p / 2.) * np.sqrt(abs(np.linalg.det(cov)))
            normalizing_const = numerator / denominator
            tmp = 1 + 1 / v * np.dot(np.dot(np.transpose(np.concatenate(x) - mean), np.linalg.inv(cov)), (np.concatenate(x) - mean))
            density = normalizing_const * pow(tmp, -((v + p) / 2.))

            return density

class RandomWalkKernel(PerturbationKernel, DiscreteKernel):
    def __init__(self, models):
        """
        This class defines a kernel perturbing discrete parameters using a naive random walk.

        Parameters
        ----------
        models: list
            List of abcpy.ProbabilisticModel objects
        """

        self.models = models

    def update(self, accepted_parameters_manager, kernel_index, row_index, rng=np.random.RandomState()):
        """
        Updates the parameter values contained in the accepted_paramters_manager using a random walk.

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
        discrete_model_values = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index]

        perturbed_discrete_values = []
        discrete_model_values = np.array(discrete_model_values)[row_index]

        # Implement a random walk for the discrete parameter values
        for discrete_value in discrete_model_values:
            perturbed_discrete_values.append(np.array([rng.randint(discrete_value - 1, discrete_value + 2)]))

        return perturbed_discrete_values


    def calculate_cov(self, accepted_parameters_manager, kernel_index):
        """
        Calculates the covariance matrix of this kernel. Since there is no covariance matrix associated with this
        random walk, it returns an empty list.
        """

        return np.array([0]).reshape(-1,)


    def pmf(self, accepted_parameters_manager, kernel_index, mean, x):
        """
        Calculates the pmf of the kernel. Commonly used to calculate weights.

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
    def __init__(self, models):
        """
        This class implements a kernel that perturbs all continuous parameters using a multivariate normal, and all
        discrete parameters using a random walk. To be used as an example for user defined kernels.

        Parameters
        ----------
        models: list
            List of abcpy.ProbabilisticModel objects, the models for which the kernel should be defined.
        """

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

