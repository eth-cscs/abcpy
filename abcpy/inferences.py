import copy
import logging
import numpy as np

import sys

from abc import ABCMeta, abstractmethod, abstractproperty
from scipy import optimize

from abcpy.acceptedparametersmanager import *
from abcpy.graphtools import GraphTools
from abcpy.jointapprox_lhd import ProductCombination
from abcpy.jointdistances import LinearCombination
from abcpy.output import Journal
from abcpy.perturbationkernel import DefaultKernel
from abcpy.probabilisticmodels import *
from abcpy.utils import cached


class InferenceMethod(GraphTools, metaclass = ABCMeta):
    """
        This abstract base class represents an inference method.

    """

    def __getstate__(self):
        """Cloudpickle is used with the MPIBackend. This function ensures that the backend itself
        is not pickled
        """
        state = self.__dict__.copy()
        del state['backend']
        return state

    @abstractmethod
    def sample(self):
        """To be overwritten by any sub-class:
        Samples from the posterior distribution of the model parameter given the observed
        data observations.
        """
        raise NotImplementedError

    @abstractproperty
    def model(self):
        """To be overwritten by any sub-class: an attribute specifying the model to be used
        """
        raise NotImplementedError

    @abstractproperty
    def rng(self):
        """To be overwritten by any sub-class: an attribute specifying the random number generator to be used
        """
        raise NotImplementedError

    @abstractproperty
    def backend(self):
        """To be overwritten by any sub-class: an attribute specifying the backend to be used."""
        raise NotImplementedError

    @abstractproperty
    def n_samples(self):
        """To be overwritten by any sub-class: an attribute specifying the number of samples to be generated
        """
        raise NotImplementedError

    @abstractproperty
    def n_samples_per_param(self):
        """To be overwritten by any sub-class: an attribute specifying the number of data points in each simulated         data set."""
        raise NotImplementedError


class BaseMethodsWithKernel(metaclass = ABCMeta):
    """
    This abstract base class represents inference methods that have a kernel.
    """

    @abstractproperty
    def kernel(self):
        """To be overwritten by any sub-class: an attribute specifying the transition or perturbation kernel."""
        raise NotImplementedError

    def perturb(self, column_index, epochs = 10, rng=np.random.RandomState()):
        """
        Perturbs all free parameters, given the current weights.
        Commonly used during inference.

        Parameters
        ----------
        column_index: integer
            The index of the column in the accepted_parameters_bds that should be used for perturbation
        epochs: integer
            The number of times perturbation should happen before the algorithm is terminated

        Returns
        -------
        boolean
            Whether it was possible to set new parameter values for all probabilistic models
        """
        current_epoch = 0

        while current_epoch < epochs:

            # Get new parameters of the graph
            new_parameters = self.kernel.update(self.accepted_parameters_manager, column_index, rng=rng)

            self._reset_flags()

            # Order the parameters provided by the kernel in depth-first search order
            correctly_ordered_parameters = self.get_correct_ordering(new_parameters)

            # Try to set new parameters
            accepted, last_index = self.set_parameters(correctly_ordered_parameters, 0)
            if accepted:
                break
            current_epoch+=1

        if current_epoch == 10:
            return [False]

        return [True, correctly_ordered_parameters]


class BaseLikelihood(InferenceMethod, BaseMethodsWithKernel, metaclass = ABCMeta):
    """
    This abstract base class represents inference methods that use the likelihood.
    """
    @abstractproperty
    def likfun(self):
        """To be overwritten by any sub-class: an attribute specifying the likelihood function to be used."""
        raise NotImplementedError


class BaseDiscrepancy(InferenceMethod, BaseMethodsWithKernel, metaclass = ABCMeta):
    """
    This abstract base class represents inference methods using descrepancy.
    """

    @abstractproperty
    def distance(self):
        """To be overwritten by any sub-class: an attribute specifying the distance function."""
        raise NotImplementedError


class RejectionABC(InferenceMethod):
    """This base class implements the rejection algorithm based inference scheme [1] for
        Approximate Bayesian Computation.

        [1] Tavaré, S., Balding, D., Griffith, R., Donnelly, P.: Inferring coalescence
        times from DNA sequence data. Genetics 145(2), 505–518 (1997).

        Parameters
        ----------
        model: list
            A list of the Probabilistic models corresponding to the observed datasets
        distance: abcpy.distances.Distance
            Distance object defining the distance measure to compare simulated and observed data sets.
        backend: abcpy.backends.Backend
            Backend object defining the backend to be used.
        seed: integer, optionaldistance
             Optional initial seed for the random number generator. The default value is generated randomly.
        """

    # TODO: defining attributes as class attributes is not correct, move to init
    model = None
    distance = None
    rng = None

    n_samples = None
    n_samples_per_param = None
    epsilon = None

    backend = None

    def __init__(self, root_models, distances, backend, seed=None):
        self.model = root_models
        # We define the joint Linear combination distance using all the distances for each individual models
        self.distance = LinearCombination(root_models, distances)
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        # An object managing the bds objects
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

        # counts the number of simulate calls
        self.simulation_counter = 0

    def sample(self, observations, n_samples, n_samples_per_param, epsilon, full_output=0):
        """
        Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations: list
            A list, containing lists describing the observed data sets
        n_samples: integer
            Number of samples to generate
        n_samples_per_param: integer
            Number of data points in each simulated data set.
        epsilon: float
            Value of threshold
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.

        Returns
        -------
        abcpy.output.Journal
            a journal containing simulation results, metadata and optionally intermediate results.
        """

        self.accepted_parameters_manager.broadcast(self.backend, observations)

        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.epsilon = epsilon

        journal = Journal(full_output)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["epsilon"] = self.epsilon

        accepted_parameters = None

        # main Rejection ABC algorithm
        seed_arr = self.rng.randint(1, n_samples * n_samples, size=n_samples, dtype=np.int32)
        rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
        rng_pds = self.backend.parallelize(rng_arr)

        accepted_parameters_distances_counter_pds = self.backend.map(self._sample_parameter, rng_pds)
        accepted_parameters_distances_counter = self.backend.collect(accepted_parameters_distances_counter_pds)
        accepted_parameters, distances, counter = [list(t) for t in zip(*accepted_parameters_distances_counter)]

        for count in counter:
            self.simulation_counter+=count

        distances = np.array(distances)

        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)
        journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
        journal.add_weights(copy.deepcopy(np.ones((n_samples, 1))))
        journal.add_distances(copy.deepcopy(distances))
        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)
        names_and_parameters = self._get_names_and_parameters()
        journal.add_user_parameters(names_and_parameters)
        journal.number_of_simulations.append(self.simulation_counter)

        return journal

    def _sample_parameter(self, rng, npc=None):
        """
        Samples a single model parameter and simulates from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        rng: random number generator
            The random number generator to be used.
        Returns
        -------
        np.array
            accepted parameter
        """
        distance = self.distance.dist_max()

        if distance < self.epsilon and self.logger:
            self.logger.warn("initial epsilon {:e} is larger than dist_max {:e}"
                             .format(float(self.epsilon), distance))

        counter = 0

        while distance > self.epsilon:
            # Accept new parameter value if the distance is less than epsilon
            self.sample_from_prior(rng=rng)
            theta = self.get_parameters(self.model)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
            if(y_sim is not None):
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
                self.logger.debug("distance after {:4d} simulations: {:e}".format(
                    counter, distance))
            else:
                distance = self.distance.dist_max()
        self.logger.debug(
                "Needed {:4d} simulations to reach distance {:e} < epsilon = {:e}".
                format(counter, distance, float(self.epsilon))
                )
        return (theta, distance, counter)


class PMCABC(BaseDiscrepancy, InferenceMethod):
    """
    This base class implements a modified version of Population Monte Carlo based inference scheme for Approximate
    Bayesian computation of Beaumont et. al. [1]. Here the threshold value at `t`-th generation are adaptively chosen by
    taking the maximum between the epsilon_percentile-th value of discrepancies of the accepted parameters at `t-1`-th
    generation and the threshold value provided for this generation by the user. If we take the value of
    epsilon_percentile to be zero (default), this method becomes the inference scheme described in [1], where the
    threshold values considered at each generation are the ones provided by the user.

    [1] M. A. Beaumont. Approximate Bayesian computation in evolution and ecology. Annual Review of Ecology,
    Evolution, and Systematics, 41(1):379–406, Nov. 2010.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distance : abcpy.distances.Distance
        Distance object defining the distance measure to compare simulated and observed data sets.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """

    model = None
    distance = None
    kernel = None
    rng = None

    #default value, set so that testing works
    n_samples = 2
    n_samples_per_param = None

    backend = None


    def __init__(self, root_models, distances, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Linear combination distance using all the distances for each individual models
        self.distance = LinearCombination(root_models, distances)
        if(kernel is None):

            mapping, garbage_index = self._get_mapping()
            models = []
            for mdl, mdl_index in mapping:
                models.append(mdl)
            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__) 

        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

        self.simulation_counter=0


    def sample(self, observations, steps, epsilon_init, n_samples = 10000, n_samples_per_param = 1, epsilon_percentile = 10, covFactor = 2, full_output=0, journal_file = None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        epsilon_init : numpy.ndarray
            An array of proposed values of epsilon to be used at each steps. Can be supplied
            A single value to be used as the threshold in Step 1 or a `steps`-dimensional array of values to be
            used as the threshold in evry steps.
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        epsilon_percentile : float, optional
            A value between [0, 100]. The default value is 10.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2 as considered in [1].
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.n_samples = n_samples
        self.n_samples_per_param=n_samples_per_param

        if(journal_file is None):
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = type(self.distance).__name__
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["steps"] = steps
            journal.configuration["epsilon_percentile"] = epsilon_percentile
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mats = None

        # Define epsilon_arr
        if len(epsilon_init) == steps:
            epsilon_arr = epsilon_init
        else:
            if len(epsilon_init) == 1:
                epsilon_arr = [None] * steps
                epsilon_arr[0] = epsilon_init
            else:
                raise ValueError("The length of epsilon_init can only be equal to 1 or steps.")

        # main PMCABC algorithm
        self.logger.info("Starting PMC iterations")
        for aStep in range(steps):
            self.logger.debug("iteration {} of PMC algorithm".format(aStep))
            if(aStep==0 and journal_file is not None):
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters, accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                # 3: calculate covariance
                self.logger.info("Calculateing covariance matrix")
                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                # Since each entry of new_cov_mats is a numpy array, we can multiply like this
                accepted_cov_mats = [covFactor * new_cov_mat for new_cov_mat in new_cov_mats]

            
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # 0: update remotely required variables
            #print("INFO: Broadcasting parameters.")
            self.logger.info("Broadcasting parameters")
            self.epsilon = epsilon_arr[aStep]
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters, accepted_weights, accepted_cov_mats)

            # 1: calculate resample parameters
            #print("INFO: Resampling parameters")
            self.logger.info("Resamping parameters")

            params_and_dists_and_counter_pds = self.backend.map(self._resample_parameter, rng_pds)
            params_and_dists_and_counter = self.backend.collect(params_and_dists_and_counter_pds)
            new_parameters, distances, counter = [list(t) for t in zip(*params_and_dists_and_counter)]
            new_parameters = np.array(new_parameters)
            distances = np.array(distances)

            for count in counter:
                self.simulation_counter+=count

            # Compute epsilon for next step
            # print("INFO: Calculating acceptance threshold (epsilon).")
            self.logger.info("Calculating acceptances threshold")
            if aStep < steps - 1:
                if epsilon_arr[aStep + 1] == None:
                    epsilon_arr[aStep + 1] = np.percentile(distances, epsilon_percentile)
                else:
                    epsilon_arr[aStep + 1] = np.max(
                        [np.percentile(distances, epsilon_percentile), epsilon_arr[aStep + 1]])

            # 2: calculate weights for new parameters
            self.logger.info("Calculating weights")

            new_parameters_pds = self.backend.parallelize(new_parameters)
            self.logger.info("Calculate weights")
            new_weights_pds = self.backend.map(self._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1, 1)
            sum_of_weights = 0.0
            for w in new_weights:
                sum_of_weights += w
            new_weights = new_weights / sum_of_weights

            # The calculation of cov_mats needs the new weights and new parameters
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters = new_parameters, accepted_weights=new_weights)

            # The parameters relevant to each kernel have to be used to calculate n_sample times. It is therefore more efficient to broadcast these parameters once,
            # instead of collecting them at each kernel in each step
            kernel_parameters = []
            for kernel in self.kernel.kernels:
                kernel_parameters.append(
                    self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
            self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

            # 3: calculate covariance
            self.logger.info("Calculating covariance matrix")
            new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
            # Since each entry of new_cov_mats is a numpy array, we can multiply like this
            new_cov_mats = [covFactor*new_cov_mat for new_cov_mat in new_cov_mats]

            # 4: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mats = new_cov_mats

            self.logger.info("Save configuration to output journal")

            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(distances))
                journal.add_weights(copy.deepcopy(accepted_weights))
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon_arr

        return journal

    def _resample_parameter(self, rng, npc=None):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        seed: integer
            initial seed for the random number generator.

        Returns
        -------
        np.array
            accepted parameter
        """

        #print(npc.communicator())
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        distance = self.distance.dist_max()

        if distance < self.epsilon and self.logger:
            self.logger.warn("initial epsilon {:e} is larger than dist_max {:e}"
                             .format(float(self.epsilon), distance))

        theta = self.get_parameters()
        counter=0

        while distance > self.epsilon:
            if self.accepted_parameters_manager.accepted_parameters_bds == None:
                self.sample_from_prior(rng=rng)
                theta = self.get_parameters()
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter+=1

            else:
                index = rng.choice(self.n_samples, size=1, p=self.accepted_parameters_manager.accepted_weights_bds.value().reshape(-1))
                # truncate the normal to the bounds of parameter space of the model
                # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf
                while True:
                    perturbation_output = self.perturb(index[0], rng=rng)
                    if(perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1])!=0):
                        theta = perturbation_output[1]
                        break
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter+=1

            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            self.logger.debug("distance after {:4d} simulations: {:e}".format(
                     counter, distance))

        self.logger.debug(
                "Needed {:4d} simulations to reach distance {:e} < epsilon = {:e}".
                format(counter, distance, float(self.epsilon))
                )

        return (theta, distance, counter)

    def _calculate_weight(self, theta, npc=None):
        """
        Calculates the weight for the given parameter using
        accepted_parameters, accepted_cov_mat

        Parameters
        ----------
        theta: np.array
            1xp matrix containing model parameter, where p is the number of parameters

        Returns
        -------
        float
            the new weight for theta
        """
        self.logger.debug("_calculate_weight")
        if self.accepted_parameters_manager.kernel_parameters_bds is None:
            return 1.0 / self.n_samples
        else:
            prior_prob = self.pdf_of_prior(self.model, theta, 0)

            denominator = 0.0

            # Get the mapping of the models to be used by the kernels
            mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(self.accepted_parameters_manager.model)

            for i in range(0, self.n_samples):
                pdf_value = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                            self.accepted_parameters_manager.accepted_parameters_bds.value()[i], theta)
                denominator += self.accepted_parameters_manager.accepted_weights_bds.value()[i, 0] * pdf_value
            return 1.0 * prior_prob / denominator


class PMC(BaseLikelihood, InferenceMethod):
    """
    Population Monte Carlo based inference scheme of Cappé et. al. [1].

    This algorithm assumes a likelihood function is available and can be evaluated
    at any parameter value given the oberved dataset.  In absence of the
    likelihood function or when it can't be evaluated with a rational
    computational expenses, we use the approximated likelihood functions in
    abcpy.approx_lhd module, for which the argument of the consistency of the
    inference schemes are based on Andrieu and Roberts [2].

    [1] Cappé, O., Guillin, A., Marin, J.-M., and Robert, C. P. (2004). Population Monte Carlo.
    Journal of Computational and Graphical Statistics, 13(4), 907–929.

    [2] C. Andrieu and G. O. Roberts. The pseudo-marginal approach for efficient Monte Carlo computations.
    Annals of Statistics, 37(2):697–725, 04 2009.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    likfun : abcpy.approx_lhd.Approx_likelihood
        Approx_likelihood object defining the approximated likelihood to be used.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed : integer, optional
        Optional initial seed for the random number generator. The default value is generated randomly.

    """

    model = None
    likfun = None
    kernel = None
    rng = None

    n_samples = None
    n_samples_per_param = None

    backend = None


    def __init__(self, root_models, likfuns, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Product of likelihood functions using all the likelihoods for each individual models
        self.likfun = ProductCombination(root_models, likfuns)

        if(kernel is None):

            mapping, garbage_index = self._get_mapping()
            models = []
            for mdl, mdl_index in mapping:
                models.append(mdl)
            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

        self.simulation_counter = 0


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 100, covFactors = None, iniPoints = None, full_output=0, journal_file = None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        steps : integer
            number of iterations in the sequential algoritm ("generations")
        n_samples : integer, optional
            number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            number of data points in each simulated data set. The default value is 100.
        covFactor : list of float, optional
            scaling parameter of the covariance matrix. The default is a p dimensional array of 1 when p is the dimension of the parameter.
        inipoints : numpy.ndarray, optional
            parameter vaulues from where the sampling starts. By default sampled from the prior.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.


        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.sample_from_prior(rng=self.rng)

        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if(journal_file is None):
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_lhd_func"] = type(self.likfun).__name__
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["steps"] = steps
            journal.configuration["covFactor"] = covFactors
            journal.configuration["iniPoints"] = iniPoints

        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mats = None
        new_theta = None

        dim = len(self.get_parameters())

        # Initialize particles: When not supplied, randomly draw them from prior distribution
        # Weights of particles: Assign equal weights for each of the particles
        if iniPoints == None:
            accepted_parameters = []
            for ind in range(0, n_samples):
                self.sample_from_prior(rng=self.rng)
                accepted_parameters.append(self.get_parameters())
            accepted_weights = np.ones((n_samples, 1), dtype=np.float) / n_samples
        else:
            accepted_parameters = iniPoints
            accepted_weights = np.ones((iniPoints.shape[0], 1), dtype=np.float) / iniPoints.shape[0]

        if covFactors is None:
            covFactors = np.ones(shape=(len(self.kernel.kernels),))

        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters, accepted_weights=accepted_weights)

        # The parameters relevant to each kernel have to be used to calculate n_sample times. It is therefore more efficient
        # to broadcast these parameters once, instead of collecting them at each kernel in each step
        kernel_parameters = []
        for kernel in self.kernel.kernels:
            kernel_parameters.append(
                self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

        self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

        # 3: calculate covariance
        self.logger.info("Calculating covariance matrix")

        new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
        # Since each entry of new_cov_mats is a numpy array, we can multiply like this

        accepted_cov_mats = [covFactor * new_cov_mat for covFactor, new_cov_mat in zip(covFactors,new_cov_mats)]

        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

        # main SMC algorithm
        self.logger.info("Starting pmc iterations")
        for aStep in range(steps):
            if(aStep==0 and journal_file is not None):
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters, accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                # 3: calculate covariance
                self.logger.info("Calculating covariance matrix")


                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                # Since each entry of new_cov_mats is a numpy array, we can multiply like this

                accepted_cov_mats = [covFactor * new_cov_mat for covFactor, new_cov_mat in zip(covFactors, new_cov_mats)]

            self.logger.info("Iteration {} of PMC algorithm".format(aStep))

            # 0: update remotely required variables
            self.logger.info("Broadcasting parameters")
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights, accepted_cov_mats=accepted_cov_mats)

            # 1: Resample parameters
            self.logger.info("Resample parameters")
            index = self.rng.choice(len(accepted_parameters), size=n_samples, p=accepted_weights.reshape(-1))
            # Choose a new particle using the resampled particle (make the boundary proper)
            # Initialize new_parameters
            new_parameters = []
            for ind in range(0, self.n_samples):
                while True:
                    perturbation_output = self.perturb(index[ind], rng=self.rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1])!= 0:
                        new_parameters.append(perturbation_output[1])
                        break

            # 2: calculate approximate lieklihood for new parameters
            self.logger.info("Calculate approximate likelihood")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=self.n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            data_arr = []
            for i in range(len(rng_arr)):
                data_arr.append([new_parameters[i], rng_arr[i]])
            data_pds = self.backend.parallelize(data_arr)

            approx_likelihood_new_parameters_and_counter_pds = self.backend.map(self._approx_lik_calc, data_pds)
            self.logger.debug("collect approximate likelihood from pds")
            approx_likelihood_new_parameters_and_counter = self.backend.collect(approx_likelihood_new_parameters_and_counter_pds)
            approx_likelihood_new_parameters, counter = [list(t) for t in
                                                         zip(*approx_likelihood_new_parameters_and_counter)]

            approx_likelihood_new_parameters = np.array(approx_likelihood_new_parameters).reshape(-1, 1)

            for count in counter:
                self.simulation_counter += count

            # 3: calculate new weights for new parameters
            self.logger.info("Calculating weights")
            new_parameters_pds = self.backend.parallelize(new_parameters)
            new_weights_pds = self.backend.map(self._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1, 1)

            sum_of_weights = 0.0
            for i in range(0, self.n_samples):
                new_weights[i] = new_weights[i] * approx_likelihood_new_parameters[i]
                sum_of_weights += new_weights[i]
            new_weights = new_weights / sum_of_weights

            self.logger.info("new_weights : ", new_weights, ", sum_of_weights : ", sum_of_weights)
            accepted_parameters = new_parameters

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters, accepted_weights=new_weights)

            # 4: calculate covariance
            # The parameters relevant to each kernel have to be used to calculate n_sample times. It is therefore more efficient to broadcast these parameters once, instead of collecting them at each kernel in each step
            self.logger.info("Calculating covariance matrix")
            kernel_parameters = []
            for kernel in self.kernel.kernels:
                kernel_parameters.append(self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

            self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

            # 3: calculate covariance
            self.logger.info("Calculating covariance matrix")

            new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
            # Since each entry of new_cov_mats is a numpy array, we can multiply like this

            new_cov_mats = [covFactor * new_cov_mat for covFactor, new_cov_mat in zip(covFactors, new_cov_mats)]


            # 5: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mat = new_cov_mats

            self.logger.info("Saving configuration to output journal")

            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_weights(copy.deepcopy(accepted_weights))
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

        return journal

    # define helper functions for map step
    def _approx_lik_calc(self, data, npc=None):
        """
        Compute likelihood for new parameters using approximate likelihood function
        Parameters
        ----------
        data: list
            A list containing a parameter value and a random numpy state, e.g. [theta, rng]
        Returns
        -------
        float
            The approximated likelihood function
        """

        # Extract theta and rng
        theta, rng = data[0], data[1]

        # Simulate the fake data from the model given the parameter value theta
        self.logger.debug("Simulate model for parameter " + str(theta))
        self.set_parameters(theta)
        y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)

        self.logger.debug("Extracting observation.")
        obs = self.accepted_parameters_manager.observations_bds.value()

        self.logger.debug("Computing likelihood...")
        total_pdf_at_theta = 1.

        lhd = self.likfun.likelihood(obs, y_sim)

        self.logger.debug("Likelihood is :" + str(lhd))
        pdf_at_theta = self.pdf_of_prior(self.model, theta)

        total_pdf_at_theta *= (pdf_at_theta * lhd)

        self.logger.debug("Prior pdf evaluated at theta is :" + str(pdf_at_theta))

        return (total_pdf_at_theta, self.n_samples_per_param)

    def _calculate_weight(self, theta, npc=None):
        """
        Calculates the weight for the given parameter using
        accepted_parameters, accepted_cov_mat

        Parameters
        ----------
        theta: np.ndarray
            1xp matrix containing the model parameters, where p is the number of parameters

        Returns
        -------
        float
            The new weight for theta
        """

        self.logger.debug("_calculate_weight")

        if self.accepted_parameters_manager.accepted_weights_bds is None:
            return 1.0 / self.n_samples
        else:
            prior_prob = self.pdf_of_prior(self.model, theta)

            denominator = 0.0

            mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
                self.accepted_parameters_manager.model)

            for i in range(0, self.n_samples):
                pdf_value = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                            self.accepted_parameters_manager.accepted_parameters_bds.value()[i], theta)
                denominator+=self.accepted_parameters_manager.accepted_weights_bds.value()[i,0]*pdf_value

            return 1.0 * prior_prob / denominator


class SABC(BaseDiscrepancy, InferenceMethod):
    """
    This base class implements a modified version of Simulated Annealing Approximate Bayesian Computation (SABC) of [1] when the prior is non-informative.

    [1] C. Albert, H. R. Kuensch and A. Scheidegger. A Simulated Annealing Approach to
    Approximate Bayes Computations. Statistics and Computing, (2014).

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distance : abcpy.distances.Distance
        Distance object defining the distance measure used to compare simulated and observed data sets.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """

    model = None
    distance = None
    kernel = None
    rng = None

    n_samples = None
    n_samples_per_param = None
    epsilon = None

    smooth_distances_bds = None
    all_distances_bds = None

    backend = None

    def __init__(self, root_models, distances, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Linear combination distance using all the distances for each individual models
        self.distance = LinearCombination(root_models, distances)

        if (kernel is None):

            mapping, garbage_index = self._get_mapping()
            models = []
            for mdl, mdl_index in mapping:
                models.append(mdl)
            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.smooth_distances_bds = None
        self.all_distances_bds = None
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

        self.simulation_counter = 0


    def sample(self, observations, steps, epsilon, n_samples = 10000, n_samples_per_param = 1, beta = 2, delta = 0.2,
               v = 0.3, ar_cutoff = 0.1, resample = None, n_update = None, full_output=0, journal_file = None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        steps : integer
            Number of maximum iterations in the sequential algoritm ("generations")
        epsilon : numpy.float
            A proposed value of threshold to start with.
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        beta : numpy.float
            Tuning parameter of SABC, default value is 2.
        delta : numpy.float
            Tuning parameter of SABC, default value is 0.2.
        v : numpy.float, optional
            Tuning parameter of SABC, The default value is 0.3.
        ar_cutoff : numpy.float
            Acceptance ratio cutoff, The default value is 0.1.
        resample: int, optional
            Resample after this many acceptance, The default value is None which takes value inside n_samples
        n_update: int, optional
            Number of perturbed parameters at each step, The default value is None which takes value inside n_samples
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        global broken_preemptively
        self.sample_from_prior(rng=self.rng)
        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if(journal_file is None):
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = type(self.distance).__name__
            journal.configuration["type_kernel_func"] = type(self.kernel)
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["beta"] = beta
            journal.configuration["delta"] = delta
            journal.configuration["v"] = v
            journal.configuration["ar_cutoff"] = ar_cutoff
            journal.configuration["resample"] = resample
            journal.configuration["n_update"] = n_update
            journal.configuration["full_output"] = full_output
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        distances = np.zeros(shape=(n_samples,))
        smooth_distances = np.zeros(shape=(n_samples,))
        accepted_weights = np.ones(shape=(n_samples, 1))
        all_distances = None
        accepted_cov_mat = None

        if resample == None:
            resample = n_samples
        if n_update == None:
            n_update = n_samples
        sample_array = np.ones(shape=(steps,))
        sample_array[0] = n_samples
        sample_array[1:] = n_update

        ## Acceptance counter to determine the resampling step
        accept = 0
        samples_until = 0

        ## Counter whether broken preemptively
        broken_preemptively = False

        for aStep in range(0, steps):
            self.logger.debug("step {}".format(aStep))
            if(aStep==0 and journal_file is not None):
                accepted_parameters=journal.get_accepted_parameters(-1)
                accepted_weights=journal.get_weights(-1)

                #Broadcast Accepted parameters and Accedpted weights
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters, accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                #Broadcast Accepted Kernel parameters
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = []
                for new_cov_mat in new_cov_mats:
                    if not(new_cov_mat.size == 1):
                        accepted_cov_mats.append(beta * new_cov_mat + 0.0001 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
                    else:
                        accepted_cov_mats.append((beta * new_cov_mat + 0.0001 * new_cov_mat).reshape(1,1))

                # Broadcast Accepted Covariance Matrix
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

            # main SABC algorithm
            # print("INFO: Initialization of SABC")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=int(sample_array[aStep]), dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            index_arr = self.rng.randint(0, self.n_samples, size=int(sample_array[aStep]), dtype=np.uint32)
            data_arr = []
            for i in range(len(rng_arr)):
                data_arr.append([rng_arr[i], index_arr[i]])
            data_pds = self.backend.parallelize(data_arr)

            # 0: update remotely required variables
            self.logger.info("Broadcasting parameters")
            self.epsilon = epsilon
            self._update_broadcasts(smooth_distances, all_distances)

            # 1: Calculate  parameters
            self.logger.info("Initial accepted parameters")
            params_and_dists_pds = self.backend.map(self._accept_parameter, data_pds)
            params_and_dists = self.backend.collect(params_and_dists_pds)
            new_parameters, new_distances, new_all_parameters, new_all_distances, index, acceptance, counter = [list(t) for t in
                                                                                                       zip(
                                                                                                           *params_and_dists)]

            # Keeping counter of number of simulations
            for count in counter:
                self.simulation_counter+=count

            #new_parameters = np.array(new_parameters)
            new_distances = np.array(new_distances)
            new_all_distances = np.concatenate(new_all_distances)
            index = np.array(index)
            acceptance = np.array(acceptance)

            # Reading all_distances at Initial step
            if aStep == 0:
                index = np.linspace(0, n_samples - 1, n_samples).astype(int).reshape(n_samples, )
                accept = 0
                all_distances = new_all_distances

            # Initialize/Update the accepted parameters and their corresponding distances
            if accepted_parameters is None:
                accepted_parameters = new_parameters
            else:
                for ind in range(len(acceptance)):
                    if acceptance[ind] == 1:
                        accepted_parameters[index[ind]] = new_parameters[ind]
            distances[index[acceptance == 1]] = new_distances[acceptance == 1]

            # 2: Smoothing of the distances
            smooth_distances[index[acceptance == 1]] = self._smoother_distance(distances[index[acceptance == 1]],
                                                                               all_distances)

            # 3: Initialize/Update U, epsilon and covariance of perturbation kernel
            if aStep == 0:
                U = self._average_redefined_distance(self._smoother_distance(all_distances, all_distances), epsilon)
            else:
                U = np.mean(smooth_distances)
            epsilon = self._schedule(U, v)

            # 4: Show progress and if acceptance rate smaller than a value break the iteration
            if aStep > 0:
                accept = accept + np.sum(acceptance)
                samples_until = samples_until + sample_array[aStep]
                acceptance_rate = accept / samples_until

                msg = ("updates= {:.2f}, epsilon= {}, u.mean={:e}, acceptance rate: {:.2f}"
                        .format(
                            np.sum(sample_array[1:aStep + 1]) / np.sum(sample_array[1:]) * 100,
                            epsilon, U, acceptance_rate
                            )
                        )
                self.logger.debug(msg)
                if acceptance_rate < ar_cutoff:
                    broken_preemptively = True
                    self.logger.debug("Stopping as acceptance rate is lower than cutoff")
                    break

            # 5: Resampling if number of accepted particles greater than resample
            if accept >= resample and U > 1e-100:
                self.logger.info("Weighted resampling")
                weight = np.exp(-smooth_distances * delta / U)
                weight = weight / sum(weight)
                index_resampled = self.rng.choice(np.arange(n_samples, dtype=int), n_samples, replace=1, p=weight)
                accepted_parameters = [accepted_parameters[i] for i in index_resampled]
                smooth_distances = smooth_distances[index_resampled]

                ## Update U and epsilon:
                epsilon = epsilon * (1 - delta)
                U = np.mean(smooth_distances)
                epsilon = self._schedule(U, v)

                ## Print effective sampling size
                print('Resampling: Effective sampling size: ', 1 / sum(pow(weight / sum(weight), 2)))
                accept = 0
                samples_until = 0

                ## Compute and broadcast accepted parameters, accepted kernel parameters and accepted Covariance matrix
                # Broadcast Accepted parameters and add to journal
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights=accepted_weights, accepted_parameters=accepted_parameters)
                # Compute Accepetd Kernel parameters and broadcast them
                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)
                # Compute Kernel Covariance Matrix and broadcast it
                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = []
                for new_cov_mat in new_cov_mats:
                    if not(new_cov_mat.size == 1):
                        accepted_cov_mats.append(beta * new_cov_mat + 0.0001 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
                    else:
                        accepted_cov_mats.append((beta * new_cov_mat + 0.0001 * new_cov_mat).reshape(1,1))

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

                if (full_output == 1 and aStep<= steps-1):
                    ## Saving intermediate configuration to output journal.
                    print('Saving after resampling')
                    journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                    journal.add_weights(copy.deepcopy(accepted_weights))
                    journal.add_distances(copy.deepcopy(distances))
                    names_and_parameters = self._get_names_and_parameters()
                    journal.add_user_parameters(names_and_parameters)
                    journal.number_of_simulations.append(self.simulation_counter)
            else:
                ## Compute and broadcast accepted parameters, accepted kernel parameters and accepted Covariance matrix
                # Broadcast Accepted parameters
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights= accepted_weights, accepted_parameters=accepted_parameters)
                # Compute Accepetd Kernel parameters and broadcast them
                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)
                # Compute Kernel Covariance Matrix and broadcast it
                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = []
                for new_cov_mat in new_cov_mats:
                    if not(new_cov_mat.size == 1):
                        accepted_cov_mats.append(beta * new_cov_mat + 0.0001 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
                    else:
                        accepted_cov_mats.append((beta * new_cov_mat + 0.0001 * new_cov_mat).reshape(1,1))

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

                if (full_output == 1 and aStep <= steps-1):
                    ## Saving intermediate configuration to output journal.
                    journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                    journal.add_weights(copy.deepcopy(accepted_weights))
                    journal.add_distances(copy.deepcopy(distances))
                    names_and_parameters = self._get_names_and_parameters()
                    journal.add_user_parameters(names_and_parameters)
                    journal.number_of_simulations.append(self.simulation_counter)

        # Add epsilon_arr, number of final steps and final output to the journal
        # print("INFO: Saving final configuration to output journal.")
        if (full_output == 0) or (full_output ==1 and broken_preemptively and aStep<= steps-1):
            journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
            journal.add_weights(copy.deepcopy(accepted_weights))
            journal.add_distances(copy.deepcopy(distances))
            self.accepted_parameters_manager.update_broadcast(self.backend,accepted_parameters=accepted_parameters,accepted_weights=accepted_weights)
            names_and_parameters = self._get_names_and_parameters()
            journal.add_user_parameters(names_and_parameters)
            journal.number_of_simulations.append(self.simulation_counter)

        journal.configuration["steps"] = aStep + 1
        journal.configuration["epsilon"] = epsilon

        return journal

    def _smoother_distance(self, distance, old_distance):
        """Smooths the distance using the Equation 14 of [1].

        [1] C. Albert, H. R. Kuensch and A. Scheidegger. A Simulated Annealing Approach to
        Approximate Bayes Computations. Statistics and Computing 0960-3174 (2014).

        Parameters
        ----------
        distance: numpy.ndarray
            Current distance between the simulated and observed data
        old_distance: numpy.ndarray
            Last distance between the simulated and observed data

        Returns
        -------
        numpy.ndarray
            Smoothed distance

        """

        smoothed_distance = np.zeros(shape=(len(distance),))

        for ind in range(0, len(distance)):
            if distance[ind] < np.min(old_distance):
                smoothed_distance[ind] = (distance[ind] / np.min(old_distance)) / len(old_distance)
            else:
                smoothed_distance[ind] = np.mean(np.array(old_distance) < distance[ind])

        return smoothed_distance

    def _average_redefined_distance(self, distance, epsilon):
        """
        Function to calculate the weighted average of the distance
        Parameters
        ----------
        distance: numpy.ndarray
            Distance between simulated and observed data set
        epsilon: float
            threshold

        Returns
        -------
        numpy.ndarray
            Weighted average of the distance
        """
        if epsilon == 0:
            U = 0
        else:
            U = np.average(distance, weights=np.exp(-distance / epsilon))

        return (U)

    def _schedule(self, rho, v):
        if rho < 1e-100:
            epsilon = 0
        else:
            fun = lambda epsilon: pow(epsilon, 2) + v * pow(epsilon, 3 / 2) - pow(rho, 2)
            epsilon = optimize.fsolve(fun, rho / 2)

        return (epsilon)

    def _update_broadcasts(self, smooth_distances, all_distances):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy
        if not smooth_distances is None:
            self.smooth_distances_bds = self.backend.broadcast(smooth_distances)
        if not all_distances is None:
            self.all_distances_bds = self.backend.broadcast(all_distances)

    # define helper functions for map step
    def _accept_parameter(self, data, npc=None):
        """
        Samples a single model parameter and simulate from it until
        accepted with probabilty exp[-rho(x,y)/epsilon].

        Parameters
        ----------
        seed_and_index: list of two integers
            Initial seed for the random number generator and the index of data to operate on

        Returns
        -------
        numpy.ndarray
            accepted parameter
        """
        if(isinstance(data,np.ndarray)):
            data = data.tolist()
        rng=data[0]
        index=data[1]
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        all_parameters = []
        all_distances = []
        acceptance = 0

        counter = 0

        if self.accepted_parameters_manager.accepted_cov_mats_bds == None:

            while acceptance == 0:
                self.sample_from_prior(rng=rng)
                new_theta = self.get_parameters()
                all_parameters.append(new_theta)
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter+=1
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
                all_distances.append(distance)
                acceptance = rng.binomial(1, np.exp(-distance / self.epsilon), 1)
            acceptance = 1
        else:
            ## Select one arbitrary particle:
            index = rng.choice(self.n_samples, size=1)[0]
            ## Sample proposal parameter and calculate new distance:
            theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]

            while True:
                perturbation_output = self.perturb(index, rng=rng)
                if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                    new_theta = perturbation_output[1]
                    break

            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            smooth_distance = self._smoother_distance([distance], self.all_distances_bds.value())

            ## Calculate acceptance probability:
            ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model,
                self.accepted_parameters_manager.accepted_parameters_bds.value()[index])
            ratio_likelihood_prob = np.exp((self.smooth_distances_bds.value()[index] - smooth_distance) / self.epsilon)
            acceptance_prob = ratio_prior_prob * ratio_likelihood_prob

            ## If accepted
            if rng.rand(1) < acceptance_prob:
                acceptance = 1
            else:
                distance = np.inf

        return (new_theta, distance, all_parameters, all_distances, index, acceptance, counter)


class ABCsubsim(BaseDiscrepancy, InferenceMethod):
    """This base class implements Approximate Bayesian Computation by subset simulation (ABCsubsim) algorithm of [1].

    [1] M. Chiachio, J. L. Beck, J. Chiachio, and G. Rus., Approximate Bayesian computation by subset
    simulation. SIAM J. Sci. Comput., 36(3):A1339–A1358, 2014/10/03 2014.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distance : abcpy.distances.Distance
        Distance object defining the distance used to compare the simulated and observed data sets.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """

    model = None
    distance = None
    kernel = None
    rng = None
    anneal_parameter = None

    n_samples = None
    n_samples_per_param = None
    chain_length = None

    backend = None

    def __init__(self, root_models, distances, backend, kernel=None,seed=None):
        self.model = root_models
        # We define the joint Linear combination distance using all the distances for each individual models
        self.distance = LinearCombination(root_models, distances)

        if kernel is None:

            mapping, garbage_index = self._get_mapping()
            models = []
            for mdl, mdl_index in mapping:
                models.append(mdl)

            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.anneal_parameter = None
        self.logger = logging.getLogger(__name__)


        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

        self.simulation_counter = 0


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, chain_length = 10, ap_change_cutoff = 10, full_output=0, journal_file = None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        chain_length : int, optional
            The length of chains, default value is 10. But should be checked such that this is an divisor of n_samples.
        ap_change_cutoff : float, optional
            The cutoff value for the percentage change in the anneal parameter. If the change is less than
            ap_change_cutoff the iterations are stopped. The default value is 10.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.sample_from_prior(rng=self.rng)

        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.chain_length = chain_length
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if(journal_file is None):
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = type(self.distance).__name__
            journal.configuration["type_kernel_func"] = type(self.kernel)
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["chain_length"] = self.chain_length
            journal.configuration["ap_change_cutoff"] = ap_change_cutoff
            journal.configuration["full_output"] = full_output
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_weights = np.ones(shape=(n_samples, 1))
        accepted_cov_mat = None
        anneal_parameter = 0
        anneal_parameter_old = 0
        temp_chain_length = 1


        for aStep in range(0, steps):
            self.logger.info("ABCsubsim step {}".format(aStep))
            if aStep==0 and journal_file is not None:
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)
                accepted_cov_mats = journal.opt_values[-1]

            # main ABCsubsim algorithm
            self.logger.info("Initialization of ABCsubsim")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=int(n_samples / temp_chain_length),
                                        dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            index_arr = np.linspace(0, n_samples // temp_chain_length - 1, n_samples // temp_chain_length).astype(
                int).reshape(int(n_samples / temp_chain_length), )
            rng_and_index_arr = np.column_stack((rng_arr, index_arr))
            rng_and_index_pds = self.backend.parallelize(rng_and_index_arr)

            # 0: update remotely required variables
            self.logger.info("Broadcasting parameters")

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights = accepted_weights, accepted_parameters=accepted_parameters)

            # 1: Calculate  parameters
            # print("INFO: Initial accepted parameter parameters")
            self.logger.info("Initial accepted parameters")
            params_and_dists_pds = self.backend.map(self._accept_parameter, rng_and_index_pds)
            self.logger.debug("Map random number to a pseudo-observation")
            params_and_dists = self.backend.collect(params_and_dists_pds)
            self.logger.debug("Collect results from the mapping")
            new_parameters, new_distances, counter = [list(t) for t in zip(*params_and_dists)]

            for count in counter:
                self.simulation_counter+=count

            if aStep > 0:
                accepted_parameters = []
                for ind in range(len(new_parameters)):
                    accepted_parameters += new_parameters[ind]
            else:
                accepted_parameters = new_parameters
            distances = np.concatenate(new_distances)

            # 2: Sort and renumber samples
            self.logger.debug("Sort and renumber samples.")

            accepted_params_and_dist = zip(distances, accepted_parameters)
            accepted_params_and_dist = sorted(accepted_params_and_dist, key = lambda x: x[0])
            distances, accepted_parameters = [list(t) for t in zip(*accepted_params_and_dist)]

            # 3: Calculate and broadcast annealling parameters
            self.logger.debug("Calculate and broadcast annealling parameters.")
            temp_chain_length = self.chain_length
            if aStep > 0:
                anneal_parameter_old = anneal_parameter
            anneal_parameter = 0.5 * (
            distances[int(n_samples / temp_chain_length)] + distances[int(n_samples / temp_chain_length) + 1])
            self.anneal_parameter = anneal_parameter


            # 4: Update proposal covariance matrix (Parallelized)
            self.logger.debug("Update proposal covariance matrix (Parallelized).")
            if aStep == 0:
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)
                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
            else:
                accepted_cov_mats = pow(2,1)*accepted_cov_mats

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=10, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            index_arr = np.linspace(0, 10 - 1, 10).astype(int).reshape(10, )
            rng_and_index_arr = np.column_stack((rng_arr, index_arr))
            rng_and_index_pds = self.backend.parallelize(rng_and_index_arr)

            self.logger.debug("Update co-variance matrix in parallel (map).")
            cov_mats_index_pds = self.backend.map(self._update_cov_mat, rng_and_index_pds)
            self.logger.debug("Collect co-variance matrix.")
            cov_mats_index = self.backend.collect(cov_mats_index_pds)
            cov_mats, T, accept_index, counter = [list(t) for t in zip(*cov_mats_index)]

            for count in counter:
                self.simulation_counter+=count

            for ind in range(10):
                if accept_index[ind] == 1:
                    accepted_cov_mats = cov_mats[ind]
                    break

            self.logger.debug("Broadcast accepted parameters.")
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

            if full_output == 1:
                self.logger.info("Saving intermediate configuration to output journal")
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(distances))
                journal.add_weights(copy.deepcopy(accepted_weights))
                journal.add_opt_values(accepted_cov_mats)
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            # Show progress
            anneal_parameter_change_percentage = 100 * abs(anneal_parameter_old - anneal_parameter) / abs(anneal_parameter)
            msg = ("step: {}, annealing parameter: {:.4f}, change(%) in annealing parameter: {:.1f}"
                   .format(aStep, anneal_parameter, anneal_parameter_change_percentage))
            self.logger.info(msg)
            if anneal_parameter_change_percentage < ap_change_cutoff:
                break

        # Add anneal_parameter, number of final steps and final output to the journal
        # print("INFO: Saving final configuration to output journal.")
        if full_output == 0:
            journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
            journal.add_distances(copy.deepcopy(distances))
            journal.add_weights(copy.deepcopy(accepted_weights))
            journal.add_opt_values(accepted_cov_mats)
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights)
            names_and_parameters = self._get_names_and_parameters()
            journal.add_user_parameters(names_and_parameters)
            journal.number_of_simulations.append(self.simulation_counter)

        journal.configuration["steps"] = aStep + 1
        journal.configuration["anneal_parameter"] = anneal_parameter

        return journal

    # define helper functions for map step
    def _accept_parameter(self, rng_and_index, npc=None):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        seed: numpy.ndarray
            2 dimensional array. The first entry defines the initial seed of therandom number generator.
            The second entry defines the index in the data set.

        Returns
        -------
        numpy.ndarray
            accepted parameter
        """

        rng = rng_and_index[0]
        index = rng_and_index[1]
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
            self.accepted_parameters_manager.model)

        result_theta = []
        result_distance = []

        counter = 0

        if self.accepted_parameters_manager.accepted_parameters_bds == None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            result_theta += self.get_parameters()
            result_distance.append(distance)
        else:
            theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]
            self.set_parameters(theta)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            result_theta.append(theta)
            result_distance.append(distance)
            for ind in range(0, self.chain_length - 1):
                while True:
                    perturbation_output = self.perturb(index, rng=rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1])!= 0:
                        break
                y_sim = self.simulate(self.n_samples_per_param, rng=rng,npc=npc)
                counter+=1
                new_distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

                ## Calculate acceptance probability:
                ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model, theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, perturbation_output[1], theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta, perturbation_output[1])
                ratio_likelihood_prob = kernel_numerator / kernel_denominator
                acceptance_prob = min(1, ratio_prior_prob * ratio_likelihood_prob) * (
                new_distance < self.anneal_parameter)

                ## If accepted
                if rng.binomial(1, acceptance_prob) == 1:
                    result_theta.append(perturbation_output[1])
                    result_distance.append(new_distance)
                    theta = perturbation_output[1]
                    distance = new_distance
                else:
                    result_theta.append(theta)
                    result_distance.append(distance)
        return result_theta, result_distance, counter

    def _update_cov_mat(self, rng_t, npc=None):
        """
        Updates the covariance matrix.

        Parameters
        ----------
        seed_t: numpy.ndarray
            2 dimensional array. The first entry defines the initial seed of the random number generator.
            The second entry defines the way in which the accepted covariance matrix is transformed.

        Returns
        -------
        numpy.ndarray
            accepted covariance matrix
        """

        rng = rng_t[0]
        t = rng_t[1]
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        acceptance = 0

        accepted_cov_mats_transformed = [cov_mat*pow(2.0, -2.0 * t) for cov_mat in self.accepted_parameters_manager.accepted_cov_mats_bds.value()]

        theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[0]

        mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
            self.accepted_parameters_manager.model)

        counter = 0

        for ind in range(0, self.chain_length):
            self.logger.debug("Parameter acceptance loop step {}.".format(ind))
            while True:
                perturbation_output = self.perturb(0, rng=rng)
                if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                    break
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
            new_distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            self.logger.debug("Calculate acceptance probability.")
            ## Calculate acceptance probability:
            ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model, theta)
            kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                               perturbation_output[1], theta)
            kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta,
                                                 perturbation_output[1])
            ratio_likelihood_prob = kernel_numerator / kernel_denominator
            acceptance_prob = min(1, ratio_prior_prob * ratio_likelihood_prob) * (new_distance < self.anneal_parameter)
            if rng.binomial(1, acceptance_prob) == 1:
                theta = perturbation_output[1]
                acceptance = acceptance + 1

        self.logger.debug("Return accepted parameters.")
        if acceptance / 10 <= 0.5 and acceptance / 10 >= 0.3:
            return (accepted_cov_mats_transformed, t, 1, counter)
        else:
            return (accepted_cov_mats_transformed, t, 0, counter)


class RSMCABC(BaseDiscrepancy, InferenceMethod):
    """This base class implements Replenishment Sequential Monte Carlo Approximate Bayesian computation of
    Drovandi and Pettitt [1].

    [1] CC. Drovandi CC and AN. Pettitt, Estimation of parameters for macroparasite population evolution using
    approximate Bayesian computation. Biometrics 67(1):225–233, 2011.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distance : abcpy.distances.Distance
        Distance object defining the distance measure used to compare simulated and observed data sets.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """

    model = None
    distance = None
    kernel = None

    R = None
    rng = None

    n_samples = None
    n_samples_per_param = None
    alpha = None

    accepted_dist_bds = None

    backend = None


    def __init__(self, root_models, distances, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Linear combination distance using all the distances for each individual models
        self.distance = LinearCombination(root_models, distances)

        if kernel is None:

            mapping, garbage_index = self._get_mapping()
            models = []
            for mdl, mdl_index in mapping:
                models.append(mdl)

            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        self.R = None
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        self.accepted_dist_bds = None

        self.simulation_counter = 0


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, alpha = 0.1, epsilon_init = 100,
               epsilon_final = 0.1, const = 0.01, covFactor = 2.0, full_output=0, journal_file = None):
        """
        Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        alpha : float, optional
            A parameter taking values between [0,1], the default value is 0.1.
        epsilon_init : float, optional
            Initial value of threshold, the default is 100
        epsilon_final : float, optional
            Terminal value of threshold, the default is 0.1
        const : float, optional
             A constant to compute acceptance probabilty, the default is 0.01.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.sample_from_prior(rng=self.rng)

        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.alpha = alpha
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if(journal_file is None):
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = type(self.distance).__name__
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["steps"] = steps
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_cov_mat = None
        accepted_dist = None
        accepted_weights = None

        # main RSMCABC algorithm
        for aStep in range(steps):
            self.logger.info("RSMCABC iteration {}".format(aStep))

            if aStep == 0 and journal_file is not None:
                accepted_parameters=journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights= accepted_weights, accepted_parameters=accepted_parameters)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)

                accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

            # 0: Compute epsilon, compute new covariance matrix for Kernel,
            # and finally Drawing new new/perturbed samples using prior or MCMC Kernel
            # print("DEBUG: Iteration " + str(aStep) + " of RSMCABC algorithm.")
            self.logger.info("Compute epsilon and calculating covariance matrix.")
            if aStep == 0:
                n_replenish = n_samples
                # Compute epsilon
                epsilon = [epsilon_init]
                R = int(1)
                if(journal_file is None):
                    accepted_cov_mats=None
            else:
                # Compute epsilon
                epsilon.append(accepted_dist[-1])
                # Calculate covariance
                # print("INFO: Calculating covariance matrix.")
                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)

                accepted_cov_mats = [covFactor*cov_mat for cov_mat in accepted_cov_mats]

            if epsilon[-1] < epsilon_final:
                self.logger("accepted epsilon {:e} < {:e}"
                            .format(epsilon[-1], epsilon_final))
                break

            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_replenish, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # update remotely required variables
            self.logger.info("Broadcasting parameters.")
            # print("INFO: Broadcasting parameters.")
            self.epsilon = epsilon
            self.R = R
            self.logger.info("Broadcast updated variable.")
            # Broadcast updated variable
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)
            self._update_broadcasts(accepted_dist)

            # calculate resample parameters
            self.logger.info("Resampling parameters")
            # print("INFO: Resampling parameters")
            params_and_dist_index_pds = self.backend.map(self._accept_parameter, rng_pds)
            params_and_dist_index = self.backend.collect(params_and_dist_index_pds)
            new_parameters, new_dist, new_index, counter = [list(t) for t in zip(*params_and_dist_index)]
            new_dist = np.array(new_dist)
            new_index = np.array(new_index)

            for count in counter:
                self.simulation_counter+=count

            # 1: Update all parameters, compute acceptance probability, compute epsilon
            self.logger.info("Append updated new parameters.")
            if len(new_dist) == self.n_samples:
                accepted_parameters = new_parameters
                accepted_dist = new_dist
                accepted_weights = np.ones(shape=(len(accepted_parameters), 1)) * (1 / len(accepted_parameters))
            else:
                accepted_parameters += new_parameters
                accepted_dist = np.concatenate((accepted_dist, new_dist))
                accepted_weights = np.ones(shape=(len(accepted_parameters), 1)) * (1 / len(accepted_parameters))

            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                self.logger.info("Saving configuration to output journal.")
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(accepted_dist))
                journal.add_weights(accepted_weights)
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights=accepted_weights, accepted_parameters=accepted_parameters)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            # 2: Compute acceptance probabilty and set R
            self.logger.info("Compute acceptance probabilty and set R")
            prob_acceptance = sum(new_index) / (R * n_replenish)
            if prob_acceptance == 1 or prob_acceptance == 0:
                R = 1
            else:
                R = int(np.log(const) / np.log(1 - prob_acceptance))

            self.logger.info("Order accepted parameters and distances")
            n_replenish = round(n_samples * alpha)
            accepted_params_and_dist = zip(accepted_dist, accepted_parameters)
            accepted_params_and_dist = sorted(accepted_params_and_dist, key = lambda x: x[0])
            accepted_dist, accepted_parameters = [list(t) for t in zip(*accepted_params_and_dist)]

            self.logger.info("Throw away N_alpha particles with largest dist")
            # Throw away N_alpha particles with largest distance

            del accepted_parameters[self.n_samples - round(n_samples * alpha):]
            accepted_dist = np.delete(accepted_dist,
                                      np.arange(round(n_samples * alpha)) + (n_samples - round(n_samples * alpha)),
                                      0)

            accepted_weights = np.ones(shape=(len(accepted_parameters), 1)) * (1 / len(accepted_parameters))
            self.logger.info("Update parameters, weights")
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights=accepted_weights,
                                                              accepted_parameters=accepted_parameters)


        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon

        return journal

    def _update_broadcasts(self, accepted_dist):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_dist is None:
            self.accepted_dist_bds = self.backend.broadcast(accepted_dist)

    # define helper functions for map step
    def _accept_parameter(self, rng, npc=None):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        seed: integer
            Initial seed for the random number generator.

        Returns
        -------
        numpy.ndarray
            accepted parameter
        """
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        distance = self.distance.dist_max()
        mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
            self.accepted_parameters_manager.model)

        counter = 0

        if self.accepted_parameters_manager.accepted_parameters_bds == None:
            while distance > self.epsilon[-1]:
                self.sample_from_prior(rng=rng)
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter+=1
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            index_accept = 1
        else:
            index = rng.choice(len(self.accepted_parameters_manager.accepted_parameters_bds.value()), size=1)
            theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index[0]]
            index_accept = 0.0
            for ind in range(self.R):
                while True:
                    perturbation_output = self.perturb(index[0], rng=rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                        break
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter+=1
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
                ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model, theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, perturbation_output[1], theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta, perturbation_output[1])
                ratio_kernel_prob = kernel_numerator / kernel_denominator
                probability_acceptance = min(1, ratio_prior_prob * ratio_kernel_prob)
                if distance < self.epsilon[-1] and rng.binomial(1, probability_acceptance) == 1:
                    index_accept += 1
                else:
                    self.set_parameters(theta)
                    distance = self.accepted_dist_bds.value()[index[0]]

        return (self.get_parameters(self.model), distance, index_accept, counter)


class APMCABC(BaseDiscrepancy, InferenceMethod):
    """This base class implements Adaptive Population Monte Carlo Approximate Bayesian computation of
    M. Lenormand et al. [1].

    [1] M. Lenormand, F. Jabot and G. Deffuant, Adaptive approximate Bayesian computation
    for complex models. Computational Statistics, 28:2777–2796, 2013.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distance : abcpy.distances.Distance
        Distance object defining the distance measure used to compare simulated and observed data sets.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """

    model = None
    distance = None
    kernel = None

    epsilon = None
    rng = None

    n_samples = None
    n_samples_per_param = None
    alpha = None

    accepted_dist = None

    backend = None

    def __init__(self,  root_models, distances, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Linear combination distance using all the distances for each individual models
        self.distance = LinearCombination(root_models, distances)

        if kernel is None:

            mapping, garbage_index = self._get_mapping()
            models = []
            for mdl, mdl_index in mapping:
                models.append(mdl)
            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        self.epsilon= None
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        self.accepted_dist_bds = None

        self.simulation_counter = 0


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, alpha = 0.1, acceptance_cutoff = 0.03, covFactor = 2.0, full_output=0, journal_file = None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        alpha : float, optional
            A parameter taking values between [0,1], the default value is 0.1.
        acceptance_cutoff : float, optional
            Acceptance ratio cutoff, should be chosen between 0.01 and 0.03
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.sample_from_prior(rng=self.rng)

        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.alpha = alpha
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if(journal_file is None):
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = type(self.distance).__name__
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["steps"] = steps
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mats = None
        accepted_dist = None
        alpha_accepted_parameters = None
        alpha_accepted_weights = None
        alpha_accepted_dist = None

        # main APMCABC algorithm
        # print("INFO: Starting APMCABC iterations.")
        for aStep in range(steps):
            self.logger.info("APMCABC iteration {}".format(aStep))
            if(aStep==0 and journal_file is not None):
                accepted_parameters=journal.get_accepted_parameters(-1)
                accepted_weights=journal.get_weights(-1)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters, accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)

                accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters, accepted_weights=accepted_weights)

                alpha_accepted_parameters=accepted_parameters
                alpha_accepted_weights=accepted_weights

            # 0: Drawing new new/perturbed samples using prior or MCMC Kernel
            if aStep > 0:
                n_additional_samples = n_samples - round(n_samples * alpha)
            else:
                n_additional_samples = n_samples

            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_additional_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # update remotely required variables
            self.logger.info("Broadcasting parameters")
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=alpha_accepted_parameters, accepted_weights=alpha_accepted_weights, accepted_cov_mats=accepted_cov_mats)
            self._update_broadcasts(alpha_accepted_dist)

            # calculate resample parameters
            self.logger.info("Resampling parameters")
            params_and_dist_weights_pds = self.backend.map(self._accept_parameter, rng_pds)
            params_and_dist_weights = self.backend.collect(params_and_dist_weights_pds)
            new_parameters, new_dist, new_weights, counter = [list(t) for t in zip(*params_and_dist_weights)]
            new_parameters = np.array(new_parameters)
            new_dist = np.array(new_dist)
            new_weights = np.array(new_weights).reshape(n_additional_samples, 1)

            for count in counter:
                self.simulation_counter+=count

            # 1: Update all parameters, compute acceptance probability, compute epsilon
            if len(new_weights) == n_samples:
                accepted_parameters = new_parameters
                accepted_dist = new_dist
                accepted_weights = new_weights
                # Compute acceptance probability
                prob_acceptance = 1
                # Compute epsilon
                epsilon = [np.percentile(accepted_dist, alpha * 100)]
            else:
                accepted_parameters = np.concatenate((alpha_accepted_parameters, new_parameters))
                accepted_dist = np.concatenate((alpha_accepted_dist, new_dist))
                accepted_weights = np.concatenate((alpha_accepted_weights, new_weights))
                # Compute acceptance probability
                prob_acceptance = sum(new_dist < epsilon[-1]) / len(new_dist)
                # Compute epsilon
                epsilon.append(np.percentile(accepted_dist, alpha * 100))

            # 2: Update alpha_parameters, alpha_dist and alpha_weights
            index_alpha = accepted_dist < epsilon[-1]
            alpha_accepted_parameters = accepted_parameters[index_alpha, :]
            alpha_accepted_weights = accepted_weights[index_alpha] / sum(accepted_weights[index_alpha])
            alpha_accepted_dist = accepted_dist[index_alpha]

            # 3: calculate covariance
            self.logger.info("Calculating covariance matrix")
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=alpha_accepted_parameters, accepted_weights=alpha_accepted_weights)

            kernel_parameters = []
            for kernel in self.kernel.kernels:
                kernel_parameters.append(
                    self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

            self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

            accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)

            accepted_cov_mats = [covFactor*cov_mat for cov_mat in accepted_cov_mats]

            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(accepted_dist))
                journal.add_weights(copy.deepcopy(accepted_weights))
                self.accepted_parameters_manager.update_broadcast(self.backend,
                                                                  accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            # 4: Check probability of acceptance lower than acceptance_cutoff
            if prob_acceptance < acceptance_cutoff:
                break

        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon

        return journal

    def _update_broadcasts(self, accepted_dist):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy
            self.accepted_dist_bds = self.backend.broadcast(accepted_dist)

    # define helper functions for map step
    def _accept_parameter(self, rng, npc=None):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        seed: integer
            Initial seed for the random number generator.

        Returns
        -------
        numpy.ndarray
            accepted parameter
        """

        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
            self.accepted_parameters_manager.model)

        counter = 0

        if self.accepted_parameters_manager.accepted_parameters_bds == None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            weight = 1.0
        else:
            index = rng.choice(len(self.accepted_parameters_manager.accepted_weights_bds.value()), size=1,
                               p=self.accepted_parameters_manager.accepted_weights_bds.value().reshape(-1))
            # trucate the normal to the bounds of parameter space of the model
            # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf
            while True:
                perturbation_output = self.perturb(index[0], rng=rng)
                if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                    break

            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            prior_prob = self.pdf_of_prior(self.model, perturbation_output[1])
            denominator = 0.0
            for i in range(len(self.accepted_parameters_manager.accepted_weights_bds.value())):
                pdf_value = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                            self.accepted_parameters_manager.accepted_parameters_bds.value()[index[0]], perturbation_output[1])
                denominator += self.accepted_parameters_manager.accepted_weights_bds.value()[i, 0] * pdf_value
            weight = 1.0 * prior_prob / denominator

        return (self.get_parameters(self.model), distance, weight, counter)


class SMCABC(BaseDiscrepancy, InferenceMethod):
    """This base class implements Adaptive Population Monte Carlo Approximate Bayesian computation of
    Del Moral et al. [1].

    [1] P. Del Moral, A. Doucet, A. Jasra, An adaptive sequential Monte Carlo method for approximate
    Bayesian computation. Statistics and Computing, 22(5):1009–1020, 2012.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distance : abcpy.distances.Distance
        Distance object defining the distance measure used to compare simulated and observed data sets.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """

    model = None
    distance = None
    kernel = None

    epsilon = None
    rng = None

    n_samples = None
    n_samples_per_param = None

    accepted_y_sim_bds = None

    backend = None

    def __init__(self, root_models, distances, backend, kernel = None, seed=None):
        self.model = root_models
        # We define the joint Linear combination distance using all the distances for each individual models
        self.distance = LinearCombination(root_models, distances)

        if (kernel is None):

            mapping, garbage_index = self._get_mapping()
            models = []
            for mdl, mdl_index in mapping:
                models.append(mdl)
            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        self.epsilon = None
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task\
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        self.accepted_y_sim_bds = None

        self.simulation_counter = 0


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, epsilon_final = 0.1, alpha = 0.95,
               covFactor = 2, resample = None, full_output=0, which_mcmc_kernel = 0, journal_file=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        epsilon_final : float, optional
            The final threshold value of epsilon to be reached. The default value is 0.1.
        alpha : float, optional
            A parameter taking values between [0,1], determinining the rate of change of the threshold epsilon. The
            default value is 0.95.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        which_mcmc_kernel: integer, optional
            Specifies which MCMC kernel to be used: '0' kernel suggestd in [1], any other value will use r-hit kernel
            suggested by Anthony Lee. The default value is 0.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.sample_from_prior(rng=self.rng)

        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if(journal_file is None):
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = type(self.distance).__name__
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["steps"] = steps
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mats = None
        accepted_y_sim = None

        # Define the resmaple parameter
        if resample == None:
            resample = n_samples * 0.5

        # Define epsilon_init
        epsilon = [10000]

        # main SMC ABC algorithm
        for aStep in range(0, steps):
            self.logger.info("SMCABC iteration {}".format(aStep))

            if(aStep==0 and journal_file is not None):
                accepted_parameters=journal.get_accepted_parameters(-1)
                accepted_weights=journal.get_weights(-1)
                accepted_y_sim = journal.opt_values[-1]

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)

                accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

            # Break if epsilon in previous step is less than epsilon_final
            if epsilon[-1] <= epsilon_final:
                break

            # 0: Compute the Epsilon
            if accepted_y_sim != None:
                self.logger.info("Compute epsilon, might take a while")
                # Compute epsilon for next step
                fun = lambda epsilon_var: self._compute_epsilon(epsilon_var, \
                                                                epsilon, observations, accepted_y_sim, accepted_weights,
                                                                n_samples, n_samples_per_param, alpha)
                epsilon_new = self._bisection(fun, epsilon_final, epsilon[-1], 0.001)
                if epsilon_new < epsilon_final:
                    epsilon_new = epsilon_final
                epsilon.append(epsilon_new)

            # 1: calculate weights for new parameters
            self.logger.info("Calculating weights")
            if accepted_y_sim != None:
                new_weights = np.zeros(shape=(n_samples), )
                for ind1 in range(n_samples):
                    numerator = 0.0
                    denominator = 0.0
                    for ind2 in range(n_samples_per_param):
                        numerator += (self.distance.distance(observations, [[accepted_y_sim[ind1][0][ind2]]]) < epsilon[-1])
                        denominator += (
                        self.distance.distance(observations, [[accepted_y_sim[ind1][0][ind2]]]) < epsilon[-2])
                    if denominator != 0.0:
                        new_weights[ind1] = accepted_weights[ind1] * (numerator / denominator)
                    else:
                        new_weights[ind1] = 0

                new_weights = new_weights / sum(new_weights)
            else:
                new_weights = np.ones(shape=(n_samples), ) * (1.0 / n_samples)

            # 2: Resample
            if accepted_y_sim != None and pow(sum(pow(new_weights, 2)), -1) < resample:
                self.logger.info("Resampling")
                # Weighted resampling:
                index_resampled = self.rng.choice(np.arange(n_samples), n_samples, replace=1, p=new_weights)
                accepted_parameters = accepted_parameters[index_resampled]
                new_weights = np.ones(shape=(n_samples), ) * (1.0 / n_samples)

            # Update the weights
            accepted_weights = new_weights.reshape(len(new_weights), 1)

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights)
            if(accepted_y_sim is not None):
                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)

                accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]

            # 3: Drawing new perturbed samples using MCMC Kernel
            self.logger.debug("drawing new pertubated samples using mcmc kernel")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            index_arr = np.arange(n_samples)
            rng_and_index_arr = np.column_stack((rng_arr, index_arr))
            rng_and_index_pds = self.backend.parallelize(rng_and_index_arr)

            # print("INFO: Broadcasting parameters.")
            self.epsilon = epsilon

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights, accepted_cov_mats=accepted_cov_mats)
            self._update_broadcasts(accepted_y_sim)

            # calculate resample parameters
            self.logger.info("Drawing perturbed sampless")
            if which_mcmc_kernel == 0:
                params_and_ysim_pds = self.backend.map(self._accept_parameter, rng_and_index_pds)
            else:
                params_and_ysim_pds = self.backend.map(self._accept_parameter_r_hit_kernel, rng_and_index_pds)
            params_and_ysim = self.backend.collect(params_and_ysim_pds)
            new_parameters, new_y_sim, distances, counter = [list(t) for t in zip(*params_and_ysim)]
            distances = np.array(distances)

            for count in counter:
                self.simulation_counter+=count

            # Update the parameters
            accepted_parameters = new_parameters
            accepted_y_sim = new_y_sim

            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                self.logger.info("Saving configuration to output journal")
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(distances))
                journal.add_weights(copy.deepcopy(accepted_weights))
                journal.add_opt_values(copy.deepcopy(accepted_y_sim))

                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon

        return journal

    def _compute_epsilon(self, epsilon_new, epsilon, observations, accepted_y_sim, accepted_weights, n_samples,
                         n_samples_per_param, alpha):
        """
        Parameters
        ----------
        epsilon_new: float
            New value for epsilon.
        epsilon: float
            Current threshold.
        observations: numpy.ndarray
            Observed data.
        accepted_y_sim: numpy.ndarray
            Accepted simulated data.
        accepted_weights: numpy.ndarray
            Accepted weights.
        n_samples: integer
            Number of samples to generate.
        n_samples_per_param: integer
            Number of data points in each simulated data set.
        alpha: float

        Returns
        -------
        float
            Newly computed value for threshold.
        """

        RHS = alpha * pow(sum(pow(accepted_weights, 2)), -1)
        LHS = np.zeros(shape=(n_samples), )
        for ind1 in range(n_samples):
            numerator = 0.0
            denominator = 0.0
            for ind2 in range(n_samples_per_param):
                numerator += (self.distance.distance(observations, [[accepted_y_sim[ind1][0][ind2]]]) < epsilon_new)
                denominator += (self.distance.distance(observations, [[accepted_y_sim[ind1][0][ind2]]]) < epsilon[-1])
            if(denominator==0):
                LHS[ind1]=0
            else:
                LHS[ind1] = accepted_weights[ind1] * (numerator / denominator)
        if sum(LHS) == 0:
            result = RHS
        else:
            LHS = LHS / sum(LHS)
            LHS = pow(sum(pow(LHS, 2)), -1)
            result = RHS - LHS
        return (result)


    def _bisection(self, func, low, high, tol):
        # cache computed values, as we call func below
        # several times for the same argument:
        func = cached(func)
        midpoint = (low + high) / 2.0
        while (high - low) / 2.0 > tol:
            self.logger.debug("bisection: distance = {:e} > tol = {:e}"
                              .format((high - low) / 2, tol))
            if func(midpoint) == 0:
                return midpoint
            elif func(low) * func(midpoint) < 0:
                high = midpoint
            else:
                low = midpoint
            midpoint = (low + high) / 2.0

        return midpoint

    def _update_broadcasts(self, accepted_y_sim):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy
        if not accepted_y_sim is None:
            self.accepted_y_sim_bds = self.backend.broadcast(accepted_y_sim)

            # define helper functions for map step

    def _accept_parameter(self, rng_and_index, npc=None):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        seed_and_index: numpy.ndarray
            2 dimensional array. The first entry specifies the initial seed for the random number generator.
            The second entry defines the index in the data set.

        Returns
        -------
        Tuple
            The first entry of the tuple is the accepted parameters. The second entry is the simulated data set.
        """

        rng = rng_and_index[0]
        index = rng_and_index[1]
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
            self.accepted_parameters_manager.model)

        counter=0
        # print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
        if self.accepted_parameters_manager.accepted_parameters_bds is None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
        else:
            if self.accepted_parameters_manager.accepted_weights_bds.value()[index] > 0:
                theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]
                while True:
                    perturbation_output = self.perturb(index, rng=rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                        break
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter+=1
                y_sim_old = self.accepted_y_sim_bds.value()[index]
                ## Calculate acceptance probability:
                numerator = 0.0
                denominator = 0.0
                for ind in range(self.n_samples_per_param):
                    numerator += (self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                                         [[y_sim[0][ind]]]) < self.epsilon[-1])
                    denominator += (self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                                           [[y_sim_old[0][ind]]]) < self.epsilon[-1])
                if denominator == 0:
                    ratio_data_epsilon = 1
                else:
                    ratio_data_epsilon = numerator / denominator

                ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model,
                                                                                                             theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, perturbation_output[1], theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta, perturbation_output[1])
                ratio_likelihood_prob = kernel_numerator / kernel_denominator

                acceptance_prob = min(1, ratio_data_epsilon * ratio_prior_prob * ratio_likelihood_prob)
                if rng.binomial(1, acceptance_prob) == 1:
                    self.set_parameters(perturbation_output[1])
                else:
                    self.set_parameters(theta)
                    y_sim = self.accepted_y_sim_bds.value()[index]
            else:
                self.set_parameters(self.accepted_parameters_manager.accepted_parameters_bds.value()[index])
                y_sim = self.accepted_y_sim_bds.value()[index]
        distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
        return (self.get_parameters(), y_sim, distance, counter)

    def _accept_parameter_r_hit_kernel(self, rng_and_index, npc=None):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        seed_and_index: numpy.ndarray
            2 dimensional array. The first entry specifies the initial seed for the random number generator.
            The second entry defines the index in the data set.

        Returns
        -------
        Tuple
            The first entry of the tuple is the accepted parameters. The second entry is the simulated data set.
        """

        rng = rng_and_index[0]
        index = rng_and_index[1]
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        # Set value of r for r-hit kernel
        r = 3
        mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(self.accepted_parameters_manager.model)

        counter=0
        # print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
        if self.accepted_parameters_manager.accepted_parameters_bds is None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter+=1
        else:
            if self.accepted_parameters_manager.accepted_weights_bds.value()[index] > 0:
                theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]

                # Sample from theta until we get 'r-1' y_sim inside the epsilon ball
                self.set_parameters(theta)
                accept_old_arr, y_sim_old_arr, N_old = [], [], 0
                while sum(accept_old_arr) < r-1:
                    y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                    y_sim_old_arr.append(y_sim)
                    if self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                              y_sim) < self.epsilon[-1]:
                        accept_old_arr.append(N_old)
                    N_old += 1
                    counter += 1

                # Perturb and sample from the perturbed theta until we get 'r' y_sim inside the epsilon ball
                while True:
                    perturbation_output = self.perturb(index, rng=rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                        break
                accept_new_arr, y_sim_new_arr, N = [], [], 0
                while sum(accept_new_arr) < r:
                    y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                    y_sim_new_arr.append(y_sim)
                    if self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                              y_sim) < self.epsilon[-1]:
                        accept_new_arr.append(N)
                    counter += 1
                    N += 1

                #Calculate acceptance probability
                ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model,
                                                                                                             theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, perturbation_output[1], theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta, perturbation_output[1])
                ratio_likelihood_prob = kernel_numerator / kernel_denominator

                acceptance_prob = min(1, (N_old/(N-1)) * ratio_prior_prob * ratio_likelihood_prob)

                if rng.binomial(1, acceptance_prob) == 1:
                    self.set_parameters(perturbation_output[1])
                    # Randomly sample index J
                    J = rng.choice(accept_new_arr).astype(int)
                    y_sim = y_sim_new_arr[J]
                else:
                    self.set_parameters(theta)
                    y_sim = self.accepted_y_sim_bds.value()[index]
            else:
                self.set_parameters(self.accepted_parameters_manager.accepted_parameters_bds.value()[index])
                y_sim = self.accepted_y_sim_bds.value()[index]
        distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
        return (self.get_parameters(), y_sim, distance, counter)