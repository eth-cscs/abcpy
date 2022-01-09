import copy
import logging
import time
import warnings
from abc import abstractproperty

import numpy as np
from scipy import optimize
from tqdm import tqdm

from abcpy.acceptedparametersmanager import *
from abcpy.backends import BackendDummy
from abcpy.distances import Divergence
from abcpy.graphtools import GraphTools
from abcpy.jointapprox_lhd import SumCombination
from abcpy.jointdistances import LinearCombination
from abcpy.output import Journal
from abcpy.perturbationkernel import DefaultKernel, JointPerturbationKernel
from abcpy.probabilisticmodels import *
from abcpy.utils import cached
from abcpy.transformers import BoundedVarTransformer, DummyTransformer


class InferenceMethod(GraphTools, metaclass=ABCMeta):
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


class BaseMethodsWithKernel(metaclass=ABCMeta):
    """
    This abstract base class represents inference methods that have a kernel.
    """

    @abstractproperty
    def kernel(self):
        """To be overwritten by any sub-class: an attribute specifying the transition or perturbation kernel."""
        raise NotImplementedError

    def perturb(self, column_index, epochs=100, rng=np.random.RandomState(), accepted_parameters_manager=None):
        """
        Perturbs all free parameters, given the current weights.
        Commonly used during inference.

        Parameters
        ----------
        column_index: integer
            The index of the column in the accepted_parameters_bds that should be used for perturbation
        epochs: integer
            The number of times perturbation should happen before the algorithm is terminated
        accepted_parameters_manager: AcceptedParametersManager
            The AcceptedParametersManager to use; if not provided, use the one stored in
            self.accepted_parameters_manager

        Returns
        -------
        boolean
            Whether it was possible to set new parameter values for all probabilistic models
        """
        current_epoch = 0

        if accepted_parameters_manager is None:
            accepted_parameters_manager = self.accepted_parameters_manager

        while current_epoch < epochs:

            # Get new parameters of the graph
            new_parameters = self.kernel.update(accepted_parameters_manager, column_index, rng=rng)

            self._reset_flags()

            # Order the parameters provided by the kernel in depth-first search order
            correctly_ordered_parameters = self.get_correct_ordering(new_parameters)

            # Try to set new parameters
            accepted, last_index = self.set_parameters(correctly_ordered_parameters, 0)
            if accepted:
                break
            current_epoch += 1

        if current_epoch == 10:
            return [False]

        return [True, correctly_ordered_parameters]


class BaseLikelihood(InferenceMethod, BaseMethodsWithKernel, metaclass=ABCMeta):
    """
    This abstract base class represents inference methods that use the likelihood.
    """

    @abstractproperty
    def likfun(self):
        """To be overwritten by any sub-class: an attribute specifying the likelihood function to be used."""
        raise NotImplementedError


class BaseDiscrepancy(InferenceMethod, BaseMethodsWithKernel, metaclass=ABCMeta):
    """
    This abstract base class represents inference methods using descrepancy.
    """

    @abstractproperty
    def distance(self):
        """To be overwritten by any sub-class: an attribute specifying the distance function."""
        raise NotImplementedError


class DrawFromPrior(InferenceMethod):
    """Helper class to obtain samples from the prior for a model.

    The `sample` method follows similar API to the other InferenceMethod's (returning a journal file), while the
    `sample_par_sim_pairs` method generates (parameter, simulation) pairs, which can be used for instance as a training
    dataset for the automatic learning of summary statistics with the StatisticsLearning classes.

    When generating large datasets with MPI backend, pickling may give overflow error; for this reason, the methods
    split the generation in "chunks" of the specified size on which the parallelization is used.

    Parameters
    ----------
    root_models: list
        A list of the Probabilistic models corresponding to the observed datasets
    backend: abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed: integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    max_chunk_size: integer, optional
        Maximum size of chunks in which to split the data generation. Defaults to 10**4
    discard_too_large_values: boolean
         If set to True, the simulation is discarded (and repeated) if at least one element of it is too large
         to fit in float32, which therefore may be converted to infinite value in numpy. Defaults to False.
    """

    model = None
    rng = None
    n_samples = None
    backend = None

    n_samples_per_param = None  # this needs to be there otherwise it does not instantiate correctly

    def __init__(self, root_models, backend, seed=None, max_chunk_size=10 ** 4, discard_too_large_values=False):
        self.model = root_models
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.max_chunk_size = max_chunk_size
        self.discard_too_large_values = discard_too_large_values
        # An object managing the bds objects
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        self.logger = logging.getLogger(__name__)

    def sample(self, n_samples, path_to_save_journal=None):
        """
        Samples model parameters from the prior distribution.

        Parameters
        ----------
        n_samples: integer
            Number of samples to generate
        path_to_save_journal: str, optional
            If provided, save the journal after inference at the provided path.

        Returns
        -------
        abcpy.output.Journal
            a journal containing results and metadata.
        """

        journal = Journal(1)
        journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
        journal.configuration["n_samples"] = n_samples

        # we split sampling in chunks to avoid error in case MPI is used
        parameters = []
        samples_to_sample = n_samples
        while samples_to_sample > 0:
            parameters_part = self._sample(min(samples_to_sample, self.max_chunk_size))
            samples_to_sample -= self.max_chunk_size
            parameters += parameters_part

        journal.add_accepted_parameters(copy.deepcopy(parameters))
        journal.add_weights(np.ones((n_samples, 1)))
        journal.add_ESS_estimate(np.ones((n_samples, 1)))
        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=parameters)
        names_and_parameters = self._get_names_and_parameters()
        journal.add_user_parameters(names_and_parameters)
        journal.number_of_simulations.append(0)

        if path_to_save_journal is not None:  # save journal
            journal.save(path_to_save_journal)

        return journal

    def sample_par_sim_pairs(self, n_samples, n_samples_per_param):
        """
        Samples (parameter, simulation) pairs from the prior distribution from the model distribution. Specifically,
        parameter values are sampled from the prior and used to generate the specified number of simulations per
        parameter value. This returns arrays.

        Parameters
        ----------
        n_samples: integer
            Number of samples to generate
        n_samples_per_param: integer
            Number of data points in each simulated data set.

        Returns
        -------
        tuple
            A tuple of numpy.ndarray's containing parameter and simulation values. The first element of the tuple is an
            array with shape (n_samples, d_theta), where d_theta is the dimension of the parameters. The second element
            of the tuple is an array with shape (n_samples, n_samples_per_param, d_x), where d_x is the dimension of
            each simulation.
        """

        # we split sampling in chunks to avoid error in case MPI is used
        parameters_list = []
        simulations_list = []
        samples_to_sample = n_samples
        while samples_to_sample > 0:
            parameters_part, simulations_part = self._sample_par_sim_pairs(min(samples_to_sample, self.max_chunk_size),
                                                                           n_samples_per_param)
            samples_to_sample -= self.max_chunk_size
            parameters_list.append(parameters_part)
            simulations_list.append(simulations_part)
        parameters = np.concatenate(parameters_list)
        simulations = np.concatenate(simulations_list)
        return parameters, simulations

    def _sample(self, n_samples):
        """
        Not for end use; please use `sample`.

        Samples model parameters from the prior distribution. This is an helper function called by the main `sample` one
        in order to split drawing from the prior in chunks to avoid parallelization issues with MPI.

        Parameters
        ----------
        n_samples: integer
            Number of samples to generate

        Returns
        -------
        list
            List containing sampled parameter values.
        """
        # the following lines are similar to the RejectionABC code but only sample from the prior.

        # generate the rng_pds
        rng_pds = self._generate_rng_pds(n_samples)

        parameters_pds = self.backend.map(self._sample_parameter_only, rng_pds)
        parameters = self.backend.collect(parameters_pds)
        return parameters

    def _sample_par_sim_pairs(self, n_samples, n_samples_per_param):
        """
        Not for end use; please use `sample_par_sim_pairs`.

        Samples (parameter, simulation) pairs from the prior distribution from the model distribution. Specifically,
        parameter values are sampled from the prior and used to generate the specified number of simulations per
        parameter value. This returns arrays.

        This is an helper function called by the main `sample_par_sim_pair` one
        in order to split drawing from the prior in chunks to avoid parallelization issues with MPI.

        Parameters
        ----------
        n_samples: integer
            Number of samples to generate
        n_samples_per_param: integer
            Number of data points in each simulated data set.

        Returns
        -------
        tuple
            A tuple of numpy.ndarray's containing parameter and simulation values. The first element of the tuple is an
            array with shape (n_samples, d_theta), where d_theta is the dimension of the parameters. The second element
            of the tuple is an array with shape (n_samples, n_samples_per_param, d_x), where d_x is the dimension of
            each simulation.
        """
        self.n_samples_per_param = n_samples_per_param
        self.accepted_parameters_manager.broadcast(self.backend, 1)

        # generate the rng_pds
        rng_pds = self._generate_rng_pds(n_samples)

        parameters_simulations_pds = self.backend.map(self._sample_parameter_simulation, rng_pds)
        parameters_simulations = self.backend.collect(parameters_simulations_pds)
        parameters, simulations = [list(t) for t in zip(*parameters_simulations)]

        parameters = np.array(parameters)
        simulations = np.array(simulations)

        parameters = parameters.reshape((parameters.shape[0], parameters.shape[1]))
        simulations = simulations.reshape((simulations.shape[0], simulations.shape[2], simulations.shape[3],))

        return parameters, simulations

    def _generate_rng_pds(self, n_samples):
        """Helper function to generate the random seeds which are used in sampling from prior and simulating from the
        model in the parallel setup.

        Parameters
        ----------
        n_samples: integer
            Number of random seeds (corresponing to number of prior samples) to generate

        Returns
        -------
        list
            A (possibly distributed according to the used backend) list containing the random seeds to be assigned to
            each worker.
        """
        # now generate an array of seeds that need to be different one from the other. One way to do it is the
        # following.
        # Moreover, you cannot use int64 as seeds need to be < 2**32 - 1. How to fix this?
        # Note that this is not perfect; you still have small possibility of having some seeds that are equal. Is there
        # a better way? This would likely not change much the performance
        # An idea would be to use rng.choice but that is too expensive
        seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
        # check how many equal seeds there are and remove them:
        sorted_seed_arr = np.sort(seed_arr)
        indices = sorted_seed_arr[:-1] == sorted_seed_arr[1:]
        if np.sum(indices) > 0:
            # the following removes the equal seeds in case there are some
            sorted_seed_arr[:-1][indices] = sorted_seed_arr[:-1][indices] + 1
        rng_arr = np.array([np.random.RandomState(seed) for seed in sorted_seed_arr])
        rng_pds = self.backend.parallelize(rng_arr)
        return rng_pds

    def _sample_parameter_simulation(self, rng, npc=None):
        """
        Samples a single model parameter and simulates from it.

        Parameters
        ----------
        rng: random number generator
            The random number generator to be used.
        Returns
        -------
        Tuple
            The first entry of the tuple is the parameter.
            The second entry is the the simulation drawn from it.
        """

        ok_flag = False

        while not ok_flag:
            self.sample_from_prior(rng=rng)
            theta = self.get_parameters(self.model)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)

            # if there are no potential infinities there (or if we do not check for those).
            # For instance, Lorenz model may give too large values sometimes (quite rarely).
            if self.discard_too_large_values and np.sum(np.isinf(np.array(y_sim).astype("float32"))) > 0:
                self.logger.warning("y_sim contained too large values for float32; simulating again.")
            else:
                ok_flag = True

        return theta, y_sim

    def _sample_parameter_only(self, rng, npc=None):
        """
        Samples a single model parameter from the prior.

        Parameters
        ----------
        rng: random number generator
            The random number generator to be used.
        Returns
        -------
        list
            The sampled parameter values
        """

        self.sample_from_prior(rng=rng)
        theta = self.get_parameters(self.model)

        return theta


class RejectionABC(InferenceMethod):
    """This class implements the rejection algorithm based inference scheme [1] for
        Approximate Bayesian Computation.

        [1] Tavaré, S., Balding, D., Griffith, R., Donnelly, P.: Inferring coalescence
        times from DNA sequence data. Genetics 145(2), 505–518 (1997).

        Parameters
        ----------
        root_models: list
            A list of the Probabilistic models corresponding to the observed datasets
        distances: list of abcpy.distances.Distance
            List of Distance objects defining the distance measure to compare simulated and observed data sets; one for 
            each model.
        backend: abcpy.backends.Backend
            Backend object defining the backend to be used.
        seed: integer, optional
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

    def sample(self, observations, n_samples=100, n_samples_per_param=1, epsilon=None, simulation_budget=None,
               quantile=None, full_output=0, path_to_save_journal=None):
        """
        Samples from the posterior distribution of the model parameter given the observed
        data observations.

        You can either specify the required number of posterior samples `n_samples`, or the
        `simulation_budget`. In the former case, the threshold value `epsilon` is required, so that the algorithm
        will produce `n_samples` posterior samples for which the ABC distance was smaller than `epsilon`. In the latter
        case, you can specify either `epsilon` or `quantile`; in this case, the number of simulations specified in the
        simulation budget will be run, and only the parameter values for which the ABC distance was smaller than
        `epsilon` (or alternatively the ones for which the ABC distance is in the smaller specified quantile) will
        be returned.

        Parameters
        ----------
        observations: list
            A list, containing lists describing the observed data sets; one for each model.
        n_samples: integer
            Number of samples to generate
        n_samples_per_param: integer
            Number of data points in each simulated data set.
        epsilon: float
            Value of threshold
        simulation_budget : integer, optional
            Simulation budget to be considered (ie number of parameter values for which the ABC distance is computed).
            Alternative to `n_samples`, which needs to be set explicitly to None to use this. Defaults to None.
        quantile : float, optional
            If `simulation_budget` is used, only the samples which achieve performance less than the specified quantile
            of the ABC distances, will be retained in the set of posterior samples. This is alternative to epsilon.
            Defaults to None.
        full_output: integer, optional
            It is actually unused in RejectionABC but left here for general compatibility with the other inference
            classes.
        path_to_save_journal: str, optional
            If provided, save the journal after inference at the provided path.
            
        Returns
        -------
        abcpy.output.Journal
            a journal containing simulation results, metadata and optionally intermediate results.
        """

        if (n_samples is None) == (simulation_budget is None):
            raise RuntimeError("One and only one of `n_samples` and `simulation_budget` needs to be specified.")
        if n_samples is not None and quantile is not None:
            raise RuntimeError("`quantile` can be specified only when the simulation budget is fixed with "
                               "`simulation_budget`.")
        if n_samples is not None and epsilon is None:
            raise RuntimeError("`epsilon` needs to be specified when the `n_samples` is given ")
        if simulation_budget is not None and ((quantile is None) == (epsilon is None)):
            raise RuntimeError("One and only one of `quantile` and `epsilon` needs to be specified when "
                               "`simulation_budget` is used.")

        self.fixed_budget = True if simulation_budget is not None else False

        self.accepted_parameters_manager.broadcast(self.backend, observations)

        self.n_samples_per_param = n_samples_per_param

        # instantiate journal (common to both approaches); this will be overwritten if self.fixed_budget is True
        journal = Journal(full_output)
        journal.configuration["n_samples"] = n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["epsilon"] = epsilon
        journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
        journal.configuration["type_dist_func"] = [type(distance).__name__ for distance in self.distance.distances]
        journal.configuration["full_output"] = full_output

        if self.fixed_budget:
            # sample simulation_budget number of samples with super high epsilon
            n_samples = simulation_budget
            self.epsilon = 1.7976931348623157e+308  # max possible numpy value

            # call sample routine
            journal = self._sample_n_samples_epsilon(n_samples, journal)

            # then replace that journal with a new one selecting the correct samples only, with either epsilon or
            # quantile (only one of them will not be None)
            journal = self._journal_cleanup(journal, quantile, epsilon)
        else:
            self.epsilon = epsilon

            # call sample routine
            journal = self._sample_n_samples_epsilon(n_samples, journal)

        if path_to_save_journal is not None:
            # save journal there
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else \
                path_to_save_journal + '.jnl'
            journal.save(path_to_save_journal)

        return journal

    def _sample_n_samples_epsilon(self, n_samples, journal):
        """Obtains `n_samples` posterior samples with threshold `self.epsilon`, and stores them into the journal.
        Parameters
        ----------
        n_samples: integer
            Number of samples to generate
        journal: abcpy.output.Journal
            Journal file where to store results
        Returns
        -------
        abcpy.output.Journal
            a journal containing simulation results, metadata and optionally intermediate results.
        """
        # main Rejection ABC algorithm
        seed_arr = self.rng.randint(1, n_samples * n_samples, size=n_samples, dtype=np.int32)
        rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
        rng_pds = self.backend.parallelize(rng_arr)

        accepted_parameters_distances_counter_pds = self.backend.map(self._sample_parameter, rng_pds)
        accepted_parameters_distances_counter = self.backend.collect(accepted_parameters_distances_counter_pds)
        accepted_parameters, distances, counter = [list(t) for t in zip(*accepted_parameters_distances_counter)]

        for count in counter:
            self.simulation_counter += count

        distances = np.array(distances)

        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)
        journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
        journal.add_weights(np.ones((n_samples, 1)))
        journal.add_ESS_estimate(np.ones((n_samples, 1)))
        journal.add_distances(copy.deepcopy(distances))
        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)
        names_and_parameters = self._get_names_and_parameters()
        journal.add_user_parameters(names_and_parameters)
        journal.number_of_simulations.append(self.simulation_counter)

        return journal

    def _journal_cleanup(self, journal, quantile=None, threshold=None):
        """This function takes a Journal file (typically produced by an Rejection ABC run with very large epsilon value)
        and keeps only the samples which achieve performance less than either some quantile of the ABC distances,
        or either some specified threshold. It is a very simple way to obtain a Rejection ABC which
        works on a percentile of the obtained distances.

        It creates a new Journal file storing the results.

        Parameters
        ----------
        journal: abcpy.output.Journal
            Journal file where to store results
        quantile : float, optional
            If `simulation_budget` is used, only the samples which achieve performance less than the specified quantile
            of the ABC distances, will be retained in the set of posterior samples. This is alternative to epsilon.
            Defaults to None.
        threshold: float
            Value of threshold

        Returns
        -------
        abcpy.output.Journal
            a new journal containing simulation results, metadata and optionally intermediate results.
        """

        if quantile is not None:
            distance_cutoff = np.quantile(journal.distances[-1], quantile)
        else:
            distance_cutoff = threshold
        picked_simulations = journal.distances[-1] < distance_cutoff
        new_distances = journal.distances[-1][picked_simulations]
        n_reduced_samples = np.sum(picked_simulations)
        if n_reduced_samples == 0:
            raise RuntimeError(
                "The specified value of threshold is too low, no simulations from the ones generated with the fixed "
                "simulation budget are accepted."
            )
        new_journal = Journal(journal._type)
        new_journal.configuration["n_samples"] = n_reduced_samples
        new_journal.configuration["n_samples_per_param"] = journal.configuration[
            "n_samples_per_param"
        ]
        new_journal.configuration["epsilon"] = distance_cutoff

        new_accepted_parameters = []
        param_names = journal.get_parameters().keys()
        new_names_and_parameters = {name: [] for name in param_names}
        for i in np.where(picked_simulations)[0]:
            if picked_simulations[i]:
                new_accepted_parameters.append(journal.get_accepted_parameters()[i])
                for name in param_names:
                    new_names_and_parameters[name].append(journal.get_parameters()[name][i])

        new_journal.add_accepted_parameters(new_accepted_parameters)
        new_journal.add_weights(np.ones((n_reduced_samples, 1)))
        new_journal.add_ESS_estimate(np.ones((n_reduced_samples, 1)))
        new_journal.add_distances(new_distances)
        new_journal.add_user_parameters(new_names_and_parameters)
        new_journal.number_of_simulations.append(journal.number_of_simulations[-1])

        return new_journal

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
        Tuple
            The first entry of the tuple is the accepted parameters.
            The second entry is the distance between the simulated data set and the observation, while the third one is
            the number of simulations needed to obtain the accepted parameter.
        """
        distance = self.distance.dist_max()

        if distance < self.epsilon and self.logger:
            self.logger.warning("initial epsilon {:e} is larger than dist_max {:e}"
                                .format(float(self.epsilon), distance))

        counter = 0

        while distance > self.epsilon:
            # Accept new parameter value if the distance is less than epsilon
            self.sample_from_prior(rng=rng)
            theta = self.get_parameters(self.model)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter += 1
            if y_sim is not None:
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
                self.logger.debug("distance after {:4d} simulations: {:e}".format(
                    counter, distance))
            else:
                distance = self.distance.dist_max()
        self.logger.debug("Needed {:4d} simulations to reach distance {:e} < epsilon = {:e}".format(counter, distance,
                                                                                                    float(
                                                                                                        self.epsilon)))
        return theta, distance, counter


class PMCABC(BaseDiscrepancy, InferenceMethod):
    """
    This class implements a modified version of Population Monte Carlo based inference scheme for Approximate
    Bayesian computation of Beaumont et. al. [1]. Here the threshold value at `t`-th generation are adaptively chosen by
    taking the maximum between the epsilon_percentile-th value of discrepancies of the accepted parameters at `t-1`-th
    generation and the threshold value provided for this generation by the user. If we take the value of
    epsilon_percentile to be zero (default), this method becomes the inference scheme described in [1], where the
    threshold values considered at each generation are the ones provided by the user.

    [1] M. A. Beaumont. Approximate Bayesian computation in evolution and ecology. Annual Review of Ecology,
    Evolution, and Systematics, 41(1):379–406, Nov. 2010.

    Parameters
    ----------
    root_models: list
        A list of the Probabilistic models corresponding to the observed datasets
    distances: list of abcpy.distances.Distance
        List of Distance objects defining the distance measure to compare simulated and observed data sets; one for 
        each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """

    model = None
    distance = None
    kernel = None
    rng = None

    # default value, set so that testing works
    n_samples = 2
    n_samples_per_param = None

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
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

        self.simulation_counter = 0

    def sample(self, observations, steps, epsilon_init, n_samples=10000, n_samples_per_param=1, epsilon_percentile=10,
               covFactor=2, full_output=0, journal_file=None, path_to_save_journal=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        steps : integer
            Number of iterations in the sequential algorithm ("generations")
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
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path. The journal is saved (and overwritten) after each step
            of the sequential inference routine, so that partial results are stored to the disk in case the
            inference routine does not end correctly; recall that you need to set full_output=1 to obtain the
            full partial results.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if path_to_save_journal is not None:
            # save journal there
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if journal_file is None:
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = [type(distance).__name__ for distance in self.distance.distances]
            journal.configuration["steps"] = steps
            journal.configuration["epsilon_init"] = epsilon_init
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["epsilon_percentile"] = epsilon_percentile
            journal.configuration["covFactor"] = covFactor
            journal.configuration["full_output"] = full_output
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mats = None

        # Define epsilon_arr
        if len(epsilon_init) == steps:
            epsilon_arr = epsilon_init
        elif len(epsilon_init) == 1:
            epsilon_arr = [None] * steps
            epsilon_arr[0] = epsilon_init[0]
        else:
            raise ValueError("The length of epsilon_init can only be equal to 1 or steps.")

        # main PMCABC algorithm
        self.logger.info("Starting PMC iterations")
        for aStep in range(steps):
            self.logger.debug("iteration {} of PMC algorithm".format(aStep))
            if aStep == 0 and journal_file is not None:
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                if hasattr(journal, "distances"):
                    # if restarting from a journal, use the previous distances to check determine a new epsilon
                    # (it if is larger than the epsilon_arr[0] provided here)
                    self.logger.info("Calculating acceptances threshold from provided journal file")
                    epsilon_arr[0] = np.max([np.percentile(journal.distances[-1], epsilon_percentile), epsilon_arr[0]])

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                # 3: calculate covariance
                self.logger.info("Calculating covariance matrix")
                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                # Since each entry of new_cov_mats is a numpy array, we can multiply like this
                # accepted_cov_mats = [covFactor * new_cov_mat for new_cov_mat in new_cov_mats]
                accepted_cov_mats = self._compute_accepted_cov_mats(covFactor, new_cov_mats)

            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            self.logger.info("Broadcasting parameters")
            self.epsilon = epsilon_arr[aStep]
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters, accepted_weights,
                                                              accepted_cov_mats)

            # 1: calculate resample parameters
            # print("INFO: Resampling parameters")
            self.logger.info("Resampling parameters")

            params_and_dists_and_counter_pds = self.backend.map(self._resample_parameter, rng_pds)
            params_and_dists_and_counter = self.backend.collect(params_and_dists_and_counter_pds)
            new_parameters, distances, counter = [list(t) for t in zip(*params_and_dists_and_counter)]
            new_parameters = np.array(new_parameters)
            distances = np.array(distances)

            for count in counter:
                self.simulation_counter += count

            # Compute epsilon for next step
            # print("INFO: Calculating acceptance threshold (epsilon).")
            self.logger.info("Calculating acceptances threshold")
            if aStep < steps - 1:
                if epsilon_arr[aStep + 1] is None:
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
            sum_of_weights = np.sum(new_weights)
            new_weights = new_weights / sum_of_weights

            # The calculation of cov_mats needs the new weights and new parameters
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=new_parameters,
                                                              accepted_weights=new_weights)

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
            # new_cov_mats = [covFactor * new_cov_mat for new_cov_mat in new_cov_mats]
            new_cov_mats = self._compute_accepted_cov_mats(covFactor, new_cov_mats)

            # 4: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mats = new_cov_mats

            self.logger.info("Save configuration to output journal")

            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(distances))
                journal.add_weights(copy.deepcopy(accepted_weights))
                journal.add_ESS_estimate(accepted_weights)
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            # Add epsilon_arr to the journal
            if journal_file is not None and "epsilon_arr" in journal.configuration.keys():
                journal.configuration["epsilon_arr"] += epsilon_arr
            else:
                journal.configuration["epsilon_arr"] = epsilon_arr

            if path_to_save_journal is not None:  # save journal
                journal.save(path_to_save_journal)

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
        Tuple
            The first entry of the tuple is the accepted parameters.
            The second entry is the distance between the simulated data set and the observation, while the third one is
            the number of simulations needed to obtain the accepted parameter.
        """

        # print(npc.communicator())
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        distance = self.distance.dist_max()

        if distance < self.epsilon and self.logger:
            self.logger.warn("initial epsilon {:e} is larger than dist_max {:e}"
                             .format(float(self.epsilon), distance))

        theta = self.get_parameters()
        counter = 0

        while distance > self.epsilon:
            if self.accepted_parameters_manager.accepted_parameters_bds is None:
                self.sample_from_prior(rng=rng)
                theta = self.get_parameters()
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter += 1

            else:
                index = rng.choice(self.n_samples, size=1,
                                   p=self.accepted_parameters_manager.accepted_weights_bds.value().reshape(-1))
                # truncate the normal to the bounds of parameter space of the model
                # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf
                while True:
                    perturbation_output = self.perturb(index[0], rng=rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                        theta = perturbation_output[1]
                        break
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter += 1

            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            self.logger.debug("distance after {:4d} simulations: {:e}".format(counter, distance))

        self.logger.debug("Needed {:4d} simulations to reach distance {:e} < epsilon = {:e}".format(counter, distance,
                                                                                                    float(
                                                                                                        self.epsilon)))

        return theta, distance, counter

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

            # Get the mapping of the models to be used by the kernels
            mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
                self.accepted_parameters_manager.model)
            pdf_values = np.array([self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                                   self.accepted_parameters_manager.accepted_parameters_bds.value()[i],
                                                   theta) for i in range(self.n_samples)])
            denominator = np.sum(self.accepted_parameters_manager.accepted_weights_bds.value().reshape(-1) * pdf_values)

            return 1.0 * prior_prob / denominator

    def _compute_accepted_cov_mats(self, covFactor, new_cov_mats):
        """
        Update the covariance matrices computed from data by multiplying them with covFactor and adding a small term in
        the diagonal for numerical stability.

        Parameters
        ----------
        covFactor : float
            factor to correct the covariance matrices
        new_cov_mats : list
            list of covariance matrices computed from data
        Returns
        -------
        list
            List of new accepted covariance matrices
        """
        # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]
        accepted_cov_mats = []
        for new_cov_mat in new_cov_mats:
            if not (new_cov_mat.size == 1):
                accepted_cov_mats.append(
                    covFactor * new_cov_mat + 1e-20 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
            else:
                accepted_cov_mats.append((covFactor * new_cov_mat + 1e-20 * new_cov_mat).reshape(1, 1))
        return accepted_cov_mats


class PMC(BaseLikelihood, InferenceMethod):
    """
    Population Monte Carlo based inference scheme of Cappé et. al. [1].

    This algorithm assumes a likelihood function is available and can be evaluated
    at any parameter value given the observed dataset.  In absence of the
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
    root_models : list
        A list of the Probabilistic models corresponding to the observed datasets
    loglikfuns : list of abcpy.approx_lhd.Approx_likelihood
        List of Approx_loglikelihood object defining the approximated loglikelihood to be used; one for each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
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

    def __init__(self, root_models, loglikfuns, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Sum of Loglikelihood functions using all the loglikelihoods for each individual models
        self.likfun = SumCombination(root_models, loglikfuns)

        if kernel is None:

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

    def sample(self, observations, steps, n_samples=10000, n_samples_per_param=100, covFactors=None, iniPoints=None,
               full_output=0, journal_file=None, path_to_save_journal=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        steps : integer
            number of iterations in the sequential algorithm ("generations")
        n_samples : integer, optional
            number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            number of data points in each simulated data set. The default value is 100.
        covFactors : list of float, optional
            scaling parameter of the covariance matrix. The default is a p dimensional array of 1 when p is the
            dimension of the parameter. One for each perturbation kernel.
        inipoints : numpy.ndarray, optional
            parameter vaulues from where the sampling starts. By default sampled from the prior.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path. The journal is saved (and overwritten) after each step
            of the sequential inference routine, so that partial results are stored to the disk in case the
            inference routine does not end correctly; recall that you need to set full_output=1 to obtain the
            full partial results.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.sample_from_prior(rng=self.rng)

        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        if path_to_save_journal is not None:
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if journal_file is None:
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_lhd_func"] = [type(likfun).__name__ for likfun in self.likfun.approx_lhds]
            journal.configuration["steps"] = steps
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["covFactor"] = covFactors
            journal.configuration["iniPoints"] = iniPoints
            journal.configuration["full_output"] = full_output
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
            accepted_weights = np.ones((n_samples, 1), dtype=np.float64) / n_samples
        else:
            accepted_parameters = iniPoints
            accepted_weights = np.ones((iniPoints.shape[0], 1), dtype=np.float64) / iniPoints.shape[0]

        if covFactors is None:
            covFactors = np.ones(shape=(len(self.kernel.kernels),))

        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                          accepted_weights=accepted_weights)

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
        accepted_cov_mats = self._compute_accepted_cov_mats(covFactors, new_cov_mats)
        # accepted_cov_mats = [covFactor * new_cov_mat for covFactor, new_cov_mat in zip(covFactors,new_cov_mats)]

        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

        # main SMC algorithm
        self.logger.info("Starting pmc iterations")
        for aStep in range(steps):
            if aStep == 0 and journal_file is not None:
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                # 3: calculate covariance
                self.logger.info("Calculating covariance matrix")

                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                # Since each entry of new_cov_mats is a numpy array, we can multiply like this

                accepted_cov_mats = self._compute_accepted_cov_mats(covFactors, new_cov_mats)
                # accepted_cov_mats = [covFactor * new_cov_mat for covFactor, new_cov_mat in zip(covFactors, new_cov_mats)]

            self.logger.info("Iteration {} of PMC algorithm".format(aStep))

            # 0: update remotely required variables
            self.logger.info("Broadcasting parameters")
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights,
                                                              accepted_cov_mats=accepted_cov_mats)

            # 1: Resample parameters
            self.logger.info("Resample parameters")
            index = self.rng.choice(len(accepted_parameters), size=n_samples, p=accepted_weights.reshape(-1))
            # Choose a new particle using the resampled particle (make the boundary proper)
            # Initialize new_parameters
            new_parameters = []
            for ind in range(0, self.n_samples):
                while True:
                    perturbation_output = self.perturb(index[ind], rng=self.rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                        new_parameters.append(perturbation_output[1])
                        break

            # 2: calculate approximate likelihood for new parameters
            self.logger.info("Calculate approximate loglikelihood")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=self.n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            data_arr = []
            for i in range(len(rng_arr)):
                data_arr.append([new_parameters[i], rng_arr[i]])
            data_pds = self.backend.parallelize(data_arr)

            approx_log_likelihood_new_parameters_and_counter_pds = self.backend.map(self._approx_log_lik_calc, data_pds)
            self.logger.debug("collect approximate likelihood from pds")
            approx_log_likelihood_new_parameters_and_counter = self.backend.collect(
                approx_log_likelihood_new_parameters_and_counter_pds)
            approx_log_likelihood_new_parameters, counter = [list(t) for t in
                                                             zip(*approx_log_likelihood_new_parameters_and_counter)]

            approx_log_likelihood_new_parameters = np.array(approx_log_likelihood_new_parameters).reshape(-1, 1)

            for count in counter:
                self.simulation_counter += count

            # 3: calculate new weights for new parameters
            self.logger.info("Calculating weights")
            new_parameters_pds = self.backend.parallelize(new_parameters)
            new_weights_pds = self.backend.map(self._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1, 1)

            new_log_weights = np.log(new_weights) + approx_log_likelihood_new_parameters

            # we get numerical issues often:
            self.logger.info("range_of_weights (log): {}".format(np.max(new_log_weights) - np.min(new_log_weights)))
            # sorted_weights = np.sort(new_log_weights)
            # self.logger.info("Difference first second largest (log): {}".format(sorted_weights[-1] - sorted_weights[-2]))

            # center on log scale (avoid numerical issues):
            new_weights = np.exp(new_log_weights - np.max(new_log_weights))
            sum_of_weights = np.sum(new_weights)
            new_weights = new_weights / sum_of_weights

            # self.logger.info("new_weights : ", new_weights, ", sum_of_weights : ", sum_of_weights)
            self.logger.info("sum_of_weights : {}".format(sum_of_weights))
            self.logger.info("ESS : {}".format(1 / sum(pow(new_weights / sum(new_weights), 2))[0]))
            accepted_parameters = new_parameters

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=new_weights)

            # 4: calculate covariance
            # The parameters relevant to each kernel have to be used to calculate n_sample times. It is therefore more efficient to broadcast these parameters once, instead of collecting them at each kernel in each step
            self.logger.info("Calculating covariance matrix")
            kernel_parameters = []
            for kernel in self.kernel.kernels:
                kernel_parameters.append(
                    self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

            self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

            # 3: calculate covariance
            self.logger.info("Calculating covariance matrix")

            new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
            # Since each entry of new_cov_mats is a numpy array, we can multiply like this
            accepted_cov_mats = self._compute_accepted_cov_mats(covFactors, new_cov_mats)
            # new_cov_mats = [covFactor * new_cov_mat for covFactor, new_cov_mat in zip(covFactors, new_cov_mats)]

            # 5: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            # accepted_cov_mats = new_cov_mats

            self.logger.info("Saving configuration to output journal")

            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_weights(copy.deepcopy(accepted_weights))
                journal.add_ESS_estimate(accepted_weights)
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            if path_to_save_journal is not None:  # save journal
                journal.save(path_to_save_journal)

        return journal

    # define helper functions for map step
    def _approx_log_lik_calc(self, data, npc=None):
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

        # lhd = self.likfun.likelihood(obs, y_sim)
        loglhd = self.likfun.loglikelihood(obs, y_sim)

        self.logger.debug("LogLikelihood is :" + str(loglhd))

        log_pdf_at_theta = np.log(self.pdf_of_prior(self.model, theta))

        self.logger.debug("Prior pdf evaluated at theta is :" + str(log_pdf_at_theta))

        log_pdf_at_theta += loglhd

        return log_pdf_at_theta, self.n_samples_per_param

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

            mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
                self.accepted_parameters_manager.model)

            pdf_values = np.array([self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                                   self.accepted_parameters_manager.accepted_parameters_bds.value()[i],
                                                   theta) for i in range(self.n_samples)])

            denominator = np.sum(self.accepted_parameters_manager.accepted_weights_bds.value().reshape(-1) * pdf_values)

            return 1.0 * prior_prob / denominator

    def _compute_accepted_cov_mats(self, covFactors, new_cov_mats):
        """
        Update the covariance matrices computed from data by multiplying them with covFactors and adding a small term in
        the diagonal for numerical stability.

        Parameters
        ----------
        covFactors : list of float
            factors to correct the covariance matrices
        new_cov_mats : list
            list of covariance matrices computed from data
        Returns
        -------
        list
            List of new accepted covariance matrices
        """
        accepted_cov_mats = []
        for covFactor, new_cov_mat in zip(covFactors, new_cov_mats):
            if not (new_cov_mat.size == 1):
                accepted_cov_mats.append(
                    covFactor * new_cov_mat + 1e-20 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
            else:
                accepted_cov_mats.append((covFactor * new_cov_mat + 1e-20 * new_cov_mat).reshape(1, 1))
        return accepted_cov_mats


class SABC(BaseDiscrepancy, InferenceMethod):
    """
    This class implements a modified version of Simulated Annealing Approximate Bayesian Computation (SABC) of [1]
    when the prior is non-informative.

    [1] C. Albert, H. R. Kuensch and A. Scheidegger. A Simulated Annealing Approach to
    Approximate Bayes Computations. Statistics and Computing, (2014).

    Parameters
    ----------
    root_models: list
        A list of the Probabilistic models corresponding to the observed datasets
    distances: list of abcpy.distances.Distance
        List of Distance objects defining the distance measure to compare simulated and observed data sets; one for
        each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
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

        # check if the distance estimators are always positive
        if np.any([isinstance(distance, Divergence) and not distance._estimate_always_positive()
                   for distance in distances]):
            raise RuntimeError("SABC does not work with estimates of divergences which may be negative. Use another "
                               "inference algorithm or a different estimator.")

        if kernel is None:

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

    def sample(self, observations, steps, epsilon, n_samples=10000, n_samples_per_param=1, beta=2, delta=0.2,
               v=0.3, ar_cutoff=0.1, resample=None, n_update=None, full_output=0, journal_file=None,
               path_to_save_journal=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        steps : integer
            Number of maximum iterations in the sequential algorithm ("generations")
        epsilon : numpy.float
            A proposed value of threshold to start with.
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        beta : numpy.float, optional
            Tuning parameter of SABC, default value is 2. Used to scale up the covariance matrices obtained from data.
        delta : numpy.float, optional
            Tuning parameter of SABC, default value is 0.2.
        v : numpy.float, optional
            Tuning parameter of SABC, The default value is 0.3.
        ar_cutoff : numpy.float, optional
            Acceptance ratio cutoff: if the acceptance rate at some iteration of the algorithm is lower than that, the
            algorithm will stop. The default value is 0.1.
        resample: int, optional
            At any iteration, perform a resampling step if the number of accepted particles is larger than resample.
            When not provided, it assumes resample to be equal to n_samples.
        n_update: int, optional
            Number of perturbed parameters at each step, The default value is None which takes value inside n_samples
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path. The journal is saved (and overwritten) after each step
            of the sequential inference routine, so that partial results are stored to the disk in case the
            inference routine does not end correctly; recall that you need to set full_output=1 to obtain the
            full partial results.

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

        if path_to_save_journal is not None:
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if journal_file is None:
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = [type(distance).__name__ for distance in self.distance.distances]
            journal.configuration["type_kernel_func"] = [type(kernel).__name__ for kernel in self.kernel.kernels] if \
                isinstance(self.kernel, JointPerturbationKernel) else type(self.kernel)
            journal.configuration["steps"] = steps
            journal.configuration["epsilon"] = self.epsilon
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

        if resample is None:
            resample = n_samples
        if n_update is None:
            n_update = n_samples
        sample_array = np.ones(shape=(steps,))
        sample_array[0] = n_samples
        sample_array[1:] = n_update

        # Acceptance counter to determine the resampling step
        accept = 0
        samples_until = 0

        # Counter whether broken preemptively
        broken_preemptively = False

        for aStep in range(0, steps):
            self.logger.debug("step {}".format(aStep))
            if aStep == 0 and journal_file is not None:
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                # Broadcast Accepted parameters and Accepted weights
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                # Broadcast Accepted Kernel parameters
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = self._compute_accepted_cov_mats(beta, new_cov_mats)

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
            self.logger.debug("Map step of parallelism is finished")
            params_and_dists = self.backend.collect(params_and_dists_pds)
            self.logger.debug("Collect step of parallelism is finished")
            new_parameters, new_distances, new_all_parameters, new_all_distances, index, acceptance, counter = [
                list(t) for t in zip(*params_and_dists)]

            # Keeping counter of number of simulations
            self.logger.debug("Counting number of simulations")
            for count in counter:
                self.simulation_counter += count

            # new_parameters = np.array(new_parameters)
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
            self.logger.info("Initialize/Update the accepted parameters and their corresponding distances")
            if accepted_parameters is None:
                accepted_parameters = new_parameters
            else:
                for ind in range(len(acceptance)):
                    if acceptance[ind] == 1:
                        accepted_parameters[index[ind]] = new_parameters[ind]
            distances[index[acceptance == 1]] = new_distances[acceptance == 1]

            # 2: Smoothing of the distances
            self.logger.info("Smoothing of the distance")
            smooth_distances[index[acceptance == 1]] = self._smoother_distance(distances[index[acceptance == 1]],
                                                                               all_distances)

            # 3: Initialize/Update U, epsilon and covariance of perturbation kernel
            self.logger.info("Initialize/Update U, epsilon and covariance of perturbation kernel")
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

                msg = ("updates= {:.2f}, epsilon= {}, u.mean={:e}, acceptance rate: {:.2f}".format(
                    np.sum(sample_array[1:aStep + 1]) / np.sum(sample_array[1:]) * 100, epsilon, U, acceptance_rate))
                self.logger.info(msg)
                if acceptance_rate < ar_cutoff:
                    broken_preemptively = True
                    self.logger.info("Stopping as acceptance rate is lower than cutoff")
                    break

            # 5: Resampling if number of accepted particles greater than resample
            if accept >= resample and U > 1e-100:
                self.logger.info("Weighted resampling")
                weight = np.exp(-smooth_distances * delta / U)
                weight = weight / sum(weight)
                index_resampled = self.rng.choice(n_samples, n_samples, replace=True, p=weight)
                accepted_parameters = [accepted_parameters[i] for i in index_resampled]
                smooth_distances = smooth_distances[index_resampled]
                distances = distances[index_resampled]

                # Update U and epsilon:
                epsilon = epsilon * (1 - delta)
                U = np.mean(smooth_distances)
                epsilon = self._schedule(U, v)

                # Print effective sampling size
                self.logger.info('Resampling: Effective sampling size: ' + str(1 / sum(pow(weight / sum(weight), 2))))
                accept = 0
                samples_until = 0

                # Compute and broadcast accepted parameters, accepted kernel parameters and accepted Covariance matrix
                # Broadcast Accepted parameters and add to journal
                self.logger.info("Broadcast Accepted parameters and add to journal")
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights=accepted_weights,
                                                                  accepted_parameters=accepted_parameters)
                # Compute Accepted Kernel parameters and broadcast them
                self.logger.debug("Compute Accepted Kernel parameters and broadcast them")
                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)
                # Compute Kernel Covariance Matrix and broadcast it
                self.logger.debug("Compute Kernel Covariance Matrix and broadcast it")
                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = self._compute_accepted_cov_mats(beta, new_cov_mats)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

                if full_output == 1 and aStep <= steps - 1:
                    # Saving intermediate configuration to output journal.
                    self.logger.info('Saving after resampling')
                    journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                    journal.add_weights(copy.deepcopy(accepted_weights))
                    journal.add_ESS_estimate(accepted_weights)
                    journal.add_distances(copy.deepcopy(distances))
                    names_and_parameters = self._get_names_and_parameters()
                    journal.add_user_parameters(names_and_parameters)
                    journal.number_of_simulations.append(self.simulation_counter)
            else:
                # Compute and broadcast accepted parameters, accepted kernel parameters and accepted Covariance matrix
                # Broadcast Accepted parameters
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights=accepted_weights,
                                                                  accepted_parameters=accepted_parameters)
                # Compute Accepted Kernel parameters and broadcast them
                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)
                # Compute Kernel Covariance Matrix and broadcast it
                new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = self._compute_accepted_cov_mats(beta, new_cov_mats)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

                if full_output == 1 and aStep <= steps - 1:
                    # Saving intermediate configuration to output journal.
                    journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                    journal.add_weights(copy.deepcopy(accepted_weights))
                    journal.add_ESS_estimate(accepted_weights)
                    journal.add_distances(copy.deepcopy(distances))
                    names_and_parameters = self._get_names_and_parameters()
                    journal.add_user_parameters(names_and_parameters)
                    journal.number_of_simulations.append(self.simulation_counter)

            # Add epsilon_arr, number of final steps and final output to the journal
            # print("INFO: Saving final configuration to output journal.")
            if (full_output == 0) or (full_output == 1 and broken_preemptively and aStep <= steps - 1):
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_weights(copy.deepcopy(accepted_weights))
                journal.add_ESS_estimate(accepted_weights)
                journal.add_distances(copy.deepcopy(distances))
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            journal.configuration["steps"] = aStep + 1
            journal.configuration["epsilon"] = epsilon

            if path_to_save_journal is not None:  # save journal
                journal.save(path_to_save_journal)

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

        return U

    def _schedule(self, rho, v):
        if rho < 1e-100:
            epsilon = 0
        else:
            fun = lambda epsilon: pow(epsilon, 2) + v * pow(epsilon, 3 / 2) - pow(rho, 2)
            epsilon = optimize.fsolve(fun, rho / 2)

        return epsilon

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
        if isinstance(data, np.ndarray):
            data = data.tolist()
        rng = data[0]
        index = data[1]
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        all_parameters = []
        all_distances = []
        acceptance = 0

        counter = 0

        if self.accepted_parameters_manager.accepted_cov_mats_bds is None:

            while acceptance == 0:
                self.sample_from_prior(rng=rng)
                new_theta = self.get_parameters()
                all_parameters.append(new_theta)
                t0 = time.time()
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                self.logger.debug("Simulation took " + str(time.time() - t0) + "sec")
                counter += 1
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
                all_distances.append(distance)
                acceptance = rng.binomial(1, np.exp(-distance / self.epsilon), 1)
            acceptance = 1
        else:
            # Select one arbitrary particle:
            index = rng.choice(self.n_samples, size=1)[0]
            # Sample proposal parameter and calculate new distance:
            theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]

            while True:
                perturbation_output = self.perturb(index, rng=rng)
                if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                    new_theta = perturbation_output[1]
                    break
            t0 = time.time()
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            self.logger.debug("Simulation took " + str(time.time() - t0) + "sec")
            counter += 1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            smooth_distance = self._smoother_distance([distance], self.all_distances_bds.value())

            # Calculate acceptance probability:
            self.logger.debug("Calulate acceptance probability")
            ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(
                self.model, self.accepted_parameters_manager.accepted_parameters_bds.value()[index])
            ratio_likelihood_prob = np.exp((self.smooth_distances_bds.value()[index] - smooth_distance) / self.epsilon)
            acceptance_prob = ratio_prior_prob * ratio_likelihood_prob

            # If accepted
            if rng.rand(1) < acceptance_prob:
                acceptance = 1
            else:
                distance = np.inf

        return new_theta, distance, all_parameters, all_distances, index, acceptance, counter

    def _compute_accepted_cov_mats(self, beta, new_cov_mats):
        """
        Update the covariance matrices computed from data by multiplying them with beta and adding a small term in
        the diagonal for numerical stability.

        Parameters
        ----------
        beta : float
            factor to correct the covariance matrices
        new_cov_mats : list
            list of covariance matrices computed from data
        Returns
        -------
        list
            List of new accepted covariance matrices
        """
        accepted_cov_mats = []
        for new_cov_mat in new_cov_mats:
            if not (new_cov_mat.size == 1):
                accepted_cov_mats.append(
                    beta * new_cov_mat + 1e-20 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
            else:
                accepted_cov_mats.append((beta * new_cov_mat + 1e-20 * new_cov_mat).reshape(1, 1))
        return accepted_cov_mats


class ABCsubsim(BaseDiscrepancy, InferenceMethod):
    """This class implements Approximate Bayesian Computation by subset simulation (ABCsubsim) algorithm of [1].

    [1] M. Chiachio, J. L. Beck, J. Chiachio, and G. Rus., Approximate Bayesian computation by subset
    simulation. SIAM J. Sci. Comput., 36(3):A1339–A1358, 2014/10/03 2014.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distances: list of abcpy.distances.Distance
        List of Distance objects defining the distance measure to compare simulated and observed data sets; one for 
        each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
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
        self.rng = np.random.RandomState(seed)
        self.anneal_parameter = None
        self.logger = logging.getLogger(__name__)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

        self.simulation_counter = 0

    def sample(self, observations, steps, n_samples=10000, n_samples_per_param=1, chain_length=10, ap_change_cutoff=10,
               full_output=0, journal_file=None, path_to_save_journal=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        steps : integer
            Number of iterations in the sequential algorithm ("generations")
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
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path. The journal is saved (and overwritten) after each step
            of the sequential inference routine, so that partial results are stored to the disk in case the
            inference routine does not end correctly; recall that you need to set full_output=1 to obtain the
            full partial results.

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
        if path_to_save_journal is not None:
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if journal_file is None:
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = [type(distance).__name__ for distance in self.distance.distances]
            journal.configuration["type_kernel_func"] = [type(kernel).__name__ for kernel in self.kernel.kernels] if \
                isinstance(self.kernel, JointPerturbationKernel) else type(self.kernel)
            journal.configuration["steps"] = steps
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
            if aStep == 0 and journal_file is not None:
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)
                accepted_cov_mats = journal.get_accepted_cov_mats()

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

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights=accepted_weights,
                                                              accepted_parameters=accepted_parameters)

            # 1: Calculate  parameters
            # print("INFO: Initial accepted parameter parameters")
            self.logger.info("Initial accepted parameters")
            params_and_dists_pds = self.backend.map(self._accept_parameter, rng_and_index_pds)
            self.logger.debug("Map random number to a pseudo-observation")
            params_and_dists = self.backend.collect(params_and_dists_pds)
            self.logger.debug("Collect results from the mapping")
            new_parameters, new_distances, counter = [list(t) for t in zip(*params_and_dists)]

            for count in counter:
                self.simulation_counter += count

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
            accepted_params_and_dist = sorted(accepted_params_and_dist, key=lambda x: x[0])
            distances, accepted_parameters = [list(t) for t in zip(*accepted_params_and_dist)]

            # 3: Calculate and broadcast annealing parameters
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
                accepted_cov_mats = pow(2, 1) * accepted_cov_mats

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
                self.simulation_counter += count

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
                journal.add_ESS_estimate(accepted_weights)
                journal.add_accepted_cov_mats(accepted_cov_mats)
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            # Show progress
            anneal_parameter_change_percentage = 100 * abs(anneal_parameter_old - anneal_parameter) / abs(
                anneal_parameter)
            msg = ("step: {}, annealing parameter: {:.4f}, change(%) in annealing parameter: {:.1f}"
                   .format(aStep, anneal_parameter, anneal_parameter_change_percentage))
            self.logger.info(msg)
            if anneal_parameter_change_percentage < ap_change_cutoff:
                break
            if path_to_save_journal is not None:  # save journal
                journal.save(path_to_save_journal)

        # Add anneal_parameter, number of final steps and final output to the journal
        # print("INFO: Saving final configuration to output journal.")
        if full_output == 0:
            journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
            journal.add_distances(copy.deepcopy(distances))
            journal.add_weights(copy.deepcopy(accepted_weights))
            journal.add_ESS_estimate(accepted_weights)
            journal.add_accepted_cov_mats(accepted_cov_mats)
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights)
            names_and_parameters = self._get_names_and_parameters()
            journal.add_user_parameters(names_and_parameters)
            journal.number_of_simulations.append(self.simulation_counter)

        journal.configuration["steps"] = aStep + 1
        journal.configuration["anneal_parameter"] = anneal_parameter

        if path_to_save_journal is not None:  # save journal
            journal.save(path_to_save_journal)

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

        if self.accepted_parameters_manager.accepted_parameters_bds is None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter += 1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            result_theta += self.get_parameters()
            result_distance.append(distance)
        else:
            theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]
            self.set_parameters(theta)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter += 1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            result_theta.append(theta)
            result_distance.append(distance)
            for ind in range(0, self.chain_length - 1):
                while True:
                    perturbation_output = self.perturb(index, rng=rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                        break
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter += 1
                new_distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

                # Calculate acceptance probability:
                ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model,
                                                                                                             theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                                   perturbation_output[1], theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta,
                                                     perturbation_output[1])
                ratio_likelihood_prob = kernel_numerator / kernel_denominator
                acceptance_prob = min(1, ratio_prior_prob * ratio_likelihood_prob) * (
                        new_distance < self.anneal_parameter)

                # If accepted
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

        accepted_cov_mats_transformed = [cov_mat * pow(2.0, -2.0 * t) for cov_mat in
                                         self.accepted_parameters_manager.accepted_cov_mats_bds.value()]

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
            counter += 1
            new_distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            self.logger.debug("Calculate acceptance probability.")
            # Calculate acceptance probability:
            ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model,
                                                                                                         theta)
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
            return accepted_cov_mats_transformed, t, 1, counter
        else:
            return accepted_cov_mats_transformed, t, 0, counter


class RSMCABC(BaseDiscrepancy, InferenceMethod):
    """This class implements Replenishment Sequential Monte Carlo Approximate Bayesian computation of
    Drovandi and Pettitt [1].

    [1] CC. Drovandi CC and AN. Pettitt, Estimation of parameters for macroparasite population evolution using
    approximate Bayesian computation. Biometrics 67(1):225–233, 2011.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distances: list of abcpy.distances.Distance
        List of Distance objects defining the distance measure to compare simulated and observed data sets; one for 
        each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
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

    def sample(self, observations, steps, n_samples=10000, n_samples_per_param=1, alpha=0.1, epsilon_init=100,
               epsilon_final=0.1, const=0.01, covFactor=2.0, full_output=0, journal_file=None,
               path_to_save_journal=None):
        """
        Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        steps : integer
            Number of iterations in the sequential algorithm ("generations")
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
             A constant to compute acceptance probability, the default is 0.01.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path. The journal is saved (and overwritten) after each step
            of the sequential inference routine, so that partial results are stored to the disk in case the
            inference routine does not end correctly; recall that you need to set full_output=1 to obtain the
            full partial results.

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
        if path_to_save_journal is not None:
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if journal_file is None:
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = [type(distance).__name__ for distance in self.distance.distances]
            journal.configuration["type_kernel_func"] = [type(kernel).__name__ for kernel in self.kernel.kernels] if \
                isinstance(self.kernel, JointPerturbationKernel) else type(self.kernel)
            journal.configuration["steps"] = steps
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["alpha"] = alpha
            journal.configuration["epsilon_init"] = epsilon_init
            journal.configuration["epsilon_final"] = epsilon_final
            journal.configuration["const"] = const
            journal.configuration["covFactor"] = covFactor
            journal.configuration["full_output"] = full_output
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
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights=accepted_weights,
                                                                  accepted_parameters=accepted_parameters)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = self._compute_accepted_cov_mats(covFactor, accepted_cov_mats)
                # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]

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
                if journal_file is None:
                    accepted_cov_mats = None
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
                accepted_cov_mats = self._compute_accepted_cov_mats(covFactor, accepted_cov_mats)
                # accepted_cov_mats = [covFactor*cov_mat for cov_mat in accepted_cov_mats]

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
                self.simulation_counter += count

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
                journal.add_ESS_estimate(accepted_weights)
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_weights=accepted_weights,
                                                                  accepted_parameters=accepted_parameters)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            # 2: Compute acceptance probability and set R
            self.logger.info("Compute acceptance probabilty and set R")
            prob_acceptance = sum(new_index) / (R * n_replenish)
            if prob_acceptance == 1 or prob_acceptance == 0:
                R = 1
            else:
                R = int(np.log(const) / np.log(1 - prob_acceptance))

            self.logger.info("Order accepted parameters and distances")
            n_replenish = round(n_samples * alpha)
            accepted_params_and_dist = zip(accepted_dist, accepted_parameters)
            accepted_params_and_dist = sorted(accepted_params_and_dist, key=lambda x: x[0])
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

            if path_to_save_journal is not None:  # save journal
                journal.save(path_to_save_journal)

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

        if self.accepted_parameters_manager.accepted_parameters_bds is None:
            while distance > self.epsilon[-1]:
                self.sample_from_prior(rng=rng)
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter += 1
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
                counter += 1
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
                ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model,
                                                                                                             theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                                   perturbation_output[1], theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta,
                                                     perturbation_output[1])
                ratio_kernel_prob = kernel_numerator / kernel_denominator
                probability_acceptance = min(1, ratio_prior_prob * ratio_kernel_prob)
                if distance < self.epsilon[-1] and rng.binomial(1, probability_acceptance) == 1:
                    index_accept += 1
                else:
                    self.set_parameters(theta)
                    distance = self.accepted_dist_bds.value()[index[0]]

        return self.get_parameters(self.model), distance, index_accept, counter

    def _compute_accepted_cov_mats(self, covFactor, new_cov_mats):
        """
        Update the covariance matrices computed from data by multiplying them with covFactor and adding a small term in
        the diagonal for numerical stability.

        Parameters
        ----------
        covFactor : float
            factor to correct the covariance matrices
        new_cov_mats : list
            list of covariance matrices computed from data
        Returns
        -------
        list
            List of new accepted covariance matrices
        """
        # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]
        accepted_cov_mats = []
        for new_cov_mat in new_cov_mats:
            if not (new_cov_mat.size == 1):
                accepted_cov_mats.append(
                    covFactor * new_cov_mat + 1e-20 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
            else:
                accepted_cov_mats.append((covFactor * new_cov_mat + 1e-20 * new_cov_mat).reshape(1, 1))
        return accepted_cov_mats


class APMCABC(BaseDiscrepancy, InferenceMethod):
    """This class implements Adaptive Population Monte Carlo Approximate Bayesian computation of
    M. Lenormand et al. [1].

    [1] M. Lenormand, F. Jabot and G. Deffuant, Adaptive approximate Bayesian computation
    for complex models. Computational Statistics, 28:2777–2796, 2013.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distances: list of abcpy.distances.Distance
        List of Distance objects defining the distance measure to compare simulated and observed data sets; one for 
        each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
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

        self.epsilon = None
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        self.accepted_dist_bds = None

        self.simulation_counter = 0

    def sample(self, observations, steps, n_samples=10000, n_samples_per_param=1, alpha=0.1, acceptance_cutoff=0.03,
               covFactor=2.0, full_output=0, journal_file=None, path_to_save_journal=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        steps : integer
            Number of iterations in the sequential algorithm ("generations")
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
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path. The journal is saved (and overwritten) after each step
            of the sequential inference routine, so that partial results are stored to the disk in case the
            inference routine does not end correctly; recall that you need to set full_output=1 to obtain the
            full partial results.

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
        if path_to_save_journal is not None:
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if journal_file is None:
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = [type(distance).__name__ for distance in self.distance.distances]
            journal.configuration["type_kernel_func"] = [type(kernel).__name__ for kernel in self.kernel.kernels] if \
                isinstance(self.kernel, JointPerturbationKernel) else type(self.kernel)
            journal.configuration["steps"] = steps
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["alpha"] = self.alpha
            journal.configuration["acceptance_cutoff"] = acceptance_cutoff
            journal.configuration["covFactor"] = covFactor
            journal.configuration["full_output"] = full_output
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
            if aStep == 0 and journal_file is not None:
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)

                accepted_cov_mats = self._compute_accepted_cov_mats(covFactor, accepted_cov_mats)
                # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)

                alpha_accepted_parameters = accepted_parameters
                alpha_accepted_weights = accepted_weights

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
            self.accepted_parameters_manager.update_broadcast(self.backend,
                                                              accepted_parameters=alpha_accepted_parameters,
                                                              accepted_weights=alpha_accepted_weights,
                                                              accepted_cov_mats=accepted_cov_mats)
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
                self.simulation_counter += count

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
            self.accepted_parameters_manager.update_broadcast(self.backend,
                                                              accepted_parameters=alpha_accepted_parameters,
                                                              accepted_weights=alpha_accepted_weights)

            kernel_parameters = []
            for kernel in self.kernel.kernels:
                kernel_parameters.append(
                    self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

            self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

            accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
            accepted_cov_mats = self._compute_accepted_cov_mats(covFactor, accepted_cov_mats)
            # accepted_cov_mats = [covFactor*cov_mat for cov_mat in accepted_cov_mats]

            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(accepted_dist))
                journal.add_weights(copy.deepcopy(accepted_weights))
                journal.add_ESS_estimate(accepted_weights)
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

            if path_to_save_journal is not None:  # save journal
                journal.save(path_to_save_journal)

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

        if self.accepted_parameters_manager.accepted_parameters_bds is None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter += 1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            weight = 1.0
        else:
            index = rng.choice(len(self.accepted_parameters_manager.accepted_weights_bds.value()), size=1,
                               p=self.accepted_parameters_manager.accepted_weights_bds.value().reshape(-1))
            # truncate the normal to the bounds of parameter space of the model
            # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf
            while True:
                perturbation_output = self.perturb(index[0], rng=rng)
                if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                    break

            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter += 1
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)

            prior_prob = self.pdf_of_prior(self.model, perturbation_output[1])
            denominator = 0.0
            for i in range(len(self.accepted_parameters_manager.accepted_weights_bds.value())):
                pdf_value = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                            self.accepted_parameters_manager.accepted_parameters_bds.value()[i],
                                            perturbation_output[1])
                denominator += self.accepted_parameters_manager.accepted_weights_bds.value()[i, 0] * pdf_value
            weight = 1.0 * prior_prob / denominator

        return self.get_parameters(self.model), distance, weight, counter

    def _compute_accepted_cov_mats(self, covFactor, new_cov_mats):
        """
        Update the covariance matrices computed from data by multiplying them with covFactor and adding a small term in
        the diagonal for numerical stability.

        Parameters
        ----------
        covFactor : float
            factor to correct the covariance matrices
        new_cov_mats : list
            list of covariance matrices computed from data
        Returns
        -------
        list
            List of new accepted covariance matrices
        """
        # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]
        accepted_cov_mats = []
        for new_cov_mat in new_cov_mats:
            if not (new_cov_mat.size == 1):
                accepted_cov_mats.append(
                    covFactor * new_cov_mat + 1e-20 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
            else:
                accepted_cov_mats.append((covFactor * new_cov_mat + 1e-20 * new_cov_mat).reshape(1, 1))
        return accepted_cov_mats


class SMCABC(BaseDiscrepancy, InferenceMethod):
    """This class implements two versions of Sequential Monte Carlo Approximate Bayesian computation, following either
     the original in Del Moral et al. [1] or the newer version in Bernton et al. [3]. The first one is commonly used
     when standard statistics based ABC is done (for instance with Euclidean distance), while the second one is instead
     used when divergence-ABC is done (for instance, ABC with the Wasserstein distance, with the KL divergence and so
     on). The first implementation in fact does not work in that case, as it assumes the distance can be computed with
     respect to one single observation. Additionally, our implementation of the algorithm by Bernton et al.
     does not work with the standard MCMC kernel, but requires using the r-hit kernel [2], which is arguably more
     efficient (even if it leads to an algorithm for which the number of simulations is not known a priori).

    [1] P. Del Moral, A. Doucet, A. Jasra, An adaptive sequential Monte Carlo method for approximate
    Bayesian computation. Statistics and Computing, 22(5):1009–1020, 2012.

    [2] Lee, Anthony. "On the choice of MCMC kernels for approximate Bayesian computation with SMC samplers.
    Proceedings of the 2012 Winter Simulation Conference (WSC). IEEE, 2012.

    [3] Bernton, E., Jacob, P. E., Gerber, M., & Robert, C. P. (2019). Approximate Bayesian computation with the
    Wasserstein distance. Journal of the Royal Statistical Society Series B, 81(2), 235-269.

    Parameters
    ----------
    model : list
        A list of the Probabilistic models corresponding to the observed datasets
    distances: list of abcpy.distances.Distance
        List of Distance objects defining the distance measure to compare simulated and observed data sets; one for
        each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
    version : string, optional
        Denotes which version to use, either the one by Del Moral et al. [1] ("DelMoral") or the newer version in
        Bernton et al. [3] ("Bernton"). By default, uses "DelMoral".
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

    def __init__(self, root_models, distances, backend, kernel=None, version="DelMoral", seed=None):
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
        if version not in ["DelMoral", "Bernton"]:
            raise RuntimeError('The only implemented SMCABC methods are the one by Del Moral et al. ("DelMoral") or'
                               ' the one by Bernton et al. [3] ("Bernton"). Please set version'
                               ' to one of these two.')
        else:
            self.bernton = version == "Bernton"
            self.version = version

        # check now if we are using divergences and in that case return error if using DelMoral version as that does
        # not work:
        if not self.bernton and np.any([isinstance(distance, Divergence) for distance in distances]):
            raise RuntimeError("You requested to use the SMCABC algorithm by Del Moral et al. "
                               "together with divergences "
                               "between empirical measures. That algorithm however only works for standard statistics "
                               "based ABC.")

        self.logger = logging.getLogger(__name__)

        self.epsilon = None
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task\
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        self.accepted_y_sim_bds = None

        self.simulation_counter = 0

    def sample(self, observations, steps, n_samples=10000, n_samples_per_param=1, epsilon_final=0.1, alpha=None,
               covFactor=2, resample=None, full_output=0, which_mcmc_kernel=None, r=None,
               store_simulations_in_journal=True, journal_file=None, path_to_save_journal=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        steps : integer
            Number of iterations in the sequential algorithm ("generations")
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        epsilon_final : float, optional
            The final threshold value of epsilon to be reached; if at some iteration you reach a lower epsilon than
            epsilon_final, the algorithm will stop and not proceed with further iterations. The default value is 0.1.
        alpha : float, optional
            A parameter taking values between [0,1], determining the rate of change of the threshold epsilon. If
            the algorithm by Bernton et al. is used, epsilon is chosen in order to guarantee a
            proportion of unique particles equal to alpha
            after resampling. If instead the algorithm by Del Moral et al. is used,
            epsilon is chosen such that the ESS (Effective Sample Size) with
            the new threshold value is alpha times the ESS with the old threshold value. The default value is None,
            in which case 0.95 is used for Del Moral et al. algorithm, or 0.5 for Bernton et al. algorithm.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2.
        resample  : float, optional
            It defines the resample step: introduce a resample step, after the particles have been
            perturbed and the new weights have been computed, if the effective sample size is smaller than resample.
            Notice that the algorithm by Bernton et al. always uses resample (as the weight values in that setup can
            only be equal to 0 or 1), so that this parameter is ignored in that case.
            If not provided, resample is set to 0.5 * n_samples.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        which_mcmc_kernel: integer, optional
            Specifies which MCMC kernel to be used: '0' is the standard MCMC kernel used in the algorithm by
            Del Moral et al [1], '1' uses the first version of r-hit kernel suggested by Anthony Lee (Alg. 5 in [2]),
            while '2' uses the second version of r-hit kernel suggested by Anthony Lee (Alg. 6 in [2]).
            The default value is 2 if the used algorithm is the one by Bernton et al, and is 0 for the algorithm by
            Del Moral et al.
        r: integer, optional:
            Specifies the value of 'r' (the number of wanted hits) in the r-hits kernels. It is therefore ignored if
            'which_mcmc_kernel==0'. If no value is provided, the first version of r-hit kernel uses r=3, while the
            second uses r=2. The default value is None.
        store_simulations_in_journal : boolean, optional
            Every step of the SMCABC algorithm uses the accepted simulations from previous step. Therefore, the accepted
            simulations at the final step are stored in the Journal file to allow restarting the inference
            correctly. If each simulation is large, however, that means that the accepted Journal will be large in
            memory. If you want to *not* save the simulations in the journal, set this to False; however, you will not
            be able to restart the inference from the returned Journal. The default value is True, meaning simulations
            are stored in the Journal.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path. The journal is saved (and overwritten) after each step
            of the sequential inference routine, so that partial results are stored to the disk in case the
            inference routine does not end correctly; recall that you need to set full_output=1 to obtain the
            full partial results.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        if path_to_save_journal is not None:
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if journal_file is None:
            journal = Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = [type(distance).__name__ for distance in self.distance.distances]
            journal.configuration["type_kernel_func"] = [type(kernel).__name__ for kernel in self.kernel.kernels] if \
                isinstance(self.kernel, JointPerturbationKernel) else type(self.kernel)
            journal.configuration["steps"] = steps
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["epsilon_final"] = epsilon_final
            journal.configuration["alpha"] = alpha
            journal.configuration["covFactor"] = covFactor
            journal.configuration["resample"] = resample
            journal.configuration["which_mcmc_kernel"] = which_mcmc_kernel
            journal.configuration["r"] = r
            journal.configuration["full_output"] = full_output
            journal.configuration["version"] = self.version
            self.sample_from_prior(rng=self.rng)  # initialize only if you are not restarting from a journal, in order
            # to ensure reproducibility
        else:
            journal = Journal.fromFile(journal_file)

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mats = None
        accepted_y_sim = None
        distances = None

        # Define the resample parameter
        if resample is None:
            resample = n_samples * 0.5

        if alpha is None:
            alpha = 0.5 if self.bernton else 0.95

        # Define maximum value of epsilon
        if not np.isinf(self.distance.dist_max()):
            epsilon = [self.distance.dist_max()]
        else:
            epsilon = [1e5]

        if which_mcmc_kernel is None:
            which_mcmc_kernel = 2 if self.bernton else 0

        if which_mcmc_kernel not in [0, 1, 2]:
            raise NotImplementedError("'which_mcmc_kernel' was given wrong value. It specifies which MCMC kernel to be"
                                      " used: '0' kernel suggested in [1], '1' uses the first version of r-hit"
                                      "kernel suggested by Anthony Lee (Alg. 5 in [2]), while '2' uses the second "
                                      "version of r-hit kernel"
                                      "suggested by Anthony Lee (Alg. 6 in [2]). The default value is 0.")

        if self.bernton and which_mcmc_kernel == 0:
            raise RuntimeError("The algorithm by Bernton et al. does not work with the standard MCMC kernel.")

        self.r = r

        # main SMC ABC algorithm
        for aStep in range(0, steps):
            self.logger.info("SMCABC iteration {}".format(aStep))

            if aStep == 0 and journal_file is not None:
                accepted_parameters = journal.get_accepted_parameters(-1)
                accepted_weights = journal.get_weights(-1)
                accepted_y_sim = journal.get_accepted_simulations()
                if accepted_y_sim is None:
                    raise RuntimeError("You cannot restart the inference from this Journal file as you did not store "
                                       "the simulations in it. In order to do that, the inference scheme needs to be"
                                       "called with `store_simulations_in_journal=True`.")
                distances = journal.get_distances(-1)

                epsilon = journal.configuration["epsilon_arr"]

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                                  accepted_weights=accepted_weights)

                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = self._compute_accepted_cov_mats(covFactor, accepted_cov_mats)
                # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]

                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

            # Break if epsilon in previous step is less than epsilon_final
            if epsilon[-1] <= epsilon_final:
                break

            # 0: Compute the Epsilon
            if distances is not None:
                self.logger.info(
                    "Compute epsilon, might take a while; previous epsilon value: {:.4f}".format(epsilon[-1]))
                if self.bernton:
                    # for the Bernton algorithm, the distances have already been computed before during the acceptance
                    # step. This however holds only when the r-hit kernels are used

                    # first compute the distances for the current set of parameters and observations
                    # (notice that may have already been done somewhere before!):
                    # current_distance_matrix = self._compute_distance_matrix_divergence(observations, accepted_y_sim,
                    #                                                                   n_samples)
                    # assert np.allclose(current_distance_matrix, distances)
                    current_distance_matrix = distances

                    # Compute epsilon for next step
                    fun = self._def_compute_epsilon_divergence_unique_particles(n_samples,
                                                                                current_distance_matrix, alpha)
                    epsilon_new = self._bisection(fun, epsilon_final, epsilon[-1], 0.001)
                else:
                    # first compute the distances for the current set of parameters and observations
                    # (notice that may have already been done somewhere before!):
                    current_distance_matrix = self._compute_distance_matrix(observations, accepted_y_sim, n_samples,
                                                                            n_samples_per_param)
                    fun = self._def_compute_epsilon(epsilon, accepted_weights, n_samples, current_distance_matrix,
                                                    alpha)
                    epsilon_new = self._bisection(fun, epsilon_final, epsilon[-1], 0.001)
                if epsilon_new < epsilon_final:
                    epsilon_new = epsilon_final
                epsilon.append(epsilon_new)

            # 1: calculate weights for new parameters
            self.logger.info("Calculating weights")
            if distances is not None:
                if self.bernton:
                    new_weights = (current_distance_matrix < epsilon[-1]) * 1
                else:
                    numerators = np.sum(current_distance_matrix < epsilon[-1], axis=1)
                    denominators = np.sum(current_distance_matrix < epsilon[-2], axis=1)

                    non_zero_denominator = denominators != 0
                    new_weights = np.zeros(shape=n_samples)

                    new_weights[non_zero_denominator] = accepted_weights.flatten()[non_zero_denominator] * (
                            numerators[non_zero_denominator] / denominators[non_zero_denominator])

                new_weights = new_weights / sum(new_weights)
            else:
                new_weights = np.ones(shape=n_samples, ) * (1.0 / n_samples)
            # 2: Resample; we resample always when using the Bernton et al. algorithm, as in that case weights
            # can only be proportional to 1 or 0; if we use the Del Moral version, instead, the
            # weights can have fractional values -> use the # resample threshold
            if distances is not None and (self.bernton or pow(sum(pow(new_weights, 2)), -1) < resample):
                self.logger.info("Resampling")
                # Weighted resampling:
                index_resampled = self.rng.choice(n_samples, n_samples, replace=True, p=new_weights)
                # accepted_parameters is a list. Then the indexing here does not work:
                # accepted_parameters = accepted_parameters[index_resampled]
                # do instead:
                accepted_parameters = [accepted_parameters[i] for i in
                                       index_resampled]  # why don't we use arrays however?
                new_weights = np.ones(shape=n_samples, ) * (1.0 / n_samples)

            # Update the weights
            accepted_weights = new_weights.reshape(len(new_weights), 1)

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights)
            if distances is not None:
                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

                accepted_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
                accepted_cov_mats = self._compute_accepted_cov_mats(covFactor, accepted_cov_mats)
                # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]

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
                                                              accepted_weights=accepted_weights,
                                                              accepted_cov_mats=accepted_cov_mats)
            self._update_broadcasts(accepted_y_sim)

            # calculate resample parameters
            self.logger.info("Drawing perturbed samples")
            if which_mcmc_kernel == 0:
                params_and_ysim_pds = self.backend.map(self._accept_parameter, rng_and_index_pds)
            elif which_mcmc_kernel == 1:
                params_and_ysim_pds = self.backend.map(self._accept_parameter_r_hit_kernel, rng_and_index_pds)
            elif which_mcmc_kernel == 2:
                params_and_ysim_pds = self.backend.map(self._accept_parameter_r_hit_kernel_version_2, rng_and_index_pds)
            params_and_ysim = self.backend.collect(params_and_ysim_pds)
            new_parameters, new_y_sim, distances, counter = [list(t) for t in zip(*params_and_ysim)]
            distances = np.array(distances)

            for count in counter:
                self.simulation_counter += count

            # Update the parameters
            accepted_parameters = new_parameters
            accepted_y_sim = new_y_sim

            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                self.logger.info("Saving configuration to output journal")
                self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(distances))
                journal.add_weights(copy.deepcopy(accepted_weights))
                journal.add_ESS_estimate(accepted_weights)
                if store_simulations_in_journal:
                    journal.add_accepted_simulations(copy.deepcopy(accepted_y_sim))

                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)

            # Add epsilon_arr to the journal
            journal.configuration["epsilon_arr"] = epsilon

            if path_to_save_journal is not None:  # save journal
                journal.save(path_to_save_journal)

        return journal

    def _compute_distance_matrix(self, observations, accepted_y_sim, n_samples, n_samples_per_param):
        distance_matrix = np.zeros((n_samples, n_samples_per_param))

        for ind1 in range(n_samples):
            for ind2 in range(n_samples_per_param):
                distance_matrix[ind1, ind2] = self.distance.distance(observations, [[accepted_y_sim[ind1][0][ind2]]])
                self.logger.debug(
                    'Computed distance inside for weights:' + str(
                        distance_matrix[ind1, ind2]))
        return distance_matrix

    def _compute_distance_matrix_divergence(self, observations, accepted_y_sim, n_samples):
        distance_matrix = np.zeros(n_samples)
        for ind1 in range(n_samples):
            distance_matrix[ind1] = self.distance.distance(observations, [accepted_y_sim[ind1][0]])
            self.logger.debug('Computed distance matrix for weights:' + str(distance_matrix[ind1]))
        return distance_matrix

    @staticmethod
    def _def_compute_epsilon(epsilon, accepted_weights, n_samples, distance_matrix, alpha):
        """
        Returns a function of 'epsilon_new' that is used in the bisection routine.
        The distances are computed just once in the definition of the function; this therefore avoids computing them
        repeatedly during the _bisection routine for the same input values, which is inefficient.

        Parameters
        ----------
        epsilon: float
            Current threshold.
        accepted_weights: numpy.ndarray
            Accepted weights.
        n_samples: integer
            Number of samples to generate.
        distance_matrix: np.ndarray:
            the distance matrix between observation and data used to compute the weights
        alpha: float

        Returns
        -------
        callable
            The function used in the bisection routine
        """

        RHS = alpha * pow(sum(pow(accepted_weights, 2)), -1)
        denominators = np.sum(distance_matrix < epsilon[-1], axis=1)
        non_zero_denominator = denominators != 0

        def _compute_epsilon(epsilon_new):
            """
            Parameters
            ----------
            epsilon_new: float
                New value for epsilon.

            Returns
            -------
            float
                Newly computed value for threshold.
            """
            # old version (not optimized):
            # for ind1 in range(n_samples):
            #     numerator = 0.0
            #     denominator = 0.0
            #     for ind2 in range(n_samples_per_param):
            #         numerator += (distance_matrix[ind1, ind2] < epsilon_new)
            #         denominator += (distance_matrix[ind1, ind2] < epsilon[-1])
            #     if denominator == 0:
            #         LHS[ind1] = 0
            #     else:
            #         LHS[ind1] = accepted_weights[ind1] * (numerator / denominator)

            numerators = np.sum(distance_matrix < epsilon_new, axis=1)

            LHS = np.zeros(shape=n_samples)

            LHS[non_zero_denominator] = accepted_weights.flatten()[non_zero_denominator] * (
                    numerators[non_zero_denominator] / denominators[non_zero_denominator])
            if sum(LHS) == 0:
                result = RHS
            else:
                LHS = LHS / sum(LHS)  # normalize weights.
                LHS = pow(sum(pow(LHS, 2)), -1)
                result = RHS - LHS

            return result

        return _compute_epsilon

    def _def_compute_epsilon_divergence_unique_particles(self, n_samples, distance_matrix, alpha):
        """
        Parameters
        ----------
        n_samples: integer
            Number of samples to generate.
        alpha: float

        Returns
        -------
        callable
            The function used in the bisection routine
        """

        def _compute_epsilon_divergence_unique_particles(epsilon_new):
            """
            Parameters
            ----------
            epsilon_new: float
                New value for epsilon.
            Returns
            -------
            float
                proportion of unique particles after resampling
            """
            new_weights = (distance_matrix < epsilon_new) * 1
            self.logger.debug('New weights:' + str(new_weights))
            if sum(new_weights) != 0:
                new_weights = new_weights / sum(new_weights)
                rng = np.random.RandomState(1)  # this fixes the randomness across iterations; it makes sense therefore
                # Here we want a proportion of unique particles equal to alpha after resampling
                result = (len(
                    np.unique(rng.choice(n_samples, n_samples, replace=True, p=new_weights))) / n_samples) - alpha
            else:
                result = - alpha
            return result

        return _compute_epsilon_divergence_unique_particles

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

        counter = 0
        # print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
        if self.accepted_parameters_manager.accepted_parameters_bds is None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            counter += 1
        else:
            if self.accepted_parameters_manager.accepted_weights_bds.value()[index] > 0:
                theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]
                while True:
                    perturbation_output = self.perturb(index, rng=rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                        break
                y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                counter += 1
                y_sim_old = self.accepted_y_sim_bds.value()[index]
                # Calculate acceptance probability:
                numerator = 0.0
                denominator = 0.0
                for ind in range(self.n_samples_per_param):
                    numerator += (self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                                         [[y_sim[0][ind]]]) < self.epsilon[-1])
                    # we have most likely already computed this distance before, but hard to keep track. Moreover this
                    # is parallelized -> should not impact too much on computing time
                    denominator += (self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                                           [[y_sim_old[0][ind]]]) < self.epsilon[-1])
                if denominator == 0:
                    ratio_data_epsilon = 1
                else:
                    ratio_data_epsilon = numerator / denominator

                ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model,
                                                                                                             theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                                   perturbation_output[1], theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta,
                                                     perturbation_output[1])
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
        return self.get_parameters(), y_sim, distance, counter

    def _accept_parameter_r_hit_kernel(self, rng_and_index, npc=None):
        """
        This implements algorithm 5 in Lee (2012) [2] which is used as an MCMC kernel in SMCABC. This implementation
        uses r=3 as default value.

        Parameters
        ----------
        rng_and_index: numpy.ndarray
            2 dimensional array. The first entry is a random number generator.
            The second entry defines the index in the data set.

        Returns
        -------
        Tuple
            The first entry of the tuple is the accepted parameters. The second entry is the simulated data set.
            The third one is the distance between the simulated data set and the observation, while the fourth one is
            the number of simulations needed to obtain the accepted parameter.
        """

        rng = rng_and_index[0]
        index = rng_and_index[1]
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        # Set value of r for r-hit kernel
        r = 3 if self.r is None else self.r
        mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
            self.accepted_parameters_manager.model)

        counter = 0
        # print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
        if self.accepted_parameters_manager.accepted_parameters_bds is None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            # the following is was probably already computed before, but hard to keep track:
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            counter += 1
        else:
            if self.accepted_parameters_manager.accepted_weights_bds.value()[index] > 0:
                theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]

                # Sample from theta until we get 'r-1' y_sim inside the epsilon ball (line 4 in Alg 5 in [3])
                self.set_parameters(theta)
                accept_old_arr, N_old = [], 0
                # y_sim_old_arr = []  this is actually not used.
                while len(accept_old_arr) < r - 1:
                    y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                    # y_sim_old_arr.append(y_sim)
                    if self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                              y_sim) < self.epsilon[-1]:
                        accept_old_arr.append(N_old)
                    N_old += 1
                    counter += 1

                # Perturb and sample from the perturbed theta until we get 'r' y_sim inside the epsilon ball 
                # (line 2 in Alg 5 in [3])
                while True:
                    perturbation_output = self.perturb(index, rng=rng)
                    if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                        break
                accept_new_arr, y_sim_new_arr, distance_new_arr, N = [], [], [], 0
                while len(accept_new_arr) < r:
                    y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                    y_sim_new_arr.append(y_sim)
                    distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                                      y_sim)
                    if distance < self.epsilon[-1]:
                        accept_new_arr.append(N)
                    distance_new_arr.append(distance)
                    counter += 1
                    N += 1

                # Calculate acceptance probability
                ratio_prior_prob = self.pdf_of_prior(self.model, perturbation_output[1]) / self.pdf_of_prior(self.model,
                                                                                                             theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                                   perturbation_output[1], theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta,
                                                     perturbation_output[1])
                ratio_likelihood_prob = kernel_numerator / kernel_denominator

                acceptance_prob = min(1, (N_old / (N - 1)) * ratio_prior_prob * ratio_likelihood_prob)

                if rng.binomial(1, acceptance_prob) == 1:
                    self.set_parameters(perturbation_output[1])
                    # Randomly sample index J between the first r-1 hits
                    J = rng.choice(accept_new_arr[:-1]).astype(int)
                    y_sim = y_sim_new_arr[J]
                    distance = distance_new_arr[J]
                else:
                    self.set_parameters(theta)
                    y_sim = self.accepted_y_sim_bds.value()[index]
                    # the following is was probably already computed before, but hard to keep track:
                    distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            else:
                self.set_parameters(self.accepted_parameters_manager.accepted_parameters_bds.value()[index])
                y_sim = self.accepted_y_sim_bds.value()[index]
                # the following distance was probably already computed before?
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
        return self.get_parameters(), y_sim, distance, counter

    def _accept_parameter_r_hit_kernel_version_2(self, rng_and_index, npc=None):
        """
        This implements algorithm 6 in Lee (2012) [2] which is used as an MCMC kernel in SMCABC. This implementation
        uses r=2 as default value.

        Parameters
        ----------
        rng_and_index: numpy.ndarray
            2 dimensional array. The first entry is a random number generator.
            The second entry defines the index in the data set.

        Returns
        -------
        Tuple
            The first entry of the tuple is the accepted parameters. The second entry is the simulated data set.
            The third one is the distance between the simulated data set and the observation, while the fourth one is
            the number of simulations needed to obtain the accepted parameter.
        """

        rng = rng_and_index[0]
        index = rng_and_index[1]
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        # Set value of r for r-hit kernel
        r = 2 if self.r is None else self.r
        mapping_for_kernels, garbage_index = self.accepted_parameters_manager.get_mapping(
            self.accepted_parameters_manager.model)

        counter = 0
        # print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
        if self.accepted_parameters_manager.accepted_parameters_bds is None:
            self.sample_from_prior(rng=rng)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
            distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            # the following is was probably already computed before, but hard to keep track:
            counter += 1
        else:
            if self.accepted_parameters_manager.accepted_weights_bds.value()[index] > 0:
                theta = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]

                # Generate different perturbed values from theta and sample from model until we get 'r' y_sim
                # inside the epsilon ball (line 1 in Alg 6 in [3])
                accept_prime_arr, z_sim_arr, theta_prime_arr, distance_arr, N_prime = [], [], [], [], 0
                while len(accept_prime_arr) < r:
                    # first perturb:
                    while True:
                        # this perturbs using the current theta value
                        perturbation_output = self.perturb(index, rng=rng)
                        if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                            break
                    theta_prime_arr.append(perturbation_output[1])
                    # now simulate
                    z_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)
                    z_sim_arr.append(z_sim)  # could store here only the accepted ones... Can improve
                    distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), z_sim)
                    if distance < self.epsilon[-1]:
                        accept_prime_arr.append(N_prime)
                    distance_arr.append(distance)
                    N_prime += 1
                    counter += 1
                # select the index among the first r-1 hits (line 2 in Alg 6)
                L = rng.choice(accept_prime_arr[:-1]).astype(int)
                theta_prime_L = theta_prime_arr[L]

                # create a new AcceptedParametersManager storing the theta_prime_L in order to draw perturbations from
                # it:
                inner_accepted_parameters_manager = AcceptedParametersManager(self.model)
                # define a dummy backend:
                backend_inner = BackendDummy()
                # need to pass the covariance matrix (take it from the overall AcceptedParametersManager) and the
                # theta_prime_L:
                inner_accepted_parameters_manager.update_broadcast(backend_inner, accepted_parameters=[theta_prime_L],
                                                                   accepted_cov_mats=self.accepted_parameters_manager.accepted_cov_mats_bds.value())

                # in kernel_parameters you need to store theta_prime_L
                # inner_accepted_parameters_manager.update_kernel_values(backend_inner,
                #                                                        kernel_parameters=[[theta_prime_L]])

                # update the kernel parameters (which is used to perturb - this is more general than the one above which
                # works with one kernel only)
                kernel_parameters = []
                for kernel in self.kernel.kernels:
                    kernel_parameters.append(
                        inner_accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))

                inner_accepted_parameters_manager.update_kernel_values(backend_inner,
                                                                       kernel_parameters=kernel_parameters)

                # in kernel_parameters you need to store theta_prime_L
                # inner_accepted_parameters_manager.update_kernel_values(backend_inner,
                #                                                        kernel_parameters=[[theta_prime_L]])

                # Generate different perturbed values from the parameter value selected above and sample from model
                # until we get 'r-1' y_sim inside the epsilon ball (line 3 in Alg 6 in [3])
                accept_arr, N = [], 0
                while len(accept_arr) < r - 1:
                    while True:
                        perturbation_output = self.perturb(0, rng=rng,
                                                           accepted_parameters_manager=inner_accepted_parameters_manager)
                        if perturbation_output[0] and self.pdf_of_prior(self.model, perturbation_output[1]) != 0:
                            break
                    x_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)  #
                    # y_sim_new_arr.append(y_sim)
                    if self.distance.distance(self.accepted_parameters_manager.observations_bds.value(),
                                              x_sim) < self.epsilon[-1]:
                        accept_arr.append(N)
                    counter += 1
                    N += 1

                # Calculate acceptance probability (line 4 in Alg 6)
                ratio_prior_prob = self.pdf_of_prior(self.model, theta_prime_L) / self.pdf_of_prior(self.model,
                                                                                                    theta)
                kernel_numerator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager,
                                                   theta_prime_L, theta)
                kernel_denominator = self.kernel.pdf(mapping_for_kernels, self.accepted_parameters_manager, theta,
                                                     theta_prime_L)
                ratio_likelihood_prob = kernel_numerator / kernel_denominator

                acceptance_prob = min(1, (N / (N_prime - 1)) * ratio_prior_prob * ratio_likelihood_prob)

                if rng.binomial(1, acceptance_prob) == 1:
                    self.set_parameters(theta_prime_L)
                    y_sim = z_sim_arr[L]
                    distance = distance_arr[L]
                else:
                    self.set_parameters(theta)
                    y_sim = self.accepted_y_sim_bds.value()[index]
                    # the following is was probably already computed before, but hard to keep track:
                    distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
            else:
                self.set_parameters(self.accepted_parameters_manager.accepted_parameters_bds.value()[index])
                y_sim = self.accepted_y_sim_bds.value()[index]
                # the following is was probably already computed before, but hard to keep track:
                distance = self.distance.distance(self.accepted_parameters_manager.observations_bds.value(), y_sim)
        return self.get_parameters(), y_sim, distance, counter

    def _compute_accepted_cov_mats(self, covFactor, new_cov_mats):
        """
        Update the covariance matrices computed from data by multiplying them with covFactor and adding a small term in
        the diagonal for numerical stability.

        Parameters
        ----------
        covFactor : float
            factor to correct the covariance matrices
        new_cov_mats : list
            list of covariance matrices computed from data
        Returns
        -------
        list
            List of new accepted covariance matrices
        """
        # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]
        accepted_cov_mats = []
        for new_cov_mat in new_cov_mats:
            if not (new_cov_mat.size == 1):
                accepted_cov_mats.append(
                    covFactor * new_cov_mat + 1e-20 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
            else:
                accepted_cov_mats.append((covFactor * new_cov_mat + 1e-20 * new_cov_mat).reshape(1, 1))
        return accepted_cov_mats


class MCMCMetropoliHastings(BaseLikelihood, InferenceMethod):
    """
    Simple Metropolis-Hastings MCMC working with the approximate likelihood functions Approx_likelihood, with
    multivariate normal proposals.

    Parameters
    ----------
    root_models : list
        A list of the Probabilistic models corresponding to the observed datasets
    loglikfuns : list of abcpy.approx_lhd.Approx_likelihood
        List of Approx_loglikelihood object defining the approximated loglikelihood to be used; one for each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
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

    def __init__(self, root_models, loglikfuns, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Sum of Loglikelihood functions using all the loglikelihoods for each individual models
        self.likfun = SumCombination(root_models, loglikfuns)

        mapping, garbage_index = self._get_mapping()
        models = []
        self.parameter_names_with_index = {}
        for mdl, mdl_index in mapping:
            models.append(mdl)
            self.parameter_names_with_index[mdl.name] = mdl_index  # dict storing param names with index

        self.parameter_names = [model.name for model in models]  # store parameter names

        if kernel is None:
            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        # this is used to handle the data for adapting the covariance:
        self.accepted_parameters_manager_adaptive_cov = AcceptedParametersManager(self.model)

        self.simulation_counter = 0

    def sample(self, observations, n_samples, n_samples_per_param=100, burnin=1000, cov_matrices=None, iniPoint=None,
               adapt_proposal_cov_interval=None, covFactor=None, bounds=None, speedup_dummy=True, use_tqdm=True,
               journal_file=None, path_to_save_journal=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations. The MCMC is run for burnin + n_samples steps, and n_samples_per_param are used at each step
        to estimate the approximate loglikelihood. The burnin steps are then discarded from the chain stored in the
        journal file.

        During burnin, the covariance matrix is adapted from the steps generated up to that point, in a way similar to
        what suggested in [1], after each adapt_proposal_cov_interval steps. Differently from the original algorithm in
        [1], here the proposal covariance matrix is fixed after the end of the burnin steps.

        In case the original parameter space is bounded (for instance with uniform prior on an interval), the MCMC can
        be optionally run on a transformed space. Therefore, the covariance matrix describes proposals on the
        transformed space; the acceptance rate then takes into account the Jacobian of the transformation. In order to
        use MCMC with transformed space, you need to specify lower and upper bounds in the corresponding parameters (see
        details in the description of `bounds`).

        The returned journal file contains also information on acceptance rates (in the configuration dictionary).

        [1] Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. Bernoulli, 7(2), 223-242.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        n_samples : integer, optional
            number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            number of data points in each simulated data set. The default value is 100.
        burnin : integer, optional
            Number of burnin steps to discard. Defaults to 1000.
        cov_matrices : list of matrices, optional
            list of initial covariance matrices for the proposals. If not provided, identity matrices are used. If the 
            sample routine is restarting from a journal file and cov_matrices is not provided, cov_matrices is set to 
            the value used for sampling after burnin in the previous journal file (ie what is stored in 
            `journal.configuration["actual_cov_matrices"]`).
        iniPoint : numpy.ndarray, optional
            parameter value from where the sampling starts. By default sampled from the prior. Not used if journal_file 
            is passed.
        adapt_proposal_cov_interval : integer, optional
            the proposal covariance matrix is adapted each adapt_cov_matrix steps during burnin, by using the chain up
            to that point. If None, no adaptation is done. Default value is None. Use with care as, if the likelihood
            estimate is very noisy, the adaptation may work pooly (see `covFactor` parameter).
        covFactor : float, optional
            the factor by which to scale the empirical covariance matrix in order to obtain the covariance matrix for
            the proposal, whenever that is updated during the burnin steps. If not provided, we use the default value
            2.4 ** 2 / dim_theta suggested in [1].
            Notice that this value was shown to be optimal (at least in some
            limit sense) in the case in which the likelihood is fully known. In the present case, in which the
            likelihood is estimated from data, that value may turn out to be too large; specifically, if
            the likelihood estimate is very noisy, that choice may lead to a very bad adaptation which may give rise
            to an MCMC which does not explore the space well (for instance, the obtained covariance matrix may turn out
            to be too small). If that happens, we suggest to set covFactor to a smaller value than the default one, in
            which case the acceptance rate of the chain will likely be smaller but the exploration will be better.
            Alternatively, it is possible to reduce the noise in the likelihood estimate by increasing
            `n_samples_per_param`.
        bounds : dictionary, optional
            dictionary containing the lower and upper bound for the transformation to be applied to the parameters. The
            key of each entry is the name of the parameter as defined in the model, while the value if a tuple (or list)
            with `(lower_bound, upper_bound)` content. If the parameter is bounded on one side only, the other bound
            should be set to 'None'. If a parameter is not in this dictionary, no transformation is applied to it.
            If a parameter is bounded on two sides, the used transformation is based on the logit. If conversely it is
            lower bounded, we apply instead a log transformation. Notice that we do not implement yet the transformation
            for upper bounded variables. If no value is provided, the default value is None, which means no
            transformation at all is applied.
        speedup_dummy: boolean, optional.
            If set to True, the map function is not used to parallelize simulations (for the new parameter value) when
            the backend is Dummy. This can improve performance as it can exploit potential vectorization in the model.
            However, this breaks reproducibility when using, for instance, BackendMPI with respect to BackendDummy, due
            to the different way the random seeds are used when speedup_dummy is set to True. Please set this to False
            if you are interested in preserving reproducibility across MPI and Dummy backend. Defaults to True.
        use_tqdm : boolean, optional
            Whether using tqdm or not to display progress. Defaults to True.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            That's the only information used (it does not use the previous covariance matrix).
            The default value is None.
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path at the end of the inference routine.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations = observations
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.speedup_dummy = speedup_dummy
        # we use this in all places which require a backend but which are not parallelized in MCMC:
        self.dummy_backend = BackendDummy()
        dim = len(self.parameter_names)

        if path_to_save_journal is not None:
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if bounds is None:
            # no transformation is performed
            self.transformer = DummyTransformer()
        else:
            if not isinstance(bounds, dict):
                raise TypeError("Argument `bounds` need to be a dictionary")
            bounds_keys = bounds.keys()
            for key in bounds_keys:
                if key not in self.parameter_names:
                    raise KeyError("The keys in argument `bounds` need to correspond to the parameter names used "
                                   "in defining the model")
                if not hasattr(bounds[key], "__len__") or len(bounds[key]) != 2:
                    raise RuntimeError("Each entry in `bounds` need to be a tuple with 2 value, representing the lower "
                                       "and upper bound of the corresponding parameter. If the parameter is bounded on "
                                       "one side only, the other bound should be set to 'None'.")

            # create lower_bounds and upper_bounds_vector:
            lower_bound_transformer = np.array([None] * dim)
            upper_bound_transformer = np.array([None] * dim)

            for key in bounds_keys:
                lower_bound_transformer[self.parameter_names_with_index[key]] = bounds[key][0]
                upper_bound_transformer[self.parameter_names_with_index[key]] = bounds[key][1]

            # initialize transformer:
            self.transformer = BoundedVarTransformer(np.array(lower_bound_transformer),
                                                     np.array(upper_bound_transformer))

        accepted_parameters = []
        accepted_parameters_burnin = []
        if journal_file is None:
            journal = Journal(0)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_lhd_func"] = [type(likfun).__name__ for likfun in self.likfun.approx_lhds]
            journal.configuration["type_kernel_func"] = [type(kernel).__name__ for kernel in self.kernel.kernels] if \
                isinstance(self.kernel, JointPerturbationKernel) else type(self.kernel)
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["burnin"] = burnin
            journal.configuration["cov_matrices"] = cov_matrices
            journal.configuration["iniPoint"] = iniPoint
            journal.configuration["adapt_proposal_cov_interval"] = adapt_proposal_cov_interval
            journal.configuration["covFactor"] = covFactor
            journal.configuration["bounds"] = bounds
            journal.configuration["speedup_dummy"] = speedup_dummy
            journal.configuration["use_tqdm"] = use_tqdm
            journal.configuration["acceptance_rates"] = []
            # Initialize chain: when not supplied, randomly draw it from prior distribution
            # It is an MCMC chain: weights are always 1; forget about them
            # accepted_parameter will keep track of the chain position
            if iniPoint is None:
                self.sample_from_prior(rng=self.rng)
                accepted_parameter = self.get_parameters()
            else:
                accepted_parameter = iniPoint
                if isinstance(accepted_parameter, np.ndarray) and len(accepted_parameter.shape) == 1 or isinstance(
                        accepted_parameter, list) and not hasattr(accepted_parameter[0], "__len__"):
                    # reshape whether we pass a 1d array or list.
                    accepted_parameter = [np.array([x]) for x in accepted_parameter]  # give correct shape for later
            if burnin == 0:
                accepted_parameters_burnin.append(accepted_parameter)
            self.logger.info("Calculate approximate loglikelihood")
            approx_log_likelihood_accepted_parameter = self._simulate_and_compute_log_lik(accepted_parameter)
            # update the number of simulations (this tracks the number of parameters for which simulations are done;
            # the actual number of simulations is this times n_samples_per_param))
            self.simulation_counter += 1
            self.acceptance_rate = 0
        else:
            # check the following:
            self.logger.info("Restarting from previous journal")
            journal = Journal.fromFile(journal_file)
            # this is used to compute the overall acceptance rate:
            self.acceptance_rate = journal.configuration["acceptance_rates"][-1] * journal.configuration["n_samples"]
            accepted_parameter = journal.get_accepted_parameters(-1)[-1]  # go on from last MCMC step
            journal.configuration["n_samples"] += self.n_samples  # add the total number of samples
            journal.configuration["burnin"] = burnin
            if journal.configuration["n_samples_per_param"] != self.n_samples_per_param:
                warnings.warn("You specified a different n_samples_per_param from the one used in the passed "
                              "journal_file; the algorithm will still work fine.")
                journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["cov_matrices"] = cov_matrices
            journal.configuration["bounds"] = bounds  # overwrite
            if cov_matrices is None:  # use the previously stored one unless the user defines it
                cov_matrices = journal.configuration["actual_cov_matrices"]
            journal.configuration["speedup_dummy"] = speedup_dummy
            approx_log_likelihood_accepted_parameter = journal.final_step_loglik
            self.simulation_counter = journal.number_of_simulations[-1]  # update the number of simulations

        if covFactor is None:
            covFactor = 2.4 ** 2 / dim

        accepted_parameter_prior_pdf = self.pdf_of_prior(self.model, accepted_parameter)

        # set the accepted parameter in the kernel (in order to correctly generate next proposal)
        # do this on transformed parameter
        accepted_parameter_transformed = self.transformer.transform(accepted_parameter)
        self._update_kernel_parameters(accepted_parameter_transformed)

        # compute jacobian
        log_det_jac_accepted_param = self.transformer.jac_log_det(accepted_parameter)

        # 3: calculate covariance
        self.logger.info("Set kernel covariance matrix ")
        if cov_matrices is None:
            # need to set that to some value (we use identity matrices). Be careful, there need to be one
            # covariance matrix for each kernel; not sure that this works in case of multivariate parameters.

            # the kernel parameters are only used to get the exact shape of cov_matrices
            cov_matrices = [np.eye(len(self.accepted_parameters_manager.kernel_parameters_bds.value()[0][kernel_index]))
                            for kernel_index in range(len(self.kernel.kernels))]

        self.accepted_parameters_manager.update_broadcast(self.dummy_backend, accepted_cov_mats=cov_matrices)

        # main MCMC algorithm
        self.logger.info("Starting MCMC")
        for aStep in tqdm(range(burnin + n_samples), disable=not use_tqdm):

            self.logger.debug("Step {} of MCMC algorithm".format(aStep))

            # 1: Resample parameters
            self.logger.debug("Generate proposal")

            # perturb element 0 of accepted_parameters_manager.kernel_parameters_bds:
            # new_parameter = self.perturb(0, rng=self.rng)[1]  # do not use this as it leads to some weird error.
            # rather do:
            new_parameters_transformed = self.kernel.update(self.accepted_parameters_manager, 0, rng=self.rng)

            self._reset_flags()  # not sure whether this is needed, leave it anyway

            # Order the parameters provided by the kernel in depth-first search order
            new_parameter_transformed = self.get_correct_ordering(new_parameters_transformed)

            # transform back
            new_parameter = self.transformer.inverse_transform(new_parameter_transformed)

            # for now we are only using a simple MVN proposal. For bounded parameter values, this is not great; we
            # could also implement a proposal on transformed space, which would be better.
            new_parameter_prior_pdf = self.pdf_of_prior(self.model, new_parameter)
            if new_parameter_prior_pdf == 0:
                self.logger.debug("Proposal parameter at step {} is out of prior region.".format(aStep))
                if aStep >= burnin:
                    accepted_parameters.append(accepted_parameter)
                else:
                    accepted_parameters_burnin.append(accepted_parameter)
                continue

            # 2: calculate approximate likelihood for new parameter. If the backend is MPI, we distribute simulations
            # and then compute the approx likelihood locally
            self.logger.debug("Calculate approximate loglikelihood")
            approx_log_likelihood_new_parameter = self._simulate_and_compute_log_lik(new_parameter)
            self.simulation_counter += 1  # update the number of simulations

            log_det_jac_new_param = self.transformer.jac_log_det(new_parameter)
            # log_det_jac_accepted_param = self.transformer.jac_log_det(accepted_parameter)
            log_jac_term = log_det_jac_accepted_param - log_det_jac_new_param

            # compute acceptance rate:
            alpha = np.exp(
                log_jac_term + approx_log_likelihood_new_parameter - approx_log_likelihood_accepted_parameter) * (
                        new_parameter_prior_pdf) / (accepted_parameter_prior_pdf)  # assumes symmetric kernel

            # Metropolis-Hastings step:
            if self.rng.uniform() < alpha:
                # update param value and approx likelihood
                accepted_parameter_transformed = new_parameter_transformed
                accepted_parameter = new_parameter
                approx_log_likelihood_accepted_parameter = approx_log_likelihood_new_parameter
                accepted_parameter_prior_pdf = new_parameter_prior_pdf
                log_det_jac_accepted_param = log_det_jac_new_param
                # set the accepted parameter in the kernel (in order to correctly generate next proposal)
                self._update_kernel_parameters(accepted_parameter_transformed)
                if aStep >= burnin:
                    self.acceptance_rate += 1

            # save to the trace:
            if aStep >= burnin:
                accepted_parameters.append(accepted_parameter)
            else:
                accepted_parameters_burnin.append(accepted_parameter)

                # adapt covariance of proposal:
                if adapt_proposal_cov_interval is not None and (aStep + 1) % adapt_proposal_cov_interval == 0:
                    # store the accepted_parameters for adapting the covariance in the kernel.
                    # I use this piece of code as it formats the data in the right way
                    # for the sake of using them to compute the kernel cov:
                    self.accepted_parameters_manager_adaptive_cov.update_broadcast(
                        self.dummy_backend, accepted_parameters=accepted_parameters_burnin)
                    kernel_parameters = []
                    for kernel in self.kernel.kernels:
                        kernel_parameters.append(
                            self.accepted_parameters_manager_adaptive_cov.get_accepted_parameters_bds_values(
                                kernel.models))
                    self.accepted_parameters_manager_adaptive_cov.update_kernel_values(
                        self.dummy_backend, kernel_parameters=kernel_parameters)

                    self.logger.info("Updating covariance matrix")
                    cov_matrices = self.kernel.calculate_cov(self.accepted_parameters_manager_adaptive_cov)
                    # this scales with the cov_Factor:
                    cov_matrices = self._compute_accepted_cov_mats(covFactor, cov_matrices)
                    # store it in the main AcceptedParametersManager in order to perturb data with it in the following:
                    self.accepted_parameters_manager.update_broadcast(self.dummy_backend,
                                                                      accepted_cov_mats=cov_matrices)

        self.acceptance_rate /= journal.configuration["n_samples"]
        self.logger.info("Saving results to output journal")
        self.accepted_parameters_manager.update_broadcast(self.dummy_backend, accepted_parameters=accepted_parameters)
        names_and_parameters = self._get_names_and_parameters()
        if journal_file is not None:  # concatenate chains
            journal.add_accepted_parameters(journal.get_accepted_parameters() + copy.deepcopy(accepted_parameters))
            names_and_parameters = [(names_and_parameters[i][0],
                                     journal.get_parameters()[names_and_parameters[i][0]] + names_and_parameters[i][1])
                                    for i in range(len(names_and_parameters))]
            journal.add_user_parameters(names_and_parameters)
        else:
            journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
            journal.add_user_parameters(names_and_parameters)
        journal.number_of_simulations.append(self.simulation_counter)
        journal.configuration["acceptance_rates"].append(self.acceptance_rate)
        journal.add_weights(np.ones((journal.configuration['n_samples'], 1)))
        # store the final loglik to be able to restart the journal correctly
        journal.final_step_loglik = approx_log_likelihood_accepted_parameter
        # store the final actual cov_matrices, in order to use this when restarting from journal
        journal.configuration["actual_cov_matrices"] = cov_matrices

        if path_to_save_journal is not None:  # save journal
            journal.save(path_to_save_journal)

        return journal

    def _sample_parameter(self, rng, npc=None):
        """
        Generate a simulation from the model with the current value of accepted_parameter

        Parameters
        ----------
        rng: random number generator
            The random number generator to be used.
        Returns
        -------
        np.array
            accepted parameter
        """

        # get the new parameter value
        theta = self.new_parameter_bds.value()
        # Simulate the fake data from the model given the parameter value theta
        self.logger.debug("Simulate model for parameter " + str(theta))
        acc = self.set_parameters(theta)
        if acc is False:
            self.logger.debug("Parameter " + str(theta) + "has not been accepted")
        y_sim = self.simulate(1, rng=rng, npc=npc)

        return y_sim

    def _approx_log_lik_calc(self, y_sim, npc=None):
        """
        Compute likelihood for new parameters using approximate likelihood function

        Parameters
        ----------
        y_sim: list
            A list containing self.n_samples_per_param simulations for the new parameter value
        Returns
        -------
        float
            The approximated likelihood function
        """
        self.logger.debug("Extracting observation.")
        obs = self.observations

        self.logger.debug("Computing likelihood...")
        loglhd = self.likfun.loglikelihood(obs, y_sim)

        self.logger.debug("LogLikelihood is :" + str(loglhd))

        return loglhd

    def _simulate_and_compute_log_lik(self, new_parameter):
        """Helper function which simulates data from `new_parameter` and computes the approximate loglikelihood.
        In case the backend is not BackendDummy (ie parallelization is available) this parallelizes the different
        simulations (which are all for the same parameter value).

        Notice that, according to the used model, spreading the simulations in different tasks can be more inefficient
        than using one single call, according to the level of vectorization that the model uses and the overhead
        associated. For this reason, we do not split the simulations in different tasks when the backend is
        BackendDummy.

        Parameters
        ----------
        new_parameter
            Parameter value from which to generate data with which to compute the approximate loglikelihood.

        Returns
        -------
        float
            The approximated likelihood function
        """
        if isinstance(self.backend, BackendDummy) and self.speedup_dummy:
            # do all the required simulations here without parallellizing; however this gives different result
            # from the other option due to the way random seeds are handled.
            self.logger.debug('simulations')
            theta = new_parameter
            # Simulate the fake data from the model given the parameter value theta
            self.logger.debug("Simulate model for parameter " + str(theta))
            acc = self.set_parameters(theta)
            if acc is False:
                self.logger.debug("Parameter " + str(theta) + "has not been accepted")
            simulations_from_new_parameter = self.simulate(n_samples_per_param=self.n_samples_per_param, rng=self.rng)
        else:
            self.logger.debug('parallelize simulations for fixed parameter value')
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=self.n_samples_per_param, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # need first to broadcast the new_parameter value:
            self.new_parameter_bds = self.backend.broadcast(new_parameter)

            # map step:
            simulations_from_new_parameter_pds = self.backend.map(self._sample_parameter, rng_pds)
            self.logger.debug("collect simulations from pds")
            simulations_from_new_parameter = self.backend.collect(simulations_from_new_parameter_pds)
            # now need to reshape that correctly. The first index has to be the model, then we need to have
            # n_samples_per_param and then the size of the simulation
            simulations_from_new_parameter = [
                [simulations_from_new_parameter[sample_index][model_index][0] for sample_index in
                 range(self.n_samples_per_param)] for model_index in range(len(self.model))]
        approx_log_likelihood_new_parameter = self._approx_log_lik_calc(simulations_from_new_parameter)

        return approx_log_likelihood_new_parameter

    def _update_kernel_parameters(self, accepted_parameter):
        """This stores the last accepted parameter in the kernel so that it will be used to generate the new proposal
        with self.perturb.

        Parameters
        ----------
        accepted_parameter
            Parameter value from which you want to generate proposal at next iteration of MCMC.
        """
        # I use this piece of code as it formats the data in the right way for the sake of using it in the kernel
        self.accepted_parameters_manager.update_broadcast(self.dummy_backend, accepted_parameters=[accepted_parameter])

        kernel_parameters = []
        for kernel in self.kernel.kernels:
            kernel_parameters.append(
                self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
        self.accepted_parameters_manager.update_kernel_values(self.dummy_backend, kernel_parameters=kernel_parameters)

    @staticmethod
    def _compute_accepted_cov_mats(covFactor, new_cov_mats):
        """
        Update the covariance matrices computed from data by multiplying them with covFactor and adding a small term in
        the diagonal for numerical stability.

        Parameters
        ----------
        covFactor : float
            factor to correct the covariance matrices
        new_cov_mats : list
            list of covariance matrices computed from data
        Returns
        -------
        list
            List of new accepted covariance matrices
        """
        # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]
        accepted_cov_mats = []
        for new_cov_mat in new_cov_mats:
            if not (new_cov_mat.size == 1):
                accepted_cov_mats.append(
                    covFactor * new_cov_mat + 1e-20 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
            else:
                accepted_cov_mats.append((covFactor * new_cov_mat + 1e-20 * new_cov_mat).reshape(1, 1))
        return accepted_cov_mats
