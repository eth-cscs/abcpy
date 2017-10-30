from abc import ABCMeta, abstractmethod, abstractproperty

from ProbabilisticModel import *
import numpy as np
from abcpy.output import Journal
from scipy import optimize

#TODO check whether if we set a seed as well as an rng for the distributions, what happens.

#TODO check whether something like for j in range(self.parent.dimension) is done, since this will be wrong for hyperparameters

#NOTE since the output of discrete and continuous parameters is split, it will not necessarily give the correct order compared to the user input

"""
- This covariance matrix calculation is done assuming all the parameters are continuous. But if you have some of the parameters being discrete, then it breaks down.
- So for the moment do the following:
a) group continuous parameters and discrete parameters.
b) for continuous parameters do as we have done before. meaning computing cov matrix and providing that to multi-normal or muti-t to generate a new sample.
c) for discrete parameters, user should specify a distribution which can take the present values as input and give a discrete output. 
"""


#NOTE since we still need the new_parameters_pds to calculate the cov matrix, we will probably need to split up our broadcast data in discrete and continuous. If we split this anyways, do we want there even to be an option to just give back the combined object, or only single ones? We could also implement set_continuous/set_discrete together with get of both, without set/get in total!
class InferenceMethod(metaclass = ABCMeta):
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


    def sample_from_prior(self, models, rng=np.random.RandomState()):
        """
        Samples values for the parameters of the specified model as well as all its parents.
        Commonly used to sample new parameter values on the whole graph.

        Parameters
        ----------
        models: list of probabilistic models
            Defines the models for which, together with their parents, new parameters will be sampled
        rng: Random number generator to be used
            Defines the random number generator to be used
        """

        #all the parents of each model recursively sample new parameter values.
        for model in models:
            for parent in model.parents:
                if(not(parent.visited)):
                    self.sample_from_prior([parent], rng=rng)
            #each model itself samples new parameters
            model.sample_parameters(rng=rng)

    def _reset_flags(self, models):
        """
        Resets all flags that say that a probabilistic model has been updated.
        Commonly used after actions on the whole graph, to ensure that new actions can take place.

        Parameters
        ----------
        models: list of probabilistic models
            The models for which, together with their parents, the flags should be reset.
        """

        #for each model, the flags of the parents get reset recursively.
        for model in models:
            for parent in model.parents:
                if(parent.visited):
                    self._reset_flags([parent])
            model.visited = False


    #NOTE not tested yet, not sure whether this works.
    #TODO CHECK WHETHER THIS COVERS ALL 3 DIFFERENT CASES
    def pdf_of_prior(self, models, parameters, index, is_root=True):
        """
        Calculates the joint probability density function of the prior of the specified models at the given parameter values.
        Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

        Parameters
        ----------
        models: list of probabilistic models
            Defines the models for which the pdf of their prior should be evaluated
        parameters: python list
            The parameters at which the pdf should be evaluated
        index: integer
            The current index to be considered within the parameters list
        is_root: boolean
            A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

        Returns:
        list
            The resulting pdf, as well as the next index to be considered in the parameters list.
        """
        result = [1.]*len(models)
        for i, model in enumerate(models):
            #if the model is not a root model, the pdf of this model, given the prior, should be calculated
            if(not(is_root)):
                helper = []
                #this loop will skip hyperparameters, which doesn't matter due to our definition of the pdf
                for j in range(model.dimension):
                    helper.append(parameters[index])
                    index+=1
                if(len(helper)==1):
                    helper = helper[0]
                else:
                    helper = np.array(helper)
                result[i]*=model.pdf(helper)
            #for each parent, the pdf of this parent has to be calculated as well.
            for parent in model.parents:
                pdf = self.pdf_of_prior([parent], parameters, index, is_root=False)
                result[i]*=pdf[0][0]
                index=pdf[1]
        return [result, index]


    def get_cont_disc_parameters(self, models):
        """
        Returns the current values of all free parameters in the model.
        Commonly used before perturbing the parameters of the model.

        Parameters
        ----------
        models: list of probabilistic models
            The models for which, together with their parents, the parameter values should be returned

        Returns
        -------
        list
            Two lists, the first containing all continuous parameter values, the second containing all discrete parameter values.
        """
        parameters_continuous = []
        parameters_discrete = []
        for model in models:
            # append the current values of the free parameters of the model
            for parameter in model.get_parameters():
                if (isinstance(parameter, list)):
                    for param in parameter:
                        if(isinstance(model, Continuous)):
                            parameters_continuous.append(param)
                        else:
                            parameters_discrete.append(param)
                else:
                    if(isinstance(model, Continuous)):
                        parameters_continuous.append(parameter)
                    else:
                        parameters_discrete.append(parameter)
            # append the current values of the free parameters of each parent in order of a dfs.
            for parent in model.parents:
                if (not (parent.visited)):
                    parent_parameters_cont, parent_parameters_disc = self.get_parameters([parent])
                    for parameter in parent_parameters_cont:
                        if (isinstance(parameter, list)):
                            for param in parameter:
                                parameters_continuous.append(param)
                        else:
                            parameters_continuous.append(parameter)
                    for parameter in parent_parameters_disc:
                        if(isinstance(parameter, list)):
                            for param in parameter:
                                parameters_discrete.append(param)
                        else:
                            parameters_discrete.append(parameter)
                    # the parent nodes are marked as visited to avoid duplicated values
                    parent.visited = True
            model.visited = True
        return parameters_continuous, parameters_discrete

    def get_parameters(self, models):
        parameters_continuous, parameters_discrete = self._get_cont_disc_parameters(models)
        parameters = []
        for parameter in parameters_continuous:
            parameters.append(parameter)
        for parameter in parameters_discrete:
            parameters.append(parameter)
        return parameters

    def number_of_continuous_parameters(self, models):
        result = 0
        for model in models:
            if(isinstance(model, Continuous)):
                result+=1
            for parent in model.parents:
                result+=self.number_of_continuous_parameters([parent])
                parent.visited = True
            model.visited = True
        return result


    #NOTE returns false iff we couldnt set some node, in that case, use the old parameters again to resample
    def set_parameters(self, models, parameters, index_continuous, index_discrete):
        """
        Sets the parameter values of the model, as well as of its parents, to the specified values.
        Commonly used after perturbing the parameter values using a kernel.

        Parameters
        ----------
        model: list of probabilistic models
             Defines all models for which, together with their parents, new values should be set
        parameters: list
            Defines the values to which the respective parameter values of the models should be set
        index: integer
            The current index to be considered in the parameters list

        Returns
        -------
        boolean
            Returns True iff it was possible to set the values for all models as well as their parents.
        """
        for model in models:
            if(isinstance(model, Continuous)):
                #if possible, set the parameters of the current model to the appropriate values
                if(not(model.set_parameters(parameters[index_continuous:model.number_of_free_parameters()]))):
                    return False
                index_continuous+=model.dimension
            else:
                if(not(model.set_parameters(parameters[index_discrete:model.number_of_free_parameters()]))):
                    return False
                index_discrete+=model.dimension
            #set the parameters of each parent recursively, in order of dfs.
            for parent in model.parents:
                if(not(parent.visited)):
                    if(not(self.set_parameters([parent], parameters, index_continuous, index_discrete))):
                        return False
                    #marks set nodes as visited since values within the parameters list are not duplicated
                    parent.visited = True
            model.visited = True
        return True



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
    def n_samples(self):
        """To be overwritten by any sub-class: an attribute specifying the number of samples to be generated
        """
        raise NotImplementedError

    @abstractproperty
    def n_samples_per_param(self):
        """To be overwritten by any sub-class: an attribute specifying the number of data points in each simulated         data set."""
        raise NotImplementedError

    @abstractproperty
    def observations_bds(self):
        """To be overwritten by any sub-class: an attribute saving the observations as bds
        """
        raise NotImplementedError


class BasePMC(InferenceMethod, metaclass = ABCMeta):
    """
            This abstract base class represents inference methods that use Population Monte Carlo.

    """
    @abstractmethod
    def _update_broadcasts(self, accepted_parameters, accepted_weights, accepted_cov_mat):
        """
        To be overwritten by any sub-class: broadcasts visited values

        Parameters
        ----------
        accepted_parameters: numpy.array
            Contains all new accepted parameters.
        accepted_weights: numpy.array
            Contains all the new accepted weights.
        accepted_cov_mat: numpy.ndarray
            Contains the new accepted covariance matrix

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_weight(self, theta):
        """
        To be overwritten by any sub-class:
        Calculates the weight for the given parameter using
        accepted_parameters, accepted_cov_mat

        Parameters
        ----------
        theta: np.array
            1xp matrix containing the model parameters, where p is the dimension of parameters

        Returns
        -------
        float
            the new weight for theta
        """
        raise NotImplementedError

    @abstractproperty
    def kernel(self):
        """To be overwritten by any sub-class: an attribute specifying the kernel to be used
        """
        raise NotImplementedError

    @abstractproperty
    def accepted_parameters_bds(self):
        """To be overwritten by any sub-class: an attribute saving the accepted parameters as bds
        """
        raise NotImplementedError

    @abstractproperty
    def accepted_weights_bds(self):
        """To be overwritten by any sub-class: an attribute saving the accepted weights as bds
        """
        raise NotImplementedError

    @abstractproperty
    def accepted_cov_mat_bds(self):
        """To be overwritten by any sub-class: an attribute saving the accepted covariance matrix as bds
        """
        raise NotImplementedError



class BaseAnnealing(InferenceMethod, metaclass = ABCMeta):
    """
            This abstract base class represents inference methods that use annealing.

    """

    @abstractmethod
    def _update_broadcasts(self):
        raise NotImplementedError

    @abstractmethod
    def _accept_parameter(self):
        raise NotImplementedError

    @abstractproperty
    def distance(self):
        """To be overwritten by any sub-class: an attribute specifying the distance measure to be used
        """
        raise NotImplementedError

    @abstractproperty
    def kernel(self):
        """To be overwritten by any sub-class: an attribute specifying the kernel to be used
        """
        raise NotImplementedError

    @abstractproperty
    def accepted_parameters_bds(self):
        """To be overwritten by any sub-class: an attribute saving the accepted parameters as bds
        """
        raise NotImplementedError

    @abstractproperty
    def accepted_cov_mat_bds(self):
        """To be overwritten by any sub-class: an attribute saving the accepted covariance matrix as bds
        """
        raise NotImplementedError

class BaseAdaptivePopulationMC(InferenceMethod, metaclass = ABCMeta):
    """
            This abstract base class represents inference methods that use Adaptive Population Monte Carlo.

    """

    @abstractmethod
    def _update_broadcasts(self):
        """
        To be overwritten by any sub-class: broadcasts visited values

        Parameters
        ----------
        accepted_parameters: numpy.array
            Contains all new accepted parameters.
        accepted_weights: numpy.array
            Contains all the new accepted weights.
        accepted_cov_mat: numpy.ndarray
            Contains the new accepted covariance matrix

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def _accept_parameter(self):
        """
        To be overwritten by any sub-class:
        Samples a single model parameter and simulate from it until
        accepted with some probability.

        """
        raise NotImplementedError

    @abstractproperty
    def distance(self):
        """To be overwritten by any sub-class: an attribute specifying the distance measure to be used
        """
        raise NotImplementedError

    @abstractproperty
    def kernel(self):
        """To be overwritten by any sub-class: an attribute specifying the kernel to be used
        """
        raise NotImplementedError

    @abstractproperty
    def accepted_parameters_bds(self):
        """To be overwritten by any sub-class: an attribute saving the accepted parameters as bds
        """
        raise NotImplementedError

    @abstractproperty
    def accepted_cov_mat_bds(self):
        """To be overwritten by any sub-class: an attribute saving the accepted covariance matrix as bds
        """
        raise NotImplementedError

class RejectionABC(InferenceMethod):
    """This base class implements the rejection algorithm based inference scheme [1] for
        Approximate Bayesian Computation.

        [1] Tavaré, S., Balding, D., Griffith, R., Donnelly, P.: Inferring coalescence
        times from DNA sequence data. Genetics 145(2), 505–518 (1997).

        Parameters
        ----------
        model: abcpy.models.Model
            Model object defining the model to be used.
        distance: abcpy.distances.Distance
            Distance object defining the distance measure to compare simulated and observed data sets.
        backend: abcpy.backends.Backend
            Backend object defining the backend to be used.
        seed: integer, optional
             Optional initial seed for the random number generator. The default value is generated randomly.
        """

    model = None
    distance = None
    rng = None

    n_samples = None
    n_samples_per_param = None
    epsilon = None

    observations_bds = None

    def __init__(self, model, distance, backend, seed=None):
        self.model = model
        self.distance = distance
        self.backend = backend
        self.rng = np.random.RandomState(seed)

    def sample(self, observations, n_samples, n_samples_per_param, epsilon, full_output=0):
        """
        Samples from the posterior distribution of the model parameter given the observed
        data observations.
        Parameters
        ----------
        observations: numpy.ndarray
            Observed data.
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

        self.observations_bds = self.backend.broadcast(observations)
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

        accepted_parameters_pds = self.backend.map(self._sample_parameter, rng_pds)
        accepted_parameters = self.backend.collect(accepted_parameters_pds)
        accepted_parameters = np.array(accepted_parameters)

        journal.add_parameters(accepted_parameters)
        journal.add_weights(np.ones((n_samples, 1)))

        return journal

    def _sample_parameter(self, rng):
        """
        Samples a single model parameter and simulates from it until
        distance between simulated outcome and the observation is
        smaller than epsilon.

        Parameters
        ----------
        seed: int
            value of a seed to be used for reseeding
        Returns
        -------
        np.array
            accepted parameter
        """
        distance = self.distance.dist_max()

        while distance > self.epsilon:
            # Accept new parameter value if the distance is less than epsilon
            self.sample_from_prior(self.model, rng=rng)
            self._reset_flags(self.model)
            #TODO WHAT SHOULD HAPPEN IF WE HAVE MORE THAN ONE MODEL
            #NOTE this gives reasonable values for the y_sim. However, the distances are very large, possibly because of the algorithm used, not sure
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
            distance = self.distance.distance(self.observations_bds.value(), y_sim)
            #print(distance)
        return self.get_parameters(self.model)

class PMCABC(BasePMC, InferenceMethod):
    """
    This base class implements a modified version of Population Monte Carlo based inference scheme
    for Approximate Bayesian computation of Beaumont et. al. [1]. Here the threshold value at `t`-th generation are adaptively chosen
    by taking the maximum between the epsilon_percentile-th value of discrepancies of the accepted
    parameters at `t-1`-th generation and the threshold value provided for this generation by the user. If we take the
    value of epsilon_percentile to be zero (default), this method becomes the inference scheme described in [1], where
    the threshold values considered at each generation are the ones provided by the user.

    [1] M. A. Beaumont. Approximate Bayesian computation in evolution and ecology. Annual Review of Ecology,
    Evolution, and Systematics, 41(1):379–406, Nov. 2010.

    Parameters
    ----------
    model : abcpy.models.Model
        Model object defining the model to be used.
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

    observations_bds = None
    accepted_parameters_bds = None
    accepted_weights_bds = None
    accepted_cov_mat_bds = None


    def __init__(self, model, distance, kernel, backend, seed=None):

        self.model = model
        self.distance = distance
        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_weights_bds = None
        self.accepted_cov_mat_bds = None



    def sample(self, observations, steps, epsilon_init, n_samples = 10000, n_samples_per_param = 1, epsilon_percentile = 0, covFactor = 2, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : numpy.ndarray
            Observed data.
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
            A value between [0, 100]. The default value is 0, meaning the threshold value provided by the user being used.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2 as considered in [1].
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations_bds = self.backend.broadcast(observations)
        self.n_samples = n_samples
        self.n_samples_per_param=n_samples_per_param

        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["steps"] = steps
        journal.configuration["epsilon_percentile"] = epsilon_percentile

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mat = None

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
        # print("INFO: Starting PMCABC iterations.")
        for aStep in range(0, steps):
            # print("DEBUG: Iteration " + str(aStep) + " of PMCABC algorithm.")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            self.epsilon = epsilon_arr[aStep]
            self._update_broadcasts(accepted_parameters, accepted_weights, accepted_cov_mat)

            # 1: calculate resample parameters
            # print("INFO: Resampling parameters")
            params_and_dists_and_ysim_pds = self.backend.map(self._resample_parameter, rng_pds)
            params_and_dists_and_ysim = self.backend.collect(params_and_dists_and_ysim_pds)
            new_parameters, distances = [list(t) for t in zip(*params_and_dists_and_ysim)]
            new_parameters = np.array(new_parameters)
            self._update_broadcasts(accepted_parameters, accepted_weights, accepted_cov_mat)

            # Compute epsilon for next step
            # print("INFO: Calculating acceptance threshold (epsilon).")
            if aStep < steps - 1:
                if epsilon_arr[aStep + 1] == None:
                    epsilon_arr[aStep + 1] = np.percentile(distances, epsilon_percentile)
                else:
                    epsilon_arr[aStep + 1] = np.max(
                        [np.percentile(distances, epsilon_percentile), epsilon_arr[aStep + 1]])

            # 2: calculate weights for new parameters
            # print("INFO: Calculating weights.")
            new_parameters_pds = self.backend.parallelize(new_parameters)
            new_weights_pds = self.backend.map(self._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1, 1)
            sum_of_weights = 0.0
            for w in new_weights:
                sum_of_weights += w
            new_weights = new_weights / sum_of_weights

            # 3: calculate covariance
            # print("INFO: Calculating covariance matrix.")
            new_cov_mat = covFactor * np.cov(new_parameters, aweights=new_weights.reshape(-1), rowvar=False)

            # 4: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mat = new_cov_mat

            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)

        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon_arr

        return journal

    def _update_broadcasts(self, accepted_parameters, accepted_weights, accepted_cov_mat):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = self.backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = self.backend.broadcast(accepted_weights)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = self.backend.broadcast(accepted_cov_mat)

    # define helper functions for map step
    def _resample_parameter(self, rng):
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
        rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        self.kernel.rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        distance = self.distance.dist_max()
        while distance > self.epsilon:
            # print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
            if self.accepted_parameters_bds == None:
                self.sample_from_prior(self.model, rng=rng)
                self._reset_flags(self.model)
                theta = self.get_parameters(self.model)
                self._reset_flags(self.model)
            else:
                index = rng.choice(self.n_samples, size=1, p=self.accepted_weights_bds.value().reshape(-1))
                theta = self.accepted_parameters_bds.value()[index[0]]
                # truncate the normal to the bounds of parameter space of the model
                # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf
                while True:
                    new_theta = self.kernel.perturb(theta, self.accepted_cov_mat_bds.value())
                    theta_is_accepted = self.set_parameters(self.model, new_theta, 0)
                    self._reset_flags(self.model)
                    if theta_is_accepted and self.pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                        break
            #TODO WHAT IF MORE THAN 1 MODEL
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()

            distance = self.distance.distance(self.observations_bds.value(), y_sim)
        return (theta, distance)

    def _calculate_weight(self, theta):
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

        if self.accepted_weights_bds is None:
            return 1.0 / self.n_samples
        else:
            prior_prob = self.pdf_of_prior(self.model, theta, 0)[0][0] #first [0] because it returns a list, second [0] because if we have multiple models

            denominator = 0.0
            for i in range(0, self.n_samples):
                pdf_value = self.kernel.pdf(self.accepted_parameters_bds.value()[i,:], self.accepted_cov_mat_bds.value(), theta)
                denominator += self.accepted_weights_bds.value()[i, 0] * pdf_value
            return 1.0 * prior_prob / denominator



class PMC(BasePMC, InferenceMethod):
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
    model : abcpy.models.Model
        Model object defining the model to be used.
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

    observations_bds = None
    accepted_parameters_bds = None
    accepted_weights_bds = None
    accepted_cov_mat_bds = None

    def __init__(self, model, likfun, kernel, backend, seed=None):
        self.model = model
        self.likfun = likfun
        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_weights_bds = None
        self.accepted_cov_mat_bds = None


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 100, covFactor = None, iniPoints = None, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : python list
            Observed data
        steps : integer
            number of iterations in the sequential algoritm ("generations")
        n_samples : integer, optional
            number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            number of data points in each simulated data set. The default value is 100.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default is a p dimensional array of 1 when p is the dimension of the parameter.
        inipoints : numpy.ndarray, optional
            parameter vaulues from where the sampling starts. By default sampled from the prior.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations_bds = self.backend.broadcast(observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_lhd_func"] = type(self.likfun)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["steps"] = steps
        journal.configuration["covFactor"] = covFactor
        journal.configuration["iniPoints"] = iniPoints

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mat = None
        new_theta = None

        dim = len(self.get_parameters(self.model))
        self._reset_flags(self.model)

        # Initialize particles: When not supplied, randomly draw them from prior distribution
        # Weights of particles: Assign equal weights for each of the particles
        if iniPoints == None:
            accepted_parameters = np.zeros(shape=(n_samples, dim))
            for ind in range(0, n_samples):
                self.sample_from_prior(self.model, rng=self.rng)
                self._reset_flags(self.model)
                accepted_parameters[ind, :] = self.get_parameters(self.model)
                self._reset_flags(self.model)
            accepted_weights = np.ones((n_samples, 1), dtype=np.float) / n_samples
        else:
            accepted_parameters = iniPoints
            accepted_weights = np.ones((iniPoints.shape[0], 1), dtype=np.float) / iniPoints.shape[0]

        if covFactor is None:
            covFactor = np.ones(shape=(dim,))

        # Calculate initial covariance matrix
        accepted_cov_mat = covFactor * np.cov(accepted_parameters, aweights=accepted_weights.reshape(-1), rowvar=False)

        # main SMC algorithm
        # print("INFO: Starting PMC iterations.")
        for aStep in range(0, steps):
            # print("DEBUG: Iteration " + str(aStep) + " of PMC algorithm.")

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            self._update_broadcasts(accepted_parameters, accepted_weights, accepted_cov_mat)

            # 1: calculate resample parameters
            # print("INFO: Resample parameters.")
            index = self.rng.choice(accepted_parameters.shape[0], size=n_samples, p=accepted_weights.reshape(-1))
            # Choose a new particle using the resampled particle (make the boundary proper)
            # Initialize new_parameters
            new_parameters = np.zeros((n_samples, dim), dtype=np.float)
            for ind in range(0, self.n_samples):
                while True:
                    new_theta = self.kernel.perturb(accepted_parameters[index[ind],:],accepted_cov_mat)
                    theta_is_accepted = self.set_parameters(self.model, new_theta, 0)
                    self._reset_flags(self.model)
                    if theta_is_accepted and self.pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                        new_parameters[ind, :] = new_theta
                        break

            # 2: calculate approximate lieklihood for new parameters
            # print("INFO: Calculate approximate likelihood.")
            new_parameters_pds = self.backend.parallelize(new_parameters)
            approx_likelihood_new_parameters_pds = self.backend.map(self._approx_lik_calc, new_parameters_pds)
            # print("DEBUG: Collect approximate likelihood from pds.")
            approx_likelihood_new_parameters = np.array(
                self.backend.collect(approx_likelihood_new_parameters_pds)).reshape(-1, 1)

            # 3: calculate new weights for new parameters
            # print("INFO: Calculating weights.")
            new_weights_pds = self.backend.map(self._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1, 1)

            #NOTE this loop can give 0, for example if the example + synliklihood are used!
            sum_of_weights = 0.0
            for i in range(0, self.n_samples):
                new_weights[i] = new_weights[i] * approx_likelihood_new_parameters[i]
                sum_of_weights += new_weights[i]
            new_weights = new_weights / sum_of_weights
            accepted_parameters = new_parameters

            # 4: calculate covariance
            # print("INFO: Calculating covariance matrix.")
            new_cov_mat = covFactor * np.cov(accepted_parameters, aweights=accepted_weights.reshape(-1), rowvar=False)

            # 5: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mat = new_cov_mat

            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)
                journal.add_opt_values(approx_likelihood_new_parameters)

        return journal

    def _update_broadcasts(self, accepted_parameters, accepted_weights, accepted_cov_mat):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = self.backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = self.backend.broadcast(accepted_weights)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = self.backend.broadcast(accepted_cov_mat)

    # define helper functions for map step
    def _approx_lik_calc(self, theta):
        """
        Compute likelihood for new parameters using approximate likelihood function

        Parameters
        ----------
        theta: numpy.ndarray
            1xp matrix containing the model parameters, where p is the number of parameters

        Returns
        -------
        float
            The approximated likelihood function
        """

        # Assign theta to model
        self.set_parameters(self.model, theta, 0)
        self._reset_flags(self.model)

        # Simulate the fake data from the model given the parameter value theta
        # print("DEBUG: Simulate model for parameter " + str(theta))

        #TODO MULTIPLE MODELS
        y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param).tolist()
        # print("DEBUG: Extracting observation.")
        obs = self.observations_bds.value()
        # print("DEBUG: Computing likelihood...")
        lhd = self.likfun.likelihood(obs, y_sim)

        # print("DEBUG: Likelihood is :" + str(lhd))
        pdf_at_theta = self.pdf_of_prior(self.model, theta, 0)[0][0]

        # print("DEBUG: prior pdf evaluated at theta is :" + str(pdf_at_theta))
        return pdf_at_theta * lhd

    def _calculate_weight(self, theta):
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

        if self.accepted_weights_bds is None:
            return 1.0 / self.n_samples
        else:
            #TODO MULTIPLE MODELS
            prior_prob = self.pdf_of_prior(self.model, theta, 0)[0][0]

            denominator = 0.0
            for i in range(0, self.n_samples):
                pdf_value = self.kernel.pdf(self.accepted_parameters_bds.value()[i,:], self.accepted_cov_mat_bds.value(), theta)
                denominator += self.accepted_weights_bds.value()[i, 0] * pdf_value

            return 1.0 * prior_prob / denominator


class SABC(BaseAnnealing, InferenceMethod):
    """
    This base class implements a modified version of Simulated Annealing Approximate Bayesian Computation (SABC) of [1] when the prior is non-informative.

    [1] C. Albert, H. R. Kuensch and A. Scheidegger. A Simulated Annealing Approach to
    Approximate Bayes Computations. Statistics and Computing, (2014).

    Parameters
    ----------
    model : abcpy.models.Model
        Model object defining the model to be used.
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

    observations_bds = None
    accepted_parameters_bds = None
    accepted_cov_mat_bds = None
    smooth_distances_bds = None
    all_distances_bds = None

    def __init__(self, model, distance, kernel, backend, seed=None):
        self.model = model
        self.distance = distance
        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_cov_mat_bds = None
        self.smooth_distances_bds = None
        self.all_distances_bds = None


    def sample(self, observations, steps, epsilon, n_samples = 10000, n_samples_per_param = 1, beta = 2, delta = 0.2, v = 0.3, ar_cutoff = 0.5, resample = None, n_update = None, adaptcov = 1, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : numpy.ndarray
            Observed data.
        steps : integer
            Number of maximum iterations in the sequential algoritm ("generations")
        epsilon : numpy.float
            An array of proposed values of epsilon to be used at each steps.
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        beta : numpy.float
            Tuning parameter of SABC
        delta : numpy.float
            Tuning parameter of SABC
        v : numpy.float, optional
            Tuning parameter of SABC, The default value is 0.3.
        ar_cutoff : numpy.float
            Acceptance ratio cutoff, The default value is 0.5
        resample: int, optional
            Resample after this many acceptance, The default value if n_samples
        n_update: int, optional
            Number of perturbed parameters at each step, The default value if n_samples
        adaptcov : boolean, optional
            Whether we adapt the covariance matrix in iteration stage. The default value TRUE.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations_bds = self.backend.broadcast(observations)
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["type_kernel_func"] = type(self.kernel)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["beta"] = beta
        journal.configuration["delta"] = delta
        journal.configuration["v"] = v
        journal.configuration["ar_cutoff"] = ar_cutoff
        journal.configuration["resample"] = resample
        journal.configuration["n_update"] = n_update
        journal.configuration["adaptcov"] = adaptcov
        journal.configuration["full_output"] = full_output

        accepted_parameters = np.zeros(shape=(n_samples, len(self.get_parameters(self.model))))
        self._reset_flags(self.model)
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

        for aStep in range(0, steps):
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
            # print("INFO: Broadcasting parameters.")
            self.epsilon = epsilon
            self._update_broadcasts(accepted_parameters, accepted_cov_mat, smooth_distances, all_distances)

            # 1: Calculate  parameters
            # print("INFO: Initial accepted parameter parameters")
            params_and_dists_pds = self.backend.map(self._accept_parameter, data_pds)
            params_and_dists = self.backend.collect(params_and_dists_pds)
            new_parameters, new_distances, new_all_parameters, new_all_distances, index, acceptance = [list(t) for t in
                                                                                                       zip(
                                                                                                           *params_and_dists)]
            new_parameters = np.array(new_parameters)
            new_distances = np.array(new_distances)
            new_all_distances = np.concatenate(new_all_distances)
            index = index_arr
            acceptance = np.array(acceptance)

            # Reading all_distances at Initial step
            if aStep == 0:
                index = np.linspace(0, n_samples - 1, n_samples).astype(int).reshape(n_samples, )
                accept = 0
                all_distances = new_all_distances

            # print(index[acceptance == 1])
            # Initialize/Update the accepted parameters and their corresponding distances
            accepted_parameters[index[acceptance == 1], :] = new_parameters[acceptance == 1, :]
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
            if accepted_parameters.shape[1] > 1:
                accepted_cov_mat = beta * np.cov(np.transpose(accepted_parameters)) + \
                                   0.0001 * np.trace(np.cov(np.transpose(accepted_parameters))) * np.eye(
                                       accepted_parameters.shape[1])
            else:
                accepted_cov_mat = beta * np.var(np.transpose(accepted_parameters)) + \
                                   0.0001 * (np.var(np.transpose(accepted_parameters))) * np.eye(
                                       accepted_parameters.shape[1])

            # 4: Show progress and if acceptance rate smaller than a value break the iteration

            # print("INFO: Saving intermediate configuration to output journal.")
            if full_output == 1:
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)

            if aStep > 0:
                accept = accept + np.sum(acceptance)
                samples_until = samples_until + sample_array[aStep]
                acceptance_rate = accept / samples_until
                print(
                'updates: ', np.sum(sample_array[1:aStep + 1]) / np.sum(sample_array[1:]) * 100, ' epsilon: ', epsilon, \
                'u.mean: ', U, 'acceptance rate: ', acceptance_rate)
                if acceptance_rate < ar_cutoff:
                    break

            # 5: Resampling if number of accepted particles greater than resample
            if accept >= resample and U > 1e-100:
                ## Weighted resampling:
                weight = np.exp(-smooth_distances * delta / U)
                weight = weight / sum(weight)
                index_resampled = self.rng.choice(np.arange(n_samples), n_samples, replace=1, p=weight)
                accepted_parameters = accepted_parameters[index_resampled, :]
                smooth_distances = smooth_distances[index_resampled]

                ## Update U and epsilon:
                epsilon = epsilon * (1 - delta)
                U = np.mean(smooth_distances)
                epsilon = self._schedule(U, v)

                ## Print effective sampling size
                print('Resampling: Effective sampling size: ', 1 / sum(pow(weight / sum(weight), 2)))
                accept = 0
                samples_until = 0

        # Add epsilon_arr, number of final steps and final output to the journal
        # print("INFO: Saving final configuration to output journal.")
        if full_output == 0:
            journal.add_parameters(accepted_parameters)
            journal.add_weights(accepted_weights)
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

    def _update_broadcasts(self, accepted_parameters, accepted_cov_mat, smooth_distances, all_distances):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = self.backend.broadcast(accepted_parameters)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = self.backend.broadcast(accepted_cov_mat)
        if not smooth_distances is None:
            self.smooth_distances_bds = self.backend.broadcast(smooth_distances)
        if not all_distances is None:
            self.all_distances_bds = self.backend.broadcast(all_distances)

    # define helper functions for map step
    def _accept_parameter(self, data):
        """
        Samples a single model parameter and simulate from it until
        accepted with probabilty exp[-rho(x,y)/epsilon].

        Parameters
        ----------
        seed: integer
            Initial seed for the random number generator.

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
        #NOTE WE RESEEDED THE PRIOR HERE -> PASS RNG TO SAMPLE_FROM_PRIOR?
        #TODO DO WHATEVER YOU NEED WITH KERNEL.RESEED
        self.kernel.rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        all_parameters = []
        all_distances = []
        acceptance = 0

        if self.accepted_cov_mat_bds == None:
            while acceptance == 0:
                self.sample_from_prior(self.model, rng=rng)
                self._reset_flags(self.model)
                new_theta = self.get_parameters(self.model)
                self._reset_flags(self.model)
                all_parameters.append(new_theta)
                #TODO MULTIPLE MODELS
                y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
                distance = self.distance.distance(self.observations_bds.value(), y_sim)
                all_distances.append(distance)
                acceptance = rng.binomial(1, np.exp(-distance / self.epsilon), 1)
            acceptance = 1
        else:
            ## Select one arbitrary particle:
            index = rng.choice(self.n_samples, size=1)[0]
            ## Sample proposal parameter and calculate new distance:
            theta = self.accepted_parameters_bds.value()[index, :]
            while True:
                #TODO not sure what difference is??
                if len(theta) > 1:
                    new_theta = self.kernel.perturb(theta, self.accepted_cov_mat_bds.value())
                    #new_theta = self.kernel.sample(1)[0, :]
                else:
                    new_theta = self.kernel.perturb(theta, self.accepted_cov_mat_bds.value())
                    #new_theta = self.kernel.sample(1)
                theta_is_accepted = self.set_parameters(self.model, new_theta, 0)
                self._reset_flags(self.model)
                if theta_is_accepted and self.pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                    break
            #TODO multiple models
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
            distance = self.distance.distance(self.observations_bds.value(), y_sim)
            smooth_distance = self._smoother_distance([distance], self.all_distances_bds.value())

            ## Calculate acceptance probability:
            ratio_prior_prob = self.pdf_of_prior(self.model, new_theta, 0)[0][0] / self.pdf_of_prior(self.model,
                self.accepted_parameters_bds.value()[index, :], 0)[0][0]
            ratio_likelihood_prob = np.exp((self.smooth_distances_bds.value()[index] - smooth_distance) / self.epsilon)
            acceptance_prob = ratio_prior_prob * ratio_likelihood_prob

            ## If accepted
            if rng.rand(1) < acceptance_prob:
                acceptance = 1
            else:
                distance = np.inf

        return (new_theta, distance, all_parameters, all_distances, index, acceptance)

class ABCsubsim(BaseAnnealing, InferenceMethod):
    """This base class implements Approximate Bayesian Computation by subset simulation (ABCsubsim) algorithm of [1].

    [1] M. Chiachio, J. L. Beck, J. Chiachio, and G. Rus., Approximate Bayesian computation by subset
    simulation. SIAM J. Sci. Comput., 36(3):A1339–A1358, 2014/10/03 2014.

    Parameters
    ----------
    model : abcpy.models.Model
        Model object defining the model to be used.
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

    observations_bds = None
    accepted_parameters_bds = None
    accepted_cov_mat_bds = None

    def __init__(self, model, distance, kernel, backend, seed=None):
        self.model = model
        self.distance = distance
        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.anneal_parameter = None


        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_cov_mat_bds = None


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, chain_length = 10, ap_change_cutoff = 10, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : numpy.ndarray
            Observed data.
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        ap_change_cutoff : float, optional
            The cutoff value for the percentage change in the anneal parameter. If the change is less than
            ap_change_cutoff the iterations are stopped. The default value is 10.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations_bds = self.backend.broadcast(observations)
        self.chain_length = chain_length
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["type_kernel_func"] = type(self.kernel)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["chain_length"] = self.chain_length
        journal.configuration["ap_change_cutoff"] = ap_change_cutoff
        journal.configuration["full_output"] = full_output

        accepted_parameters = None
        accepted_weights = np.ones(shape=(n_samples, 1))
        accepted_cov_mat = None
        anneal_parameter = 0
        anneal_parameter_old = 0
        temp_chain_length = 1


        for aStep in range(0, steps):
            # main ABCsubsim algorithm
            # print("INFO: Initialization of ABCsubsim")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=int(n_samples / temp_chain_length),
                                        dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            index_arr = np.linspace(0, n_samples / temp_chain_length - 1, n_samples / temp_chain_length).astype(
                int).reshape(int(n_samples / temp_chain_length), )
            rng_and_index_arr = np.column_stack((rng_arr, index_arr))
            rng_and_index_pds = self.backend.parallelize(rng_and_index_arr)

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            self._update_broadcasts(accepted_parameters, accepted_cov_mat)

            # 1: Calculate  parameters
            # print("INFO: Initial accepted parameter parameters")
            params_and_dists_pds = self.backend.map(self._accept_parameter, rng_and_index_pds)
            params_and_dists = self.backend.collect(params_and_dists_pds)
            new_parameters, new_distances = [list(t) for t in zip(*params_and_dists)]
            accepted_parameters = np.concatenate(new_parameters)
            distances = np.concatenate(new_distances)

            # 2: Sort and renumber samples
            SortIndex = sorted(range(len(distances)), key=lambda k: distances[k])
            distances = distances[SortIndex]
            accepted_parameters = accepted_parameters[SortIndex, :]

            # 3: Calculate and broadcast annealling parameters
            temp_chain_length = chain_length
            if aStep > 0:
                anneal_parameter_old = anneal_parameter
            anneal_parameter = 0.5 * (
            distances[int(n_samples / temp_chain_length)] + distances[int(n_samples / temp_chain_length) + 1])
            self.anneal_parameter = anneal_parameter

            # 4: Update proposal covariance matrix (Parallelized)
            if aStep == 0:
                accepted_cov_mat = np.cov(accepted_parameters, rowvar=False)
            else:
                accepted_cov_mat = pow(2, 1) * accepted_cov_mat
            self._update_broadcasts(accepted_parameters, accepted_cov_mat)

            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=10, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            index_arr = np.linspace(0, 10 - 1, 10).astype(int).reshape(10, )
            rng_and_index_arr = np.column_stack((rng_arr, index_arr))
            rng_and_index_pds = self.backend.parallelize(rng_and_index_arr)

            cov_mat_index_pds = self.backend.map(self._update_cov_mat, rng_and_index_pds)
            cov_mat_index = self.backend.collect(cov_mat_index_pds)
            cov_mat, T, accept_index = [list(t) for t in zip(*cov_mat_index)]

            for ind in range(10):
                if accept_index[ind] == 1:
                    accepted_cov_mat = cov_mat[ind]
                    break

            # print("INFO: Saving intermediate configuration to output journal.")
            if full_output == 1:
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)

            # Show progress
            anneal_parameter_change_percentage = 100 * abs(anneal_parameter_old - anneal_parameter) / anneal_parameter
            print('Steps: ', aStep, 'annealing parameter: ', anneal_parameter, 'change (%) in annealing parameter: ',
                  anneal_parameter_change_percentage)
            if anneal_parameter_change_percentage < ap_change_cutoff:
                break

        # Add anneal_parameter, number of final steps and final output to the journal
        # print("INFO: Saving final configuration to output journal.")
        if full_output == 0:
            journal.add_parameters(accepted_parameters)
            journal.add_weights(accepted_weights)
        journal.configuration["steps"] = aStep + 1
        journal.configuration["anneal_parameter"] = anneal_parameter

        return journal

    def _update_broadcasts(self, accepted_parameters, accepted_cov_mat):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = self.backend.broadcast(accepted_parameters)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = self.backend.broadcast(accepted_cov_mat)

    # define helper functions for map step
    def _accept_parameter(self, rng_and_index):
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
        #NOTE AGAIN DELETED PRIOR.RESEED
        #TODO KERNEL RESEEDING -> DO WE WANT A DIFFERENT RNG?
        self.kernel.rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        result_theta = []
        result_distance = []

        if self.accepted_parameters_bds == None:
            self.sample_from_prior(self.model, rng=rng)
            self._reset_flags(self.model)
            #TODO MULTIPLE MODELS
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
            distance = self.distance.distance(self.observations_bds.value(), y_sim)
            result_theta.append(self.get_parameters(self.model))
            self._reset_flags(self.model)
            result_distance.append(distance)
        else:
            self._reset_flags(self.model)
            theta = self.accepted_parameters_bds.value()[index]
            self.set_parameters(self.model, theta, 0)
            self._reset_flags(self.model)
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
            distance = self.distance.distance(self.observations_bds.value(), y_sim)
            result_theta.append(theta)
            result_distance.append(distance)
            for ind in range(0, self.chain_length - 1):
                while True:
                    new_theta = self.kernel.perturb(theta, self.accepted_cov_mat_bds.value())
                    theta_is_accepted = self.set_parameters(self.model, new_theta, 0)
                    self._reset_flags(self.model)
                    if theta_is_accepted and self.pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                        break
                y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param,rng=rng).tolist()
                new_distance = self.distance.distance(self.observations_bds.value(), y_sim)

                ## Calculate acceptance probability:
                ratio_prior_prob = self.pdf_of_prior(self.model, new_theta, 0)[0][0] / self.pdf_of_prior(self.model, theta, 0)[0][0]
                kernel_numerator = self.kernel.pdf(new_theta, self.accepted_cov_mat_bds.value(), theta)
                kernel_denominator = self.kernel.pdf(theta, self.accepted_cov_mat_bds.value(), new_theta)
                ratio_likelihood_prob = kernel_numerator / kernel_denominator
                acceptance_prob = min(1, ratio_prior_prob * ratio_likelihood_prob) * (
                new_distance < self.anneal_parameter)

                ## If accepted
                if rng.binomial(1, acceptance_prob) == 1:
                    result_theta.append(new_theta)
                    result_distance.append(new_distance)
                    theta = new_theta
                    distance = new_distance
                else:
                    result_theta.append(theta)
                    result_distance.append(distance)

        return (result_theta, result_distance)

    def _update_cov_mat(self, rng_t):
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

        #TODO AGAIN PRIOR RESEED AND KERNEL
        self.kernel.rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        acceptance = 0

        accepted_cov_mat_transformed = self.accepted_cov_mat_bds.value() * pow(2.0, -2.0 * t)

        theta = self.accepted_parameters_bds.value()[0]

        #NOTE left this out, since it doesnt seem to do anything anymore?
        #self.model.set_parameters(theta)

        for ind in range(0, self.chain_length):
            while True:
                self._reset_flags(self.model)
                new_theta = self.kernel.perturb(theta, accepted_cov_mat_transformed)
                theta_is_accepted = self.set_parameters(self.model, new_theta, 0)
                if theta_is_accepted and self.pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                    break
                y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
                new_distance = self.distance.distance(self.observations_bds.value(), y_sim)

                ## Calculate acceptance probability:
                ratio_prior_prob = self.pdf_of_prior(self.model, new_theta, 0)[0][0] / self.pdf_of_prior(self.model, theta, 0)[0][0]
                kernel_numerator = self.kernel.pdf(new_theta, accepted_cov_mat_transformed, theta)
                kernel_denominator = self.kernel.pdf(theta, accepted_cov_mat_transformed, new_theta)
                ratio_likelihood_prob = kernel_numerator / kernel_denominator
                acceptance_prob = min(1, ratio_prior_prob * ratio_likelihood_prob) * (
                new_distance < self.anneal_parameter)
                ## If accepted
                if rng.binomial(1, acceptance_prob) == 1:
                    theta = new_theta
                    acceptance = acceptance + 1
        if acceptance / 10 <= 0.5 and acceptance / 10 >= 0.3:
            return (accepted_cov_mat_transformed, t, 1)
        else:
            return (accepted_cov_mat_transformed, t, 0)

#NOTE when testing with example values -> raises singular matrix error during calculation of pdf of kernel. I think this might happen in general, not a mistake of the code ---> desired behavior?
class RSMCABC(BaseAdaptivePopulationMC, InferenceMethod):
    """This base class implements Adaptive Population Monte Carlo Approximate Bayesian computation of
    Drovandi and Pettitt [1].

    [1] CC. Drovandi CC and AN. Pettitt, Estimation of parameters for macroparasite population evolution using
    approximate Bayesian computation. Biometrics 67(1):225–233, 2011.

    Parameters
    ----------
    model : abcpy.models.Model
        Model object defining the model to be used.
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

    observations_bds = None
    accepted_parameters_bds = None
    accepted_dist_bds = None
    accepted_cov_mat_bds = None


    def __init__(self, model, distance, kernel, backend, seed=None):
        self.model = model
        self.distance = distance
        self.kernel = kernel
        self.backend = backend

        self.R=None
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_dist_bds = None
        self.accepted_cov_mat_bds = None


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, alpha = 0.1, epsilon_init = 100, epsilon_final = 0.1, const = 1, covFactor = 2.0, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : numpy.ndarray
            Observed data.
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
             A constant to compute acceptance probabilty
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations_bds = self.backend.broadcast(observations)
        self.alpha = alpha
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["steps"] = steps

        accepted_parameters = None
        accepted_cov_mat = None
        accepted_dist = None

        # main RSMCABC algorithm
        # print("INFO: Starting RSMCABC iterations.")
        for aStep in range(steps):

            # 0: Compute epsilon, compute new covariance matrix for Kernel,
            # and finally Drawing new new/perturbed samples using prior or MCMC Kernel
            # print("DEBUG: Iteration " + str(aStep) + " of RSMCABC algorithm.")
            if aStep == 0:
                n_replenish = n_samples
                # Compute epsilon
                epsilon = [epsilon_init]
                R = int(1)
            else:
                n_replenish = round(n_samples * alpha)
                # Throw away N_alpha particles with largest dist
                accepted_parameters = np.delete(accepted_parameters, np.arange(round(n_samples * alpha)) + (
                self.n_samples - round(n_samples * alpha)), 0)
                accepted_dist = np.delete(accepted_dist,
                                          np.arange(round(n_samples * alpha)) + (n_samples - round(n_samples * alpha)),
                                          0)
                # Compute epsilon
                epsilon.append(accepted_dist[-1])
                # Calculate covariance
                # print("INFO: Calculating covariance matrix.")
                new_cov_mat = covFactor * np.cov(accepted_parameters, rowvar=False)
                accepted_cov_mat = new_cov_mat

            if epsilon[-1] < epsilon_final:
                break

            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_replenish, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # update remotely required variables
            # print("INFO: Broadcasting parameters.")
            self.epsilon = epsilon
            self.R = R
            # Broadcast updated variable
            self._update_broadcasts(accepted_parameters, accepted_dist, accepted_cov_mat)

            # calculate resample parameters
            # print("INFO: Resampling parameters")
            params_and_dist_index_pds = self.backend.map(self._accept_parameter, rng_pds)
            params_and_dist_index = self.backend.collect(params_and_dist_index_pds)
            new_parameters, new_dist, new_index = [list(t) for t in zip(*params_and_dist_index)]
            new_parameters = np.array(new_parameters)
            new_dist = np.array(new_dist)
            new_index = np.array(new_index)

            # 1: Update all parameters, compute acceptance probability, compute epsilon
            if len(new_dist) == self.n_samples:
                accepted_parameters = new_parameters
                accepted_dist = new_dist
            else:
                accepted_parameters = np.concatenate((accepted_parameters, new_parameters))
                accepted_dist = np.concatenate((accepted_dist, new_dist))

            # 2: Compute acceptance probabilty and set R
            # print(aStep)
            # print(new_index)
            prob_acceptance = sum(new_index) / (R * n_replenish)
            if prob_acceptance == 1 or prob_acceptance == 0:
                R = 1
            else:
                R = int(np.log(const) / np.log(1 - prob_acceptance))

            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_parameters(accepted_parameters)
                journal.add_weights(np.ones(shape=(n_samples, 1)) * (1 / n_samples))

        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon

        return journal

    def _update_broadcasts(self, accepted_parameters, accepted_dist, accepted_cov_mat):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = self.backend.broadcast(accepted_parameters)
        if not accepted_dist is None:
            self.accepted_dist_bds = self.backend.broadcast(accepted_dist)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = self.backend.broadcast(accepted_cov_mat)

    # define helper functions for map step
    def _accept_parameter(self, rng):
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
        #TODO AGAIN PRIOR AND KERNEL
        self.kernel.rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        self._reset_flags(self.model)

        distance = self.distance.dist_max()
        if self.accepted_parameters_bds == None:
            while distance > self.epsilon[-1]:
                self.sample_from_prior(self.model, rng=rng)
                self._reset_flags(self.model)
                y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
                distance = self.distance.distance(self.observations_bds.value(), y_sim)
            index_accept = 1
        else:
            index = rng.choice(len(self.accepted_parameters_bds.value()), size=1)
            theta = self.accepted_parameters_bds.value()[index[0]]
            index_accept = 0.0
            for ind in range(self.R):
                while True:
                    new_theta = self.kernel.perturb(theta, self.accepted_cov_mat_bds.value())
                    theta_is_accepted = self.set_parameters(self.model, new_theta, 0)
                    self._reset_flags(self.model)
                    if theta_is_accepted and self.pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                        break
                y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
                distance = self.distance.distance(self.observations_bds.value(), y_sim)
                ratio_prior_prob = self.pdf_of_prior(self.model, new_theta, 0)[0][0] / self.pdf_of_prior(self.model, theta, 0)[0][0]
                kernel_numerator = self.kernel.pdf(new_theta, self.accepted_cov_mat_bds.value(), theta)
                kernel_denominator = self.kernel.pdf(theta, self.accepted_cov_mat_bds.value(), new_theta)
                ratio_kernel_prob = kernel_numerator / kernel_denominator
                probability_acceptance = min(1, ratio_prior_prob * ratio_kernel_prob)
                if distance < self.epsilon[-1] and rng.binomial(1, probability_acceptance) == 1:
                    index_accept += 1
                else:
                    self.set_parameters(self.model, theta, 0)
                    distance = self.accepted_dist_bds.value()[index[0]]

        return (self.get_parameters(self.model), distance, index_accept)

class APMCABC(BaseAdaptivePopulationMC, InferenceMethod):
    """This base class implements Adaptive Population Monte Carlo Approximate Bayesian computation of
    M. Lenormand et al. [1].

    [1] M. Lenormand, F. Jabot and G. Deffuant, Adaptive approximate Bayesian computation
    for complex models. Computational Statistics, 28:2777–2796, 2013.

    Parameters
    ----------
    model : abcpy.models.Model
        Model object defining the model to be used.
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

    observations_bds = None
    accepted_parameters_bds = None
    accepted_weights_bds = None
    accepted_dist = None
    accepted_cov_mat_bds = None

    def __init__(self,  model, distance, kernel, backend, seed=None):
        self.model = model
        self.distance = distance
        self.kernel = kernel
        self.backend = backend

        self.epsilon= None
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_weights_bds = None
        self.accepted_dist = None
        self.accepted_cov_mat_bds = None


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, alpha = 0.9, acceptance_cutoff = 0.2, covFactor = 2.0, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : numpy.ndarray
            Observed data.
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        alpha : float, optional
            A parameter taking values between [0,1], the default value is 0.1.
        acceptance_cutoff : float, optional
            Acceptance ratio cutoff, The default value is 0.2
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations_bds = self.backend.broadcast(observations)
        self.alpha = alpha
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["steps"] = steps

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mat = None
        accepted_dist = None
        alpha_accepted_parameters = None
        alpha_accepted_weights = None
        alpha_accepted_dist = None

        # main APMCABC algorithm
        # print("INFO: Starting APMCABC iterations.")
        for aStep in range(steps):

            # 0: Drawing new new/perturbed samples using prior or MCMC Kernel
            # print("DEBUG: Iteration " + str(aStep) + " of APMCABC algorithm.")
            if aStep > 0:
                n_additional_samples = n_samples - round(n_samples * alpha)
            else:
                n_additional_samples = n_samples

            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_additional_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # update remotely required variables
            # print("INFO: Broadcasting parameters.")
            self._update_broadcasts(alpha_accepted_parameters, alpha_accepted_weights, alpha_accepted_dist,
                                  accepted_cov_mat)

            # calculate resample parameters
            # print("INFO: Resampling parameters")
            params_and_dist_weights_pds = self.backend.map(self._accept_parameter, rng_pds)
            params_and_dist_weights = self.backend.collect(params_and_dist_weights_pds)
            new_parameters, new_dist, new_weights = [list(t) for t in zip(*params_and_dist_weights)]
            new_parameters = np.array(new_parameters)
            new_dist = np.array(new_dist)
            new_weights = np.array(new_weights).reshape(n_additional_samples, 1)

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
            # print("INFO: Calculating covariance matrix.")
            new_cov_mat = covFactor * np.cov(alpha_accepted_parameters, aweights=alpha_accepted_weights.reshape(-1),
                                             rowvar=False)
            accepted_cov_mat = new_cov_mat

            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)

            # 4: Check probability of acceptance lower than acceptance_cutoff
            if prob_acceptance < acceptance_cutoff:
                break

        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon

        return journal

    def _update_broadcasts(self, accepted_parameters, accepted_weights, accepted_dist,
                           accepted_cov_mat):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = self.backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = self.backend.broadcast(accepted_weights)
        if not accepted_dist is None:
            self.accepted_dist_bds = self.backend.broadcast(accepted_dist)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = self.backend.broadcast(accepted_cov_mat)

    # define helper functions for map step
    def _accept_parameter(self, rng):
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
        #TODO AGAIN PRIOR AND KERNEL
        self.kernel.rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        self._reset_flags(self.model)

        if self.accepted_parameters_bds == None:
            self.sample_from_prior(self.model, rng=rng)
            self._reset_flags(self.model)
            #TODO multiple models
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
            dist = self.distance.distance(self.observations_bds.value(), y_sim)
            weight = 1.0
        else:
            index = rng.choice(len(self.accepted_weights_bds.value()), size=1,
                               p=self.accepted_weights_bds.value().reshape(-1))
            theta = self.accepted_parameters_bds.value()[index[0]]
            # trucate the normal to the bounds of parameter space of the model
            # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf
            while True:
                new_theta = self.kernel.perturb(theta, self.accepted_cov_mat_bds.value())
                theta_is_accepted = self.set_parameters(self.model, new_theta, 0)
                self._reset_flags(self.model)
                if theta_is_accepted and self.pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                    break
            #todo multiple models
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
            dist = self.distance.distance(self.observations_bds.value(), y_sim)

            prior_prob = self.pdf_of_prior(self.model, new_theta, 0)[0][0]
            denominator = 0.0
            for i in range(0, len(self.accepted_weights_bds.value())):
                pdf_value = self.kernel.pdf(self.accepted_parameters_bds.value()[i,:], self.accepted_cov_mat_bds.value(), new_theta)
                denominator += self.accepted_weights_bds.value()[i, 0] * pdf_value
            weight = 1.0 * prior_prob / denominator

        return (self.get_parameters(self.model), dist, weight)


#NOTE takes long time to test with example, but no obvious mistakes so far
class SMCABC(BaseAdaptivePopulationMC, InferenceMethod):
    """This base class implements Adaptive Population Monte Carlo Approximate Bayesian computation of
    Del Moral et al. [1].

    [1] P. Del Moral, A. Doucet, A. Jasra, An adaptive sequential Monte Carlo method for approximate
    Bayesian computation. Statistics and Computing, 22(5):1009–1020, 2012.

    Parameters
    ----------
    model : abcpy.models.Model
        Model object defining the model to be used.
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

    observations_bds = None
    accepted_parameters_bds = None
    accepted_weights_bds = None
    accepted_cov_mat_bds = None
    accepted_y_sim_bds = None

    def __init__(self, model, distance, kernel, backend, seed=None):
        self.model = model
        self.distance = distance
        self.kernel = kernel
        self.backend = backend

        self.epsilon = None
        self.rng = np.random.RandomState(seed)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_weights_bds = None
        self.accepted_cov_mat_bds = None
        self.accepted_y_sim_bds = None


    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, epsilon_final = 0.1, alpha = 0.95, covFactor = 2, resample = None, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.

        Parameters
        ----------
        observations : numpy.ndarray
            Observed data.
        steps : integer
            Number of iterations in the sequential algoritm ("generations")
        epsilon_final : float, optional
            The final threshold value of epsilon to be reached. The default value is 0.1.
        n_samples : integer, optional
            Number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        alpha : float, optional
            A parameter taking values between [0,1], determinining the rate of change of the threshold epsilon. The
            default value is 0.5.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations_bds= self.backend.broadcast(observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["steps"] = steps

        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mat = None
        accepted_y_sim = None

        # Define the resmaple parameter
        if resample == None:
            resample = n_samples * 0.5

        # Define epsilon_init
        epsilon = [10000]

        # main SMC ABC algorithm
        # print("INFO: Starting SMCABC iterations.")
        for aStep in range(0, steps):

            # Break if epsilon in previous step is less than epsilon_final
            if epsilon[-1] == epsilon_final:
                break

            # 0: Compute the Epsilon
            if accepted_y_sim != None:
                # Compute epsilon for next step
                fun = lambda epsilon_var: self._compute_epsilon(epsilon_var, \
                                                                epsilon, observations, accepted_y_sim, accepted_weights,
                                                                n_samples, n_samples_per_param, alpha)
                epsilon_new = self._bisection(fun, epsilon_final, epsilon[-1], 0.001)
                if epsilon_new < epsilon_final:
                    epsilon_new = epsilon_final
                epsilon.append(epsilon_new)

            # 1: calculate weights for new parameters
            # print("INFO: Calculating weights.")
            if accepted_y_sim != None:
                new_weights = np.zeros(shape=(n_samples), )
                for ind1 in range(n_samples):
                    numerator = 0.0
                    denominator = 0.0
                    for ind2 in range(n_samples_per_param):
                        numerator += (self.distance.distance(observations, [accepted_y_sim[ind1][ind2]]) < epsilon[-1])
                        denominator += (
                        self.distance.distance(observations, [accepted_y_sim[ind1][ind2]]) < epsilon[-2])
                    if denominator != 0.0:
                        new_weights[ind1] = accepted_weights[ind1] * (numerator / denominator)
                    else:
                        new_weights[ind1] = 0
                #NOTE gain new_weights can be 0
                new_weights = new_weights / sum(new_weights)
            else:
                new_weights = np.ones(shape=(n_samples), ) * (1.0 / n_samples)

            # 2: Resample
            if accepted_y_sim != None and pow(sum(pow(new_weights, 2)), -1) < resample:
                print('Resampling')
                # Weighted resampling:
                index_resampled = self.rng.choice(np.arange(n_samples), n_samples, replace=1, p=new_weights)
                accepted_parameters = accepted_parameters[index_resampled, :]
                new_weights = np.ones(shape=(n_samples), ) * (1.0 / n_samples)

            # Update the weights
            accepted_weights = new_weights.reshape(len(new_weights), 1)

            # 3: Drawing new perturbed samples using MCMC Kernel
            # print("DEBUG: Iteration " + str(aStep) + " of SMCABC algorithm.")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            index_arr = np.arange(n_samples)
            rng_and_index_arr = np.column_stack((rng_arr, index_arr))
            rng_and_index_pds = self.backend.parallelize(rng_and_index_arr)

            # print("INFO: Broadcasting parameters.")
            self.epsilon = epsilon
            self._update_broadcasts(accepted_parameters, accepted_weights, accepted_cov_mat, accepted_y_sim)

            # calculate resample parameters
            # print("INFO: Resampling parameters")
            params_and_ysim_pds = self.backend.map(self._accept_parameter, rng_and_index_pds)
            params_and_ysim = self.backend.collect(params_and_ysim_pds)
            new_parameters, new_y_sim = [list(t) for t in zip(*params_and_ysim)]
            new_parameters = np.array(new_parameters)

            # Update the parameters
            accepted_parameters = new_parameters
            accepted_y_sim = new_y_sim

            # 4: calculate covariance
            # print("INFO: Calculating covariance matrix.")
            new_cov_mat = covFactor * np.cov(accepted_parameters, aweights=accepted_weights.reshape(-1), rowvar=False)
            accepted_cov_mat = new_cov_mat

            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)
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
                numerator += (self.distance.distance(observations, [accepted_y_sim[ind1][ind2]]) < epsilon_new)
                denominator += (self.distance.distance(observations, [accepted_y_sim[ind1][ind2]]) < epsilon[-1])
            LHS[ind1] = accepted_weights[ind1] * (numerator / denominator)
        if sum(LHS) == 0:
            result = RHS
        else:
            LHS = LHS / sum(LHS)
            LHS = pow(sum(pow(LHS, 2)), -1)
            result = RHS - LHS
        return (result)

    def _bisection(self, func, low, high, tol):
        midpoint = (low + high) / 2.0
        while (high - low) / 2.0 > tol:
            if func(midpoint) == 0:
                return midpoint
            elif func(low) * func(midpoint) < 0:
                high = midpoint
            else:
                low = midpoint
            midpoint = (low + high) / 2.0

        return midpoint

    def _update_broadcasts(self, accepted_parameters, accepted_weights, accepted_cov_mat, accepted_y_sim):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = self.backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = self.backend.broadcast(accepted_weights)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = self.backend.broadcast(accepted_cov_mat)
        if not accepted_y_sim is None:
            self.accepted_y_sim_bds = self.backend.broadcast(accepted_y_sim)

            # define helper functions for map step

    def _accept_parameter(self, rng_and_index):
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
        #TODO AGAIN PRIOR AND KERNEL
        self.kernel.rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        self._reset_flags(self.model)

        # print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
        if self.accepted_parameters_bds == None:
            self.sample_from_prior(self.model, rng=rng)
            self._reset_flags(self.model)
            #todo multiple models
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
        else:
            if self.accepted_weights_bds.value()[index] > 0:
                theta = self.accepted_parameters_bds.value()[index]
                while True:
                    new_theta = self.kernel.perturb(theta, self.accepted_cov_mat_bds.value())
                    theta_is_accepted = self.set_parameters(self.model, new_theta, 0)
                    self._reset_flags(self.model)
                    if theta_is_accepted and self.pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                        break
                #todo multiple models
                y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
                y_sim_old = self.accepted_y_sim_bds.value()[index]
                ## Calculate acceptance probability:
                numerator = 0.0
                denominator = 0.0
                for ind in range(self.n_samples_per_param):
                    numerator += (
                    self.distance.distance(self.observations_bds.value(), [y_sim[ind]]) < self.epsilon[-1])
                    denominator += (
                    self.distance.distance(self.observations_bds.value(), [y_sim_old[ind]]) < self.epsilon[-1])
                ratio_data_epsilon = numerator / denominator
                ratio_prior_prob = self.pdf_of_prior(self.model, new_theta, 0)[0][0] / self.pdf_of_prior(self.model, theta, 0)[0][0]
                kernel_numerator = self.kernel.pdf(new_theta, self.accepted_cov_mat_bds.value(), theta)
                kernel_denominator = self.kernel.pdf(theta, self.accepted_cov_mat_bds.value(), new_theta)
                ratio_likelihood_prob = kernel_numerator / kernel_denominator
                acceptance_prob = min(1, ratio_data_epsilon * ratio_prior_prob * ratio_likelihood_prob)
                if rng.binomial(1, acceptance_prob) == 1:
                    self.set_parameters(self.model, new_theta, 0)
                    self._reset_flags(self.model)
                else:
                    self.set_parameters(self.model, theta, 0)
                    self._reset_flags(self.model)
                    y_sim = self.accepted_y_sim_bds.value()[index]
            else:
                self.set_parameters(self.model, self.accepted_parameters_bds.value()[index], 0)
                self._reset_flags(self.model)
                y_sim = self.accepted_y_sim_bds.value()[index]

        return (self.get_parameters(self.model), y_sim)