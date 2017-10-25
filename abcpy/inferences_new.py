from abc import ABCMeta, abstractmethod, abstractproperty

from ProbabilisticModel import *
import numpy as np
from abcpy.output import Journal
from scipy import optimize


#TODO if we first sample from the kernel, and then set the values of our graph: we will need a set_parameters for the whole inferencemethod, else a set_parameters is not needed glbally!
#TODO if we send the kernel, and sample at each node individually, we will need a "send kernel" function of the InferenceMethod ----> discuss which of the two would be appropriate and implement accordingly



#TODO I am not sure, but if we do sample_from_prior: if we sample some value from a distribution for some child, then for all other children of that node, the same value should be used!!!! we need to somehow implement that!

#NOTE WE DEFINITELY NEED TO SOMEHOW PASS THE INFORMATION ABOUT THE VALUE THAT WAS SAMPLED FOR ONE CHILD TO THE OTHER CHILDREN! HOW DO WE DO THAT?
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


    def sample_from_prior(self, model):
        for i in range(len(model)):
            for parent in model[i].parents:
                if(isinstance(parent, ProbabilisticModel) and not(parent.visited)):
                    self.sample_from_prior([parent])
            model[i].sample_parameters()

    def _reset_flags(self, model):
        for i in range(len(model)):
            for parent in model[i].parents:
                if(isinstance(parent, ProbabilisticModel) and parent.visited):
                    self._reset_flags([parent])
            model[i].visited = False


    #NOTE not tested yet, not sure whether this works.
    #TODO CHECK WHETHER THIS COVERS ALL 3 DIFFERENT CASES
    def pdf_of_prior(self, model, parameters, index):
        result = [1.]*len(model)
        for i in range(len(model)):
            #NOTE this might give out of range -> need another way to check whether it is bottom or not (for example flag passed to function)
            if(not(model[i]==self.model[i])):
                helper = []
                for j in range(model[i].dimension):
                    helper.append(parameters[index])
                    index+=1
                if(len(helper)==1):
                    helper = helper[0]
                else:
                    helper = np.array(helper)
                result[i]*=model[i].pdf(helper)
            for parent in model[i].parents:
                if(isinstance(parent, ProbabilisticModel)):
                    pdf = self.pdf_of_prior([parent], parameters, index)
                    result[i]*=pdf[0][0]
                    index=pdf[1]
        return [result, index]

    def get_parameters(self, model):
        parameters = []
        for i in range(len(model)):
            for parameter in model[i].get_parameters():
                if(isinstance(parameter, list)):
                    for param in parameter:
                        parameters.append(param)
                else:
                    parameters.append(parameter)
            for parent in model[i].parents:
                if(isinstance(parent, ProbabilisticModel) and not(parent.visited)):
                    parent_parameters = self.get_parameters([parent])
                    for parameter in parent_parameters:
                        if(isinstance(parameter, list)):
                            for param in parameter:
                                parameters.append(param)
                        else:
                            parameters.append(parameter)
                    parent.visited=True
            model[i].visited = True
        return parameters

    #NOTE if we have a Uniform distribution, this fails because we cant set like that, we would there have to set as a 2d list, but that is also not really feasible, or is it? ->REWRITE UNIFORM TO TAKE 1 LONG STRING, AND SAME FOR OUTPUT, WIHCH MEANS WE CANT REALLY TELL WHETHER BLA, BUT DO IT ANYWAYS
    #NOTE IT NEEDS TO BE SAME FOR ALL, OTHERWISE IT WONT WORK TO SET IT GLOBALLY! HOWEVER, THIS SHOULD NEVER BE CALLED BY HAND, SO IT SHOULD BE FINE!
    #NOTE returns false iff we couldnt set some node, in that case, use the old parameters again to resample
    def set_parameters(self, model, parameters, index):
        for i in range(len(model)):
            model[i].set_parameters(parameters[index:model[i].dimension])
            index+=model[i].dimension
            for parent in model[i].parents:
                if(isinstance(parent, ProbabilisticModel) and not(parent.visited)):
                    if(not(self.set_parameters([parent], parameters, index))):
                        return False
                    parent.visited = True
            model[i].visited = True
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
#NOTE returns model.get_parameters -> do we want to receive ALL parameters, or just the ones just above it, because all would make sense
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
            super(RejectionABC, self).sample_from_prior(self.model)
            super(RejectionABC, self)._reset_flags(self.model)
            #TODO WHAT SHOULD HAPPEN IF WE HAVE MORE THAN ONE MODEL
            #NOTE this gives reasonable values for the y_sim. However, the distances are very large, possibly because of the algorithm used, not sure
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param, rng=rng).tolist()
            distance = self.distance.distance(self.observations_bds.value(), y_sim)
            #print(distance)
        return super(RejectionABC, self).get_parameters(self.model)

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
        #NOTE WE RESEEDED THE PRIOR HERE -> GIVE RNG TO SAMPLE_FROM_PRIOR?
        #TODO WHAT DO WE DO WITH KERNEL RESEEDS
        self.kernel.rng.seed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        distance = self.distance.dist_max()
        while distance > self.epsilon:
            # print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
            if self.accepted_parameters_bds == None:
                super(PMCABC, self).sample_from_prior(self.model)
                super(PMCABC, self)._reset_flags(self.model)
                theta = super(PMCABC, self).get_parameters(self.model)
                super(PMCABC, self)._reset_flags(self.model)
            else:
                index = rng.choice(self.n_samples, size=1, p=self.accepted_weights_bds.value().reshape(-1))
                theta = self.accepted_parameters_bds.value()[index[0]]
                # truncate the normal to the bounds of parameter space of the model
                # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf


                #TODO we define the kernel either as 1 distribution, or multiple (check whether it makes difference when indep! If 1, we first gather all parameters, set it, sample, send all. If multiple/indep -> send individual kernels and sample at the node
                while True:
                    new_theta = self.kernel.perturb(theta, self.accepted_cov_mat_bds.value())
                    theta_is_accepted = super(PMCABC, self).set_parameters(self.model, new_theta, 0)
                    self._reset_flags(self.model)
                    if theta_is_accepted and super(PMCABC, self).pdf_of_prior(self.model, new_theta, 0)[0][0] != 0:
                        break
            #TODO WHAT IF MORE THAN 1 MODEL
            y_sim = self.model[0].sample_from_distribution(self.n_samples_per_param).tolist()

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