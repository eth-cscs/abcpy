import numpy as np
from abcpy.output import Journal
class RejectionABC:
    """This base class implements the rejection algorithm based inference scheme [1] for
    Approximate Bayesian Computation. 

    [1] Tavaré, S., Balding, D., Griffith, R., Donnelly, P.: Inferring coalescence
    times from DNA sequence data. Genetics 145(2), 505–518 (1997).

    Parameters
    ----------
    model: abcpy.models.Model
        Model object that conforms to the Model class.
    distance: abcpy.distances.Distance
        Distance object that conforms to the Distance class.
    backend: abcpy.backends.Backend
        Backend object that conforms to the Backend class.
    seed: integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """
    def __init__(self, model, distance, backend, seed=None):
        self.model = model
        self.dist_calc = distance
        self.backend = backend
        self.rng = np.random.RandomState(seed)


    def sample(self, observations, n_samples, n_samples_per_param, epsilon, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed 
        data observations.
        
        Parameters
        ----------
        observations : python list
            The observed data set.
        n_samples : integer
            Number of samples to generate.
        n_samples_per_param : integer
            Number of data points in each simulated dataset.  
        epsilon: float 
            Value of threshold.
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal. 
            The default value is 0, meaning the intermediate results are not saved.
            
        Returns
        -------
        abcpy.output.Journal
            a journal containing simulation results, metadata and optionally intermediate results.
        """

        journal = Journal(full_output)
        journal.configuration["n_samples"] = n_samples
        journal.configuration["n_samples_per_param"] = n_samples_per_param
        journal.configuration["epsilon"] = epsilon
        
        accepted_parameters = None                        
        
        # Initialize variables that need to be available remotely
        rc = _RemoteContextRejectionABC(self.backend, self.model, self.dist_calc, observations, n_samples, n_samples_per_param, epsilon)
        
        # main Rejection ABC algorithm                 
        seed_arr = self.rng.randint(1, n_samples*n_samples, size=n_samples, dtype=np.int32)
        seed_pds = self.backend.parallelize(seed_arr)     

        accepted_parameters_pds = self.backend.map(rc._sample_parameter, seed_pds)
        accepted_parameters = self.backend.collect(accepted_parameters_pds)
        accepted_parameters = np.array(accepted_parameters)

        journal.add_parameters(accepted_parameters)
        journal.add_weights(np.ones((n_samples, 1)))
    
        return journal


    
class _RemoteContextRejectionABC:
    """
    Contains everything that is sent over the network like broadcast vars and map functions
    """
    
    def __init__(self, backend, model, dist_calc, observations, n_samples, n_samples_per_param, epsilon):
        self.model = model
        self.dist_calc = dist_calc
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        
        self.epsilon = epsilon

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = backend.broadcast(observations)


        
    def _sample_parameter(self, seed):
        """
        Samples a single model parameter and simulates from it until
        distance between simulated outcome and the observation is
        smaller than eplison.
        
        Parameters
        ----------            
        seed: int
            value of a seed to be used for reseeding
        Returns
        -------
        np.array
            accepted parameter
        """
    
        distance = self.dist_calc.dist_max()
        self.model.prior.reseed(seed)
        
        while distance > self.epsilon:
            # Accept new parameter value if the distance is less than epsilon
            self.model.sample_from_prior()
            y_sim = self.model.simulate(self.n_samples_per_param)
            distance = self.dist_calc.distance(self.observations_bds.value(), y_sim)

        return self.model.get_parameters()




class PMCABC:
    """This base class implements a modified version of Population Monte Carlo based inference scheme 
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
        Model object that conforms to the Model class.
    distance : abcpy.distances.Distance
        Distance object that conforms to the Distance class.
    kernel : abcpy.distributions.Distribution
        Distribution object defining the perturbation kernel needed for the sampling
    backend : abcpy.backends.Backend
        Backend object that conforms to the Backend class.
    seed : integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    """
    def __init__(self, model, distance, kernel, backend, seed=None):       
        self.model = model
        self.distance = distance
        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        

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
        
        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["n_samples"] = n_samples
        journal.configuration["n_samples_per_param"] = n_samples_per_param
        journal.configuration["steps"] = steps
        journal.configuration["epsilon_percentile"] = epsilon_percentile
        
        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mat = None
        #Define epsilon_arr
        if len(epsilon_init) == steps:
            epsilon_arr = epsilon_init
        else:
            if len(epsilon_init) == 1:
                epsilon_arr = [None]*steps
                epsilon_arr[0] = epsilon_init  
            else:
                raise ValueError("The length of epsilon_init can only be of 1 or steps.")
        
        
        # Initialize variables that need to be available remotely
        rc = _RemoteContextPMCABC(self.backend, self.model, self.distance, self.kernel, observations, n_samples, n_samples_per_param)

        # main SMC ABC algorithm
        # print("INFO: Starting SMCABC iterations.")
        for aStep in range(0, steps):
            # print("DEBUG: Iteration " + str(aStep) + " of SMCABC algorithm.")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
            seed_pds = self.backend.parallelize(seed_arr)

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            rc.epsilon = epsilon_arr[aStep] 
            rc.update_broadcasts(self.backend, accepted_parameters, accepted_weights, accepted_cov_mat)

            # 1: calculate resample parameters
            # print("INFO: Resampling parameters")
            params_and_dists_and_ysim_pds = self.backend.map(rc._resample_parameter, seed_pds)
            params_and_dists_and_ysim = self.backend.collect(params_and_dists_and_ysim_pds)
            new_parameters, distances = [list(t) for t in zip(*params_and_dists_and_ysim)]
            new_parameters = np.array(new_parameters)
            rc.update_broadcasts(self.backend, accepted_parameters, accepted_weights, accepted_cov_mat)
            
            # Compute epsilon for next step
            # print("INFO: Calculating acceptance threshold (epsilon).")
            if aStep < steps-1:
                if epsilon_arr[aStep + 1] == None:
                    epsilon_arr[aStep + 1] = np.percentile(distances, epsilon_percentile)
                else:
                    epsilon_arr[aStep + 1] = np.max([np.percentile(distances, epsilon_percentile), epsilon_arr[aStep+1]]) 

            # 2: calculate weights for new parameters 
            # print("INFO: Calculating weights.")
            new_parameters_pds = self.backend.parallelize(new_parameters)
            new_weights_pds = self.backend.map(rc._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1,1)
            sum_of_weights = 0.0
            for w in new_weights:
                sum_of_weights += w
            new_weights = new_weights / sum_of_weights

            # 3: calculate covariance
            # print("INFO: Calculating covariance matrix.")
            new_cov_mat = covFactor * np.cov(new_parameters, aweights = new_weights.reshape(-1), rowvar=False)
                                
            # 4: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mat = new_cov_mat

            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps-1) or (full_output == 0 and aStep == steps-1):
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)
                
        #Add epsilon_arr to the journal                
        journal.configuration["epsilon_arr"] = epsilon_arr

        return journal

class _RemoteContextPMCABC:
    """
    Contains everything that is sent over the network like broadcast vars and map functions
    """
    
    def __init__(self, backend, model, distance, kernel, observations, n_samples, n_samples_per_param):
        self.model = model
        self.distance = distance
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.kernel = kernel
        
        self.epsilon = None

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = backend.broadcast(observations)
        self.accepted_parameters_bds = None
        self.accepted_weights_bds = None
        self.accepted_cov_mat_bds = None


    def update_broadcasts(self, backend, accepted_parameters, accepted_weights, accepted_cov_mat):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                #bc.destroy
                
        if not accepted_parameters is None:
            self.accepted_parameters_bds = backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = backend.broadcast(accepted_weights)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = backend.broadcast(accepted_cov_mat)
                

    # define helper functions for map step
    def _resample_parameter(self, seed):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than eplison.
            
        :type seed: int
        :rtype: np.array
        :return: accepted parameter
        """
        rng = np.random.RandomState(seed)
        self.model.prior.reseed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        self.kernel.reseed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        
        distance = self.distance.dist_max()
        while distance > self.epsilon:
            #print("on seed " + str(seed) + " distance: " + str(distance) + " epsilon: " + str(self.epsilon))
            if self.accepted_parameters_bds == None:
                self.model.sample_from_prior()
            else:
                index = rng.choice(self.n_samples, size=1, p=self.accepted_weights_bds.value().reshape(-1))
                theta = self.accepted_parameters_bds.value()[index[0]]
                # trucate the normal to the bounds of parameter space of the model
                # truncating the normal like this is fine: https://arxiv.org/pdf/0907.4010v1.pdf
                while True:
                    self.kernel.set_parameters([theta, self.accepted_cov_mat_bds.value()])
                    new_theta = self.kernel.sample(1)[0,:]
                    theta_is_accepted = self.model.set_parameters(new_theta)
                    if theta_is_accepted and self.model.prior.pdf(self.model.get_parameters()) != 0:
                        break

            y_sim = self.model.simulate(self.n_samples_per_param)

            distance = self.distance.distance(self.observations_bds.value(), y_sim)
        return (self.model.get_parameters(), distance)
    
    def _calculate_weight(self, theta):
        """
        Calculates the weight for the given parameter using
        accepted_parameters, accepted_cov_mat

        Parameters
        ----------
        theta: np.array
            1xp matrix containing model parameter, where p is the dimension of parameter
            
        Returns
        -------
        float
            the new weight for theta
        """
        
        if self.accepted_weights_bds is None:
            return 1.0 / self.n_samples
        else:
            prior_prob = self.model.prior.pdf(theta)

            denominator = 0.0
            for i in range(0, self.n_samples):
                self.kernel.set_parameters([self.accepted_parameters_bds.value()[i,:], self.accepted_cov_mat_bds.value()])
                pdf_value = self.kernel.pdf(theta)
                denominator += self.accepted_weights_bds.value()[i,0] * pdf_value
            return 1.0 * prior_prob / denominator


class PMC:
    """Population Monte Carlo based inference scheme of Cappé et. al. [1]. 

    This algorithm assumes a likelihood function is available and can be evaluated
    at any parameter value given the oberved dataset.  In absence of the
    likelihood function or when it can't be evaluated with a rational
    computational expenses, we use the approximated likleihood functions in
    abcpy.approx_lhd module, for which the argument of the consistency of the
    inference schemes are based on Andrieu and Roberts [2].

    [1] Cappé, O., Guillin, A., Marin, J.-M., and Robert, C. P. (2004). Population Monte Carlo.
    Journal of Computational and Graphical Statistics, 13(4), 907–929. 
     
    [2] C. Andrieu and G. O. Roberts. The pseudo-marginal approach for efficient Monte Carlo computations.
    Annals of Statistics, 37(2):697–725, 04 2009.
     
    """        
        
    def __init__(self, model, likfun, kernel, backend, seed=None):
        """Constructor of PMC inference schemes.

        Parameters
        ----------
        model : abcpy.models.Model
            Model object that conforms to the Model class 
        likfun : abcpy.approx_lhd.Approx_likelihood
            Approx_likelihood object that conforms to the Approx_likelihood class
        kernel : abcpy.distributions.Distribution
            Distribution object defining the perturbation kernel needed for the sampling
        backend : abcpy.backends.Backend
            Backend object that conforms to the Backend class
        seed : integer, optional
            Optional initial seed for the random number generator. The default value is generated randomly.

        """
        self.model = model
        self.likfun = likfun
        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        
    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 100, covFactor = None, iniPoints = None, full_output=0):
        """Samples from the posterior distribution of the model parameter given the observed 
        data observations.
        
        Parameters
        ----------
        observations : python list 
            Observed data
        steps : integer        
            number of iterations in the sequential algoritm ("generations") 
        n_sample : integer, optional
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
        
        Parameters
        ----------

        """

        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_lhd_func"] = type(self.likfun)
        journal.configuration["n_samples"] = n_samples
        journal.configuration["n_samples_per_param"] = n_samples_per_param
        journal.configuration["steps"] = steps
        journal.configuration["covFactor"] = covFactor
        journal.configuration["iniPoints"] = iniPoints
        
        accepted_parameters = None
        accepted_weights = None
        accepted_cov_mat = None
        new_theta = None
        dim = len(self.model.get_parameters())
        
        # Initialize variables that need to be available remotely
        rc = _RemoteContextPMC(self.backend, self.model, self.likfun, self.kernel, observations, n_samples, n_samples_per_param)
        
        # Initialize particles: When not supplied, randomly draw them from prior distribution
        # Weights of particles: Assign equal weights for each of the particles
        if iniPoints == None:
            accepted_parameters = np.zeros(shape = (n_samples,dim))
            for ind in range(0,n_samples):
                self.model.sample_from_prior()
                accepted_parameters[ind,:] = self.model.get_parameters()
            accepted_weights = np.ones((n_samples,1), dtype=np.float)/n_samples 
        else:
            accepted_parameters = iniPoints
            accepted_weights = np.ones((iniPoints.shape[0],1), dtype=np.float)/iniPoints.shape[0] 
        
        if covFactor == None:
            covFactor = np.ones(shape=(dim,))

        # Calculate initial covariance matrix
        accepted_cov_mat = covFactor * np.cov(accepted_parameters, aweights = accepted_weights.reshape(-1), rowvar=False)                            
        
        # main SMC algorithm  
        # print("INFO: Starting SMC iterations.")
        for aStep in range(0, steps):
            # print("DEBUG: Iteration " + str(aStep) + " of SMC algorithm.")
            
            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            rc.update_broadcasts(self.backend, accepted_parameters, accepted_weights, accepted_cov_mat)

            # 1: calculate resample parameters
            # print("INFO: Resample parameters.")
            index = self.rng.choice(accepted_parameters.shape[0], size = n_samples, p = accepted_weights.reshape(-1))
            #Choose a new particle using the resampled particle (make the boundary proper)
            #Initialize new_parameters
            new_parameters = np.zeros((n_samples,dim), dtype=np.float) 
            for ind in range(0,n_samples):
                self.kernel.set_parameters([accepted_parameters[index[ind],:], accepted_cov_mat])
                while True:
                    new_theta = self.kernel.sample(1)[0,:]
                    theta_is_accepted = self.model.set_parameters(new_theta)
                    if theta_is_accepted and self.model.prior.pdf(self.model.get_parameters()) != 0:
                        new_parameters[ind,:] = new_theta
                        break
            # 2: calculate approximate lieklihood for new parameters
            # print("INFO: Calculate approximate likelihood.")
            new_parameters_pds = self.backend.parallelize(new_parameters)
            approx_likelihood_new_parameters_pds = self.backend.map(rc._approx_lik_calc, new_parameters_pds)
            # print("DEBUG: Collect approximate likelihood from pds.")
            approx_likelihood_new_parameters = np.array(self.backend.collect(approx_likelihood_new_parameters_pds)).reshape(-1,1)         
            
            # 3: calculate new weights for new parameters 
            # print("INFO: Calculating weights.")
            new_weights_pds = self.backend.map(rc._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1,1)
            
            sum_of_weights = 0.0
            for i in range(0, n_samples):
                new_weights[i] = new_weights[i]*approx_likelihood_new_parameters[i]
                sum_of_weights += new_weights[i]
            new_weights = new_weights / sum_of_weights
            accepted_parameters = new_parameters

            # 4: calculate covariance
            # print("INFO: Calculating covariance matrix.")
            new_cov_mat = covFactor * np.cov(accepted_parameters, aweights = accepted_weights.reshape(-1), rowvar=False)
                       
            # 5: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mat = new_cov_mat
            
            # print("INFO: Saving configuration to output journal.")
            if (full_output == 1 and aStep <= steps-1) or (full_output == 0 and aStep == steps-1):
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)
                journal.add_opt_values(approx_likelihood_new_parameters)

        return journal

class _RemoteContextPMC:
    """
    Contains everything that is sent over the network like broadcast vars and map functions
    """        

    def __init__(self, backend, model, likfun, kernel, observations, n_samples, n_samples_per_param,):
        self.model = model
        self.likfun = likfun
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.kernel = kernel

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = backend.broadcast(observations)
        self.accepted_parameters_bds = None
        self.accepted_weights_bds = None
        self.accepted_cov_mat_bds = None

    def update_broadcasts(self, backend, accepted_parameters, accepted_weights, accepted_cov_mat):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                #bc.destroy
                
        if not accepted_parameters is None:
            self.accepted_parameters_bds = backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = backend.broadcast(accepted_weights)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = backend.broadcast(accepted_cov_mat)

    # define helper functions for map step
    def _approx_lik_calc(self, theta):
        """
        Compute likelihood for new parameters using approximate likelihood function
        
        :seed = aTuple[0]
        :theta = aTuple[1]
        :type seed: int
        :rtype: np.arraybg

        :return: likelihood values of new parameter
        """  

        # Assign theta to model
        self.model.set_parameters(theta)
        # Simulate the fake data from the model given the parameter value theta
        # print("DEBUG: Simulate model for parameter " + str(theta))
        y_sim = self.model.simulate(self.n_samples_per_param)
        # print("DEBUG: Extracting observation.")
        obs = self.observations_bds.value()
        # print("DEBUG: Computing likelihood...")
        lhd = self.likfun.likelihood(obs, y_sim)
        # print("DEBUG: Likelihood is :" + str(lhd))
        pdf_at_theta = self.model.prior.pdf(theta)
        # print("DEBUG: prior pdf evaluated at theta is :" + str(pdf_at_theta))
        return pdf_at_theta * lhd
        
    def _calculate_weight(self, theta):
        """
        Calculates the weight for the given parameter using
        accepted_parameters, accepted_cov_mat

        :type theta: np.array
        :param theta: 1xp matrix containing model parameters, where p is the number of parameters
        :rtype: float
        :return: the new weight for theta
        """
        
        if self.accepted_weights_bds is None:
            return 1.0 / self.n_samples
        else:
            prior_prob = self.model.prior.pdf(theta)

            denominator = 0.0
            for i in range(0, self.n_samples):
                self.kernel.set_parameters([self.accepted_parameters_bds.value()[i,:], self.accepted_cov_mat_bds.value()])
                pdf_value = self.kernel.pdf(theta)
                denominator += self.accepted_weights_bds.value()[i,0] * pdf_value
            return 1.0 * prior_prob / denominator        
