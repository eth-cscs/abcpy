import numpy as np
from abcpy.output import Journal
from scipy import optimize

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
            rc._update_broadcasts(self.backend, accepted_parameters, accepted_weights, accepted_cov_mat)

            # 1: calculate resample parameters
            # print("INFO: Resampling parameters")
            params_and_dists_and_ysim_pds = self.backend.map(rc._resample_parameter, seed_pds)
            params_and_dists_and_ysim = self.backend.collect(params_and_dists_and_ysim_pds)
            new_parameters, distances = [list(t) for t in zip(*params_and_dists_and_ysim)]
            new_parameters = np.array(new_parameters)
            rc._update_broadcasts(self.backend, accepted_parameters, accepted_weights, accepted_cov_mat)
            
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


    def _update_broadcasts(self, backend, accepted_parameters, accepted_weights, accepted_cov_mat):
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
        
        if covFactor is None:
            covFactor = np.ones(shape=(dim,))

        # Calculate initial covariance matrix
        accepted_cov_mat = covFactor * np.cov(accepted_parameters, aweights = accepted_weights.reshape(-1), rowvar=False)                            
        
        # main SMC algorithm  
        # print("INFO: Starting SMC iterations.")
        for aStep in range(0, steps):
            # print("DEBUG: Iteration " + str(aStep) + " of SMC algorithm.")
            
            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            rc._update_broadcasts(self.backend, accepted_parameters, accepted_weights, accepted_cov_mat)

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

    def _update_broadcasts(self, backend, accepted_parameters, accepted_weights, accepted_cov_mat):
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


class SABC:
    """This base class implements a modified version of Simulated Annealing Approximate Bayesian Computation (SABC) of [1] when the prior is non-informative.
        
    [1] C. Albert, H. R. Kuensch and A. Scheidegger. A Simulated Annealing Approach to 
    Approximate Bayes Computations. Statistics and Computing, (2014). 
    
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
        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["type_kernel_func"] = type(self.kernel)
        journal.configuration["n_samples"] = n_samples
        journal.configuration["n_samples_per_param"] = n_samples_per_param
        journal.configuration["beta"] = beta
        journal.configuration["delta"] = delta
        journal.configuration["v"] = v
        journal.configuration["ar_cutoff"] = ar_cutoff
        journal.configuration["resample"] = resample
        journal.configuration["n_update"] = n_update
        journal.configuration["adaptcov"] = adaptcov
        journal.configuration["full_output"] = full_output
        
        accepted_parameters = np.zeros(shape=(n_samples,len(self.model.get_parameters())))
        distances = np.zeros(shape=(n_samples,))
        smooth_distances = np.zeros(shape=(n_samples,))
        accepted_weights = np.ones(shape=(n_samples,1))
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
        
        # Initialize variables that need to be available remotely
        rc = _RemoteContextSABC(self.backend, self.model, self.distance, self.kernel, \
        observations, n_samples, n_samples_per_param)
        
        for aStep in range(0,steps):
            # main SABC algorithm 
            # print("INFO: Initialization of SABC")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=int(sample_array[aStep]), dtype=np.uint32)
            seed_pds = self.backend.parallelize(seed_arr)

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            rc.epsilon = epsilon
            rc._update_broadcasts(self.backend, accepted_parameters, accepted_cov_mat, smooth_distances, all_distances)
        

            # 1: Calculate  parameters
            # print("INFO: Initial accepted parameter parameters")
            params_and_dists_pds = self.backend.map(rc._accept_parameter, seed_pds)
            params_and_dists = self.backend.collect(params_and_dists_pds)
            new_parameters, new_distances, new_all_parameters, new_all_distances, index, acceptance = [list(t) for t in zip(*params_and_dists)]
            new_parameters = np.array(new_parameters)            
            new_distances = np.array(new_distances)
            new_all_distances = np.concatenate(new_all_distances)
            index = np.array(index)
            acceptance = np.array(acceptance)
            
            # Reading all_distances at Initial step 
            if aStep == 0:
                index = np.linspace(0,n_samples-1,n_samples).astype(int).reshape(n_samples,)
                accept = 0
                all_distances = new_all_distances

            #print(index[acceptance == 1])              
            # Initialize/Update the accepted parameters and their corresponding distances            
            accepted_parameters[index[acceptance==1],:] = new_parameters[acceptance==1,:]
            distances[index[acceptance==1]] = new_distances[acceptance==1]
            
                
            # 2: Smoothing of the distances
            smooth_distances[index[acceptance==1]] = self._smoother_distance(distances[index[acceptance==1]],all_distances)
            
            # 3: Initialize/Update U, epsilon and covariance of perturbation kernel
            if aStep == 0:
                U = self._avergae_redefined_distance(self._smoother_distance(all_distances,all_distances), epsilon)
            else:               
                U = np.mean(smooth_distances)
            epsilon = self._schedule(U,v) 
            if accepted_parameters.shape[1] > 1:
                accepted_cov_mat = beta*np.cov(np.transpose(accepted_parameters)) + \
                0.0001*np.trace(np.cov(np.transpose(accepted_parameters)))*np.eye(accepted_parameters.shape[1])
            else:
                accepted_cov_mat = beta*np.cov(np.transpose(accepted_parameters)) + \
                0.0001*(np.cov(np.transpose(accepted_parameters)))*np.eye(accepted_parameters.shape[1]) 
            ## 4: Show progress and if acceptance rate smaller than a value break the iteration
            
            # print("INFO: Saving intermediate configuration to output journal.")
            if full_output == 1:
                journal.add_parameters(accepted_parameters)
                journal.add_weights(accepted_weights)
                
            if aStep > 0:
                accept = accept + np.sum(acceptance)
                samples_until = samples_until + sample_array[aStep]
                acceptance_rate = accept/samples_until
                print('updates: ',np.sum(sample_array[1:aStep+1])/np.sum(sample_array[1:])*100,' epsilon: ' ,epsilon,\
                'u.mean: ', U, 'acceptance rate: ', acceptance_rate)
                if acceptance_rate < ar_cutoff:
                    break
       
            # 5: Resampling if number of accepted particles greater than resample
            if accept >= resample and U > 1e-100:
                ## Weighted resampling:
                weight = np.exp(-smooth_distances*delta/U)
                weight = weight/sum(weight)
                index_resampled = self.rng.choice(np.arange(n_samples), n_samples, replace = 1, p = weight)
                accepted_parameters = accepted_parameters[index_resampled,:]
                smooth_distances = smooth_distances[index_resampled]
  
                ## Update U and epsilon:
                epsilon = epsilon*(1-delta)
                U = np.mean(smooth_distances)
                epsilon = self._schedule(U,v) 
  
                ## Print effective sampling size
                print('Resampling: Effective sampling size: ', 1/sum(pow(weight/sum(weight),2)))
                accept = 0      
                samples_until = 0
      
        #Add epsilon_arr, number of final steps and final output to the journal 
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
        """
        
        smoothed_distance = np.zeros(shape=(len(distance),))        
        
        for ind in range(0,len(distance)):
            if distance[ind] < np.min(old_distance):
                smoothed_distance[ind] = (distance[ind]/np.min(old_distance))/len(old_distance)
            else:
                smoothed_distance[ind] = np.mean(np.array(old_distance)<distance[ind])
                
        return smoothed_distance
        
    def _avergae_redefined_distance(self, distance, epsilon):        
        if epsilon==0:
            U = 0
        else:
            U = np.average(distance, weights = np.exp(-distance/epsilon))

        return(U)
        
    def _schedule(self, rho, v):
      if rho < 1e-100:
          epsilon = 0
      else:
          fun = lambda epsilon: pow(epsilon,2) + v*pow(epsilon,3/2) - pow(rho,2)
          epsilon = optimize.fsolve(fun,rho/2)
             
      return(epsilon)
      

class _RemoteContextSABC:
    """
    Contains everything that is sent over the network like broadcast vars and map functions
    """
    
    def __init__(self, backend, model, distance, kernel, observations, n_samples, n_samples_per_param):
        self.model = model
        self.distance = distance
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param     
        self.epsilon = None
        self.kernel = kernel
        #self._smoother_distance = _smoother_distance

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = backend.broadcast(observations)               
        self.accepted_parameters_bds = None
        self.accepted_cov_mat_bds = None
        self.smooth_distances_bds = None
        self.all_distances_bds = None
        
    def _update_broadcasts(self, backend, accepted_parameters, accepted_cov_mat, smooth_distances, all_distances):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                #bc.destroy
                
        if not accepted_parameters is None:
            self.accepted_parameters_bds = backend.broadcast(accepted_parameters)  
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = backend.broadcast(accepted_cov_mat)
        if not smooth_distances is None:
            self.smooth_distances_bds = backend.broadcast(smooth_distances)
        if not all_distances is None:
            self.all_distances_bds = backend.broadcast(all_distances)
    
    def _smoother_distance_remote(self, distance, old_distance):
        """Smooths the distance using the Equation 14 of [1].
        
        [1] C. Albert, H. R. Kuensch and A. Scheidegger. A Simulated Annealing Approach to 
        Approximate Bayes Computations. Statistics and Computing 0960-3174 (2014).     
        """
        
        smoothed_distance = np.zeros(shape=(len(distance),))        
        
        for ind in range(0,len(distance)):
            if distance[ind] < np.min(old_distance):
                smoothed_distance[ind] = (distance[ind]/np.min(old_distance))/len(old_distance)
            else:
                smoothed_distance[ind] = np.mean(np.array(old_distance)<distance[ind])
                
        return smoothed_distance
        
    # define helper functions for map step
    def _accept_parameter(self, seed):
        """
        Samples a single model parameter and simulate from it until
        accepted with probabilty exp[-rho(x,y)/epsilon].
            
        :type seed: int
        :rtype: np.array
        :return: accepted parameter
        """
        rng = np.random.RandomState(seed)
        self.model.prior.reseed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        self.kernel.reseed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))        
        
        
        
        all_parameters = []
        all_distances = []  
        index = []          
        acceptance = 0 
        
        
        
        if self.accepted_cov_mat_bds == None:
            while acceptance == 0:
                self.model.sample_from_prior()
                new_theta = self.model.get_parameters()
                all_parameters.append(self.model.get_parameters())
                y_sim = self.model.simulate(self.n_samples_per_param)
                distance = self.distance.distance(self.observations_bds.value(), y_sim)
                all_distances.append(distance)
                acceptance = rng.binomial(1,np.exp(-distance/self.epsilon),1)
            acceptance = 1
        else:                    
            ## Select one arbitrary particle:
            index = rng.choice(self.n_samples, size=1)[0]
            ## Sample proposal parameter and calculate new distance:
            theta = self.accepted_parameters_bds.value()[index,:]
            while True:
                self.kernel.set_parameters([theta, self.accepted_cov_mat_bds.value()])
                new_theta = self.kernel.sample(1)[0,:]
                theta_is_accepted = self.model.set_parameters(new_theta)
                if theta_is_accepted and self.model.prior.pdf(self.model.get_parameters()) != 0:
                    break
            y_sim = self.model.simulate(self.n_samples_per_param)
            distance = self.distance.distance(self.observations_bds.value(), y_sim)          
            smooth_distance = self._smoother_distance_remote([distance],self.all_distances_bds.value())
            
            ## Calculate acceptance probability:
            ratio_prior_prob  = self.model.prior.pdf(new_theta)/self.model.prior.pdf(self.accepted_parameters_bds.value()[index,:])
            ratio_likelihood_prob = np.exp((self.smooth_distances_bds.value()[index] - smooth_distance) / self.epsilon)
            acceptance_prob = ratio_prior_prob*ratio_likelihood_prob
            
            ## If accepted
            if rng.rand(1) < acceptance_prob:
                acceptance = 1
            else:
                distance = np.inf
            
        return (new_theta, distance, all_parameters, all_distances, index, acceptance)

class ABCsubsim:
    """This base class implements Approximate Bayesian Computation by subset simulation (ABCsubsim) algorithm of [1].
        
    [1] M. Chiachio, J. L. Beck, J. Chiachio, and G. Rus., Approximate Bayesian computation by subset
    simulation. SIAM J. Sci. Comput., 36(3):A1339–A1358, 2014/10/03 2014.
    
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
        

    def sample(self, observations, steps, n_samples = 10000, n_samples_per_param = 1, chain_length = 10, ap_change_cutoff = 10, full_output=0):
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
        chain_length : integer, optional
            Chain length of the MCMC. n_samples should be divisable by chain_length. The default value is 10.
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
        
        journal = Journal(full_output)
        journal.configuration["type_model"] = type(self.model)
        journal.configuration["type_dist_func"] = type(self.distance)
        journal.configuration["type_kernel_func"] = type(self.kernel)
        journal.configuration["n_samples"] = n_samples
        journal.configuration["n_samples_per_param"] = n_samples_per_param
        journal.configuration["chain_length"] = chain_length
        journal.configuration["ap_change_cutoff"] = ap_change_cutoff
        journal.configuration["full_output"] = full_output
        
        accepted_parameters = None
        accepted_weights = np.ones(shape=(n_samples,1))
        accepted_cov_mat = None  
        anneal_parameter = 0
        anneal_parameter_old = 0
        temp_chain_length = 1
        
                # Initialize variables that need to be available remotely
        rc = _RemoteContextABCsubsim(self.backend, self.model, self.distance, self.kernel, observations, n_samples, n_samples_per_param, chain_length)
        
        for aStep in range(0,steps):
            # main SABC algorithm 
            # print("INFO: Initialization of SABC")
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=int(n_samples/temp_chain_length), dtype=np.uint32)
            index_arr = np.linspace(0,n_samples/temp_chain_length-1,n_samples/temp_chain_length).astype(int).reshape(int(n_samples/temp_chain_length),)
            seed_index_arr = np.column_stack((seed_arr,index_arr))            
            seed_index_pds = self.backend.parallelize(seed_index_arr)
            

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            rc._update_broadcasts(self.backend, accepted_parameters, accepted_cov_mat)       

            # 1: Calculate  parameters
            # print("INFO: Initial accepted parameter parameters")
            params_and_dists_pds = self.backend.map(rc._accept_parameter, seed_index_pds)
            params_and_dists = self.backend.collect(params_and_dists_pds)
            new_parameters, new_distances = [list(t) for t in zip(*params_and_dists)]
            accepted_parameters = np.concatenate(new_parameters)  
            distances = np.concatenate(new_distances)

            # 2: Sort and renumber samples   
            SortIndex = sorted(range(len(distances)), key=lambda k: distances[k])
            distances = distances[SortIndex]
            accepted_parameters = accepted_parameters[SortIndex,:]
            
            # 3: Calculate and broadcast annealling parameters
            temp_chain_length = chain_length
            if aStep > 0:
                anneal_parameter_old = anneal_parameter
            anneal_parameter = 0.5*(distances[int(n_samples/temp_chain_length)]+distances[int(n_samples/temp_chain_length)+1])
            rc.anneal_parameter = anneal_parameter
            
            # 4: Update proposal covariance matrix (Parallelized)
            if aStep == 0:
                accepted_cov_mat = np.cov(accepted_parameters, rowvar=False)
            else:
                accepted_cov_mat = pow(2,1)*accepted_cov_mat
            rc._update_broadcasts(self.backend, accepted_parameters, accepted_cov_mat)
            
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=10, dtype=np.uint32)
            index_arr = np.linspace(0,10-1,10).astype(int).reshape(10,)
            seed_index_arr = np.column_stack((seed_arr,index_arr))
            seed_index_pds = self.backend.parallelize(seed_index_arr)
            
            cov_mat_index_pds = self.backend.map(rc._update_cov_mat, seed_index_pds)
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
            anneal_parameter_change_percentage = 100*abs(anneal_parameter_old-anneal_parameter)/anneal_parameter
            print('Steps: ', aStep, 'annealing parameter: ', anneal_parameter, 'change (%) in annealing parameter: ', anneal_parameter_change_percentage )
            if anneal_parameter_change_percentage < ap_change_cutoff:
                break
            

        #Add anneal_parameter, number of final steps and final output to the journal 
        # print("INFO: Saving final configuration to output journal.")
        if full_output == 0:
            journal.add_parameters(accepted_parameters)
            journal.add_weights(accepted_weights)
        journal.configuration["steps"] = aStep+1              
        journal.configuration["anneal_parameter"] = anneal_parameter

        return journal

class _RemoteContextABCsubsim:
    """
    Contains everything that is sent over the network like broadcast vars and map functions
    """
    
    def __init__(self, backend, model, distance, kernel, observations, n_samples, n_samples_per_param, chain_length):
        self.model = model
        self.distance = distance
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.kernel = kernel
        self.chain_length = chain_length
        
        self.anneal_parameter = None
        
        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = backend.broadcast(observations)
        self.accepted_parameters_bds = None
        self.accepted_cov_mat_bds = None


    def _update_broadcasts(self, backend, accepted_parameters, accepted_cov_mat):
        def destroy(bc):
            if bc != None:
                bc.unpersist
                #bc.destroy
                
        if not accepted_parameters is None:
            self.accepted_parameters_bds = backend.broadcast(accepted_parameters)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = backend.broadcast(accepted_cov_mat)
                

    # define helper functions for map step
    def _accept_parameter(self, seed_index):
        """
        Samples a single model parameter and simulate from it until
        distance between simulated outcome and the observation is
        smaller than eplison.
            
        :type seed: int
        :rtype: np.array
        :return: accepted parameter
        """
        seed = seed_index[0]
        index = seed_index[1]
        rng = np.random.RandomState(seed)
        self.model.prior.reseed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        self.kernel.reseed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))

        result_theta = []        
        result_distance = []       
        
        if self.accepted_parameters_bds == None:
            self.model.sample_from_prior()
            y_sim = self.model.simulate(self.n_samples_per_param)
            distance = self.distance.distance(self.observations_bds.value(), y_sim)
            result_theta.append(self.model.get_parameters())
            result_distance.append(distance)
        else:
            theta = self.accepted_parameters_bds.value()[index]
            self.model.set_parameters(theta)
            y_sim = self.model.simulate(self.n_samples_per_param)
            distance = self.distance.distance(self.observations_bds.value(), y_sim)
            result_theta.append(theta)
            result_distance.append(distance)
            for ind in range(0,self.chain_length-1):
                while True:
                    self.kernel.set_parameters([theta, self.accepted_cov_mat_bds.value()])
                    new_theta = self.kernel.sample(1)[0,:]
                    theta_is_accepted = self.model.set_parameters(new_theta)
                    if theta_is_accepted and self.model.prior.pdf(self.model.get_parameters()) != 0:
                        break
                y_sim = self.model.simulate(self.n_samples_per_param)
                new_distance = self.distance.distance(self.observations_bds.value(), y_sim)

                ## Calculate acceptance probability:
                ratio_prior_prob  = self.model.prior.pdf(new_theta)/self.model.prior.pdf(theta)
                self.kernel.set_parameters([new_theta, self.accepted_cov_mat_bds.value()])
                kernel_numerator = self.kernel.pdf(theta)
                self.kernel.set_parameters([theta, self.accepted_cov_mat_bds.value()])                
                kernel_denominator = self.kernel.pdf(new_theta)
                ratio_likelihood_prob = kernel_numerator/kernel_denominator
                acceptance_prob = min(1,ratio_prior_prob*ratio_likelihood_prob)*(new_distance<self.anneal_parameter)
            
                ## If accepted
                if rng.binomial(1,acceptance_prob)==1:
                    result_theta.append(new_theta)
                    result_distance.append(new_distance)
                    theta = new_theta
                    distance = new_distance
                else:
                    result_theta.append(theta)
                    result_distance.append(distance)
                    
        return (result_theta, result_distance)
        
    def _update_cov_mat(self, seed_t):
        
        seed = seed_t[0]
        t = seed_t[1]
        rng = np.random.RandomState(seed)
        self.model.prior.reseed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        self.kernel.reseed(rng.randint(np.iinfo(np.uint32).max, dtype=np.uint32))
        acceptance = 0
        accepted_cov_mat_transformed = self.accepted_cov_mat_bds.value()*pow(2.0,-2.0*t)
        theta = self.accepted_parameters_bds.value()[0]
        self.model.set_parameters(theta)
        for ind in range(0,self.chain_length):
            while True:
                self.kernel.set_parameters([theta, accepted_cov_mat_transformed])
                new_theta = self.kernel.sample(1)[0,:]
                theta_is_accepted = self.model.set_parameters(new_theta)
                if theta_is_accepted and self.model.prior.pdf(self.model.get_parameters()) != 0:
                    break
                y_sim = self.model.simulate(self.n_samples_per_param)
                new_distance = self.distance.distance(self.observations_bds.value(), y_sim)
                        
                ## Calculate acceptance probability:
                ratio_prior_prob  = self.model.prior.pdf(new_theta)/self.model.prior.pdf(theta)
                self.kernel.set_parameters([new_theta, accepted_cov_mat_transformed])
                kernel_numerator = self.kernel.pdf(theta)
                self.kernel.set_parameters([theta, accepted_cov_mat_transformed])                
                kernel_denominator = self.kernel.pdf(new_theta)
                ratio_likelihood_prob = kernel_numerator/kernel_denominator
                acceptance_prob = min(1,ratio_prior_prob*ratio_likelihood_prob)*(new_distance<self.anneal_parameter)
                ## If accepted
                if rng.binomial(1,acceptance_prob)==1:
                    theta = new_theta
                    acceptance = acceptance + 1
        if acceptance/10<=0.5 and acceptance/10>=0.3:
            return(accepted_cov_mat_transformed, t, 1)
        else:
            return(accepted_cov_mat_transformed, t, 0)