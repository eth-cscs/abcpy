from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn import ensemble



class ModelSelections(metaclass = ABCMeta):
    """This abstract base class defines a model selection rule of how to choose a model from a set of models
    given an observation. 
 
            
    """
    
    @abstractmethod
    def __init__(self, model_array, statistics_calc, backend, seed = None):
        """Constructor that must be overwritten by the sub-class.

        The constructor of a sub-class must accept an array of models to choose the model 
        from, and two non-optional parameters statistics calculator and backend stored in self.statistics_calc 
        and self.backend defining how to calculate sumarry statistics from data and what kind of parallelization
        to use.
        
        Parameters
        ----------
        model_array: list
            A list of models which are of type abcpy.models.Model
        statistics: abcpy.statistics.Statistics
            Statistics object that conforms to the Statistics class.
        backend: abcpy.backends.Backend
            Backend object that conforms to the Backend class.
        seed: integer, optional
            Optional initial seed for the random number generator. The default value is generated randomly.    
        """   
        self.model_array = model_array
        self.statistics_calc = statistics_calc
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.reference_table_calculated = None
        

        raise NotImplemented
        
        
    @abstractmethod
    def select_model(self, observations, n_samples = 1000, n_samples_per_param = 100):
        """To be overwritten by any sub-class: returns a model selected by the modelselection
        procedure most suitable to the obersved data set observations. It is assumed that observations is a 
        list of n same type elements(eg., The observations can be a list containing n timeseries, n 
        graphs or n np.ndarray). Further two optional integer arguments n_samples and n_samples_per_param
        is supplied denoting the number of samples in the refernce table and the data points in each 
        simulated data set.
        
        Parameters
        ----------
        observations: python list
            The observed data set.
        n_samples : integer, optional
            Number of samples to generate for reference table.
        n_samples_per_param : integer, optional 
            Number of data points in each simulated data set.  
        Returns
        -------
        abcpy.models.Model
            A model which are of type abcpy.models.Model
            
        """
        
        raise NotImplemented

    @abstractmethod
    def posterior_probability(self, observations):
        """To be overwritten by any sub-class: returns the approximate posterior probability
        of the chosen model given the observed data set observations. It is assumed that observations 
        is a  list of n same type elements(eg., The observations can be a list containing n timeseries, n graphs or n np.ndarray).
        
        Parameters
        ----------
        observations: python list
            The observed data set.
        Returns
        -------
        np.ndarray
            A vector containing the approximate posterior probability of the model chosen.                
        """
        
        raise NotImplemented
        
class RandomForest(ModelSelections):
    """
    This class implements the model selection procedure based on the Random Forest ensemble learner
    as described in Pudlo et. al. [1].
    
    [1] Pudlo, P., Marin, J.-M., Estoup, A., Cornuet, J.-M., Gautier, M. and Robert, C.
    (2016). Reliable ABC model choice via random forests. Bioinformatics, 32 859–866.
    """
    def __init__(self, model_array, statistics_calc, backend, N_tree = 100, n_try_fraction = 0.5, seed = None):
        """        
        Parameters
        ----------
        N_tree : integer, optional
            Number of trees in the random forest. The default value is 100.
        n_try_fraction : float, optional 
            The fraction of number of summary statistics to be considered as the size of 
            the number of covariates randomly sampled at each node by the randomised CART.
            The default value is 0.5.           
        """
        
        self.model_array = model_array
        self.statistics_calc = statistics_calc
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.seed = seed 
        self.reference_table_calculated = 0
        self.N_tree = N_tree
        self.n_try_fraction = n_try_fraction

    def select_model(self, observations, n_samples = 1000, n_samples_per_param = 1):
        """        
        Parameters
        ----------
        observations: python list
            The observed data set.
        n_samples : integer, optional
            Number of samples to generate for reference table. The default value is 1000.
        n_samples_per_param : integer, optional 
            Number of data points in each simulated data set. The default value is 1.
        Returns
        -------
        abcpy.models.Model
            A model which are of type abcpy.models.Model
            
        """
        # Creation of reference table
        if self.reference_table_calculated is 0:        
            rc = _RemoteContextForReferenceTable(self.backend, self.model_array, self.statistics_calc, observations, n_samples_per_param)

            # Simulating the data, distance and statistics                
            seed_arr = self.rng.randint(1, n_samples*n_samples, size=n_samples, dtype=np.int32)
            seed_pds = self.backend.parallelize(seed_arr)     

            model_data_pds = self.backend.map(rc._simulate_model_data, seed_pds)
            model_data = self.backend.collect(model_data_pds)
            models, data, statistics = [list(t) for t in zip(*model_data)]
            self.reference_table_models = models
            self.reference_table_data = data
            self.reference_table_statistics = np.concatenate(statistics)
            self.reference_table_calculated = 1

        # Construct a label for the model_array
        label = np.zeros(shape=(len(self.reference_table_models)))
        for ind1 in range(len(self.reference_table_models)):  
            for ind2 in range(len(self.model_array)):
                if self.reference_table_models[ind1] == self.model_array[ind2]:
                    label[ind1] = ind2 

        # Define the classifier 
        classifier = ensemble.RandomForestClassifier(n_estimators = self.N_tree, \
        max_features=int(self.n_try_fraction*self.reference_table_statistics.shape[1]), bootstrap=True, random_state=self.seed)
        classifier.fit(self.reference_table_statistics, label)            

        return(self.model_array[int(classifier.predict(self.statistics_calc.statistics(observations)))])

    def posterior_probability(self, observations, n_samples = 1000, n_samples_per_param = 1):
        """        
        Parameters
        ----------
        observations: python list
            The observed data set.
        n_samples : integer, optional
            Number of samples to generate for reference table. The default value is 1000.
        n_samples_per_param : integer, optional 
            Number of data points in each simulated data set. The default value is 1.
        Returns
        -------
        abcpy.models.Model
            A model which are of type abcpy.models.Model
            
        """        
        # Creation of reference table
        if self.reference_table_calculated is 0:        
            rc = _RemoteContextForReferenceTable(self.backend, self.model_array, self.statistics_calc, observations, n_samples_per_param)

            # Simulating the data, distance and statistics                
            seed_arr = self.rng.randint(1, n_samples*n_samples, size=n_samples, dtype=np.int32)
            seed_pds = self.backend.parallelize(seed_arr)     

            model_data_pds = self.backend.map(rc._simulate_model_data, seed_pds)
            model_data = self.backend.collect(model_data_pds)
            models, data, statistics = [list(t) for t in zip(*model_data)]
            self.reference_table_models = models
            self.reference_table_data = data
            self.reference_table_statistics = np.concatenate(statistics)
            self.reference_table_calculated = 1
        
        # Construct a label for the model_array
        label = np.zeros(shape=(len(self.reference_table_models)))
        for ind1 in range(len(self.reference_table_models)):  
            for ind2 in range(len(self.model_array)):
                if self.reference_table_models[ind1] == self.model_array[ind2]:
                    label[ind1] = ind2 

        # Define the classifier 
        classifier = ensemble.RandomForestClassifier(n_estimators = self.N_tree, \
        max_features=int(self.n_try_fraction*self.reference_table_statistics.shape[1]), bootstrap=True, random_state=self.seed)
        classifier.fit(self.reference_table_statistics, label)  

        pred_error = np.zeros(len(self.reference_table_models),)
        # Compute missclassification error rate
        for ind in range(len(self.reference_table_models)):
            pred_error[ind] = 1 - classifier.predict_proba(self.statistics_calc.statistics(self.reference_table_data[ind]))[0][int(label[ind])] 

        # Estimate a regression function with prediction error as response on summary statitistics of the reference table                
        regressor = ensemble.RandomForestRegressor(n_estimators = self.N_tree)
        regressor.fit(self.reference_table_statistics,pred_error)

        return(1-regressor.predict(self.statistics_calc.statistics(observations)))
        
class _RemoteContextForReferenceTable:
    """
    Contains everything that is sent over the network like broadcast vars and map functions
    """
    
    def __init__(self, backend, model_array, stat_calc, observations, n_samples_per_param):       
        self.model_array = model_array
        self.stat_calc = stat_calc
        self.n_samples_per_param = n_samples_per_param
        
        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = backend.broadcast(observations)
        
    def _simulate_model_data(self, seed):
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
        #reseed random number genearator
        rng = np.random.RandomState(seed)
        len_model_array = len(self.model_array)
        model = self.model_array[int(sum(np.linspace(0,len_model_array-1,len_model_array)\
        *rng.multinomial(1,(1/len_model_array)*np.ones(len_model_array))))]
        #reseed prior
        model.prior.reseed(seed)
        #sample from prior
        model.sample_from_prior()
        #sample data set, extract statistics and compute distance from y_obs
        y_sim = model.simulate(self.n_samples_per_param)
        statistics = self.stat_calc.statistics(y_sim)
        
        return (model, y_sim, statistics)