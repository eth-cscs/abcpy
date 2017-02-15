from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier



class ModelSelections(metaclass = ABCMeta):
    """This abstract base class defines how to calculate statistics from dataset.

    The base class also implements a polynomial expansion with cross-product
    terms that can be used to get desired polynomial expansion of the calculated statistics.
    
            
    """
    
    @abstractmethod
    def __init__(self, model_array, statistics_calc, distance_calc, backend, seed = None):
        """Constructor that must be overwritten by the sub-class.

        The constructor of a sub-class must accept arguments for the polynomial 
        expansion after extraction of the summary statistics, one has to define 
        the degree of polynomial expansion and cross, indicating whether cross-prodcut
        terms are included.
        
        Parameters
        ----------
        model_array: list
            A list of models which are of type abcpy.models.Model
        statistics: abcpy.statistics.Statistics
            Statistics object that conforms to the Statistics class.
        distance: abcpy.distances.Distance
            Distance object that conforms to the Distance class.
        backend: abcpy.backends.Backend
            Backend object that conforms to the Backend class.
        seed: integer, optional
            Optional initial seed for the random number generator. The default value is generated randomly.    
        """   
        self.model_array = model_array
        self.statistics_calc = statistics_calc
        self.distance_calc = distance_calc
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.reference_table_calculated = None
        

        raise NotImplemented
        
        
    @abstractmethod
    def modelchoice(self, observations, n_samples = 1000, n_samplee_per_param = 100):
        """To be overwritten by any sub-class: returns a model chosen by the modelselection
        procedure most suitable to the obersved data set observations. It is assumed that observations is a 
        list of n same type elements(eg., The observations can be a list containing n timeseries, n 
        graphs or n np.ndarray). Further two optional integer arguments n_samples and n_samples_per_param
        is supplied denoting the number of samples in the refernce table and the data points in each 
        simulated data set.
        
        Parameters
        ----------
        data: python list
            Contains n data sets.
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
    def posteriorprobability(self, observations):
        """To be overwritten by any sub-class: returns the approximate posterior probabilities 
        of the models in model_array given the observed data set observations. It is assumed that observations 
        is a  list of n same type elements(eg., The observations can be a list containing n timeseries, n graphs or n np.ndarray).
        
        Parameters
        ----------
        observations: python list
            Contains n data sets.
        Returns
        -------
        np.ndarray
            A vector containing the approximate posterior probabilities of the models in 
            model_array. The elements in the array should sum upto 1.            
                
        """
        
        raise NotImplemented

class Knn(ModelSelections):
    """
    This class implements the model selection procedure based on the k-nearest neighbour 
    classifier as described in Stoehr et. al. [1].
    
    [1] Stoehr, J., Pudlo, P. and Cucala, L. (2015). Adaptive ABC model choice and geometric
    summary statistics for hidden Gibbs random fields. Statistics and Computing, 25, 129-141.
    """
    def __init__(self, model_array, statistics_calc, distance_calc, backend, seed = None):
        self.model_array = model_array
        self.statistics_calc = statistics_calc
        self.distance_calc = distance_calc
        self.backend = backend
        self.rng = np.random.RandomState(seed)  
        self.reference_table_calculated = 0

    def modelchoice(self, observations, n_samples = 1000, n_samples_per_param = 100, k = 50):
        """        
        Parameters
        ----------
        data: python list
            Contains n data sets.
        n_samples : integer, optional
            Number of samples to generate for reference table.
        n_samples_per_param : integer, optional 
            Number of data points in each simulated data set. 
        k : integer, optional
            The parameter for K-nearest neighbour classifier. The default calue if 50
        Returns
        -------
        abcpy.models.Model
            A model which are of type abcpy.models.Model
            
        """
        
        # Creation of reference table
        if self.reference_table_calculated is 0:        
            rc = _RemoteContextgeneratereferencetable(self.backend, self.model_array, self.distance_calc, self.statistics_calc, observations, n_samples_per_param)

            # Simulating the data, distance and statistics                
            seed_arr = self.rng.randint(1, n_samples*n_samples, size=n_samples, dtype=np.int32)
            seed_pds = self.backend.parallelize(seed_arr)     

            model_data_pds = self.backend.map(rc._simulate_model_data, seed_pds)
            model_data = self.backend.collect(model_data_pds)
            models, data, distances, statistics = [list(t) for t in zip(*model_data)]
            self.reference_table_models = models
            self.reference_table_data = data
            self.reference_table_distances = np.array(distances)
            self.reference_table_statistics = statistics
            self.reference_table_calculated = 1
        
        # Sort the array
        Indices = np.array(sorted(range(len(self.reference_table_distances)), key=lambda k: self.reference_table_distances[k])).astype(int)

        # Compute the frequence of each model chosen in the k models having the smalles distance
        count = np.zeros(shape=(len(self.model_array),1))
        for ind1 in range(k):  
            for ind2 in range(len(self.model_array)):
                if self.reference_table_models[Indices[ind1]] == self.model_array[ind2]:
                    count[ind2] = count[ind2] + 1

        return(self.model_array[np.argmax(count)])

    def posteriorprobability(self, observations, n_samples = 1000, n_samples_per_param = 100, k = 50):
        """        
        Parameters
        ----------
        data: python list
            Contains n data sets.
        n_samples : integer, optional
            Number of samples to generate for reference table.
        n_samples_per_param : integer, optional 
            Number of data points in each simulated data set. 
        k : integer, optional
            The parameter for K-nearest neighbour classifier. The default calue if 50
        Returns
        -------
        abcpy.models.Model
            A model which are of type abcpy.models.Model
            
        """            
        
        # Creation of reference table
        if self.reference_table_calculated is 0:        
            rc = _RemoteContextgeneratereferencetable(self.backend, self.model_array, self.distance_calc, self.statistics_calc, observations, n_samples_per_param)

            # Simulating the data, distance and statistics                
            seed_arr = self.rng.randint(1, n_samples*n_samples, size=n_samples, dtype=np.int32)
            seed_pds = self.backend.parallelize(seed_arr)     

            model_data_pds = self.backend.map(rc._simulate_model_data, seed_pds)
            model_data = self.backend.collect(model_data_pds)
            models, data, distances, statistics = [list(t) for t in zip(*model_data)]
            self.reference_table_models = models
            self.reference_table_data = data
            self.reference_table_distances = np.array(distances)
            self.reference_table_statistics = statistics
            self.reference_table_calculated = 1
        
        # Sort the array
        Indices = np.array(sorted(range(len(self.reference_table_distances)), key=lambda k: self.reference_table_distances[k])).astype(int)
        
        # Compute the frequence of each model chosen in the k models having the smalles distance
        count = np.zeros(shape=(len(self.model_array),1))
        for ind1 in range(k):  
            for ind2 in range(len(self.model_array)):
                if self.reference_table_models[Indices[ind1]] == self.model_array[ind2]:
                    count[ind2] = count[ind2] + 1

        return(count/k)
        
class RandomForest(ModelSelections):
    """
    This class implements the model selection procedure based on the k-nearest neighbour 
    classifier as described in Pudlo et. al. [1].
    
    [1] Pudlo, P., Marin, J.-M., Estoup, A., Cornuet, J.-M., Gautier, M. and Robert, C.
    (2016). Reliable ABC model choice via random forests. Bioinformatics, 32 859–866.
    """
    def __init__(self, model_array, statistics_calc, distance_calc, backend, seed = None):
        self.model_array = model_array
        self.statistics_calc = statistics_calc
        self.distance_calc = distance_calc
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.seed = seed 
        

    def modelchoice(self, observations, n_samples = 1000, n_samples_per_param = 1, N_tree = 100, n_try_fraction = 0.5):
        """        
        Parameters
        ----------
        data: python list
            Contains n data sets.
        n_samples : integer, optional
            Number of samples to generate for reference table.
        n_samples_per_param : integer, optional 
            Number of data points in each simulated data set. 
        N_tree : integer, optional
            Number of trees in the random forest. The default value if 100
        n_try_fraction : float, optional 
            The fraction of number of summary statistics to be considered as the size of 
            the number of covariates randomly sampled at each node by the randomised CART.
        Returns
        -------
        abcpy.models.Model
            A model which are of type abcpy.models.Model
            
        """
        # Creation of reference table
        if self.reference_table_calculated is 0:        
            rc = _RemoteContextgeneratereferencetable(self.backend, self.model_array, self.distance_calc, self.statistics_calc, observations, n_samples_per_param)

            # Simulating the data, distance and statistics                
            seed_arr = self.rng.randint(1, n_samples*n_samples, size=n_samples, dtype=np.int32)
            seed_pds = self.backend.parallelize(seed_arr)     

            model_data_pds = self.backend.map(rc._simulate_model_data, seed_pds)
            model_data = self.backend.collect(model_data_pds)
            models, data, distances, statistics = [list(t) for t in zip(*model_data)]
            self.reference_table_models = models
            self.reference_table_data = data
            self.reference_table_distances = np.array(distances)
            self.reference_table_statistics = statistics
            self.reference_table_calculated = 1

        # Construct a label for the model_array
        label = np.zeros(shape=(len(self.reference_table_models)))
        for ind1 in range(len(self.reference_table_models)):  
            for ind2 in range(len(self.model_array)):
                if self.reference_table_models[ind1] == self.model_array[ind2]:
                    label[ind1] = ind2 

        # Define the classifier 
        classifier = RandomForestClassifier(n_estimators = N_tree, max_features=int(n_try_fraction*self.reference_table_statistics.shape[1]), bootstrap=True, random_state=self.seed)
        classifier.fit(self.reference_table_statistics, label)            

        return(self.model_array[int(classifier.predict(self.statistics_calc.statistics(observations)))])

    def posteriorprobability(self, observations, n_samples = 1000, n_samples_per_param = 1, N_tree = 100, n_try_fraction = 0.5):
        """        
        Parameters
        ----------
        data: python list
            Contains n data sets.
        n_samples : integer, optional
            Number of samples to generate for reference table.
        n_samples_per_param : integer, optional 
            Number of data points in each simulated data set. 
        N_tree : integer, optional
            Number of trees in the random forest. The default value if 100
        n_try_fraction : float, optional 
            The fraction of number of summary statistics to be considered as the size of 
            the number of covariates randomly sampled at each node by the randomised CART.
        Returns
        -------
        abcpy.models.Model
            A model which are of type abcpy.models.Model
            
        """        
        # Creation of reference table
        if self.reference_table_calculated is 0:        
            rc = _RemoteContextgeneratereferencetable(self.backend, self.model_array, self.distance_calc, self.statistics_calc, observations, n_samples_per_param)

            # Simulating the data, distance and statistics                
            seed_arr = self.rng.randint(1, n_samples*n_samples, size=n_samples, dtype=np.int32)
            seed_pds = self.backend.parallelize(seed_arr)     

            model_data_pds = self.backend.map(rc._simulate_model_data, seed_pds)
            model_data = self.backend.collect(model_data_pds)
            models, data, distances, statistics = [list(t) for t in zip(*model_data)]
            self.reference_table_models = models
            self.reference_table_data = data
            self.reference_table_distances = np.array(distances)
            self.reference_table_statistics = statistics
            self.reference_table_calculated = 1
        
        # Construct a label for the model_array
        label = np.zeros(shape=(len(self.reference_table_models)))
        for ind1 in range(len(self.reference_table_models)):  
            for ind2 in range(len(self.model_array)):
                if self.reference_table_models[ind1] == self.model_array[ind2]:
                    label[ind1] = ind2 

        # Define the classifier 
        classifier = RandomForestClassifier(n_estimators = N_tree, max_features=int(n_try_fraction*statistics.shape[1]), bootstrap=True, random_state=self.seed)
        classifier.fit(self.reference_table_statistics, label)            


        return(classifier.predict_proba(self.statistics_calc.statistics(observations)))
        
class _RemoteContextgeneratereferencetable:
    """
    Contains everything that is sent over the network like broadcast vars and map functions
    """
    
    def __init__(self, backend, model_array, dist_calc, stat_calc, observations, n_samples_per_param):       
        self.model_array = model_array
        self.dist_calc = dist_calc
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
        distance = self.dist_calc.distance(self.observations_bds.value(), y_sim)
        statistics = self.stat_calc.statistics(y_sim)
        
        return (model, y_sim, distance, statistics)