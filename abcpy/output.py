import numpy as np
import pickle



class Journal:
    """The journal holds information created by the run of inference schemes.

    It can be configured to even hold intermediate.

    Attributes
    ----------
    parameters : numpy.array
        a nxpxt matrix
    weights : numpy.array
        a nxt matrix
    opt_value: numpy.array
        nxp matrix containing for each parameter the evaluated objective function for every time step
    configuration: Python dictionary
        dictionary containing the schemes configuration parameters

    """
    
    def __init__(self, type):
        """
        Initializes a new output journal of given type.

        Parameters
        ----------
        type: int (identifying type)
            type=0 only logs final parametersa and weight (production use);
            type=1 logs all generated information (reproducibily use).
        """
        
        self.parameters = []
        self.weights = []
        self.opt_values = []
        self.configuration = {}

        if type not in [0, 1]:
            raise ValueError("Parameter type has not valid value.")
        else:
            self._type = type



    @classmethod
    def fromFile(cls, filename):
        """This method reads a saved journal from disk an returns it as an object.

        Notes
        -----
        To store a journal use Journal.save(filename).

        Parameters
        ----------
        filename: string
            The string representing the location of a file
            
        Returns
        -------
        abcpy.output.Journal
            The journal object serialized in <filename>


        Example
        --------
        >>> jnl = Journal.fromFile('example_output.jnl')

        """
        
        with open(filename, 'rb') as input:
            journal = pickle.load(input)
        return journal
        

        
    def add_parameters(self, params):
        """
        Saves provided parameters by appending them to the journal. If type==0, old parameters get overwritten.

        Parameters
        ----------
        params: numpy.array 
            nxp matrix containing n parameters of dimension p
        """

        if self._type == 0:
            self.parameters = [params]

        if self._type == 1:
            self.parameters.append(params)



    def get_parameters(self, iteration=None):
        """
        Returns the parameters from a sampling scheme.

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return parameters
        """

        if iteration is None:
            endp = len(self.parameters) - 1
            params = self.parameters[endp]
            return params
        else:
            return self.parameters[iteration]
        


    def get_weights(self, iteration=None):
        """
        Returns the weights from a sampling scheme.

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return weights
        """

        if iteration is None:
            end = len(self.weights) - 1
            return self.weights[end]
        else:
            return self.weights[iteration]


    
    def add_weights(self, weights):
        """
        Saves provided weights by appending them to the journal. If type==0, old weights get overwritten.

        Parameters
        ----------
        weights: numpy.array
            vector containing n weigths
        """

        if self._type == 0:
            self.weights = [weights]

        if self._type == 1:
            self.weights.append(weights)



    def add_opt_values(self, opt_values):
        """
        Saves provided values of the evaluation of the schemes objective function. If type==0, old values get overwritten

        Parameters
        ----------
        opt_value: numpy.array
            vector containing n evaluations of the schemes objective function
        """

        if self._type == 0:
            self.opt_values = [opt_values]

        if self._type == 1:
            self.opt_values.append(opt_values)


            
    def save(self, filename):
        """
        Stores the journal to disk.

        Parameters
        ----------
        filename: string
            the location of the file to store the current object to.
        """
        
        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)
            


    def posterior_mean(self):
        """
        Computes posterior mean from the samples drawn from posterior distribution

        Returns
        -------
        np.ndarray
            posterior mean        
        """
        endp = len(self.parameters) - 1
        endw = len(self.weights) - 1

        params = self.parameters[endp]
        weights = self.weights[endw]

        return np.average(params, weights = weights.reshape(len(weights),), axis = 0)

    

    def posterior_cov(self):
        """
        Computes posterior covariance from the samples drawn from posterior distribution

        Returns
        -------
        np.ndarray
            posterior covariance        
        """
        endp = len(self.parameters) - 1
        endw = len(self.weights) - 1

        params = self.parameters[endp]
        weights = self.weights[endw]
        
        return np.cov(np.transpose(params), aweights = weights.reshape(len(weights),))


    
    def posterior_histogram(self, n_bins = 10):
        """
        Computes a weighted histogram of multivariate posterior samples
        andreturns histogram H and A list of p arrays describing the bin 
        edges for each dimension.
        
        Returns
        -------
        python list 
            containing two elements (H = np.ndarray, edges = list of p arraya)
        """        
        endp = len(self.parameters) - 1
        endw = len(self.weights) - 1

        params = self.parameters[endp]
        weights = self.weights[endw]
        weights.shape
        H, edges = np.histogramdd(params, bins = n_bins, weights = weights.reshape(len(weights),))
        
        return [H, edges]
