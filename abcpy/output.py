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
        
        self.accepted_parameters = []
        self.names_and_parameters = []
        self.weights = []
        self.distances = []
        self.opt_values = []
        self.configuration = {}



        if type not in [0, 1]:
            raise ValueError("Parameter type has not valid value.")
        else:
            self._type = type

        self.number_of_simulations =[]

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

    def add_user_parameters(self, names_and_params):
        """
        Saves the provided parameters and names of the probabilistic models corresponding to them. If type==0, old parameters get overwritten.

        Parameters
        ----------
        names_and_params: list
            Each entry is a tupel, where the first entry is the name of the probabilistic model, and the second entry is the parameters associated with this model.
        """
        if(self._type == 0):
            self.names_and_parameters = [dict(names_and_params)]
        else:
            self.names_and_parameters.append(dict(names_and_params))

    def add_accepted_parameters(self, accepted_parameters):
        """
        Saves provided weights by appending them to the journal. If type==0, old weights get overwritten.

        Parameters
        ----------
        accepted_parameters: list
        """

        if self._type == 0:
            self.accepted_parameters = [accepted_parameters]

        if self._type == 1:
            self.accepted_parameters.append(accepted_parameters)

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

    def add_distances(self, distances):
        """
        Saves provided distances by appending them to the journal. If type==0, old weights get overwritten.

        Parameters
        ----------
        distances: numpy.array
            vector containing n distances
        """

        if self._type == 0:
            self.distances = [distances]

        if self._type == 1:
            self.distances.append(distances)

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

    def get_parameters(self, iteration=None):
        """
        Returns the parameters from a sampling scheme.

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return parameters

        Returns
        -------
        names_and_parameters: dictionary
            Samples from the specified iteration (last, if not specified) returned as a disctionary with names of the
            random variables
        """

        if iteration is None:
            endp = len(self.names_and_parameters) - 1
            params = self.names_and_parameters[endp]
            return params
        else:
            return self.names_and_parameters[iteration]

    def get_accepted_parameters(self, iteration=None):
        """
        Returns the accepted parameters from a sampling scheme.

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return parameters

        Returns
        -------
        accepted_parameters: dictionary
            Samples from the specified iteration (last, if not specified) returned as a disctionary with names of the
            random variables
        """

        if iteration is None:
            return self.accepted_parameters[-1]

        else:
            return self.accepted_parameters[iteration]

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

    def get_distances(self, iteration=None):
        """
        Returns the distances from a sampling scheme.

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return distances
        """

        if iteration is None:
            end = len(self.distances) - 1
            return self.distances[end]
        else:
            return self.distances[iteration]

    def posterior_mean(self, iteration=None):
        """
        Computes posterior mean from the samples drawn from posterior distribution

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return posterior mean

        Returns
        -------
        posterior mean: dictionary
            Posterior mean from the specified iteration (last, if not specified) returned as a disctionary with names of the
            random variables
        """

        if iteration is None:
            endp = len(self.names_and_parameters) - 1
            params = self.names_and_parameters[endp]
            weights = self.weights[endp]
        else:
            params = self.names_and_parameters[iteration]
            weights = self.weights[iteration]

        return_value = []
        for keyind in params.keys():
            return_value.append((keyind, np.average(np.array(params[keyind]).squeeze(), weights = weights.reshape(len(weights),), axis = 0)))

        return dict(return_value)

    def posterior_cov(self, iteration=None):
        """
        Computes posterior covariance from the samples drawn from posterior distribution

        Returns
        -------
        np.ndarray
            posterior covariance
        dic
            order of the variables in the covariance matrix
        """

        if iteration is None:
            endp = len(self.names_and_parameters) - 1
            params = self.names_and_parameters[endp]
            weights = self.weights[endp]
        else:
            params = self.names_and_parameters[iteration]
            weights = self.weights[iteration]

        joined_params = []
        for keyind in params.keys():
            joined_params.append(np.array(params[keyind]).squeeze(axis=1))

        return np.cov(np.transpose(np.hstack(joined_params)), aweights = weights.reshape(len(weights),)), params.keys()

    def posterior_histogram(self, iteration=None, n_bins = 10):
        """
        Computes a weighted histogram of multivariate posterior samples
        andreturns histogram H and A list of p arrays describing the bin 
        edges for each dimension.
        
        Returns
        -------
        python list 
            containing two elements (H = np.ndarray, edges = list of p arraya)
        """
        if iteration is None:
            endp = len(self.names_and_parameters) - 1
            params = self.names_and_parameters[endp]
            weights = self.weights[endp]
        else:
            params = self.names_and_parameters[iteration]
            weights = self.weights[iteration]

        joined_params = []
        for keyind in params.keys():
            joined_params.append(np.array(params[keyind]).squeeze(axis=1))

        H, edges = np.histogramdd(np.hstack(joined_params), bins = n_bins, weights = weights.reshape(len(weights),))
        return [H, edges]