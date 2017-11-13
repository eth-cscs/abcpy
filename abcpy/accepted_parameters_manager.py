from ProbabilisticModel import Hyperparameter

#in kernel it should have self.accepted_parameters_bds.value() without [index[0]] but instead it should look which model corresponds to which entry and pack these togehter (as rows, it is nxd)

class AcceptedParametersManager():
    """
    This class managed the accepted parameters and other bds objects

    Parameters
    ----------
    model: list
        List of all root probabilistic models
    """
    def __init__(self, model):
        self.model = model

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_weights_bds = None
        self.accepted_cov_mat_bds = None

    def broadcast(self, backend, observations):
        """Broadcasts the observations to observations_bds using the specified backend."""
        self.observations_bds = backend.broadcast(observations)

    def update_broadcast(self, backend, accepted_parameters=None, accepted_weights=None, accepted_cov_mat=None):
        """Updates the broadcasted values using the specified backend

        Parameters
        ----------
        backend: abcpy.backend object
            The backend to be used for broadcasting
        accepted_parameters: list
            The accepted parameters to be broadcasted
        accepted_weights: list
            The accepted weights to be broadcasted
        accepted_cov_mat: np.ndarray
            The accepted covariance matrix to be broadcasted
        """
        #NOTE what does this do????
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = backend.broadcast(accepted_weights)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = backend.broadcast(accepted_cov_mat)

    def _get_mapping(self, models, is_root=True, index=0):
        """Returns the order in which the models are discovered during recursive depth-first search.
        Commonly used when returning the accepted_parameters_bds for certain models.
        Parameters
        ----------
        models: list
            List of the root probabilistic models of the graph.
        is_root: boolean
            Specifies whether the current list of models is the list of overall root models
        index: integer
            The current index in depth-first search.

        Returns
        -------
        list
            The first entry corresponds to the mapping of the root model, as well as all its parents. The second entry corresponds to the next index in depth-first search.
        """

        # Implement a dfs to discover all nodes of the model
        mapping = []

        for model in models:
            if(not(model.visited) and not(isinstance(model, Hyperparameter))):
                model.visited = True

                # Only parameters that are neither root nor Hyperparameters are included in the mapping
                if(not(is_root)):
                    for i in range(model.dimension):
                        mapping.append((model, index))
                        index+=1

                for parent, parent_index in model.parents:
                    parent_mapping, index = self._get_mapping([parent], is_root= False, index=index)
                    for element in parent_mapping:
                        mapping.append(element)

        return [mapping, index]

    # NOTE after this, _reset_flags should be called
    def get_accepted_parameters_bds_values(self, models, index=0):
        """
        Returns the accepted bds values for the specified models.

        Parameters
        ----------.
        models: list
            Contains the probabilistic models for which the accepted bds values should be returned
        index: integer
            The current index to be considered in the accepted_parameters_bds list

        Returns
        -------
        list:
            The accepted_parameters_bds values of all the probabilistic models specified in models.

        """
        mapping, mapping_index = self._get_mapping(self.model)
        accepted_bds_values = []
        for model in models:
            for prob_model, index in mapping:
                if(model==prob_model):
                    #TODO not sure if correct, see test_pmcabc
                    accepted_bds_values.append(self.accepted_parameters_bds.value()[index,:])
        return accepted_bds_values
