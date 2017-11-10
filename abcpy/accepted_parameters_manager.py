#in kernel it should have self.accepted_parameters_bds.value() without [index[0]] but instead it should look which model corresponds to which entry and pack these togehter (as rows, it is nxd)

class AcceptedParameterManager():
    """
    This class managed the accepted parameters and other bds objects
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

    def update_broadcast(self, backend, accepted_parameters, accepted_weights, accepted_cov_mat):
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
            self.accepted_parameters_bds = self.backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = self.backend.broadcast(accepted_weights)
        if not accepted_cov_mat is None:
            self.accepted_cov_mat_bds = self.backend.broadcast(accepted_cov_mat)

    # todo use self.model instead of graph_models here
    # NOTE after this, reset_flags shoudl be called
    def get_accepted_bds_values(self, graph_models, models, index=0):
        """
        Returns the accepted bds values for the specified models.

        Parameters
        ----------
        graph_models: list
            Contains all the root probabilistic models of the graph.
        models: list
            Contains the probabilistic models for which the accepted bds values should be returned
        index: integer
            The current index to be considered in the accepted_parameters_bds list

        Returns
        -------
        list:
            The first entry corresponds to the accepted_bds_values of the current root models being considered as well as their parent models. The second entry corresponds to the next index to be considerd in the accepted_parameters_bds list.

        """
        accepted_bds_values = []
        for graph_model in graph_models:
            if(not(model.visited)):
                model.visited = True
                for model in models:
                    if(graph_model==model):
                        accepted_bds_values.append(self.accepted_parameters_bds.value()[index])
                        break
                index+=1
            for parent, parent_index in model.parents:
                if(not(parent.visited)):
                    parent_accepted_values, index = self.get_accepted_bds_values([parent], models, index)
                    for parent_value in parent_accepted_values:
                        accepted_bds_values.append(parent_value)
        return [accepted_bds_values, index]




