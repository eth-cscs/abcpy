from abcpy.probabilisticmodels import Hyperparameter, ModelResultingFromOperation
import numpy as np


class AcceptedParametersManager:
    def __init__(self, model):
        """
        This class manages the accepted parameters and other bds objects.

        Parameters
        ----------
        model: list
            List of all root probabilistic models
        """
        self.model = model

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.observations_bds = None
        self.accepted_parameters_bds = None
        self.accepted_weights_bds = None
        self.accepted_cov_mats_bds = None

        # saves the current parameters relevant to each kernel
        self.kernel_parameters_bds = None

    def broadcast(self, backend, observations):
        """Broadcasts the observations to observations_bds using the specified backend.

        Parameters
        ----------
        backend: abcpy.backends object
            The backend used by the inference algorithm
        observations: list
            A list containing all observed data
        """
        self.observations_bds = backend.broadcast(observations)

    def update_kernel_values(self, backend, kernel_parameters):
        """Broadcasts new parameters for each kernel

        Parameters
        ----------
        backend: abcpy.backends object
            The backend used by the inference algorithm
        kernel_parameters: list
            A list, in which each entry contains the values of the parameters associated with the corresponding kernel in the joint perturbation kernel
        """

        self.kernel_parameters_bds = backend.broadcast(kernel_parameters)

    def update_broadcast(self, backend, accepted_parameters=None, accepted_weights=None, accepted_cov_mats=None):
        """Updates the broadcasted values using the specified backend

        Parameters
        ----------
        backend: abcpy.backend object
            The backend to be used for broadcasting
        accepted_parameters: list
            The accepted parameters to be broadcasted
        accepted_weights: list
            The accepted weights to be broadcasted
        accepted_cov_mats: np.ndarray
            The accepted covariance matrix to be broadcasted
        """
        # Used for Spark backend
        def destroy(bc):
            if bc != None:
                bc.unpersist
                # bc.destroy

        if not accepted_parameters is None:
            self.accepted_parameters_bds = backend.broadcast(accepted_parameters)
        if not accepted_weights is None:
            self.accepted_weights_bds = backend.broadcast(accepted_weights)
        if not accepted_cov_mats is None:
            self.accepted_cov_mats_bds = backend.broadcast(accepted_cov_mats)

    def get_mapping(self, models, is_root=True, index=0):
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
            The first entry corresponds to the mapping of the root model, as well as all its parents. The second entry
            corresponds to the next index in depth-first search.
        """

        # Implement a dfs to discover all nodes of the model
        mapping = []

        for model in models:
            if(not(model.visited) and not(isinstance(model, Hyperparameter))):
                model.visited = True

                # Only parameters that are neither root nor Hyperparameters are included in the mapping
                if(not(is_root) and not(isinstance(model, ModelResultingFromOperation))):
                    #for i in range(model.get_output_dimension()):
                    mapping.append((model, index))
                    index+=1

                for parent in model.get_input_models():
                    parent_mapping, index = self.get_mapping([parent], is_root= False, index=index)
                    for element in parent_mapping:
                        mapping.append(element)

        # Reset the flags of all models
        if(is_root):
            self._reset_flags()

        return [mapping, index]

    def get_accepted_parameters_bds_values(self, models):
        """
        Returns the accepted bds values for the specified models.

        Parameters
        ----------
        models: list
            Contains the probabilistic models for which the accepted bds values should be returned

        Returns
        -------
        list:
            The accepted_parameters_bds values of all the probabilistic models specified in models.
        """

        # Get the enumerated recursive depth-first search ordering
        mapping, mapping_index = self.get_mapping(self.model)

        # The self.accepted_parameters_bds.value() list has dimensions d x n_steps, where d is the number of free parameters
        accepted_bds_values = [[] for i in range(len(self.accepted_parameters_bds.value()))]

        # Add all columns that correspond to desired parameters to the list that is returned
        for model in models:
            for prob_model, index in mapping:
                if(model==prob_model):
                    for i in range(len(self.accepted_parameters_bds.value())):
                        accepted_bds_values[i].append(self.accepted_parameters_bds.value()[i][index])
        #accepted_bds_values = [np.array(x).reshape(-1, ) for x in accepted_bds_values]

        return accepted_bds_values

    def _reset_flags(self, models=None):
        """Resets the visited flags of all models specified, such that other functions can act on the graph freely.
        Commonly used after calling the get_mapping method.

        Parameters
        ----------
        models: list
            List of abcpy.ProbabilisticModel objects, the models the root models for which, together with their parents, the flags should be reset
        """
        if(models is None):
            models = self.model

        for model in models:
            for parent in model.get_input_models():
                if(parent.visited):
                    self._reset_flags([parent])
            model.visited=False
