import numpy as np
from probabilisticmodels import Hyperparameter

class GraphTools():
    """This class implements all methods that will be called recursively on the graph structure."""
    def sample_from_prior(self, rng=np.random.RandomState()):
        """
       Samples values for all random variables of the model.
        Commonly used to sample new parameter values on the whole graph.

        Parameters
        ----------
        rng: Random number generator
            Defines the random number generator to be used
        """

        # If it was at some point not possible to sample (due to incompatible parameter values provided by the parents), we start from scratch
        while(not(self._sample_from_prior(self.model, rng=rng))):
            self._reset_flags()

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        self._reset_flags()

    def _sample_from_prior(self, models, is_not_root=False, was_accepted=True, rng=np.random.RandomState()):
        """
        Recursive version of sample_from_prior. Commonly called from within sample_from_prior.

        Parameters
        ----------
        models: list of probabilistc models
            Defines the models for which, together with their parents, new parameters will be sampled
        is_root: boolean
            Whether the probabilistic models provided in models are root models.
        was_accepted: boolean
            Whether the sampled values for all previous/parent models were accepted.
        rng: Random number generator
            Defines the random number generator to be used

        Returns
        -------
        boolean:
            Whether it was possible to sample new values for all nodes of the graph.
        """

        # If it was so far possible to sample parameters for all nodes, the current node as well as its parents are sampled, using depth-first search
        if(was_accepted):
            for model in models:

                for parent, index in model.parents:
                    if(not(parent.visited)):
                        parent.visited = True
                        was_accepted = self._sample_from_prior([parent], is_not_root = True, was_accepted=was_accepted, rng=rng)
                        if(not(was_accepted)):
                            return False

                if(is_not_root and not(model.sample_parameters(rng=rng))):
                    return False

                model.visited = True

        return was_accepted

    def _reset_flags(self, models=None):
        """
        Resets all flags that say that a probabilistic model has been updated.
        Commonly used after actions on the whole graph, to ensure that new actions can take place.

        Parameters
        ----------
        models: list of probabilistic models
            The models for which, together with their parents, the flags should be reset. If no value is provided, the root models are assumed to be the model of the inference method.
        """
        if(not(models)):
            models=self.model

        # For each model, the flags of the parents get reset recursively.
        for model in models:
            for parent, parent_index in model.parents:
                if(parent.visited):
                    self._reset_flags([parent])
            model.visited = False

    # todo could we somehow use a set?
    def pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
        """
        Calculates the joint probability density function of the prior of the specified models at the given parameter values.
        Commonly used to check whether new parameters are valid given the prior, as well as to calculate acceptance probabilities.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel objects
            Defines the models for which the pdf of their prior should be evaluated
        parameters: python list
            The parameters at which the pdf should be evaluated
        mapping: list of tupels
            Defines the mapping of probabilistic models and index in a parameter list.
        is_root: boolean
            A flag specifying whether the provided models are the root models. This is to ensure that the pdf is calculated correctly.

        Returns:
        list
            The resulting pdf, given as a list, as well as the next index to be considered in the parameters list.
        """
        # At the beginning of calculation, obtain the mapping
        if(is_root):
            mapping, garbage_index = self._get_mapping()

        # The pdf of each root model is first calculated seperately
        result = [1.]*len(models)

        for i, model in enumerate(models):
            # If the model is not a root model, the pdf of this model, given the prior, should be calculated
            if(not(is_root)):
                # Define a helper list which will contain the parameters relevant to the current model for pdf calculation
                relevant_parameters = []

                for mapped_model, model_index in mapping:
                    if(mapped_model==model):
                        parameter_index = model_index
                        for j in range(model.dimension):
                            relevant_parameters.append(parameters[parameter_index])
                            parameter_index+=1
                        break
                if(len(relevant_parameters)==1):
                    relevant_parameters = relevant_parameters[0]
                else:
                    relevant_parameters = np.array(relevant_parameters)
                result[i]*=model.pdf(relevant_parameters)

            # Mark whether the parents of each model have been visited before for this model to avoid repeated calculation
            visited_parents = [False for j in range(len(model.parents))]

            # For each parent, the pdf of this parent has to be calculated as well.
            for parent_index, parents in enumerate(model.parents):
                parent = parents[0]

                # Only calculate the pdf if the parent has never been visited for this model
                if(not(visited_parents[parent_index])):
                    pdf = self.pdf_of_prior([parent], parameters, mapping=mapping, is_root=False)
                    for j in range(len(model.parents)):
                        if(model.parents[j][0]==parent):
                            visited_parents[j]=True
                    result[i]*=pdf

        temporary_result = result
        result = 1.
        for individual_result in temporary_result:
            result*=individual_result

        return result

    def _get_mapping(self, models=None, index=0, is_not_root=False):
        """Returns a mapping of model and first index corresponding to the outputs in this model in parameter lists.

        Parameters
        ----------
        models: list
            List of abcpy.ProbabilisticModel objects
        index: integer
            Next index to be mapped in a parameter list
        is_not_root: boolean
            Specifies whether the models specified are root models.

        Returns
        -------
        list
            A list containing two entries. The first entry corresponds to the mapping of the root models, including their parents. The second entry corresponds to the next index to be considered in a parameter list.
        """

        if(models is None):
            models = self.model

        mapping = []

        for model in models:
            # If this model corresponds to an unvisited free parameter, add it to the mapping
            if(is_not_root and not(model.visited) and not(isinstance(model, Hyperparameter))):
                mapping.append((model, index))
                index+=model.dimension
            # Add all parents to the mapping, if applicable
            for parent, parent_index in model.parents:
                parent_mapping, index = self._get_mapping([parent], index=index, is_not_root=True)
                parent.visited=True
                for mappings in parent_mapping:
                    mapping.append(mappings)

            model.visited=True

        # At the end of the algorithm, reset all flags such that another method can act on the graph freely.
        if(not(is_not_root)):
            self._reset_flags()

        return [mapping, index]

    def get_parameters(self, models=None, is_root=True):
        """
        Returns the current values of all free parameters in the model.
        Commonly used before perturbing the parameters of the model.

        Parameters
        ----------
        models: list of probabilistic models
            The models for which, together with their parents, the parameter values should be returned. If no value is provided, the root models are assumed to be the model of the inference method.
        is_root: boolean
            Specifies whether the current models are at the root. This ensures that the values corresponding to simulated observations will not be returned.

        Returns
        -------
        list
            A list containing all currently sampled values of the free parameters.
        """
        parameters = []

        # If we are at the root, we sed models to the model attribute of the inference method
        if(is_root):
            models = self.model

        for model in models:
            # If we are not at the root, the sampled values for the current node should be returned
            if(not(is_root)):
                model_parameters = model.get_parameters()
                for parameter in model_parameters:
                    parameters.append(parameter)
                model.visited = True

            # Implement a depth-first search to return also the sampled values associated with each parent of the model
            for parent, parent_index in model.parents:
                if(not(parent.visited)):
                    parent_parameters = self.get_parameters(models=[parent], is_root=False)
                    for parameter in parent_parameters:
                        parameters.append(parameter)
                    parent.visited = True

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        if(is_root):
            self._reset_flags()

        return parameters

    def set_parameters(self, parameters, models=None, index=0, is_root=True):
        """
        Sets new values for the currently used values of each random variable.
        Commonly used after perturbing the parameter values using a kernel.

        Parameters
        ----------
        parameters: list
            Defines the values to which the respective parameter values of the models should be set
        model: list of probabilistic models
             Defines all models for which, together with their parents, new values should be set. If no value is provided, the root models are assumed to be the model of the inference method.
        index: integer
            The current index to be considered in the parameters list
        is_root: boolean
            Defines whether the current models are at the root. This ensures that only values corresponding to random variables will be set.

        Returns
        -------
        list: [boolean, integer]
            Returns whether it was possible to set all parameters and the next index to be considered in the parameters list.
        """
        # If we are at the root, we set models to the model attribute of the inference method
        if(is_root):
            models = self.model

        for model in models:
            # New parameters should only be set in case we are not at the root
            if(not(is_root)):
                if(not(model.set_parameters(parameters[index:index+model.dimension]))):
                    return [False, index]
                index+=model.dimension
                model.visited = True

            # New parameters for all parents are set using a depth-first search
            for parent, parent_index in model.parents:
                if(not(parent.visited)):
                    is_set, index = self.set_parameters(parameters,models=[parent],index=index,is_root=False)
                    if(not(is_set)):
                        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
                        if(is_root):
                            self._reset_flags()
                        return [False, index]
            model.visited = True

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        if(is_root):
            self._reset_flags()

        return [True, index]

    def get_correct_ordering(self, parameters_and_models, models=None, is_root = True):
        """
        Orders the parameters returned by a kernel in the order required by the graph.
        Commonly used when perturbing the parameters.

        Parameters
        ----------
        parameters_and_models: list of tuples
            Contains tuples containing as the first entry the probabilistic model to be considered and as the second entry the parameter values associated with this model
        models: list
            Contains the root probabilistic models that make up the graph. If no value is provided, the root models are assumed to be the model of the inference method.

        Returns
        -------
        list
            The ordering which can be used by recursive functions on the graph.
        """
        ordered_parameters = []

        # If we are at the root, we set models to the model attribute of the inference method
        if(is_root):
            models=self.model

        for model in models:
            if(not(model.visited)):
                model.visited = True

                # Check all entries in parameters_and_models to determine whether the current model is contained within it
                for corresponding_model, parameter in parameters_and_models:
                    if(corresponding_model==model):
                        for param in parameter:
                            ordered_parameters.append(param)
                        break

                # Recursively order all the parents of the current model
                for parent, parents_index in model.parents:
                    if(not(parent.visited)):
                        parent_ordering = self.get_correct_ordering(parameters_and_models, models=[parent],is_root=False)
                        for parent_parameters in parent_ordering:
                            ordered_parameters.append(parent_parameters)

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        if(is_root):
            self._reset_flags()

        return ordered_parameters

    def simulate(self, rng=np.random.RandomState()):
        """Simulates data of each model using the currently sampled or perturbed parameters.

        Parameters
        ----------
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list
            Each entry corresponds to the simulated data of one model.
        """
        result = []
        for model in self.model:
            simulation_result = model.sample_from_distribution(self.n_samples_per_param, rng=rng)
            if(simulation_result[0]):
                result.append(simulation_result[1].tolist())
            else:
                return None
        return result
