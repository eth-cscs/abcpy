import numpy as np
from abcpy.probabilisticmodels import Hyperparameter, ModelResultingFromOperation


class GraphTools():
    """This class implements all methods that will be called recursively on the graph structure."""

    def sample_from_prior(self, model=None, rng=np.random.RandomState()):
        """
        Samples values for all random variables of the model.
        Commonly used to sample new parameter values on the whole graph.

        Parameters
        ----------
        model: abcpy.ProbabilisticModel object
            The root model for which sample_from_prior should be called.
        rng: Random number generator
            Defines the random number generator to be used
        """
        if(model is None):
            model = self.model
        # If it was at some point not possible to sample (due to incompatible parameter values provided by the parents), we start from scratch
        while(not(self._sample_from_prior(model, rng=rng))):
            self._reset_flags(model)

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        self._reset_flags(model)

    def _sample_from_prior(self, models, is_not_root=False, was_accepted=True, rng=np.random.RandomState()):
        """
        Recursive version of sample_from_prior. Commonly called from within sample_from_prior.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel objects
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
                for parent in model.get_input_models():
                    if(not(parent.visited)):
                        parent.visited = True
                        was_accepted = self._sample_from_prior([parent], is_not_root = True, was_accepted=was_accepted, rng=rng)
                        if(not(was_accepted)):
                            return False

                if(is_not_root and not(model._forward_simulate_and_store_output(rng=rng))):
                    return False

                model.visited = True

        return was_accepted

    def _reset_flags(self, models=None):
        """
        Resets all flags that say that a probabilistic model has been updated. Commonly used after actions on the whole
        graph, to ensure that new actions can take place.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel
            The models for which, together with their parents, the flags should be reset. If no value is provided, the
            root models are assumed to be the model of the inference method.
        """
        if not models:
            models = self.model

        # For each model, the flags of the parents get reset recursively.
        for model in models:
            for parent in model.get_input_models():
                self._reset_flags([parent])
            model.visited = False
            model.calculated_pdf = None

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

        Returns
        -------
        list
            The resulting pdf,as well as the next index to be considered in the parameters list.
        """
        self.set_parameters(parameters)
        result = self._recursion_pdf_of_prior(models, parameters, mapping, is_root)
        return result

    def _recursion_pdf_of_prior(self, models, parameters, mapping=None, is_root=True):
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

        Returns
        -------
        list
            The resulting pdf,as well as the next index to be considered in the parameters list.
        """
        # At the beginning of calculation, obtain the mapping
        if(is_root):
            mapping, garbage_index = self._get_mapping()

        # The pdf of each root model is first calculated seperately
        result = [1.]*len(models)

        for i, model in enumerate(models):
            # If the model is not a root model, the pdf of this model, given the prior, should be calculated
            if(not(is_root) and not(isinstance(model, ModelResultingFromOperation))):
                # Define a helper list which will contain the parameters relevant to the current model for pdf calculation
                relevant_parameters = []

                for mapped_model, model_index in mapping:
                    if(mapped_model==model):
                        parameter_index = model_index
                        #for j in range(model.get_output_dimension()):
                        relevant_parameters.append(parameters[parameter_index])
                        #parameter_index+=1
                        break
                if(len(relevant_parameters)==1):
                    relevant_parameters = relevant_parameters[0]
                else:
                    relevant_parameters = np.array(relevant_parameters)
            else:
                relevant_parameters=[]

            # Mark whether the parents of each model have been visited before for this model to avoid repeated calculation.
            visited_parents = [False for j in range(len(model.get_input_models()))]
            # For each parent, the pdf of this parent has to be calculated as well.
            for parent_index, parent in enumerate(model.get_input_models()):
                # Only calculate the pdf if the parent has never been visited for this model
                if(not(visited_parents[parent_index])):
                    pdf = self._recursion_pdf_of_prior([parent], parameters, mapping=mapping, is_root=False)
                    input_models = model.get_input_models()
                    for j in range(len(input_models)):
                        if input_models[j][0] == parent:
                            visited_parents[j]=True
                    result[i] *= pdf
            if(not(is_root)):
                if(model.calculated_pdf is None):
                    result[i] *= model.pdf(model.get_input_values(),relevant_parameters)
                else:
                    result[i] *= 1 

        # Multiply the pdfs of all roots together to give an overall pdf.
        temporary_result = result
        result = 1.
        for individual_result in temporary_result:
            result *= individual_result

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
            if(is_not_root and not(model.visited) and not(isinstance(model, Hyperparameter)) and not(isinstance(model, ModelResultingFromOperation))):
                mapping.append((model, index))
                index+= 1 #model.get_output_dimension()
            # Add all parents to the mapping, if applicable
            for parent in model.get_input_models():
                parent_mapping, index = self._get_mapping([parent], index=index, is_not_root=True)
                parent.visited=True
                for mappings in parent_mapping:
                    mapping.append(mappings)

            model.visited=True

        # At the end of the algorithm, reset all flags such that another method can act on the graph freely.
        if(not(is_not_root)):
            self._reset_flags()

        return [mapping, index]

    def _get_names_and_parameters(self):
        """
        A function returning the name of each model and the corresponding parameters to this model

        Returns
        -------
        list:
            Each entry is a tupel, the first entry of which is the name of the model and the second entry is the parameter values associated with it
        """
        mapping = self._get_mapping()[0]

        return_value = []

        for model, index in mapping:

            return_value.append((model.name, self.accepted_parameters_manager.get_accepted_parameters_bds_values([model])))

        return return_value


    def get_parameters(self, models=None, is_root=True):
        """
        Returns the current values of all free parameters in the model. Commonly used before perturbing the parameters
        of the model.

        Parameters
        ----------
        models: list of abcpy.ProbabilisticModel objects
            The models for which, together with their parents, the parameter values should be returned. If no value is
            provided, the root models are assumed to be the model of the inference method.
        is_root: boolean
            Specifies whether the current models are at the root. This ensures that the values corresponding to
            simulated observations will not be returned.

        Returns
        -------
        list
            A list containing all currently sampled values of the free parameters.
        """
        parameters = []

        # If we are at the root, we set models to the model attribute of the inference method
        if is_root:
            models = self.model

        for model in models:
            # If we are not at the root, the sampled values for the current node should be returned
            if is_root == False and not isinstance(model, (ModelResultingFromOperation, Hyperparameter)):
                parameters.append(model.get_stored_output_values())
                model.visited = True

            # Implement a depth-first search to return also the sampled values associated with each parent of the model
            for parent in model.get_input_models():
                if not parent.visited:
                    parameters += self.get_parameters(models=[parent], is_root=False)
                    parent.visited = True

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        if is_root:
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
        model: list of abcpy.ProbabilisticModel objects
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
        if is_root:
            models = self.model

        for model in models:
            # New parameters should only be set in case we are not at the root
            if not is_root and not isinstance(model, ModelResultingFromOperation):
                #new_output_values = np.array(parameters[index:index + model.get_output_dimension()])
                new_output_values = np.array(parameters[index]).reshape(-1,)
                if not model.set_output_values(new_output_values):
                    return [False, index]
                index += 1 #model.get_output_dimension()
                model.visited = True

            # New parameters for all parents are set using a depth-first search
            for parent in model.get_input_models():
                if not parent.visited and not isinstance(parent, Hyperparameter):
                    is_set, index = self.set_parameters(parameters, models=[parent], index=index, is_root=False)
                    if not is_set:
                        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
                        if is_root:
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
                for parent in model.get_input_models():
                    if(not(parent.visited)):
                        parent_ordering = self.get_correct_ordering(parameters_and_models, models=[parent],is_root=False)
                        for parent_parameters in parent_ordering:
                            ordered_parameters.append(parent_parameters)

        # At the end of the algorithm, are flags are reset such that new methods can act on the graph freely
        if(is_root):
            self._reset_flags()

        return ordered_parameters

    def simulate(self, n_samples_per_param, rng=np.random.RandomState(), npc=None):
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
            parameters_compatible = model._check_input(model.get_input_values())
            if parameters_compatible:
                if npc is not None and npc.communicator().Get_size() > 1:
                    simulation_result = npc.run_nested(model.forward_simulate, model.get_input_values(), n_samples_per_param, rng=rng)
                else:
                    simulation_result = model.forward_simulate(model.get_input_values(),n_samples_per_param, rng=rng)
                result.append(simulation_result)
            else:
                return None
        return result
