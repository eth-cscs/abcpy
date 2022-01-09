import copy
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from abcpy.acceptedparametersmanager import AcceptedParametersManager
from abcpy.graphtools import GraphTools
from abcpy.utils import wass_dist


class Journal:
    """The journal holds information created by the run of inference schemes.

    It can be configured to even hold intermediate.

    Attributes
    ----------
    accepted_parameters : list
        List of lists containing posterior samples
    names_and_parameters : list
        List of dictionaries containing posterior samples with parameter names as keys
    accepted_simulations : list
        List of lists containing simulations corresponding to posterior samples (this could be empty if the sampling
        routine does not store those)
    accepted_cov_mats : list
        List of lists containing covariance matrices from accepted posterior samples (this could be empty if
        the sampling routine does not store those)
    weights : list
        List containing posterior weights
    ESS : list
        List containing the Effective Sample Size (ESS) at each iteration
    distances : list
        List containing the ABC distance at each iteration
    configuration : Python dictionary
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
        self.accepted_simulations = []
        self.accepted_cov_mats = []
        self.weights = []
        self.ESS = []
        self.distances = []
        self.configuration = {}

        if type not in [0, 1]:
            raise ValueError("Parameter type has not valid value.")
        else:
            self._type = type

        self.number_of_simulations = []

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
        Saves the provided parameters and names of the probabilistic models corresponding to them. If type==0, old
        parameters get overwritten.

        Parameters
        ----------
        names_and_params: list
            Each entry is a tuple, where the first entry is the name of the probabilistic model, and the second entry is
            the parameters associated with this model.
        """
        if self._type == 0:
            self.names_and_parameters = [dict(names_and_params)]
        else:
            self.names_and_parameters.append(dict(names_and_params))

    def add_accepted_parameters(self, accepted_parameters):
        """
        FIX THIS!
        Saves provided accepted parameters by appending them to the journal. If type==0, old accepted parameters get
        overwritten.

        Parameters
        ----------
        accepted_parameters: list
        """

        if self._type == 0:
            self.accepted_parameters = [accepted_parameters]

        if self._type == 1:
            self.accepted_parameters.append(accepted_parameters)

    def add_accepted_simulations(self, accepted_simulations):
        """
        Saves provided accepted simulations by appending them to the journal. If type==0, old accepted simulations get
        overwritten.

        Parameters
        ----------
        accepted_simulations: list
        """

        if self._type == 0:
            self.accepted_simulations = [accepted_simulations]

        if self._type == 1:
            self.accepted_simulations.append(accepted_simulations)

    def add_accepted_cov_mats(self, accepted_cov_mats):
        """
        Saves provided accepted cov_mats by appending them to the journal. If type==0, old accepted cov_mats get
        overwritten.

        Parameters
        ----------
        accepted_cov_mats: list
        """

        if self._type == 0:
            self.accepted_cov_mats = [accepted_cov_mats]

        if self._type == 1:
            self.accepted_cov_mats.append(accepted_cov_mats)

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

    def add_ESS_estimate(self, weights):
        """
        Computes and saves Effective Sample Size (ESS) estimate starting from provided weights; ESS is estimated as sum
        the inverse of sum of squared normalized weights. The provided weights are normalized before computing ESS.
        If type==0, old ESS estimate gets overwritten.

        Parameters
        ----------
        weights: numpy.array
            vector containing n weigths
        """

        # normalize weights:
        normalized_weights = weights / np.sum(weights)

        ESS = 1 / sum(sum(pow(normalized_weights, 2)))

        if self._type == 0:
            self.ESS = [ESS]

        if self._type == 1:
            self.ESS.append(ESS)

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
        accepted_parameters: list
            List containing samples from the specified iteration (last, if not specified)
        """

        if iteration is None:
            return self.accepted_parameters[-1]

        else:
            return self.accepted_parameters[iteration]

    def get_accepted_simulations(self, iteration=None):
        """
        Returns the accepted simulations from a sampling scheme. Notice not all sampling schemes store those in the
        Journal, so this may return None.

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return accepted simulations

        Returns
        -------
        accepted_simulations: list
            List containing simulations corresponding to accepted samples from the specified
            iteration (last, if not specified)
        """

        if iteration is None:
            if len(self.accepted_simulations) == 0:
                return None
            return self.accepted_simulations[-1]

        else:
            return self.accepted_simulations[iteration]

    def get_accepted_cov_mats(self, iteration=None):
        """
        Returns the accepted cov_mats used in a sampling scheme. Notice not all sampling schemes store those in the
        Journal, so this may return None.

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return accepted cov_mats

        Returns
        -------
        accepted_cov_mats: list
            List containing accepted cov_mats from the specified
            iteration (last, if not specified)
        """

        if iteration is None:
            if len(self.accepted_cov_mats) == 0:
                return None
            return self.accepted_cov_mats[-1]

        else:
            return self.accepted_cov_mats[iteration]

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

    def get_ESS_estimates(self, iteration=None):
        """
        Returns the estimate of Effective Sample Size (ESS) from a sampling scheme.

        For intermediate results, pass the iteration.

        Parameters
        ----------
        iteration: int
            specify the iteration for which to return ESS
        """
        if iteration is None:
            iteration = -1

        return self.ESS[iteration]

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
            Posterior mean from the specified iteration (last, if not specified) returned as a disctionary with names of
            the random variables
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
            return_value.append((keyind,
                                 np.average(np.array(params[keyind]).squeeze(), weights=weights.reshape(len(weights), ),
                                            axis=0)))

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

        return np.cov(np.transpose(np.hstack(joined_params)), aweights=weights.reshape(len(weights), )), params.keys()

    def posterior_histogram(self, iteration=None, n_bins=10):
        """
        Computes a weighted histogram of multivariate posterior samples
        and returns histogram H and a list of p arrays describing the bin
        edges for each dimension.
        
        Returns
        -------
        python list 
            containing two elements (H = np.ndarray, edges = list of p arrays)
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

        H, edges = np.histogramdd(np.hstack(joined_params), bins=n_bins, weights=weights.reshape(len(weights), ))
        return [H, edges]

    # TODO this does not work for multidimensional parameters and discrete distributions
    def plot_posterior_distr(self, parameters_to_show=None, ranges_parameters=None, iteration=None, show_samples=None,
                             single_marginals_only=False, double_marginals_only=False, write_posterior_mean=True,
                             show_posterior_mean=True, true_parameter_values=None, contour_levels=14, figsize=None,
                             show_density_values=True, bw_method=None, path_to_save=None):
        """
        Produces a visualization of the posterior distribution of the parameters of the model.

        A Gaussian kernel density estimate (KDE) is used to approximate the density starting from the sampled
        parameters. Specifically, it produces a scatterplot matrix, where the diagonal contains single parameter
        marginals, while the off diagonal elements contain the contourplot for the paired marginals for each possible
        pair of parameters.

        This visualization is not satisfactory for parameters that take on discrete values, specially in the case where
        the number of values it can assume are small, as it obtains the posterior by KDE in this case as well. We need
        to improve on that, considering histograms.

        Parameters
        ----------
        parameters_to_show : list, optional
            a list of the parameters for which you want to plot the posterior distribution. For each parameter, you need
            to provide the name string as it was defined in the model. For instance,
            `jrnl.plot_posterior_distr(parameters_to_show=["mu"])` will only plot the posterior distribution for the
            parameter named "mu" in the list of parameters.
            If `None`, then all parameters will be displayed.
        ranges_parameters : Python dictionary, optional
            a dictionary in which you can optionally provide the plotting range for the parameters that you chose to
            display. You can use this even if `parameters_to_show=None`. The dictionary key is the name of parameter,
            and the range needs to be an array-like of the form [lower_limit, upper_limit]. For instance:
            {"theta" : [0,2]} specifies that you want to plot the posterior distribution for the parameter "theta" in
            the range [0,2].
        iteration : int, optional
            specify the iteration for which to plot the posterior distribution, in the case of a sequential algorithm.
            If `None`, then the last iteration will be used.
        show_samples : boolean, optional
            specifies if you want to show the posterior samples overimposed to the contourplots of the posterior
            distribution. If `None`, the default behaviour is the following: if the posterior samples are associated
            with importance weights, then the samples are not showed (in fact, the KDE for the posterior distribution
            takes into account the weights, and showing the samples may be misleading). Otherwise, if the posterior
            samples are not associated with weights, they are displayed by default.
        single_marginals_only : boolean, optional
            if `True`, the method does not show the paired marginals but only the single parameter marginals;
            otherwise, it shows the paired marginals as well. Default to `False`.
        double_marginals_only : boolean, optional
            if `True`, the method shows the contour plot for the marginal posterior for each possible pair of parameters
            in the parameters that have to be shown (all parameters of the model if `parameters_to_show` is None).
            Default to `False`.
        write_posterior_mean : boolean, optional
            Whether to write or not the posterior mean on the single marginal plots. Default to `True`.
        show_posterior_mean: boolean, optional
            Whether to display a line corresponding to the posterior mean value in the plot. Default to `True`.
        true_parameter_values: array-like, optional
            you can provide here the true values of the parameters, if known, and that will be displayed in the
            posterior plot. It has to be an array-like of the same length of `parameters_to_show` (if that is provided),
            otherwise of length equal to the number of parameters in the model, and with entries corresponding to the
            true value of that parameter (in case `parameters_to_show` is not provided, the order of the parameters is
            the same order the model `forward_simulate` step takes.
        contour_levels: integer, optional
            The number of levels to be used in the contour plots. Default to 14.
        figsize: float, optional
            Denotes the size (in inches) of the smaller dimension of the plot; the other dimension is automatically
            determined. If None, then figsize is chosen automatically. Default to `None`.
        show_density_values: boolean, optional
            If `True`, the method displays the value of the density at each contour level in the contour plot. Default
            to `True`.
        bw_method: str, scalar or callable, optional
            The parameter of the `scipy.stats.gaussian_kde` defining the method used to calculate the bandwith in the
            Gaussian kernel density estimator. Please refer to the documentation therein for details. Default to `None`.
        path_to_save : string, optional
            if provided, save the figure in png format in the specified path.

        Returns
        -------
        tuple
            a tuple containing the matplotlib "fig, axes" objects defining the plot. Can be useful for further
            modifications.
        """

        # if you do not pass any name, then we plot all parameters.
        # you can get the list of parameters from the journal file as:
        if parameters_to_show is None:
            parameters_to_show = list(self.names_and_parameters[0].keys())
        else:
            for param_name in parameters_to_show:
                if param_name not in self.names_and_parameters[0].keys():
                    raise KeyError("Parameter '{}' is not a parameter of the model.".format(param_name))

        if single_marginals_only and double_marginals_only:
            raise RuntimeError("You cannot ask to produce only plots for single marginals and double marginals only at "
                               "the same time. Either or both of `single_marginal_only` or `double_marginal_only` have "
                               "to be False.")

        if len(parameters_to_show) == 1 and double_marginals_only:
            raise RuntimeError("It is not possible to plot a bivariate marginal if only one parameter is in the "
                               "`parameters_to_show` list")

        if true_parameter_values is not None:
            if len(true_parameter_values) != len(parameters_to_show):
                raise RuntimeError("You need to provide values for all the parameters to be shown.")

        meanpost = np.array([self.posterior_mean(iteration=iteration)[x] for x in parameters_to_show])

        post_samples_dict = {name: np.concatenate(self.get_parameters(iteration)[name]) for name in parameters_to_show}

        weights = np.concatenate(self.get_weights(iteration))
        all_weights_are_1 = all([weights[i] == 1 for i in range(len(weights))])

        if show_samples is None:
            # by default, we display samples if the weights are all 1. Otherwise, we choose not to display them by
            # default as they may be misleading
            if all_weights_are_1:
                show_samples = True
            else:
                show_samples = False

        elif not all_weights_are_1 and show_samples is True:
            # in this case, the user required to show the sample points but importance weights are present. Then warn
            # the user about this
            warnings.warn(
                "You requested to show the sample points; however, note that the algorithm generated posterior samples "
                "associated with weights, and the kernel density estimate used to produce the density plots consider "
                "those. Therefore, the resulting plot may be misleading. ", RuntimeWarning)

        data = np.hstack(list(post_samples_dict.values()))
        datat = data.transpose()

        # now set up the range of parameters. If the range for a given parameter is not given, use the min and max
        # value of posterior samples.

        if ranges_parameters is None:
            ranges_parameters = {}

        # check if the user provided range for some parameters that are not requested to be displayed:
        if not all([key in post_samples_dict for key in ranges_parameters.keys()]):
            warnings.warn("You provided range for a parameter that was not requested to be displayed (or it may be "
                          "that you misspelled something). This will be ignored.", RuntimeWarning)

        for name in parameters_to_show:
            if name in ranges_parameters:
                # then check the format is correct
                if isinstance(ranges_parameters[name], list):
                    if not (len(ranges_parameters[name]) == 2 and isinstance(ranges_parameters[name][0], (int, float))):
                        raise TypeError(
                            "The range for the parameter {} should be an array-like with two numbers.".format(name))
                elif isinstance(ranges_parameters[name], np.ndarray):
                    if not (ranges_parameters[name].shape == 2 or ranges_parameters[name].shape == (2, 1)):
                        raise TypeError(
                            "The range for the parameter {} should be an array-like with two numbers.".format(name))
            else:
                # add as new range the min and max values. We add also a 5% of the whole range on both sides to have
                # better visualization
                difference = np.max(post_samples_dict[name]) - np.min(post_samples_dict[name])
                ranges_parameters[name] = [np.min(post_samples_dict[name]) - 0.05 * difference,
                                           np.max(post_samples_dict[name]) + 0.05 * difference]

        def write_post_mean_function(ax, post_mean, size):
            ax.text(0.15, 0.06, r"Post. mean = {:.2f}".format(post_mean), size=size,
                    transform=ax.transAxes)

        def scatterplot_matrix(data, meanpost, names, single_marginals_only=False, **kwargs):
            """
            Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
            against other rows, resulting in a nrows by nrows grid of subplots with the
            diagonal subplots labeled with "parameters_to_show".  Additional keyword arguments are
            passed on to matplotlib's "plot" command. Returns the matplotlib figure
            object containg the subplot grid.
            """
            if figsize is None:
                figsize_actual = 4 * len(names)
            else:
                figsize_actual = figsize
            numvars, numdata = data.shape
            if single_marginals_only:
                fig, axes = plt.subplots(nrows=1, ncols=numvars, figsize=(figsize_actual, figsize_actual / len(names)))
            else:
                fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(figsize_actual, figsize_actual))
                fig.subplots_adjust(hspace=0.08, wspace=0.08)

            # if we have to plot 1 single parameter value, envelop the ax in an array, so that it gives no troubles:
            if len(names) == 1:
                if not single_marginals_only:
                    axes = np.array([[axes]])
                else:
                    axes = np.array([axes])

            if not single_marginals_only:
                if len(names) > 1:
                    for ax in axes.flat:
                        # Hide all ticks and labels
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)

                        # Set up ticks only on one side for the "edge" subplots...
                        if ax.is_first_col():
                            ax.yaxis.set_ticks_position('left')
                        if ax.is_last_col():
                            ax.yaxis.set_ticks_position('right')
                        if ax.is_first_row():
                            ax.xaxis.set_ticks_position('top')
                        if ax.is_last_row():
                            ax.xaxis.set_ticks_position('bottom')

                # off diagonal plots:
                for x in range(numvars):
                    for y in range(numvars):
                        if x is not y:
                            # this plots the posterior samples
                            if show_samples:
                                axes[x, y].plot(data[y], data[x], '.k', markersize='1')

                            xmin, xmax = ranges_parameters[names[y]]
                            ymin, ymax = ranges_parameters[names[x]]

                            # then you need to perform the KDE and plot the contour of posterior
                            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                            positions = np.vstack([X.ravel(), Y.ravel()])
                            values = np.vstack([data[y].T, data[x].T])
                            kernel = gaussian_kde(values, weights=weights, bw_method=bw_method)
                            Z = np.reshape(kernel(positions).T, X.shape)
                            # axes[x, y].plot(meanpost[y], meanpost[x], 'r+', markersize='10')
                            if show_posterior_mean:
                                axes[x, y].plot([xmin, xmax], [meanpost[x], meanpost[x]], "red", markersize='20',
                                                linestyle='solid')
                                axes[x, y].plot([meanpost[y], meanpost[y]], [ymin, ymax], "red", markersize='20',
                                                linestyle='solid')
                            if true_parameter_values is not None:
                                axes[x, y].plot([xmin, xmax], [true_parameter_values[x], true_parameter_values[x]],
                                                "green",
                                                markersize='20', linestyle='dashed')
                                axes[x, y].plot([true_parameter_values[y], true_parameter_values[y]], [ymin, ymax],
                                                "green",
                                                markersize='20', linestyle='dashed')

                            CS = axes[x, y].contour(X, Y, Z, contour_levels, linestyles='solid')
                            if show_density_values:
                                axes[x, y].clabel(CS, fontsize=figsize_actual / len(names) * 2.25, inline=1)
                            axes[x, y].set_xlim([xmin, xmax])
                            axes[x, y].set_ylim([ymin, ymax])

            # diagonal plots

            if not single_marginals_only:
                diagonal_axes = np.array([axes[i, i] for i in range(len(axes))])
            else:
                diagonal_axes = axes
            label_size = figsize_actual / len(names) * 4
            title_size = figsize_actual / len(names) * 4.25
            post_mean_size = figsize_actual / len(names) * 4
            ticks_size = figsize_actual / len(names) * 3

            for i, label in enumerate(names):
                xmin, xmax = ranges_parameters[names[i]]
                positions = np.linspace(xmin, xmax, 100)
                gaussian_kernel = gaussian_kde(data[i], weights=weights, bw_method=bw_method)
                diagonal_axes[i].plot(positions, gaussian_kernel(positions), color='k', linestyle='solid', lw="1",
                                      alpha=1, label="Density")
                values = gaussian_kernel(positions)
                # axes[i, i].plot([positions[np.argmax(values)], positions[np.argmax(values)]], [0, np.max(values)])
                if show_posterior_mean:
                    diagonal_axes[i].plot([meanpost[i], meanpost[i]], [0, 1.1 * np.max(values)], "red", alpha=1,
                                          label="Posterior mean")
                if true_parameter_values is not None:
                    diagonal_axes[i].plot([true_parameter_values[i], true_parameter_values[i]],
                                          [0, 1.1 * np.max(values)], "green",
                                          alpha=1, label="True value")
                if write_posterior_mean:
                    write_post_mean_function(diagonal_axes[i], meanpost[i], size=post_mean_size)
                diagonal_axes[i].set_xlim([xmin, xmax])
                diagonal_axes[i].set_ylim([0, 1.1 * np.max(values)])

            # labels and ticks:
            if not single_marginals_only:
                for j, label in enumerate(names):
                    axes[0, j].set_title(label, size=title_size)

                    if len(names) > 1:
                        axes[j, 0].set_ylabel(label, size=label_size)
                        axes[-1, j].set_xlabel(label, size=label_size)
                    else:
                        axes[j, 0].set_ylabel("Density", size=label_size)

                    axes[j, 0].tick_params(axis='both', which='major', labelsize=ticks_size)
                    axes[j, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    axes[j, 0].yaxis.offsetText.set_fontsize(ticks_size)
                    axes[j, 0].yaxis.set_visible(True)

                    axes[-1, j].tick_params(axis='both', which='major', labelsize=ticks_size)
                    axes[-1, j].ticklabel_format(style='sci', axis='x')  # , scilimits=(0, 0))
                    axes[-1, j].xaxis.offsetText.set_fontsize(ticks_size)
                    axes[-1, j].xaxis.set_visible(True)
                    axes[j, -1].tick_params(axis='both', which='major', labelsize=ticks_size)
                    axes[j, -1].yaxis.offsetText.set_fontsize(ticks_size)
                    axes[j, -1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    axes[j, -1].yaxis.set_visible(True)

            else:
                for j, label in enumerate(names):
                    axes[j].set_title(label, size=figsize_actual / len(names) * 4.25)
                axes[0].set_ylabel("Density")

            return fig, axes

        def double_marginals_plot(data, meanpost, names, **kwargs):
            """
            Plots contour plots of all possible pairs of samples. Additional keyword arguments are
            passed on to matplotlib's "plot" command. Returns the matplotlib figure
            object containg the subplot grid.
            """
            numvars, numdata = data.shape
            numplots = int(numvars * (numvars - 1) / 2)
            if figsize is None:
                figsize_actual = 4 * numplots
            else:
                figsize_actual = figsize

            fig, axes = plt.subplots(nrows=1, ncols=numplots, figsize=(figsize_actual, figsize_actual / numplots))

            if numplots == 1:  # in this case you will only have one plot. Envelop it to avoid problems.
                axes = [axes]

            # if we have to plot 1 single parameter value, envelop the ax in an array, so that it gives no troubles:

            ax_counter = 0

            for x in range(numvars):
                for y in range(0, x):
                    # this plots the posterior samples
                    if show_samples:
                        axes[ax_counter].plot(data[y], data[x], '.k', markersize='1')

                    xmin, xmax = ranges_parameters[names[y]]
                    ymin, ymax = ranges_parameters[names[x]]

                    # then you need to perform the KDE and plot the contour of posterior
                    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    values = np.vstack([data[y].T, data[x].T])
                    kernel = gaussian_kde(values, weights=weights, bw_method=bw_method)
                    Z = np.reshape(kernel(positions).T, X.shape)
                    # axes[x, y].plot(meanpost[y], meanpost[x], 'r+', markersize='10')
                    if show_posterior_mean:
                        axes[ax_counter].plot([xmin, xmax], [meanpost[x], meanpost[x]], "red", markersize='20',
                                              linestyle='solid')
                        axes[ax_counter].plot([meanpost[y], meanpost[y]], [ymin, ymax], "red", markersize='20',
                                              linestyle='solid')
                    if true_parameter_values is not None:
                        axes[ax_counter].plot([xmin, xmax], [true_parameter_values[x], true_parameter_values[x]],
                                              "green",
                                              markersize='20', linestyle='dashed')
                        axes[ax_counter].plot([true_parameter_values[y], true_parameter_values[y]], [ymin, ymax],
                                              "green",
                                              markersize='20', linestyle='dashed')

                    CS = axes[ax_counter].contour(X, Y, Z, contour_levels, linestyles='solid')
                    if show_density_values:
                        axes[ax_counter].clabel(CS, fontsize=figsize_actual / numplots * 2.25, inline=1)
                    axes[ax_counter].set_xlim([xmin, xmax])
                    axes[ax_counter].set_ylim([ymin, ymax])

                    label_size = figsize_actual / numplots * 4
                    ticks_size = figsize_actual / numplots * 3
                    axes[ax_counter].set_ylabel(names[x], size=label_size)
                    axes[ax_counter].set_xlabel(names[y], size=label_size)

                    axes[ax_counter].tick_params(axis='both', which='major', labelsize=ticks_size)
                    axes[ax_counter].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    axes[ax_counter].yaxis.offsetText.set_fontsize(ticks_size)
                    axes[ax_counter].yaxis.set_visible(True)
                    axes[ax_counter].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    axes[ax_counter].yaxis.offsetText.set_fontsize(ticks_size)
                    axes[ax_counter].xaxis.set_visible(True)

                    ax_counter += 1

            return fig, axes

        if not double_marginals_only:
            fig, axes = scatterplot_matrix(datat, meanpost, parameters_to_show,
                                           single_marginals_only=single_marginals_only)
        else:
            fig, axes = double_marginals_plot(datat, meanpost, parameters_to_show)

        if path_to_save is not None:
            plt.savefig(path_to_save, bbox_inches="tight")

        return fig, axes

    def plot_ESS(self):
        """
        Produces a plot showing the evolution of the estimated ESS (from sample weights) across iterations; it also
        shows as a baseline the maximum possible ESS which can be achieved, corresponding to the case of independent
        samples, which is equal to the total number of samples.

        Returns
        -------
        tuple
            a tuple containing the matplotlib "fig, ax" objects defining the plot. Can be useful for further
            modifications.
        """

        if self._type == 0:
            raise RuntimeError("ESS plot is available only if the journal was created with full_output=1; otherwise, "
                               "ESS is saved only for the last iteration.")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

        ax.scatter(np.arange(len(self.ESS)) + 1, self.ESS, label="Estimated ESS")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ESS")
        # put horizontal line at the largest value ESS can get:
        ax.axhline(len(self.weights[-1]), label="Max theoretical value", ls="dashed")
        ax.legend()
        ax.set_xticks(np.arange(len(self.ESS)) + 1)

        return fig, ax

    def Wass_convergence_plot(self, num_iter_max=1e8, **kwargs):
        """
        Computes the Wasserstein distance between the empirical distribution at subsequent iterations to see whether
        the approximation of the posterior is converging. Then, it produces a plot displaying that. The approximation of
        the posterior is converging if the Wass distance between subsequent iterations decreases with iteration and gets
        close to 0, as that means there is no evolution of the posterior samples. The Wasserstein distance is estimated
        using the POT library).

        This method only works when the Journal stores results from all the iterations (ie it was generated with
        full_output=1). Moreover, this only works when all the parameters in the model are univariate.

        Parameters
        ----------
        num_iter_max : integer, optional
            The maximum number of iterations in the linear programming algorithm to estimate the Wasserstein distance.
            Default to 1e8.
        kwargs
            Additional arguments passed to the wass_dist calculation function.

        Returns
        -------
        tuple
            a tuple containing the matplotlib "fig, ax" objects defining the plot and the list of the computed
            Wasserstein distances. "fig" and "ax" can be useful for further modifying the plot.
        """
        if self._type == 0:
            raise RuntimeError("Wasserstein distance convergence test is available only if the journal was created with"
                               " full_output=1; in fact, this works by comparing the saved empirical distribution at "
                               "different iterations, and the latter is saved only if full_output=1.")

        if len(self.weights) == 1:
            raise RuntimeError("Only a set of posterior samples has been saved, corresponding to either running a "
                               "sequential algorithm for one iteration only or to using non-sequential algorithms (as"
                               "RejectionABC). Wasserstein distance convergence test requires at least samples from at "
                               "least 2 iterations.")
        last_params = np.array(self.get_accepted_parameters())
        if last_params.dtype == "object":
            raise RuntimeError("This error was probably raised due to the parameters in your model having different "
                               "dimensions (and specifically not being univariate). For now, Wasserstein distance"
                               " convergence test is available only if the different parameters have the same "
                               "dimension.")

        wass_dist_lists = [None] * (len(self.weights) - 1)

        for i in range(len(self.weights) - 1):
            params_1 = np.array(self.get_accepted_parameters(i))
            params_2 = np.array(self.get_accepted_parameters(i + 1))
            weights_1 = self.get_weights(i)
            weights_2 = self.get_weights(i + 1)
            if len(params_1.shape) == 1:  # we assume that the dimension of parameters is 1
                params_1 = params_1.reshape(-1, 1)
            else:
                params_1 = params_1.reshape(params_1.shape[0], -1)
            if len(params_2.shape) == 1:  # we assume that the dimension of parameters is 1
                params_2 = params_2.reshape(-1, 1)
            else:
                params_2 = params_2.reshape(params_2.shape[0], -1)

            if len(weights_1.shape) == 2:  # it can be that the weights have shape (-1,1); reshape therefore
                weights_1 = weights_1.reshape(-1)
            if len(weights_2.shape) == 2:  # it can be that the weights have shape (-1,1); reshape therefore
                weights_2 = weights_2.reshape(-1)

            wass_dist_lists[i] = wass_dist(samples_1=params_1, samples_2=params_2, weights_1=weights_1,
                                           weights_2=weights_2, num_iter_max=num_iter_max, **kwargs)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        ax.scatter(np.arange(len(self.weights) - 1) + 1, wass_dist_lists,
                   label="Estimated Wass. distance\nbetween iteration i and i+1")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Wasserstein distance")
        ax.legend()
        ax.set_xticks(np.arange(len(self.weights) - 1) + 1)

        return fig, ax, wass_dist_lists

    def traceplot(self, parameters_to_show=None, iteration=None, **kwargs):
        """
        Produces a traceplot for the MCMC inference scheme. This only works for journal files which were created by the
        `MCMCMetropolisHastings` inference scheme.


        Parameters
        ----------
        parameters_to_show : list, optional
            a list of the parameters for which you want to plot the traceplot. For each parameter, you need
            to provide the name string as it was defined in the model. For instance,
            `jrnl.traceplot(parameters_to_show=["mu"])` will only plot the traceplot for the
            parameter named "mu" in the list of parameters.
            If `None`, then all parameters will be displayed.
        iteration : int, optional
            specify the iteration for which to plot the posterior distribution, in the case of a sequential algorithm.
            If `None`, then the last iteration will be used.
        kwargs
            Additional arguments passed to matplotlib.pyplot.plot

        Returns
        -------
        tuple
            a tuple containing the matplotlib "fig, axes" objects defining the plot. Can be useful for further
            modifications.
        """

        if not "acceptance_rates" in self.configuration.keys():
            raise RuntimeError("The traceplot can be produced only for journal files which were created by the"
                               " MCMCMetropolisHastings inference scheme")

        if parameters_to_show is None:
            parameters_to_show = list(self.names_and_parameters[0].keys())
        else:
            for param_name in parameters_to_show:
                if param_name not in self.names_and_parameters[0].keys():
                    raise KeyError("Parameter '{}' is not a parameter of the model.".format(param_name))

        param_dict = self.get_parameters(iteration)
        n_plots = len(parameters_to_show)

        fig, ax = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        fig.suptitle("Traceplot")
        if n_plots == 1:
            ax = [ax]
        for i, name in enumerate(parameters_to_show):
            ax[i].plot(np.squeeze(param_dict[name]))
            ax[i].set_title(name)
            ax[i].set_xlabel("MCMC step")

        return fig, ax

    def resample(self, n_samples=None, replace=True, path_to_save_journal=None, seed=None):
        """
        Helper method to resample (by bootstrapping or subsampling) the posterior samples stored in the Journal.
        This can be used for instance to obtain an unweighted set of
        posterior samples from a weighted one (via bootstrapping) or
        to subsample a given number of posterior samples from a larger set. The new set of (unweighted)
        samples are stored in a new journal which is returned by the method.

        In order to bootstrap/subsample, the ``np.random.choice`` method is used, with the posterior sample
        weights used as
        probabilities (p) for resampling each sample. ``np.random.choice`` performs resampling with or without
        replacement according to whether ``replace=True`` or ``replace=False``. Moreover, the parameter
        ``n_samples`` specifies the number of resampled samples
        (the ``size`` argument of ``np.ranodom.choice``) and is set by
         default to the number of samples in the journal). Therefore, different combinations of these
        two parameters can be used to bootstrap or to subsample a set of posterior samples (see the examples below);
        the default parameter values perform bootstrap.

        Parameters
        ----------
        n_samples: integer, optional
            The number of posterior samples which you want to resample. Defaults to the number of posterior samples
            currently stored in the Journal.
        replace: boolean, optional
            If True, sampling with replacement is performed; if False, sampling without replacement. Defaults to False.
        path_to_save_journal: str, optional
            If provided, save the journal with the resampled posterior samples at the provided path.
        seed: integer, optional
             Optional initial seed for the random number generator. The default value is generated randomly.

        Returns
        -------
        abcpy.output.Journal
            a journal containing the resampled posterior samples

        Examples
        --------
        If ``journal`` contains a weighted set of posterior samples, the following returns an unweighted bootstrapped
        set of posterior samples, stored in ``new_journal``:

        >>> new_journal = journal.resample()

        The above of course also works when the original posterior samples are unweighted.

        If ``journal`` contains a here a large number of posterior sampling, you can subsample (without replacement)
        a smaller number of them (say 100) with the following line (and store them in ``new_journal``):

        >>> new_journal = journal.resample(n_samples=100, replace=False)

        Notice that the above takes into account the weights in the original ``journal``.

        """

        # instantiate the random number generator
        rng = np.random.RandomState(seed)

        # this extracts the parameters from the journal
        accepted_parameters = self.get_accepted_parameters(-1)
        accepted_weights = self.get_weights(-1)
        n_samples_old = self.configuration["n_samples"]
        normalized_weights = accepted_weights.reshape(-1) / np.sum(accepted_weights)

        n_samples = n_samples_old if n_samples is None else n_samples

        if n_samples > n_samples_old and not replace:
            raise RuntimeError("You cannot draw without replacement a larger number of samples than the posterior "
                               "samples currently stored in the journal.")

        # here you just need to bootstrap (or subsample):
        bootstrapped_parameter_indices = rng.choice(np.arange(n_samples_old), size=n_samples, replace=replace,
                                                    p=normalized_weights)
        bootstrapped_parameters = [accepted_parameters[index] for index in bootstrapped_parameter_indices]

        # define new journal
        journal_new = Journal(0)
        journal_new.configuration["type_model"] = self.configuration["type_model"]
        journal_new.configuration["n_samples"] = n_samples

        # store bootstrapped parameters in new journal
        journal_new.add_accepted_parameters(copy.deepcopy(bootstrapped_parameters))
        journal_new.add_weights(np.ones((n_samples, 1)))
        journal_new.add_ESS_estimate(np.ones((n_samples, 1)))

        # the next piece of code build the list to be passed to add_user_parameter in order to build the dictionary;
        # this mimics the behavior of what is done in the InferenceMethod's using the lines:
        # `self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)`
        # `names_and_parameters = self._get_names_and_parameters()`

        names_par_dicts = self.get_parameters()
        par_names = list(names_par_dicts.keys())
        new_names_par_list = []
        start_index = 0
        for name in par_names:
            parameter_size = len(names_par_dicts[name][0])  # the size of that parameter
            name_par_tuple = (
                name, [bootstrapped_parameters[i][start_index:start_index + parameter_size] for i in range(n_samples)])
            new_names_par_list.append(name_par_tuple)
            start_index += parameter_size
        journal_new.add_user_parameters(new_names_par_list)
        journal_new.number_of_simulations.append(0)

        if path_to_save_journal is not None:  # save journal
            journal_new.save(path_to_save_journal)

        return journal_new


class GenerateFromJournal(GraphTools):
    """Helper class to generate simulations from a model starting from the parameter values stored in a Journal file.

    Parameters
    ----------
    root_models: list
        A list of the Probabilistic models corresponding to the observed datasets
    backend: abcpy.backends.Backend
        Backend object defining the backend to be used.
    seed: integer, optional
         Optional initial seed for the random number generator. The default value is generated randomly.
    discard_too_large_values: boolean
         If set to True, the simulation is discarded (and repeated) if at least one element of it is too large
         to fit in float32, which therefore may be converted to infinite value in numpy. Defaults to False.

    Examples
    --------
    Simplest possible usage is:

    >>> generate_from_journal = GenerateFromJournal([model], backend=backend)
    >>> parameters, simulations, normalized_weights = generate_from_journal.generate(journal)

    which takes the parameter values stored in journal and generated simulations from them. Notice how the method
    returns (in this order) the parameter values used for the simulations, the simulations themselves and the
    posterior weights associated to the parameters. All of these three objects are numpy arrays.

    """

    def __init__(self, root_models, backend, seed=None, discard_too_large_values=False):
        self.model = root_models
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.discard_too_large_values = discard_too_large_values
        # An object managing the bds objects
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

    def generate(self, journal, n_samples_per_param=1, iteration=None):
        """
        Method to generate simulations using parameter values stored in the provided Journal.

        Parameters
        ----------
        journal: abcpy.output.Journal
            the Journal containing the parameter values from which to generate simulations from the model.
        n_samples_per_param: integer, optional
            Number of simulations for each parameter value. Defaults to 1.
        iteration: integer, optional
            specifies the iteration from which the parameter samples in the Journal are taken to generate simulations.
            If None (default), it uses the last iteration.

        Returns
        -------
        tuple
            A tuple of numpy ndarray's containing the parameter values (first element, with shape n_samples x d_theta),
            the generated
            simulations (second element, with shape n_samples x n_samples_per_param x d_x, where d_x is the dimension of
            each simulation) and the normalized weights attributed to each parameter value
            (third element, with shape n_samples).

        Examples
        --------
        Simplest possible usage is:

        >>> generate_from_journal = GenerateFromJournal([model], backend=backend)
        >>> parameters, simulations, normalized_weights = generate_from_journal.generate(journal)

        which takes the parameter values stored in journal and generated simulations from them. Notice how the method
        returns (in this order) the parameter values used for the simulations, the simulations themselves and the
        posterior weights associated to the parameters. All of these three objects are numpy arrays.

        """
        # check whether the model corresponds to the one for which the journal was generated
        if journal.configuration["type_model"] != [type(model).__name__ for model in self.model]:
            raise RuntimeError("You are not using the same model as the one with which the journal was generated.")

        self.n_samples_per_param = n_samples_per_param

        accepted_parameters = journal.get_accepted_parameters(iteration)
        accepted_weights = journal.get_weights(iteration)
        normalized_weights = accepted_weights.reshape(-1) / np.sum(accepted_weights)
        n_samples = len(normalized_weights)

        self.accepted_parameters_manager.broadcast(self.backend, [None])
        # Broadcast Accepted parameters
        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)

        seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
        # no need to check if the seeds are the same here as they are assigned to different parameter values
        rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
        index_arr = np.arange(0, n_samples, 1)
        data_arr = []
        for i in range(len(rng_arr)):
            data_arr.append([rng_arr[i], index_arr[i]])
        data_pds = self.backend.parallelize(data_arr)

        simulations_pds = self.backend.map(self._sample_parameter, data_pds)
        simulations = self.backend.collect(simulations_pds)

        parameters = np.array(accepted_parameters)
        simulations = np.array(simulations)

        parameters = parameters.reshape((parameters.shape[0], parameters.shape[1]))
        simulations = simulations.reshape((simulations.shape[0], simulations.shape[2], simulations.shape[3],))

        return parameters, simulations, normalized_weights

    def _sample_parameter(self, data, npc=None):
        """
        Simulates from a single model parameter.

        Parameters
        ----------
        data: list
            A list containing a random numpy state and a parameter index, e.g. [rng, index]

        Returns
        -------
        numpy.ndarray
            The simulated dataset.
        """

        if isinstance(data, np.ndarray):
            data = data.tolist()
        rng = data[0]
        index = data[1]

        parameter = self.accepted_parameters_manager.accepted_parameters_bds.value()[index]
        ok_flag = False

        while not ok_flag:
            self.set_parameters(parameter)
            y_sim = self.simulate(n_samples_per_param=self.n_samples_per_param, rng=rng, npc=npc)
            # if there are no potential infinities there (or if we do not check for those).
            # For instance, Lorenz model may give too large values sometimes (quite rarely).
            if self.discard_too_large_values and np.sum(np.isinf(np.array(y_sim).astype("float32"))) > 0:
                self.logger.warning("y_sim contained too large values for float32; simulating again.")
            else:
                ok_flag = True

        return y_sim
