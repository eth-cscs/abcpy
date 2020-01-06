import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


class Journal:
    """The journal holds information created by the run of inference schemes.

    It can be configured to even hold intermediate.

    Attributes
    ----------
    parameters : numpy.array
        a nxpxt matrix
    weights : numpy.array
        a nxt matrix
    opt_value : numpy.array
        nxp matrix containing for each parameter the evaluated objective function for every time step
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
        self.weights = []
        self.distances = []
        self.opt_values = []
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
        Saves the provided parameters and names of the probabilistic models corresponding to them. If type==0, old parameters get overwritten.

        Parameters
        ----------
        names_and_params: list
            Each entry is a tupel, where the first entry is the name of the probabilistic model, and the second entry is the parameters associated with this model.
        """
        if (self._type == 0):
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

    def plot_posterior_distr(self, parameters_to_show=None, ranges_parameters=None, iteration=None, show_samples=None,
                             single_marginals_only=False, double_marginals_only=False, write_posterior_mean=True,
                             true_parameter_values=None, contour_levels=14, show_density_values=True, bw_method=None,
                             path_to_save=None):
        """
        Produces a visualization of the posterior distribution of the parameters of the model.

        A Gaussian kernel density estimate (KDE) is used to approximate the density starting from the sampled
        parameters. Specifically, it produces a scatterplot matrix, where the diagonal contains single parameter
        marginals, while the off diagonal elements contain the contourplot for the paired marginals for each possible
        pair of parameters.

        Parameters
        ----------
        parameters_to_show : list, optional
            a list of the parameters for which you want to plot the posterior distribution. For each parameter, you need
            to provide the name string as it was defined in the model.
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
            takes into account the weights, and showing the samples may be misleading). Otherwise, if the posterior #
            samples are not associated with weights, they are displayed by defauly.
        single_marginals_only : boolean, optional
            if `True`, the method does not show the paired marginals but only the single parameter marginals;
            otherwise, it shows the paired marginals as well. Default to `False`.
        double_marginals_only : boolean, optional
            if `True`, the method shows the contour plot for the marginal posterior for each possible pair of parameters
            in the parameters that have to be shown (all parameters of the model if `parameters_to_show` is None).
            Default to `False`.
        write_posterior_mean : boolean, optional
            Whether to write or not the posterior mean on the single marginal plots. Default to `True`.
        true_parameter_values: array-like, optional
            you can provide here the true values of the parameters, if known, and that will be displayed in the
            posterior plot. It has to be an array-like of the same length of `parameters_to_show` (if that is provided),
            otherwise of length equal to the number of parameters in the model, and with entries corresponding to the
            true value of that parameter (in case `parameters_to_show` is not provided, the order of the parameters is
            the same order the model `forward_simulate` step takes.
        contour_levels: integer, optional
            The number of levels to be used in the contour plots. Default to 14.
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
        list
            a list containing the matplotlib "fig, axes" objects defining the plot. Can be useful for further
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

        meanpost = np.array([self.posterior_mean()[x] for x in parameters_to_show])

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

        def write_post_mean_function(ax, post_mean, name):
            ax.text(0.15, 0.06, r"Post. mean = {:.2f}".format(post_mean), size=14.5 * 2 / len(meanpost),
                    transform=ax.transAxes)

        def scatterplot_matrix(data, meanpost, names, single_marginals_only=False, **kwargs):
            """
            Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
            against other rows, resulting in a nrows by nrows grid of subplots with the
            diagonal subplots labeled with "parameters_to_show".  Additional keyword arguments are
            passed on to matplotlib's "plot" command. Returns the matplotlib figure
            object containg the subplot grid.
            """
            numvars, numdata = data.shape
            if single_marginals_only:
                fig, axes = plt.subplots(nrows=1, ncols=numvars, figsize=(4 * len(names), 4))
            else:
                fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
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
                            axes[x, y].plot([xmin, xmax], [meanpost[x], meanpost[x]], "red", markersize='20',
                                            linestyle='solid')
                            axes[x, y].plot([meanpost[y], meanpost[y]], [ymin, ymax], "red", markersize='20',
                                            linestyle='solid')
                            if true_parameter_values is not None:
                                axes[x, y].plot([xmin, xmax], [true_parameter_values[x], true_parameter_values[x]], "green",
                                                markersize='20', linestyle='dashed')
                                axes[x, y].plot([true_parameter_values[y], true_parameter_values[y]], [ymin, ymax], "green",
                                                markersize='20', linestyle='dashed')

                            CS = axes[x, y].contour(X, Y, Z, contour_levels, linestyles='solid')
                            if show_density_values:
                                axes[x, y].clabel(CS, fontsize=9, inline=1)
                            axes[x, y].set_xlim([xmin, xmax])
                            axes[x, y].set_ylim([ymin, ymax])

            # diagonal plots

            if not single_marginals_only:
                diagonal_axes = np.array([axes[i, i] for i in range(len(axes))])
            else:
                diagonal_axes = axes

            for i, label in enumerate(names):
                xmin, xmax = ranges_parameters[names[i]]
                positions = np.linspace(xmin, xmax, 100)
                gaussian_kernel = gaussian_kde(data[i], weights=weights, bw_method=bw_method)
                diagonal_axes[i].plot(positions, gaussian_kernel(positions), color='k', linestyle='solid', lw="1",
                                      alpha=1, label="Density")
                values = gaussian_kernel(positions)
                # axes[i, i].plot([positions[np.argmax(values)], positions[np.argmax(values)]], [0, np.max(values)])
                diagonal_axes[i].plot([meanpost[i], meanpost[i]], [0, 1.1 * np.max(values)], "red", alpha=1,
                                      label="Posterior mean")
                if true_parameter_values is not None:
                    diagonal_axes[i].plot([true_parameter_values[i], true_parameter_values[i]], [0, 1.1 * np.max(values)], "green",
                                           alpha=1, label="True value")
                if write_posterior_mean:
                    write_post_mean_function(diagonal_axes[i], meanpost[i], label)
                diagonal_axes[i].set_xlim([xmin, xmax])
                diagonal_axes[i].set_ylim([0, 1.1 * np.max(values)])

            # labels and ticks:
            if not single_marginals_only:
                for j, label in enumerate(names):
                    axes[0, j].set_title(label, size=17)

                    if len(names) > 1:
                        axes[j, 0].set_ylabel(label)
                        axes[-1, j].set_xlabel(label)
                    else:
                        axes[j, 0].set_ylabel("Density")

                    axes[j, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    axes[j, 0].yaxis.set_visible(True)

                    axes[-1, j].ticklabel_format(style='sci', axis='x')  # , scilimits=(0, 0))
                    axes[-1, j].xaxis.set_visible(True)
                    axes[j, -1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    axes[j, -1].yaxis.set_visible(True)

            else:
                for j, label in enumerate(names):
                    axes[j].set_title(label, size=17)
                axes[0].set_ylabel("Density")

            return fig, axes

        def double_marginals_plot(data, meanpost, names, **kwargs):
            """
            Plots contour plots of all possible pairs of samples. Additional keyword arguments are
            passed on to matplotlib's "plot" command. Returns the matplotlib figure
            object containg the subplot grid.
            """
            numvars, numdata = data.shape
            numplots = np.int(numvars * (numvars - 1) / 2)
            fig, axes = plt.subplots(nrows=1, ncols=numplots, figsize=(4 * numplots, 4))

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
                    axes[ax_counter].plot([xmin, xmax], [meanpost[x], meanpost[x]], "red", markersize='20',
                                          linestyle='solid')
                    axes[ax_counter].plot([meanpost[y], meanpost[y]], [ymin, ymax], "red", markersize='20',
                                          linestyle='solid')
                    if true_parameter_values is not None:
                        axes[ax_counter].plot([xmin, xmax], [true_parameter_values[x], true_parameter_values[x]], "green",
                                              markersize='20', linestyle='dashed')
                        axes[ax_counter].plot([true_parameter_values[y], true_parameter_values[y]], [ymin, ymax], "green",
                                              markersize='20', linestyle='dashed')

                    CS = axes[ax_counter].contour(X, Y, Z, contour_levels, linestyles='solid')
                    if show_density_values:
                        axes[ax_counter].clabel(CS, fontsize=9, inline=1)
                    axes[ax_counter].set_xlim([xmin, xmax])
                    axes[ax_counter].set_ylim([ymin, ymax])

                    axes[ax_counter].set_ylabel(names[x])
                    axes[ax_counter].set_xlabel(names[y])

                    axes[ax_counter].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    axes[ax_counter].yaxis.set_visible(True)
                    axes[ax_counter].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
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
