from abc import ABCMeta, abstractmethod

import cloudpickle
import numpy as np

try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True
    from abcpy.NN_utilities.utilities import load_net, save_net
    from abcpy.NN_utilities.networks import createDefaultNN, ScalerAndNet


class Statistics(metaclass=ABCMeta):
    """This abstract base class defines how to calculate statistics from dataset.

    The base class also implements a polynomial expansion with cross-product
    terms that can be used to get desired polynomial expansion of the calculated statistics.

    """

    def __init__(self, degree=1, cross=False, reference_simulations=None, previous_statistics=None):
        """
        Initialization of the parent class. All sub-classes must call this at the end of their __init__,
         as it takes care of initializing the correct attributes to self for the other methods to work.

        `degree` and `cross` specify the polynomial expansion you want to apply to the statistics.

        If `reference_simulations` are provided, the standard deviation of the different statistics on the set 
        of reference simulations is computed and stored; these will then be used to rescale
        the statistics for each new simulation or observation. 
        If no set of reference simulations are provided, then this is not done.
        
        `previous_statistics` allows different Statistics object to be pipelined. Specifically, if the final 
        statistic to be used is determined by the
        composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
        is sufficient to call the `statistics` method of the second one, and that will automatically apply both
        transformations.  
        
        Parameters
        ----------
        degree: integer, optional
            Of polynomial expansion. The default value is 2 meaning second order polynomial expansion.
        cross: boolean, optional
            Defines whether to include the cross-product terms. The default value is True, meaning the cross product term
            is included.
        reference_simulations: array, optional
            A numpy array with shape (n_samples, output_size) containing a set of reference simulations. If provided, 
            statistics are computed at initialization for all reference simulations, and the standard deviation of the 
            different statistics is extracted. The standard deviation is then used to standardize the summary 
            statistics each time they are compute on a new observation or simulation. Defaults to None, in which case 
            standardization is not applied.
        previous_statistics: abcpy.statistics.Statistics, optional
            It allows pipelining of Statistics. Specifically, if the final statistic to be used is determined by the
            composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
            is sufficient to call the `statistics` method of the second one, and that will automatically apply both
            transformations.
        """

        self.degree = degree
        self.cross = cross
        self.previous_statistics = previous_statistics
        if reference_simulations is not None:
            training_statistics = self.statistics(
                [reference_simulations[i] for i in range(reference_simulations.shape[0])])
            self.std_statistics = np.std(training_statistics, axis=0)
            # we store this and use it to rescale the statistics

    @abstractmethod
    def statistics(self, data: object) -> object:
        """To be overwritten by any sub-class: should extract statistics from the
        data set data. It is assumed that data is a  list of n same type
        elements(eg., The data can be a list containing n timeseries, n graphs or n np.ndarray).

        All statistics implementation should follow this structure:

        >>> # need to call this first which takes care of calling the
        >>> # previous statistics if that is defined and of properly
        >>> # formatting data
        >>> data = self._preprocess(data)
        >>>
        >>> # !!! here do all the processing on the statistics (data) !!!
        >>>
        >>> # Expand the data with polynomial expansion
        >>> result = self._polynomial_expansion(data)
        >>>
        >>> # now call the _rescale function which automatically rescales
        >>> # the different statistics using the standard
        >>> # deviation of them on the training set provided at initialization.
        >>> result = self._rescale(result)


        Parameters
        ----------
        data: python list
            Contains n data sets with length p.
        Returns
        -------
        numpy.ndarray
            nxp matrix where for each of the n data points p statistics are calculated.

        """

        raise NotImplementedError

    def _polynomial_expansion(self, summary_statistics):
        """Helper function that does the polynomial expansion and includes cross-product
        terms of summary_statistics, already calculated summary statistics. It is tipically called in the `statistics`
        method of a `Statistics` class, after the statistics have been computed from data but before the statistics
        are (optionally) rescaled.

        Parameters
        ----------
        summary_statistics: numpy.ndarray
            nxp matrix where n is number of data points in the datasets data set and p number os
            summary statistics calculated.
        Returns
        -------
        numpy.ndarray
            nx(p+degree*p+cross*nchoosek(p,2)) matrix where for each of the n points with
            p statistics, degree*p polynomial expansion term and cross*nchoosek(p,2) many
            cross-product terms are calculated.

        """

        # Check summary_statistics is a np.ndarray
        if not isinstance(summary_statistics, np.ndarray):
            raise TypeError('Summary statistics is not of allowed types')
        # Include the polynomial expansion
        result = summary_statistics
        for ind in range(2, self.degree + 1):
            result = np.column_stack((result, np.power(summary_statistics, ind)))

        # Include the cross-product term
        if self.cross and summary_statistics.shape[1] > 1:
            # Convert to a matrix
            for ind1 in range(0, summary_statistics.shape[1]):
                for ind2 in range(ind1 + 1, summary_statistics.shape[1]):
                    result = np.column_stack((result, summary_statistics[:, ind1] * summary_statistics[:, ind2]))
        return result

    def _rescale(self, result):
        """Rescales the final summary statistics using the standard deviations computed at initialization on the set of
        reference simulations. If that was not done, no rescaling is done.

        Parameters
        ----------
        result: numpy.ndarray
            Final summary statistics (after polynomial expansion)

        Returns
        -------
        numpy.ndarray
            Rescaled summary statistics, with the same shape as the input.
        """
        if hasattr(self, "std_statistics"):
            if result.shape[-1] != self.std_statistics.shape[-1]:
                raise RuntimeError("The size of the statistics is not the same as the stored standard deviations for "
                                   "rescaling! Please check that you initialized the statistics with the correct set "
                                   "of reference samples.")

            result = result / self.std_statistics

        return result

    def _preprocess(self, data):
        """Utility which needs to be called at the beginning of the `statistics` method for all `Statistics` classes.
        It takes care of calling the `previous_statistics` if that is available (pipelining)
        and of correctly formatting the data.

        Parameters
        ----------
        data: python list
            Contains n data sets with length p.

        Returns
        -------
        numpy.ndarray
            Formatted statistics after pipelining.
        """

        # pipeline: first call the previous statistics:
        if self.previous_statistics is not None:
            data = self.previous_statistics.statistics(data)
        # the first of the statistics need to take list as input, in order to match the API. Then actually the
        # transformations work on np.arrays. In fact the first statistic transforms the list to array. Therefore, the
        # following code needs to be called only if the self statistic is the first, i.e. it does not have a
        # previous_statistic element.
        else:
            data = self._check_and_transform_input(data)

        return data

    def _check_and_transform_input(self, data):
        """ Formats the input in the correct way for computing summary statistics; specifically takes as input a
        list and returns a numpy.ndarray.

        Parameters
        ----------
        data: python list
            Contains n data sets with length p.

        Returns
        -------
        numpy.ndarray
            Formatted statistics after pipelining.
        """
        if isinstance(data, list):
            if np.array(data).shape == (len(data),):
                if len(data) == 1:
                    data = np.array(data).reshape(1, 1)
                data = np.array(data).reshape(len(data), 1)
            else:
                data = np.concatenate(data).reshape(len(data), -1)
        else:
            raise TypeError('Input data should be of type list, but found type {}'.format(type(data)))

        return data


class Identity(Statistics):
    """
    This class implements identity statistics not applying any transformation to the data, before the optional
    polynomial expansion step. If the data set contains n numpy.ndarray of length p, it returns therefore an
    nx(p+degree*p+cross*nchoosek(p,2)) matrix, where for each of the n points with p statistics, degree*p polynomial
    expansion term and cross*nchoosek(p,2) many cross-product terms are calculated.
    """

    def statistics(self, data):
        """
        Parameters
        ----------
        data: python list
            Contains n data sets with length p.
        Returns
        -------
        numpy.ndarray
            nx(p+degree*p+cross*nchoosek(p,2)) matrix where for each of the n data points with length p,
            (p+degree*p+cross*nchoosek(p,2)) statistics are calculated.
        """

        # need to call this first which takes care of calling the previous statistics if that is defined and of properly
        # formatting data
        data = self._preprocess(data)

        # Expand the data with polynomial expansion
        result = self._polynomial_expansion(data)

        # now call the _rescale function which automatically rescales the different statistics using the standard
        # deviation of them on the training set provided at initialization.
        result = self._rescale(result)

        return result


class LinearTransformation(Statistics):
    """Applies a linear transformation to the data to get (usually) a lower dimensional statistics. Then you can apply
    an additional polynomial expansion step.
    """

    def __init__(self, coefficients, degree=1, cross=False, reference_simulations=None, previous_statistics=None):
        """
        `degree` and `cross` specify the polynomial expansion you want to apply to the statistics.

        If `reference_simulations` are provided, the standard deviation of the different statistics on the set 
        of reference simulations is computed and stored; these will then be used to rescale
        the statistics for each new simulation or observation. 
        If no set of reference simulations are provided, then this is not done.

        `previous_statistics` allows different Statistics object to be pipelined. Specifically, if the final 
        statistic to be used is determined by the
        composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
        is sufficient to call the `statistics` method of the second one, and that will automatically apply both
        transformations.  

        Parameters
        ----------
        coefficients: coefficients is a matrix with size d x p, where d is the dimension of the summary statistic that
            is obtained after applying the linear transformation (i.e. before a possible polynomial expansion is
            applied), while d is the dimension of each data.
        degree : integer, optional
            Of polynomial expansion. The default value is 2 meaning second order polynomial expansion.
        cross : boolean, optional
            Defines whether to include the cross-product terms. The default value is True, meaning the cross product term
            is included.
        reference_simulations: array, optional
            A numpy array with shape (n_samples, output_size) containing a set of reference simulations. If provided, 
            statistics are computed at initialization for all reference simulations, and the standard deviation of the 
            different statistics is extracted. The standard deviation is then used to standardize the summary 
            statistics each time they are compute on a new observation or simulation. Defaults to None, in which case 
            standardization is not applied.
        previous_statistics : abcpy.statistics.Statistics, optional
            It allows pipelining of Statistics. Specifically, if the final statistic to be used is determined by the
            composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
            is sufficient to call the `statistics` method of the second one, and that will automatically apply both
            transformations.
        """
        self.coefficients = coefficients

        super(LinearTransformation, self).__init__(degree, cross, reference_simulations, previous_statistics)

    def statistics(self, data):
        """
        Parameters
        ----------
        data: python list
            Contains n data sets with length p.
        Returns
        -------
        numpy.ndarray
            nx(d+degree*d+cross*nchoosek(d,2)) matrix where for each of the n data points with length p you apply the
            linear transformation to get to dimension d, from where (d+degree*d+cross*nchoosek(d,2)) statistics are
            calculated.
        """

        # need to call this first which takes care of calling the previous statistics if that is defined and of properly
        # formatting data
        data = self._preprocess(data)

        # Apply now the linear transformation
        if not data.shape[1] == self.coefficients.shape[0]:
            raise ValueError('Mismatch in dimension of summary statistics and coefficients')
        data = np.dot(data, self.coefficients)

        # Expand the data with polynomial expansion
        result = self._polynomial_expansion(data)

        # now call the _rescale function which automatically rescales the different statistics using the standard
        # deviation of them on the training set provided at initialization.
        result = self._rescale(result)

        return result


class NeuralEmbedding(Statistics):
    """Computes the statistics by applying a neural network transformation. 
    
    It is essentially a wrapper for the application of a neural network transformation to the data. Note that the
    neural network has had to be trained in some way (for instance check the statistics learning routines) and that 
    Pytorch is required for this part to work.   
    """

    def __init__(self, net, degree=1, cross=False, reference_simulations=None, previous_statistics=None):

        """
        `degree` and `cross` specify the polynomial expansion you want to apply to the statistics.

        If `reference_simulations` are provided, the standard deviation of the different statistics on the set
        of reference simulations is computed and stored; these will then be used to rescale
        the statistics for each new simulation or observation. 
        If no set of reference simulations are provided, then this is not done.

        `previous_statistics` allows different Statistics object to be pipelined. Specifically, if the final 
        statistic to be used is determined by the
        composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
        is sufficient to call the `statistics` method of the second one, and that will automatically apply both
        transformations.

        Parameters
        ----------
        net : torch.nn object
            the embedding neural network. The input size of the neural network must coincide with the size of each of
            the datapoints.
        degree: integer, optional
            Of polynomial expansion. The default value is 2 meaning second order polynomial expansion.
        cross: boolean, optional
            Defines whether to include the cross-product terms. The default value is True, meaning the cross product term
            is included.
        reference_simulations: array, optional
            A numpy array with shape (n_samples, output_size) containing a set of reference simulations. If provided, 
            statistics are computed at initialization for all reference simulations, and the standard deviation of the 
            different statistics is extracted. The standard deviation is then used to standardize the summary 
            statistics each time they are compute on a new observation or simulation. Defaults to None, in which case 
            standardization is not applied.
        previous_statistics: abcpy.statistics.Statistics, optional
            It allows pipelining of Statistics. Specifically, if the final statistic to be used is determined by the
            composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
            is sufficient to call the `statistics` method of the second one, and that will automatically apply both
            transformations.
        """
        if not has_torch:
            raise ImportError(
                "Pytorch is required to instantiate an element of the {} class, in order to handle "
                "neural networks. Please install it. ".format(self.__class__.__name__))

        self.net = net

        # init of super class
        super(NeuralEmbedding, self).__init__(degree, cross, reference_simulations, previous_statistics)

    @classmethod
    def fromFile(cls, path_to_net_state_dict, network_class=None, path_to_scaler=None, input_size=None,
                 output_size=None, hidden_sizes=None, degree=1, cross=False, reference_simulations=None,
                 previous_statistics=None):
        """If the neural network state_dict was saved to the disk, this method can be used to instantiate a
        NeuralEmbedding object with that neural network.

        In order for the state_dict to be read correctly, the network class is needed. Therefore, we provide 2 options:
        1) the Pytorch neural network class can be passed (if the user defined it, for instance)
        2) if the neural network was defined by using the DefaultNN class in abcpy.NN_utilities.networks, you can
        provide arguments `input_size`, `output_size` and `hidden_sizes` (the latter is optional) that define
        the sizes of a fully connected network; then a DefaultNN is instantiated with those sizes. This can be used
        if for instance the neural network was trained using the utilities in abcpy.statisticslearning and you did
        not provide explicitly the neural network class there, but defined it through the sizes of the different layers.

        In both cases, note that the input size of the neural network must coincide with the size of each of the
        datapoints generated from the model (unless some other statistics are computed beforehand).

        Note that if the neural network was of the class `ScalerAndNet`, ie a scaler was applied before the data is fed
        through it, you need to pass `path_to_scaler` as well. Then this method will instantiate the network in the
        correct way.

        Parameters
        ----------
        path_to_net_state_dict : basestring
            the path where the state-dict is saved
        network_class : torch.nn class, optional
            if the neural network class is known explicitly (for instance if the used defined it), then it has to be
             passed here. This must not be provided together with `input_size` or `output_size`.
        path_to_scaler: basestring, optional
            The path where the scaler which was applied before the neural network is saved. Note that if the neural
            network was trained on scaled data and now you do not pass the correct scaler, the behavior will not be
            correct, leading to wrong inference. Default to None.
        input_size : integer, optional
            if the neural network is an instance of abcpy.NN_utilities.networks.DefaultNN with some input and
            output size, then you should provide here the input size of the network. It has to be provided together with
            the corresponding output_size, and it must not be provided with `network_class`.
        output_size : integer, optional
            if the neural network is an instance of abcpy.NN_utilities.networks.DefaultNN with some input and
            output size, then you should provide here the output size of the network. It has to be provided together
            with the corresponding input_size, and it must not be provided with `network_class`.
        hidden_sizes : array-like, optional
            if the neural network is an instance of abcpy.NN_utilities.networks.DefaultNN with some input and
            output size, then you can provide here an array-like with the size of the hidden layers (for instance
            [5,7,5] denotes 3 hidden layers with correspondingly 5,7,5 neurons). In case this parameter is not provided,
            the hidden sizes are determined from the input and output sizes as determined in
            abcpy.NN_utilities.networks.DefaultNN. Note that this must not be provided together with `network_class`.
        degree: integer, optional
            Of polynomial expansion. The default value is 2 meaning second order polynomial expansion.
        cross: boolean, optional
            Defines whether to include the cross-product terms. The default value is True, meaning the cross product term
            is included.
        reference_simulations: array, optional
            A numpy array with shape (n_samples, output_size) containing a set of reference simulations. If provided,
            statistics are computed at initialization for all reference simulations, and the standard deviation of the
            different statistics is extracted. The standard deviation is then used to standardize the summary
            statistics each time they are compute on a new observation or simulation. Defaults to None, in which case
            standardization is not applied.
        previous_statistics : abcpy.statistics.Statistics, optional
            It allows pipelining of Statistics. Specifically, if the final statistic to be used is determined by the
            composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
            is sufficient to call the `statistics` method of the second one, and that will automatically apply both
            transformations. In this case, this is the statistics that has to be computed before the neural network
            transformation is applied.
        Returns
        -------
        abcpy.statistics.NeuralEmbedding
            the `NeuralEmbedding` object with the neural network obtained from the specified file.
        """
        if not has_torch:
            raise ImportError(
                "Pytorch is required to instantiate an element of the {} class, in order to handle "
                "neural networks. Please install it. ".format(cls.__name__))

        if network_class is None and (input_size is None or output_size is None):
            raise RuntimeError("You need to pass either network class or both input_size and output_size.")
        if network_class is not None and (input_size is not None or output_size is not None):
            raise RuntimeError("You can't pass together network_class and one of input_size, output_size")
        if network_class is not None and hidden_sizes is not None:
            raise RuntimeError("You passed hidden_sizes as an argument, but that may be passed only if you are passing "
                               "input_size and input_size as well, and you are not passing network_class.")

        if network_class is not None:  # user explicitly passed the NN class
            net = load_net(path_to_net_state_dict, network_class)
        else:  # the user passed the input_size, output_size and (maybe) the hidden_sizes
            net = load_net(path_to_net_state_dict, createDefaultNN(input_size=input_size, output_size=output_size,
                                                                   hidden_sizes=hidden_sizes))

        if path_to_scaler is not None:
            f = open(path_to_scaler, 'rb')
            scaler = cloudpickle.load(f)
            f.close()
            net = ScalerAndNet(net, scaler)

        statistic_object = cls(net, degree=degree, cross=cross, reference_simulations=reference_simulations,
                               previous_statistics=previous_statistics)

        return statistic_object

    def save_net(self, path_to_net_state_dict, path_to_scaler=None):
        """Method to save the neural network state dict to a file. If the network is of the class ScalerAndNet, ie a
        scaler is applied before the data is fed through the network, then you are required to pass the path where you
        want the scaler to be saved.

        Parameters
        ----------
        path_to_net_state_dict: basestring
            Path where the state dict of the neural network is saved.
        path_to_scaler: basestring
            Path where the scaler is saved (with pickle); this is required if the neural network is of the class
            ScalerAndNet, and is ignored otherwise.
        """
        # if the net is of the class ScalerAndNet
        if hasattr(self.net, "scaler") and path_to_scaler is None:
            raise RuntimeError("You did not specify path_to_scaler, which is required as the neural network is an "
                               "element of the class `ScalerAndNet`, ie a scaler is applied before the data is fed"
                               " through the network")

        if hasattr(self.net, "scaler"):
            save_net(path_to_net_state_dict, self.net.net)
            f = open(path_to_scaler, 'wb')
            cloudpickle.dump(self.net.scaler, f)
            f.close()
        else:
            save_net(path_to_net_state_dict, self.net)

    def statistics(self, data):
        """
        Parameters
        ----------
        data: python list
            Contains n data sets with length p.

        Returns
        -------
        numpy.ndarray
            the statistics computed by applying the neural network.
        """

        # need to call this first which takes care of calling the previous statistics if that is defined and of properly
        # formatting data
        data = self._preprocess(data)

        data = torch.from_numpy(data.astype("float32"))

        # move data to gpu if the net is on gpu
        if next(self.net.parameters()).is_cuda:
            data = data.cuda()

        # simply apply the network transformation.
        data = self.net(data).cpu().detach().numpy()
        data = np.array(data)

        # Expand the data with polynomial expansion
        result = self._polynomial_expansion(data)

        # now call the _rescale function which automatically rescales the different statistics using the standard
        # deviation of them on the training set provided at initialization.
        result = self._rescale(result)

        return result
