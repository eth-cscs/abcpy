import numpy as np


# these transformers are used in the MCMC inference scheme, in order to run MCMC of an unbounded transformed space in
# case the original space is bounded. It therefore also implements the jacobian terms which appear in the acceptance
# rate.

class BoundedVarTransformer:
    """
    This scaler implements both lower bounded and two sided bounded transformations according to the provided bounds;
    """

    def __init__(self, lower_bound, upper_bound):

        # upper and lower bounds can be both scalar or array-like with size the size of the variable
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if not hasattr(lower_bound, "shape") or not hasattr(upper_bound, "shape"):
            raise RuntimeError("Provided lower and upper bounds need to be arrays.")
        elif hasattr(lower_bound, "shape") and hasattr(upper_bound, "shape") and lower_bound.shape != upper_bound.shape:
            raise RuntimeError("Provided lower and upper bounds need to have same shape.")

        # note that == None checks if the array is None element wise.
        self.unbounded_vars = np.logical_and(np.equal(lower_bound, None), np.equal(upper_bound, None))
        self.lower_bounded_vars = np.logical_and(np.not_equal(lower_bound, None), np.equal(upper_bound, None))
        self.upper_bounded_vars = np.logical_and(np.equal(lower_bound, None), np.not_equal(upper_bound, None))
        self.two_sided_bounded_vars = np.logical_and(np.not_equal(lower_bound, None), np.not_equal(upper_bound, None))
        if self.upper_bounded_vars.any():
            raise NotImplementedError("We do not yet implement the transformation for upper bounded random variables")

        self.lower_bound_lower_bounded = self.lower_bound[self.lower_bounded_vars].astype("float32")
        self.lower_bound_two_sided = self.lower_bound[self.two_sided_bounded_vars].astype("float32")
        self.upper_bound_two_sided = self.upper_bound[self.two_sided_bounded_vars].astype("float32")

    @staticmethod
    def logit(x):
        return np.log(x) - np.log(1 - x)

    def _check_data_in_bounds(self, x):
        if np.any(x[self.lower_bounded_vars] <= self.lower_bound_lower_bounded):
            raise RuntimeError("The provided data are out of the bounds.")
        if (x[self.two_sided_bounded_vars] <= self.lower_bound[self.two_sided_bounded_vars]).any() or (
                x[self.two_sided_bounded_vars] >= self.upper_bound_two_sided).any():
            raise RuntimeError("The provided data is out of the bounds.")

    def _apply_nonlinear_transf(self, x):
        # apply the different scalers to the different kind of variables:
        x_transf = x.copy()
        x_transf[self.lower_bounded_vars] = np.log(x[self.lower_bounded_vars] - self.lower_bound_lower_bounded)
        x_transf[self.two_sided_bounded_vars] = self.logit(
            (x[self.two_sided_bounded_vars] - self.lower_bound_two_sided) / (
                    self.upper_bound_two_sided - self.lower_bound_two_sided))
        return x_transf

    @staticmethod
    def _array_from_list(x):
        return np.array(x).reshape(-1)

    @staticmethod
    def _list_from_array(x_arr, x):
        # transforms the array x to the list structure that contains x
        x_new = [None] * len(x)
        for i in range(len(x)):
            if isinstance(x[i], np.ndarray):
                x_new[i] = np.array(x_arr[i].reshape(x[i].shape))
            else:
                x_new[i] = x_arr[i]
        return x_new

    def transform(self, x):
        """Scale features of x according to feature_range.
        Parameters
        ----------
        x : list of len n_parameters
            Input data that will be transformed.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        # convert data to array:
        x_arr = self._array_from_list(x)

        # need to check if we can apply the log first:
        self._check_data_in_bounds(x_arr)

        # we transform the data with the log transformation:
        x_arr = self._apply_nonlinear_transf(x_arr)

        # convert back to the list structure:
        x = self._list_from_array(x_arr, x)

        return x

    def inverse_transform(self, x):
        """Undo the scaling of x according to feature_range.
        Parameters
        ----------
        x : list of len n_parameters
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        # now apply the inverse transform
        x_arr = self._array_from_list(x)

        inv_x = x_arr.copy()
        inv_x[self.two_sided_bounded_vars] = (self.upper_bound_two_sided - self.lower_bound_two_sided) * np.exp(
            x_arr[self.two_sided_bounded_vars]) / (1 + np.exp(
            x_arr[self.two_sided_bounded_vars])) + self.lower_bound_two_sided
        inv_x[self.lower_bounded_vars] = np.exp(x_arr[self.lower_bounded_vars]) + self.lower_bound_lower_bounded

        # convert back to the list structure:
        inv_x = self._list_from_array(inv_x, x)

        return inv_x

    def jac_log_det(self, x):
        """Returns the log determinant of the Jacobian: log |J_t(x)|.

        Parameters
        ----------
        x : list of len n_parameters
            Input data, living in the original space (with lower bound constraints).
        Returns
        -------
        res : float
            log determinant of the jacobian
        """
        x = self._array_from_list(x)
        self._check_data_in_bounds(x)

        results = np.zeros_like(x)
        results[self.two_sided_bounded_vars] = np.log(
            (self.upper_bound_two_sided - self.lower_bound_two_sided).astype("float64") / (
                    (x[self.two_sided_bounded_vars] - self.lower_bound_two_sided) * (
                    self.upper_bound_two_sided - x[self.two_sided_bounded_vars])))
        results[self.lower_bounded_vars] = - np.log(x[self.lower_bounded_vars] - self.lower_bound_lower_bounded)

        return np.sum(results)

    def jac_log_det_inverse_transform(self, x):
        """Returns the log determinant of the Jacobian evaluated in the inverse transform:
        log |J_t(t^{-1}(x))| = - log |J_{t^{-1}}(x)|

        Parameters
        ----------
        x : list of len n_parameters
            Input data, living in the transformed space (spanning the whole R^d).
        Returns
        -------
        res : float
            log determinant of the jacobian evaluated in t^{-1}(x)
        """
        x = self._array_from_list(x)

        results = np.zeros_like(x)
        results[self.lower_bounded_vars] = - x[self.lower_bounded_vars]
        # two sided: need some tricks to avoid numerical issues:
        results[self.two_sided_bounded_vars] = - np.log(
            self.upper_bound_two_sided - self.lower_bound_two_sided)

        indices = x[self.two_sided_bounded_vars] < 100  # for avoiding numerical overflow
        res_b = np.copy(x)[self.two_sided_bounded_vars]
        res_b[indices] = np.log(1 + np.exp(x[self.two_sided_bounded_vars][indices]))
        results[self.two_sided_bounded_vars] += res_b

        indices = x[self.two_sided_bounded_vars] > - 100  # for avoiding numerical overflow
        res_c = np.copy(- x)[self.two_sided_bounded_vars]
        res_c[indices] = np.log(1 + np.exp(- x[self.two_sided_bounded_vars][indices]))
        results[self.two_sided_bounded_vars] += res_c

        # res = res_b + res_c - res_a

        return np.sum(results)


class DummyTransformer:
    """Dummy transformer which does nothing, and for which the jacobian is 1"""

    def __init__(self):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def jac_log_det(self, x):
        return 0

    def jac_log_det_inverse_transform(self, x):
        return 0
