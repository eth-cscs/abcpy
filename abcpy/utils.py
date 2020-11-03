from functools import wraps

import numpy as np
import ot


def cached(func):
    cache = {}

    @wraps(func)
    def wrapped(x):
        if x not in cache:
            cache[x] = func(x)
        return cache[x]

    return wrapped


def wass_dist(samples_1, samples_2, weights_1=None, weights_2=None, num_iter_max=100000, **kwargs):
    """
    Computes the Wasserstein 2 distance between two empirical distributions with weights. This uses the POT library to 
    estimate Wasserstein distance. The Wasserstein distance computation can take long if the number of samples in the 
    two datasets is large (cost of the computation scales in fact quadratically with the number of samples).

    Parameters
    ----------
    samples_1 : np.ndarray
         Samples defining the first empirical distribution, with shape (nxd), n being the number of samples in the
         first empirical distribution and d the dimension of the random variable.
    samples_2 : np.ndarray
         Samples defining the second empirical distribution, with shape (mxd), m being the number of samples in the
         second empirical distribution and d the dimension of the random variable.
    weights_1 : np.ndarray, optional
         Weights defining the first empirical distribution, with shape (n), n being the number of samples in the
         first empirical distribution. Weights are normalized internally to the function. If not provided, they are
         assumed to be identical for all samples.
    weights_2 : np.ndarray, optional
         Weights defining the second empirical distribution, with shape (m), m being the number of samples in the
         second empirical distribution. Weights are normalized internally to the function. If not provided, they are
         assumed to be identical for all samples.
    num_iter_max : integer, optional
        The maximum number of iterations in the linear programming algorithm to estimate the Wasserstein distance. 
        Default to 100000. 
    kwargs 
        Additional arguments passed to ot.emd2

    Returns
    -------
    float
        The estimated 2-Wasserstein distance.
    """
    n = samples_1.shape[0]
    m = samples_2.shape[0]

    if weights_1 is None:
        a = np.ones((n,)) / n
    else:
        if len(weights_1) != n:
            raise RuntimeError("Number of weights and number of samples need to be the same.")
        a = weights_1 / np.sum(weights_1)
    if weights_2 is None:
        b = np.ones((m,)) / m
    else:
        if len(weights_2) != m:
            raise RuntimeError("Number of weights and number of samples need to be the same.")
        b = weights_2 / np.sum(weights_2)

    # loss matrix
    M = ot.dist(x1=samples_1, x2=samples_2)  # this returns squared distance!
    cost = ot.emd2(a, b, M, numItermax=num_iter_max, **kwargs)

    return np.sqrt(cost)
