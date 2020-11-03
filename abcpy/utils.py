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


def wass_dist(post_samples_1, post_samples_2, weights_post_1=None, weights_post_2=None):
    """Computes the Wasserstein 2 distance.

    post_samples_1 and post_post_samples_2 are 2 dimensional arrays: first dim is the number of samples, 2nd dim is the
    number of coordinates in the each sample.

    We allow to give weights to the posterior distribution. Leave weights_post_1 and weights_post_2 to None if your
    samples do not have weights. """

    n = post_samples_1.shape[0]

    if weights_post_1 is None:
        a = np.ones((n,)) / n
    else:
        a = weights_post_1 / np.sum(weights_post_1)
    if weights_post_2 is None:
        b = np.ones((n,)) / n
    else:
        b = weights_post_2 / np.sum(weights_post_2)

    # loss matrix
    M = ot.dist(x1=post_samples_1, x2=post_samples_2)  # this returns squared distance!
    G0 = ot.emd(a, b, M, log=True)

    # print('EMD cost:', G0[1].get('cost'))

    return G0
