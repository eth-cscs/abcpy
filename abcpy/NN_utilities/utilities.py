try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True

import logging

import numpy as np


def dist2(x, y):
    """Compute the square of the Euclidean distance between 2 arrays of same length"""
    return np.dot(x - y, x - y)


def compute_similarity_matrix(target, quantile=0.1, return_pairwise_distances=False):
    """Compute the similarity matrix between some values given a given quantile of the Euclidean distances.

    If return_pairwise_distances is True, it also returns a matrix with the pairwise distances with every distance."""

    logger = logging.getLogger("Compute_similarity_matrix")

    n_samples = target.shape[0]

    pairwise_distances = np.zeros([n_samples] * 2)

    for i in range(n_samples):
        for j in range(n_samples):
            pairwise_distances[i, j] = dist2(target[i], target[j])

    q = np.quantile(pairwise_distances[~np.eye(n_samples, dtype=bool)].reshape(-1), quantile)

    similarity_set = pairwise_distances < q

    logger.info("Fraction of similar pairs (epurated by self-similarity): {}".format(
        (np.sum(similarity_set) - n_samples) / n_samples ** 2))

    if (np.sum(similarity_set) - n_samples) / n_samples ** 2 == 0:
        raise RuntimeError("The chosen quantile is too small, as there are no similar samples according to the "
                           "corresponding threshold.\nPlease increase the quantile.")

    return (similarity_set, pairwise_distances) if return_pairwise_distances else similarity_set


def save_net(path, net):
    """Function to save the Pytorch state_dict of a network to a file."""
    torch.save(net.state_dict(), path)


def load_net(path, network_class, *network_args, **network_kwargs):
    """Function to load a network from a Pytorch state_dict, given the corresponding network_class."""
    net = network_class(*network_args, **network_kwargs)
    net.load_state_dict(torch.load(path))
    return net.eval()  # call the network to eval model. Needed with batch normalization and dropout layers.
