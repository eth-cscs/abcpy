try:
    import torch
except ImportError:
    has_torch = False
else:
    has_torch = True

import logging
from functools import reduce
from operator import mul

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


def jacobian(input, output, diffable=True):
    '''
    Returns the Jacobian matrix (batch x in_size x out_size) of the function that produced the output evaluated at the
    input

    From https://github.com/mwcvitkovic/MASS-Learning/blob/master/models/utils.py

    Important: need to use diffable=True in order for the training routines based on these to work!

    '''
    assert len(output.shape) == 2
    assert input.shape[0] == output.shape[0]
    in_size = reduce(mul, list(input.shape[1:]), 1)
    if (input.sum() + output.sum()).item() in [np.nan, np.inf]:
        raise ValueError
    J = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)
    # they are able here to do the gradient computation one batch at a time, of course still considering only one output coordinate at a time
    for i in range(output.shape[1]):
        g = torch.zeros(output.shape).to(input)
        g[:, i] = 1
        if diffable:
            J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
        else:
            J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True)[0]
    J = J.reshape(output.shape[0], output.shape[1], in_size)
    return J.transpose(2, 1)


def jacobian_second_order(input, output, diffable=True):
    '''
    Returns the Jacobian matrix (batch x in_size x out_size) of the function that produced the output evaluated at the
    input, as well as
    the matrix of second derivatives of outputs with respect to inputs (batch x in_size x out_size)

    Adapted from https://github.com/mwcvitkovic/MASS-Learning/blob/master/models/utils.py

    Important: need to use diffable=True in order for the training routines based on these to work!
    '''
    assert len(output.shape) == 2
    assert input.shape[0] == output.shape[0]
    in_size = reduce(mul, list(input.shape[1:]), 1)
    if (input.sum() + output.sum()).item() in [np.nan, np.inf]:
        raise ValueError
    J = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)
    J2 = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)

    for i in range(output.shape[1]):
        g = torch.zeros(output.shape).to(input)
        g[:, i] = 1
        J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
    J = J.reshape(output.shape[0], output.shape[1], in_size)

    for i in range(output.shape[1]):
        for j in range(input.shape[1]):
            g = torch.zeros(J.shape).to(input)
            g[:, i, j] = 1
            if diffable:
                J2[:, i, j] = torch.autograd.grad(J, input, g, only_inputs=True, retain_graph=True, create_graph=True)[
                                  0][:, j]
            else:
                J2[:, i, j] = torch.autograd.grad(J, input, g, only_inputs=True, retain_graph=True)[0][:, j]

    J2 = J2.reshape(output.shape[0], output.shape[1], in_size)

    return J.transpose(2, 1), J2.transpose(2, 1)


def jacobian_hessian(input, output, diffable=True):
    '''
    Returns the Jacobian matrix (batch x in_size x out_size) of the function that produced the output evaluated at the
    input, as well as the Hessian matrix (batch x in_size x in_size x out_size).

    This takes slightly more than the jacobian_second_order routine.

    Adapted from https://github.com/mwcvitkovic/MASS-Learning/blob/master/models/utils.py

    Important: need to use diffable=True in order for the training routines based on these to work!
    '''
    assert len(output.shape) == 2
    assert input.shape[0] == output.shape[0]
    in_size = reduce(mul, list(input.shape[1:]), 1)
    if (input.sum() + output.sum()).item() in [np.nan, np.inf]:
        raise ValueError
    J = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)
    H = torch.zeros(list(output.shape) + list(input.shape[1:]) + list(input.shape[1:])).to(input)

    for i in range(output.shape[1]):
        g = torch.zeros(output.shape).to(input)
        g[:, i] = 1
        J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
    J = J.reshape(output.shape[0], output.shape[1], in_size)

    for i in range(output.shape[1]):
        for j in range(input.shape[1]):
            g = torch.zeros(J.shape).to(input)
            g[:, i, j] = 1
            if diffable:
                H[:, i, j] = torch.autograd.grad(J, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
            else:
                H[:, i, j] = torch.autograd.grad(J, input, g, only_inputs=True, retain_graph=True)[0]

    return J.transpose(2, 1), H.transpose(3, 1)


def set_requires_grad(net, value):
    for param in net.parameters():
        param.requires_grad = value
