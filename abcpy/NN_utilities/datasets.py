import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


# DATASETS DEFINITION FOR DISTANCE LEARNING:

class Similarities(Dataset):
    """A dataset class that considers a set of samples and pairwise similarities defined between them.
    Note that, for our application of computing distances, we are not interested in train/test split. """

    def __init__(self, samples, similarity_matrix, device):
        """
        Parameters:

        samples: n_samples x n_features
        similarity_matrix: n_samples x n_samples
        """
        if isinstance(samples, np.ndarray):
            self.samples = torch.from_numpy(samples.astype("float32")).to(device)
        else:
            self.samples = samples.to(device)
        if isinstance(similarity_matrix, np.ndarray):
            self.similarity_matrix = torch.from_numpy(similarity_matrix.astype("int")).to(device)
        else:
            self.similarity_matrix = similarity_matrix.to(device)

    def __getitem__(self, index):
        """Return the required sample along with the similarities of the sample with all the others."""
        return self.samples[index], self.similarity_matrix[index]

    def __len__(self):
        return self.samples.shape[0]


class SiameseSimilarities(Dataset):
    """
    This class defines a dataset returning pairs of similar and dissimilar samples. It has to be instantiated with a
    dataset of the class Similarities
    """

    def __init__(self, similarities_dataset, positive_weight=None):

        """If positive_weight=None, then for each sample we pick another random element to form a pair.
        If positive_weight is a number (in [0,1]), we will pick positive samples with that probability
        (if there are some)."""
        self.dataset = similarities_dataset
        self.positive_weight = positive_weight
        self.samples = similarities_dataset.samples
        self.similarity_matrix = similarities_dataset.similarity_matrix

    def __getitem__(self, index):
        """If self.positive_weight is None, or if the sample denoted by index has no similar elements, choose another
        random sample to build the pair. If instead self.positive_weight is a number, choose a similar element with
        that probability.
        """
        if self.positive_weight is None or (torch.sum(self.similarity_matrix[index]) < 2):
            # sample a new index different from the present one
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(range(self.samples.shape[0]))

            target = self.similarity_matrix[index, siamese_index]

        else:
            # pick positive target with probability self.positive_weight
            target = int(np.random.uniform() < self.positive_weight)
            if target:
                # sample a new index different from the present one
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(np.where(self.similarity_matrix[index].cpu())[0])
            else:
                # sample a new index different from the present one. This would not be necessary in theory,
                # as a sample is always similar to itself.
                # Leave this check anyway, to avoid problems in case the dataset is not perfectly defined.
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(np.where(self.similarity_matrix[index].cpu() == False)[0])

        return (self.samples[index], self.samples[siamese_index]), target

    def __len__(self):
        return self.samples.shape[0]


class TripletSimilarities(Dataset):
    """
    This class defines a dataset returning triplets of anchor, positive and negative samples.
    It has to be instantiated with a dataset of the class Similarities.
    """

    def __init__(self, similarities_dataset, ):
        self.dataset = similarities_dataset
        self.samples = similarities_dataset.samples
        self.similarity_matrix = similarities_dataset.similarity_matrix

    def __getitem__(self, index):
        # sample a new index different from the present one
        if torch.sum(self.similarity_matrix[index]) < 2:
            # then we pick a new sample that has at least one similar example
            warnings.warn("Sample {} in the dataset has no similar samples. \nIncrease the quantile defining the"
                          " similarity matrix to avoid such problems.\nExecution will continue taking another sample "
                          "instead of that as anchor.".format(index), RuntimeWarning)
            new_anchor = index
            while new_anchor == index:
                new_anchor = np.random.randint(0, self.dataset.__len__())
                # if this other sample does not have a similar one as well -> sample another one.
                if torch.sum(self.similarity_matrix[new_anchor]) < 2:
                    new_anchor = index
            index = new_anchor

        positive_index = index
        while positive_index == index:
            # this loops indefinitely if some sample has no other similar samples!
            positive_index = np.random.choice(np.where(self.similarity_matrix[index].cpu())[0])

            # sample a new index different from the present one. This would not be necessary in theory,
            # as a sample is always similar to itself.
        # Leave this check anyway, to avoid problems in case the dataset is not perfectly defined.
        negative_index = index
        while negative_index == index:
            negative_index = np.random.choice(np.where(self.similarity_matrix[index].cpu() == False)[0])

        return (self.samples[index], self.samples[positive_index], self.samples[negative_index]), []

    def __len__(self):
        return self.samples.shape[0]


# DATASET DEFINITION FOR SUFFICIENT STATS LEARNING:

class ParameterSimulationPairs(Dataset):
    """A dataset class that consists of pairs of parameters-simulation pairs, in which the data contains the
    simulations, with shape (n_samples, n_features), and targets contains the ground truth of the parameters,
    with shape (n_samples, 2). Note that n_features could also have more than one dimension here. """

    def __init__(self, simulations, parameters, device):
        """
        Parameters:

        simulations: (n_samples,  n_features)
        parameters: (n_samples, 2)
        """
        if simulations.shape[0] != parameters.shape[0]:
            raise RuntimeError("The number of simulations must be the same as the number of parameters.")

        if isinstance(simulations, np.ndarray):
            self.simulations = torch.from_numpy(simulations.astype("float32")).to(device)
        else:
            self.simulations = simulations.to(device)
        if isinstance(parameters, np.ndarray):
            self.parameters = torch.from_numpy(parameters.astype("float32")).to(device)
        else:
            self.parameters = parameters.to(device)

    def __getitem__(self, index):
        """Return the required sample along with the ground truth parameter."""
        return self.simulations[index], self.parameters[index]

    def __len__(self):
        return self.parameters.shape[0]
