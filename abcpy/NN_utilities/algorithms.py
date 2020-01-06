try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torch.utils.data import Dataset
    from abcpy.NN_utilities.datasets import Similarities, SiameseSimilarities, TripletSimilarities, \
        ParameterSimulationPairs
    from abcpy.NN_utilities.losses import ContrastiveLoss, TripletLoss
    from abcpy.NN_utilities.networks import SiameseNet, TripletNet
    from abcpy.NN_utilities.trainer import fit
except ModuleNotFoundError:
    has_torch = False
else:
    has_torch = True


def contrastive_training(samples, similarity_set, embedding_net, cuda, batch_size=16, n_epochs=200,
                         positive_weight=None, load_all_data_GPU=False, margin=1., lr=None, optimizer=None,
                         scheduler=None, start_epoch=0, verbose=False, optimizer_kwargs={}, scheduler_kwargs={},
                         loader_kwargs={}):
    """ Implements the algorithm for the contrastive distance learning training of a neural network; need to be
     provided with a set of samples and the corresponding similarity matrix"""

    # If the dataset is small enough, we can speed up training by loading all on the GPU at beginning, by using
    # load_all_data_GPU=True. It may crash if the dataset is too large. Note that in some cases using only CPU may still
    # be quicker.

    # Do all the setups

    # need to use the Similarities and SiameseSimilarities datasets

    similarities_dataset = Similarities(samples, similarity_set, "cuda" if cuda and load_all_data_GPU else "cpu")
    pairs_dataset = SiameseSimilarities(similarities_dataset, positive_weight=positive_weight)

    if cuda:
        if load_all_data_GPU:
            loader_kwargs_2 = {'num_workers': 0, 'pin_memory': False}
        else:
            loader_kwargs_2 = {'num_workers': 1, 'pin_memory': True}
    else:
        loader_kwargs_2 = {}

    pairs_train_loader = torch.utils.data.DataLoader(pairs_dataset, batch_size=batch_size, shuffle=True,
                                                     **loader_kwargs, **loader_kwargs_2)

    model_contrastive = SiameseNet(embedding_net)

    if cuda:
        model_contrastive.cuda()
    loss_fn = ContrastiveLoss(margin)

    if lr is None:
        lr = 1e-3

    if optimizer is None:  # default value
        optimizer = optim.Adam(embedding_net.parameters(), lr=lr, **optimizer_kwargs)
    else:
        optimizer = optimizer(embedding_net.parameters(), lr=lr, **optimizer_kwargs)

    if scheduler is None:  # default value, i.e. a dummy scheduler
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=1, last_epoch=-1)
    else:
        scheduler = scheduler(optimizer, **scheduler_kwargs)

    # now train:
    fit(pairs_train_loader, model_contrastive, loss_fn, optimizer, scheduler, n_epochs, cuda, start_epoch=start_epoch)

    return embedding_net


def triplet_training(samples, similarity_set, embedding_net, cuda, batch_size=16, n_epochs=400,
                     load_all_data_GPU=False, margin=1., lr=None, optimizer=None, scheduler=None, start_epoch=0,
                     verbose=False, optimizer_kwargs={}, scheduler_kwargs={}, loader_kwargs={}):
    """ Implements the algorithm for the triplet distance learning training of a neural network; need to be
     provided with a set of samples and the corresponding similarity matrix"""

    # If the dataset is small enough, we can speed up training by loading all on the GPU at beginning, by using
    # load_all_data_GPU=True. It may crash if the dataset is too large. Note that in some cases using only CPU may still
    # be quicker.
    # Do all the setups

    # need to use the Similarities and TripletSimilarities datasets

    similarities_dataset = Similarities(samples, similarity_set, "cuda" if cuda and load_all_data_GPU else "cpu")
    triplets_dataset = TripletSimilarities(similarities_dataset)

    if cuda:
        if load_all_data_GPU:
            loader_kwargs_2 = {'num_workers': 0, 'pin_memory': False}
        else:
            loader_kwargs_2 = {'num_workers': 1, 'pin_memory': True}
    else:
        loader_kwargs_2 = {}

    triplets_train_loader = torch.utils.data.DataLoader(triplets_dataset, batch_size=batch_size, shuffle=True,
                                                        **loader_kwargs, **loader_kwargs_2)

    model_triplet = TripletNet(embedding_net)

    if cuda:
        model_triplet.cuda()
    loss_fn = TripletLoss(margin)

    if lr is None:
        lr = 1e-3

    if optimizer is None:  # default value
        optimizer = optim.Adam(embedding_net.parameters(), lr=lr, **optimizer_kwargs)
    else:
        optimizer = optimizer(embedding_net.parameters(), lr=lr, **optimizer_kwargs)

    if scheduler is None:  # default value, i.e. a dummy scheduler
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=1, last_epoch=-1)
    else:
        scheduler = scheduler(optimizer, **scheduler_kwargs)

    # now train:
    fit(triplets_train_loader, model_triplet, loss_fn, optimizer, scheduler, n_epochs, cuda, start_epoch=start_epoch)

    return embedding_net


def FP_nn_training(samples, target, embedding_net, cuda, batch_size=1, n_epochs=50, load_all_data_GPU=False,
                   lr=1e-3, optimizer=None, scheduler=None, start_epoch=0, verbose=False, optimizer_kwargs={},
                   scheduler_kwargs={}, loader_kwargs={}):
    """ Implements the algorithm for the training of a neural network based on regressing the values of the parameters
    on the corresponding simulation outcomes; it is effectively a training with a mean squared error loss. Needs to be
    provided with a set of samples and the corresponding parameters that generated the samples. Note that in this case
    the network has to have same output size as the number of parameters, as the learned summary statistic will have the
    same dimension as the parameter."""

    # If the dataset is small enough, we can speed up training by loading all on the GPU at beginning, by using
    # load_all_data_GPU=True. It may crash if the dataset is too large. Note that in some cases using only CPU may still
    # be quicker.

    # Do all the setups

    dataset_FP_nn = ParameterSimulationPairs(samples, target, "cuda" if cuda and load_all_data_GPU else "cpu")

    if cuda:
        if load_all_data_GPU:
            loader_kwargs_2 = {'num_workers': 0, 'pin_memory': False}
        else:
            loader_kwargs_2 = {'num_workers': 1, 'pin_memory': True}
    else:
        loader_kwargs_2 = {}

    data_loader_FP_nn = torch.utils.data.DataLoader(dataset_FP_nn, batch_size=batch_size, shuffle=True, **loader_kwargs,
                                                    **loader_kwargs_2)

    if cuda:
        embedding_net.cuda()
    loss_fn = nn.MSELoss(reduction="mean")

    if optimizer is None:  # default value
        optimizer = optim.Adam(embedding_net.parameters(), lr=lr, **optimizer_kwargs)
    else:
        optimizer = optimizer(embedding_net.parameters(), lr=lr, **optimizer_kwargs)

    if scheduler is None:  # default value, i.e. a dummy scheduler
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=1, last_epoch=-1)
    else:
        scheduler = scheduler(optimizer, **scheduler_kwargs)

    # now train:
    fit(data_loader_FP_nn, embedding_net, loss_fn, optimizer, scheduler, n_epochs, cuda, start_epoch=start_epoch)

    return embedding_net
