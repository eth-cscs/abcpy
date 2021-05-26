import logging

import torch
from tqdm import tqdm


def fit(train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, val_loader=None, early_stopping=False,
        epochs_early_stopping_interval=1, start_epoch_early_stopping=10, start_epoch_training=0, use_tqdm=True):
    """
    Basic function to train a neural network given a train_loader, a loss function and an optimizer.

    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Adapted from https://github.com/adambielski/siamese-triplet
    """

    logger = logging.getLogger("NN Trainer")
    train_loss_list = []
    if val_loader is not None:
        test_loss_list = []
        if early_stopping:
            early_stopping_loss_list = []  # list of losses used for early stopping
    else:
        test_loss_list = None
    if early_stopping and val_loader is None:
        raise RuntimeError("You cannot perform early stopping if a validation loader is not provided to the training "
                           "routine")

    for epoch in range(0, start_epoch_training):
        scheduler.step()

    for epoch in tqdm(range(start_epoch_training, n_epochs), disable=not use_tqdm):
        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda)
        train_loss_list.append(train_loss)

        logger.debug('Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

        # Validation stage
        if val_loader is not None:
            val_loss = test_epoch(val_loader, model, loss_fn, cuda)
            test_loss_list.append(val_loss)

            logger.debug('Epoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss))

            # early stopping:
            if early_stopping and (epoch + 1) % epochs_early_stopping_interval == 0:
                early_stopping_loss_list.append(val_loss)  # save the previous validation loss. It is actually
                # we need to have at least two saved test losses for performing early stopping (in which case we know
                # we have saved the previous state_dict as well).
                if epoch + 1 >= start_epoch_early_stopping and len(early_stopping_loss_list) > 1:
                    if early_stopping_loss_list[-1] > early_stopping_loss_list[-2]:
                        logger.info("Training has been early stopped at epoch {}.".format(epoch + 1))
                        # reload the previous state dict:
                        model.load_state_dict(net_state_dict)
                        break  # stop training
                # if we did not stop: update the state dict to the next value
                net_state_dict = model.state_dict()

        scheduler.step()

    return train_loss_list, test_loss_list

def train_epoch(train_loader, model, loss_fn, optimizer, cuda):
    """Function implementing the training in one epoch.

    Adapted from https://github.com/adambielski/siamese-triplet
    """
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / (batch_idx + 1)  # divide here by the number of elements in the batch.


def test_epoch(val_loader, model, loss_fn, cuda):
    """Function implementing the computation of the validation error, in batches.

    Adapted from https://github.com/adambielski/siamese-triplet
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

    return val_loss / (batch_idx + 1)  # divide here by the number of elements in the batch.
