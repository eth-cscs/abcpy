from tqdm import tqdm
import logging


def fit(train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, start_epoch=0):
    """
    Basic function to train a neural network given a train_loader, a loss function and an optimizer.

    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss

    Adapted from https://github.com/adambielski/siamese-triplet
    """

    logger = logging.getLogger("NN Trainer")

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in tqdm(range(start_epoch, n_epochs)):
        scheduler.step()

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda)

        logger.debug('Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))


def train_epoch(train_loader, model, loss_fn, optimizer, cuda):
    """Function implementing the training in one epoch.

    Adapted from https://github.com/adambielski/siamese-triplet
    """
    model.train()
    losses = []
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
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        losses = []

    total_loss /= (batch_idx + 1)
    return total_loss
