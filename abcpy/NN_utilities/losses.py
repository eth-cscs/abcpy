import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise.

    Code from https://github.com/adambielski/siamese-triplet"""

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample.

    Code from https://github.com/adambielski/siamese-triplet"""

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def Fisher_divergence_loss(first_der_t, second_der_t, eta, lam=0):
    """lam is the regularization parameter of the Kingma & LeCun (2010) regularization"""
    inner_prod_second_der_eta = torch.bmm(second_der_t, eta.unsqueeze(-1))  # this is used twice

    if lam == 0:
        return sum(
            (0.5 * torch.bmm(first_der_t, eta.unsqueeze(-1)) ** 2 + inner_prod_second_der_eta).view(-1))
    else:
        return sum(
            (0.5 * torch.bmm(first_der_t, eta.unsqueeze(-1)) ** 2 +
             inner_prod_second_der_eta + lam * inner_prod_second_der_eta ** 2).view(-1))


def Fisher_divergence_loss_with_c_x(first_der_t, second_der_t, eta, lam=0):
    # this enables to use the term c(x) in the approximating family, ie a term that depends only on x and not on theta.
    new_eta = torch.cat((eta, torch.ones(eta.shape[0], 1).to(eta)),
                        dim=1)  # the one tensor need to be on same device as eta.
    # then call the other loss function with this new_eta:
    return Fisher_divergence_loss(first_der_t, second_der_t, new_eta, lam=lam)
