import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_accuracy(preds, target):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    omg = torch.sum(target,0)
    len_omg = len(torch.nonzero(omg))
    preds = torch.max(preds, 0)[1].float()
    target = torch.max(target, 0)[1].float()

    correct_prediction = torch.mul(omg, (preds == target).float())
    return torch.sum(correct_prediction)/len_omg


def rmse(logits, labels):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    omg = torch.sum(labels, 0).view(-1)
    len_omg = len(torch.nonzero(omg))

    pred_y = logits
    y = torch.max(labels, 0)[1].float() + 1.

    se = torch.sub(y, pred_y).pow_(2)
    rmse = torch.sqrt(torch.mean(se))

    return torch.sum(torch.mul(omg, rmse))/len_omg


def softmax_cross_entropy(input, target):
    """ computes average softmax cross entropy """

    omg = torch.sum(target,0).view(-1)
    len_omg = len(torch.nonzero(omg))
    target = torch.max(target, 0)[1].view(-1)

    loss = F.cross_entropy(input=input.view(-1,input.size(0)), target=target, reduction='none')
    loss = torch.sum(torch.mul(omg, loss))/len_omg

    return loss
