import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).view(-1)
    #r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.mm(torch.mm(r_mat_inv_sqrt, mx), r_mat_inv_sqrt)

    # T
    #mx = torch.mm(mx, mx)
    return mx

def softmax_accuracy(preds, target):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    omg = torch.nonzero(torch.sum(target,0))
    preds = torch.stack([torch.max(preds[:, x, y], 0)[1] for (x,y) in omg], 0)
    target = torch.stack([torch.max(target[:,x,y], 0)[1] for (x,y) in omg], 0)

    correct_prediction = (preds == target).float()
    return torch.mean(correct_prediction)


def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable. :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    probs = tf.nn.softmax(logits)
    if class_values is None:
        scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        scores = class_values
        y = tf.gather(class_values, labels)

    pred_y = tf.reduce_sum(probs * scores, 1)

    diff = tf.subtract(y, pred_y)
    exp_rmse = tf.square(diff)
    exp_rmse = tf.cast(exp_rmse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(exp_rmse))


def rmse(logits, labels, class_values=None):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    if class_values is None:
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        y = tf.gather(class_values, labels)

    pred_y = logits

    diff = tf.subtract(y, pred_y)
    mse = tf.square(diff)
    mse = tf.cast(mse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(mse))


def softmax_cross_entropy(input, target):
    """ computes average softmax cross entropy """

    omg = torch.nonzero(torch.sum(target,0))
    input = torch.stack([input[:, x, y] for (x,y) in omg], 0)
    target = torch.stack([torch.max(target[:,x,y], 0)[1] for (x,y) in omg], 0)

    loss = F.cross_entropy(input=input, target=target)

    return loss
