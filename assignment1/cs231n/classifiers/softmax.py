import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    num_train = X.shape[0]
    num_classes = W.shape[1]
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    for i in range(num_train):
        scores = X[i].dot(W)
        adj_const = np.max(scores)
        scores = scores - adj_const  # for numerical stability
        scores = np.exp(scores)
        score = scores[y[i]] / np.sum(scores)
        loss += - np.log(score)
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] -= X[i]
            if j != y[i]:
                weight = scores[j] / np.sum(scores)
                dW[:, j] += weight * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    scores = X.dot(W)
    adj_consts = scores.max(axis=1)
    # for numerical stability
    scores = scores - adj_consts.reshape(num_train, 1)
    scores = np.exp(scores)
    score = scores[range(num_train), y] / np.sum(scores, axis=1)
    score = - np.log(score)
    loss = np.sum(score)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    weights = scores
    weights = weights / weights.sum(axis=1).reshape(num_train, 1)
    weights[range(num_train), y] += -1
    dW = X.T.dot(weights)
    dW /= num_train
    dW += reg * W

    return loss, dW
