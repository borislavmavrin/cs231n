import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        pos_margin_counter = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                pos_margin_counter += 1
                dW[:, j] += X[i]
        dW[:, y[i]] += -pos_margin_counter * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dim = X.shape[1]

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    scores = X.dot(W)
    diff = scores - scores[range(num_train), y].reshape((num_train, 1)) + 1
    margins = np.maximum(0, diff)
    margins[range(num_train), y] = 0
    loss = np.sum(margins)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    # convert margines to 0/1
    margins[margins > 0] = 1
    margins[range(num_train), y] = (
        -margins.dot(np.ones((num_classes, 1))).
        reshape(num_train,)
        )
    dW = X.T.dot(margins)
    dW /= num_train
    return loss, dW
