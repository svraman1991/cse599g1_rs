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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_scores = scores[y[i]]
        sm_sum = 0.0

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_scores + 1

            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                sm_sum += np.exp(scores[j])

        dW[:, y[i]] -= X[i] - (X[i] * np.exp(correct_class_scores) / (sm_sum + np.exp(correct_class_scores)))

    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)

    dW /= num_train
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_features = X.shape[1]

    scores = X.dot(W)

    max_scores = np.max(scores, axis=1, keepdims=True)

    loss_i = -scores[range(num_train), y] + max_scores.flatten() + np.log(np.sum(np.exp(scores - max_scores), axis=1))
    loss = np.sum(loss_i)

    softmax_probabilities = np.exp(scores - max_scores) / np.sum(np.exp(scores - max_scores), axis=1, keepdims=True)

    correct_class = np.zeros_like(softmax_probabilities)
    correct_class[range(num_train), y] = 1

    dW = X.T.dot(softmax_probabilities - correct_class)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)

    dW /= num_train
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
