import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    # summing up exponentials
    sum_exp = 0
    for j in xrange(num_classes):
      sum_exp += np.exp(scores[j])
    for j in xrange(num_classes):
      dW[:, j] += (np.exp(scores[j])/sum_exp) * X[i]
    loss -= np.log(np.exp(scores[y[i]])/ sum_exp)
    dW[:, y[i]] -= X[i]

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Add regularization gradient
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  scores = X.dot(W)

  scores_exp = np.exp(scores)
  correct_class_score = scores[range(num_train), y]
  sum_exp_scores = np.sum(scores_exp, axis=1, keepdims=True)
  loss += -np.sum(correct_class_score) + np.sum(np.log(sum_exp_scores))
  loss /= num_train

  mask = np.zeros_like(scores)
  mask[range(num_train), y] = 1
  aux = scores_exp/sum_exp_scores
  dW += (X.T).dot(aux - mask)
  # divide by the number of training examples
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Add regularization gradient
  dW += 2 * reg * W

  return loss, dW

