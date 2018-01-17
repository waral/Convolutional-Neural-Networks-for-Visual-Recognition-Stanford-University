import numpy as np
from random import shuffle
from past.builtins import xrange

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
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train #same from gradient

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Add regularization gradient
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  # Loss
  scores = X.dot(W)
  correct_class_scores_init = scores[range(num_train), y]
  correct_class_scores = np.tile(correct_class_scores_init, (10, 1))
  margin = np.maximum(0, scores - correct_class_scores.T + 1)
  # not to count stuff twice
  margin[range(num_train), y] = 0
  loss = np.sum(margin)/num_train

  # Derivative
  margin_mask = np.zeros(margin.shape)
  margin_mask[margin > 0] = 1
  #substract what we counted too much
  sub = np.sum(margin_mask, axis=1)
  margin_mask[range(num_train), y] = -sub
  dW += (X.T).dot(margin_mask)
  #divide by the number of training examples
  dW /=num_train
  # Add regularization gradient
  dW += 2 * reg * W

  return loss, dW
