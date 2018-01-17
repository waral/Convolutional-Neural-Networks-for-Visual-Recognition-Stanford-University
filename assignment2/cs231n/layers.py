from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    out = (x.reshape(N, -1)).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]

    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)

    dw = (x.reshape(N, -1).T).dot(dout)

    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout
    dx[x < 0] = 0

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache, xhat, mu, sigma2 = None, None, None, None, None
    if mode == 'train':
        mu = np.mean(x, axis=0)
        sigma2 = np.var(x, axis=0)
        xhat = (x - mu) / np.sqrt(sigma2 + eps)
        out = gamma * xhat + beta

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * sigma2

    elif mode == 'test':
        xhat = (x - running_mean)/np.sqrt(running_var + eps)
        out = gamma * xhat + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    cache = (x, xhat, mu, sigma2, eps, N, gamma, beta)
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    dx, dgamma, dbeta = None, None, None
    x, xhat, mu, sigma2, eps, N, gamma, beta = cache

    dxhat = gamma * dout

    dsigma2 = (-0.5) * ((sigma2 + eps)**(-1.5)) * np.sum((x - mu) * dxhat, axis=0)

    dmu = (-1) * np.sum(((sigma2 + eps)**(-0.5)) * dxhat, axis=0)
    dmu += (-2/float(N)) * dsigma2 * np.sum(x - mu, axis=0)

    dx = ((sigma2 + eps)**(-0.5)) * dxhat
    dx += (2/float(N)) * dsigma2 * (x - mu)
    dx += (1/float(N)) * np.ones((dx.shape)) * dmu

    dgamma = np.sum(xhat * dout, axis=0)

    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    p = dropout_param['p']
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    P, S = conv_param['pad'], conv_param['stride']

    assert (H + 2 * P - HH) % S == 0, 'height does not work'
    assert (W + 2 * P - WW) % S == 0, 'width does not work'

    H_new = (H + 2 * P - HH) // S + 1
    W_new = (W + 2 * P - WW) // S + 1
    out = np.zeros((N, F, H_new, W_new))
    x_new = np.lib.pad(x, ((0, 0), (0, 0), (P, P), (P, P)),  'constant',
                       constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

    for ex in range(N):
        for fi in range(F):
            for he in range(H_new):
                for wi in range(W_new):
                    out[ex, fi, he, wi] = \
                        np.sum(x_new[ex, :, (S * he):(S * he + HH), (S * wi):(S * wi + WW)] * w[fi, :, :, :]) + b[fi]

    cache = (x, x_new, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x, x_new, w, b, conv_param = cache
    P, S = conv_param['pad'], conv_param['stride']

    N, C, H, W = x.shape
    N, F, H_new, W_new = dout.shape
    F, C, HH, WW = w.shape

    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    dx_new = np.zeros_like(x_new)
    db = np.zeros_like(b)

    for ex in range(N):
        for fi in range(F):
            for he in range(H_new):
                for wi in range(W_new):
                    db[fi] += dout[ex, fi, he, wi]
                    for hh in range(HH):
                        for ww in range(WW):
                            dw[fi, :, hh, ww] += x_new[ex, :, (S * he + hh), (S * wi + ww)] * dout[ex, fi, he, wi]
                            dx_new[ex, :, (S * he + hh), (S * wi + ww)] += w[fi, :, hh, ww] * dout[ex, fi, he, wi]

    dx = dx_new[:, :, P:(P + H), P:(P + W)]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    HH, WW, S = pool_param['pool_height'], pool_param['pool_width'],\
                                           pool_param['stride']
    N, C, H, W = x.shape
    assert (H - HH) % S == 0, 'height does not work'
    assert (W - WW) % S == 0, 'width does not work'

    H_new = (H - HH) // S + 1
    W_new = (W - WW) // S + 1
    out = np.zeros((N, C, H_new, W_new))

    for ex in range(N):
        for ca in range(C):
            for he in range(H_new):
                for wi in range(W_new):
                    out[ex, ca, he, wi] = np.amax(x[ex, ca, (S * he): (S * he + HH), (S * wi): (S * wi + WW)])

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    dx = np.zeros_like(x)
    HH, WW, S = pool_param['pool_height'], pool_param['pool_width'], \
                pool_param['stride']
    N, C, H, W = x.shape
    H_new = (H - HH) // S + 1
    W_new = (W - WW) // S + 1

    for ex in range(N):
        for ca in range(C):
            for he in range(H_new):
                for wi in range(W_new):
                    pool_mat = x[ex, ca, (S * he): (S * he + HH), (S * wi): (S * wi + WW)]
                    index1, index2 = np.unravel_index(pool_mat.argmax(), pool_mat.shape)
                    dx[ex, ca, S * he + index1, S * wi + index2] += dout[ex, ca, he, wi]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    x_new = np.reshape(np.swapaxes(x, 1, 3), (-1, C))
    out_new, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = np.swapaxes(np.reshape(out_new, (N, W, H, C)), 1, 3)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """

    N, C, H, W = dout.shape
    dout_new = np.reshape(np.swapaxes(dout, 1, 3), (-1, C))
    dx_new, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = np.swapaxes(np.reshape(dx_new, (N, W, H, C)), 1, 3)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
