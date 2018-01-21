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
    D = np.prod(x.shape[1:])
    out = np.dot(x.reshape(N, D), w) + b
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
    N, dims = x.shape[0], x.shape[1:]
    D, M = w.shape
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(N, D).T, dout)
    db = np.dot(np.ones((1, N)), dout)
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
    x = cache
    dout_dx = np.maximum(0, x)
    dout_dx[dout_dx > 0] = 1
    dx = np.multiply(dout_dx, dout)
    return dx

'''
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
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

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #######################################################################

        # Forward pass
        # Step 1 - shape of mu (D,)
        mu = 1 / float(N) * np.sum(x, axis=0)

        # Step 2 - shape of var (N,D)
        xmu = x - mu

        # Step 3 - shape of carre (N,D)
        carre = xmu ** 2

        # Step 4 - shape of var (D,)
        var = 1 / float(N) * np.sum(carre, axis=0)

        # Step 5 - Shape sqrtvar (D,)
        sqrtvar = np.sqrt(var + eps)

        # Step 6 - Shape invvar (D,)
        invvar = 1. / sqrtvar

        # Step 7 - Shape va2 (N,D)
        va2 = xmu * invvar

        # Step 8 - Shape va3 (N,D)
        va3 = gamma * va2

        # Step 9 - Shape out (N,D)
        out = va3 + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

        cache = (mu, xmu, carre, var, sqrtvar, invvar,
                 va2, va3, gamma, beta, x, bn_param)
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #######################################################################
        mu = running_mean
        var = running_var
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

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

    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    ##########################################################################
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    # Backprop Step 9
    dva3 = dout
    dbeta = np.sum(dout, axis=0)

    # Backprop step 8
    dva2 = gamma * dva3
    dgamma = np.sum(va2 * dva3, axis=0)

    # Backprop step 7
    dxmu = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)

    # Backprop step 6
    dsqrtvar = -1. / (sqrtvar ** 2) * dinvvar

    # Backprop step 5
    dvar = 0.5 * (var + eps) ** (-0.5) * dsqrtvar

    # Backprop step 4
    dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar

    # Backprop step 3
    dxmu += 2 * xmu * dcarre

    # Backprop step 2
    dx = dxmu
    dmu = - np.sum(dxmu, axis=0)

    # Basckprop step 1
    dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

    return dx, dgamma, dbeta

'''
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

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

    out, cache = None, None

    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.sum((x - sample_mean) ** 2, axis=0) / N
        # stabilize by adding epsilon
        x_hat = np.divide((x - sample_mean), np.sqrt((sample_var + eps)))
        out = gamma * x_hat + beta
        running_mean = momentum * running_mean + \
        (1 - momentum) * sample_mean
        running_var = momentum * running_var + \
        (1 - momentum) * sample_var
        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        cache = (x, sample_mean, sample_var, x_hat, gamma, beta, bn_param)
        #######################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #######################################################################
        #######################################################################
        #                             END OF YOUR CODE                              #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #######################################################################
        out = (x - running_mean) / np.sqrt((running_var + eps))
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache


def batchnorm_backward(dout, cache):

    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    ##########################################################################
    x, sample_mean, sample_var, x_hat, gamma, beta, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    # forward pass
    N, D = x.shape
    mu = x.sum(axis=0) / N
    xmu = x - mu
    sq = xmu ** 2
    var = sq.sum(axis=0) / N
    sqrtvar = np.sqrt(var + eps)
    ivar = 1. / sqrtvar
    xhat = xmu * ivar
    gammax = gamma * xhat
    out = gammax + beta

    # backward pass
    dbeta = dout.sum(axis=0)
    dgammax = dout
    dgamma = np.sum(dgammax * xhat, axis=0)
    dxhat = dgammax * gamma

    dxmu1 = dxhat * ivar
    divar = np.sum(dxhat * xmu, axis=0)
    dsqrtvar = - 1. / (sqrtvar ** 2) * divar
    dvar = 0.5 * dsqrtvar / sqrtvar
    dsq = dvar / N
    dxmu2 = 2. * xmu * dsq

    dmu = - np.sum(dxmu1 + dxmu2, axis=0)
    dx1 = dxmu1 + dxmu2
    dx2 = dmu / N
    dx = dx1 + dx2

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
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
    x, sample_mean, sample_var, x_hat, gamma, beta, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = x.shape

    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = dout.sum(axis=0)
    a = np.divide(gamma, (N * np.sqrt((sample_var + eps))))
    b = dout.sum(axis=0)
    C = np.divide((x - sample_mean), ((sample_var + eps)))
    d = np.sum(dout * (x - sample_mean), axis=0)
    dx = a * (N * dout - b - C * d)

    # dx, dgamma, dbeta = None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    ##########################################################################
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
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    if mode == 'train':
        #######################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                            END OF YOUR CODE                             #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.       #
        #######################################################################
        mask = None
        out = x
        #######################################################################
        #                            END OF YOUR CODE                             #
        #######################################################################

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
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                            END OF YOUR CODE                             #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

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
    stride, pad = conv_param['stride'], conv_param['pad']

    if (H + 2 * pad - HH) % stride is not 0 or (W + 2 * pad - WW) % stride is not 0:
        raise ValueError("Dimensions do not match!")

    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    img_pad = np.zeros((C, H + 2 * pad, W + 2 * pad))
    out = np.zeros((N, F, H_out, W_out))
    for n in xrange(N):
        for c in xrange(C):
            img_pad[c, :, :] = np.pad(x[n, c, :, :], pad, 'constant', constant_values=0)
        for i in xrange(W_out):
            for j in xrange(H_out):
                for f in xrange(F):
                    out[n, f, j, i] = np.sum(img_pad[:, j * stride:j * stride + HH,
                                                 i * stride:i * stride + WW] *
                                         w[f, :, :, :]) + b[f]


    ##########################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    ##########################################################################

    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################
    cache = (x, w, b, conv_param)
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
    ##########################################################################
    # TODO: Implement the convolutional backward pass.                          #
    ##########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    N, F, H_out, W_out = dout.shape

    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    img_pad = np.zeros((C, H + 2 * pad, W + 2 * pad))
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx = np.zeros_like(x)
    for n in xrange(N):
        dimg_pad = np.zeros((C, H + 2 * pad, W + 2 * pad))
        for c in xrange(C):
            img_pad[c, :, :] = np.pad(x[n, c, :, :], pad, 'constant', constant_values=0)
        for i in xrange(W_out):
            for j in xrange(H_out):
                for f in xrange(F):
                    dw[f, :, :, :] += img_pad[:, j * stride : j * stride + HH,
                                                 i * stride : i * stride + WW] * \
                                      dout[n, f, j, i]
                    db[f] += dout[n, f, j, i]
                    dimg_pad[:, j * stride : j * stride + HH,
                             i * stride : i * stride + WW] += \
                             w[f, :, :, :] * dout[n, f, j, i]
        dx[n, :, :, :] = dimg_pad[:, pad:pad + H, pad:pad + W]


    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################
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
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    if (H - pool_height) % stride is not 0 or (W - pool_width) % stride is not 0:
        raise ValueError("Dimensions do not match!")
    H_out = 1 + (H - pool_height) / stride
    W_out = 1 + (W - pool_width) / stride
    out = np.zeros((N, C, H_out, W_out))
    ##########################################################################
    # TODO: Implement the max pooling forward pass                              #
    ##########################################################################
    for n in xrange(N):
        for c in xrange(C):
            for i in xrange(W_out):
                for j in xrange(H_out):
                    out[n, c, j, i] = \
                        np.max(x[n, c, j * stride:j * stride + pool_height,
                                 i * stride:i * stride + pool_height])

    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################
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
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    if (H - pool_height) % stride is not 0 or (W - pool_width) % stride is not 0:
        raise ValueError("Dimensions do not match!")
    H_out = 1 + (H - pool_height) / stride
    W_out = 1 + (W - pool_width) / stride
    dx = np.zeros_like(x)
    for n in xrange(N):
        for c in xrange(C):
            for i in xrange(W_out):
                for j in xrange(H_out):
                    window = x[n, c, j * stride:j * stride + pool_height,
                        i * stride:i * stride + pool_height]
                    mask = np.zeros_like(window)
                    max_ind = np.unravel_index(window.argmax(), window.shape)
                    mask[max_ind] = 1
                    dx[n, c, j * stride:j * stride + pool_height,
                        i * stride:i * stride + pool_height] = \
                        dout[n, c, j, i] * mask

    ##########################################################################
    # TODO: Implement the max pooling backward pass                             #
    ##########################################################################
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################
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
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, C, H, W = x.shape
    sample_size = N * H * W
    running_mean = bn_param.get('running_mean', np.zeros((1, C, 1, 1), dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros((1, C, 1, 1), dtype=x.dtype))
    out, cache = None, None

    if mode == 'train':
        sample_mean = np.mean(x, keepdims=True, axis=(0, 2, 3))
        sample_var = np.mean((x - sample_mean) ** 2, keepdims=True, axis=(0, 2, 3))

        # stabilize by adding epsilon
        x_hat = np.divide((x - sample_mean), np.sqrt((sample_var + eps)))
        out = gamma.reshape((1, C, 1, 1)) * x_hat + beta.reshape((1, C, 1, 1))
        running_mean = momentum * running_mean + \
        (1 - momentum) * sample_mean
        running_var = momentum * running_var + \
        (1 - momentum) * sample_var
        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        cache = (x, sample_mean, sample_var, x_hat, gamma, beta, bn_param)
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #######################################################################
        out = ((x - running_mean.reshape((1, C, 1, 1))) /
               np.sqrt((running_var.reshape((1, C, 1, 1)) + eps)))
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


    ##########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    ##########################################################################
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

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
    x, sample_mean, sample_var, x_hat, gamma, beta, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, C, H, W = x.shape
    sample_size = N * H * W

    dgamma = np.sum(x_hat * dout, axis=(0, 2, 3))
    dbeta = dout.sum(axis=(0, 2, 3))
    a = np.divide(gamma.reshape((1, C, 1, 1)),
                  (sample_size * np.sqrt((sample_var + eps))))
    b = dout.sum(keepdims=True, axis=(0, 2, 3))
    C = np.divide((x - sample_mean), ((sample_var + eps)))
    d = np.sum(dout * (x - sample_mean), keepdims=True, axis=(0, 2, 3))
    dx = a * (sample_size * dout - b - C * d)
    print(dx.shape)

    # dx, dgamma, dbeta = None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    ##########################################################################
    return dx, dgamma, dbeta

    ##########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    ##########################################################################
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
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
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
