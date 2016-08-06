import math

from theano import tensor as T

c = - .5 * math.log(2 * math.pi)


def log_normal(x, mean, std, eps=.0):
    """Compute density of N(mean,std^2) at point x

    Parameters
    ----------
    x : Theano tensor
    mean : Theano tensor
    std : Theano tensor
    eps : float

    Returns
    -------
    Theano tensor
        pointwise density of N(mean, std^2) at x
    """
    std += eps
    return c - T.log(T.abs_(std)) - (x - mean) ** 2 / (2 * std ** 2)


def log_normal3(x, mean, rho, eps=.0):
    """Compute density of N(mean,(log(1+exp(rho)))^2) at point x

    Parameters
    ----------
    x : Theano tensor
    mean : Theano tensor
    rho : Theano tensor
    eps : float

    Returns
    -------
    Theano tensor
        pointwise density of N(mean,(log(1+exp(rho)))^2) at x
    """
    std = T.log1p(T.exp(rho))
    return log_normal(x, mean, std, eps)
