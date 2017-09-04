"""
A module with a package-wide random number generator,
used for weight initialization and seeding noise layers.
This can be replaced by a :class:`numpy.random.RandomState` instance with a
particular seed to facilitate reproducibility.

Note: When using cuDNN, the backward passes of convolutional and max-pooling
layers will introduce additional nondeterminism (for performance reasons).
For 2D convolutions, you can enforce a deterministic backward pass
implementation via the Theano flags ``dnn.conv.algo_bwd_filter=deterministic``
and ``dnn.conv.algo_bwd_data=deterministic``. Alternatively, you can disable
cuDNN completely with ``dnn.enabled=False``.
"""

import numpy as np


_rng = np.random


def get_rng():
    """Get the package-level random number generator.

    Returns
    -------
    :class:`numpy.random.RandomState` instance
        The :class:`numpy.random.RandomState` instance passed to the most
        recent call of :func:`set_rng`, or ``numpy.random`` if :func:`set_rng`
        has never been called.
    """
    return _rng


def set_rng(new_rng):
    """Set the package-level random number generator.

    Parameters
    ----------
    new_rng : ``numpy.random`` or a :class:`numpy.random.RandomState` instance
        The random number generator to use.
    """
    global _rng
    _rng = new_rng
