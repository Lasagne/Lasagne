import numpy as np
import theano.tensor as T
import theano

from .. import init
from .. import nonlinearities

from .base import Layer
from .. import utils

__all__ = [
    "PReLULayer"
]

class PReLULayer(Layer):
    """
    A layer used to implement Parametric Rectified Linear Units (PReLU).
    This layer simply applies the parametric nonlinearity to the previous
    layer's output. 

    Parameters
    -----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    alpha : float or None
        The initial "leakiness" of the rectifier which is then learned.
        If argument not provided, initial leakiness will be set to 0.25.

    untie_alphas : int, iterable, or None
        When passed this argument creates a tensor of alphas rather than a
        single theano scalar. The argument dictates which dimension(s)
        the alphas are untied across. 

    Examples
    -------
        >>> from lasagne.layers import InputLayer, DenseLayer, PReLULayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50, nonlinearity=None)
        >>> l1_prelu = PReLULayer(l1, alpha=0.35)
    """
    def __init__(self, incoming, alpha=0.25, untie_alphas=None, 
                 **kwargs):
        super(PReLULayer, self).__init__(incoming, **kwargs)
        if untie_alphas is None:
            alpha_shape = ()
        else:
            untie_alphas = [untie_alphas] if isinstance(untie_alphas, int) else sorted(set(untie_alphas))
            alpha_shape = tuple([self.input_shape[axis] for axis in untie_alphas])
        alpha = np.zeros(alpha_shape, dtype=theano.config.floatX) + alpha
        self.alpha = self.add_param(utils.floatX(alpha), alpha_shape, name='alpha', regularizable=False)
        self.untie_alphas = untie_alphas

    def get_output_for(self, input, **kwargs):
        if self.untie_alphas is None:
            alpha = self.alpha
        else:
            pattern = ['x'] * input.ndim
            for idx, dim in enumerate(self.untie_alphas):
                pattern[dim] = idx
            alpha = self.alpha.dimshuffle(pattern)
        return nonlinearities.LeakyRectify(alpha)(input)
