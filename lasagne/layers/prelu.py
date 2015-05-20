import numpy as np
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer

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

    untie_alphas : int, array or None
        When passed this argument creates a tensor of alphas rather than a
        single theano scalar. The argument dictates which dimension(s)
        the alphas are untied across. 

    Examples
    -------
        >>> from lasagne.layers import InputLayer, DenseLayer, PReLULayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50, nonlinearity=identity)
        >>> l1_prelu = PReLULayer(l1, alpha=0.35)
    """
    def __init__(self, incoming, alpha=0.25, untie_alphas=None, 
                 **kwargs):
        super(PReLULayer, self).__init__(incoming, **kwargs)
        self.untie_alphas = untie_alphas
        if untie_alphas == None:
            self.alpha_shape = ()
            self.alpha = self.create_param(alpha, self.alpha_shape, name="alpha")
        elif isinstance(untie_alphas, int) and untie_alphas >= 0 and untie_alphas < len(self.input_shape):
            self.alpha_shape = tuple([self.input_shape[untie_alphas]])
            self.alpha = self.create_param(np.zeros(self.alpha_shape)+alpha, self.alpha_shape, name="alpha")
        elif isinstance(untie_alphas, list) and np.min(untie_alphas) >= 0 and np.max(untie_alphas) < len(self.input_shape):
            self.alpha_shape = tuple([self.input_shape[x] for x in set(untie_alphas)])
            self.alpha = self.create_param(np.zeros(self.alpha_shape)+alpha, self.alpha_shape, name="alpha")
        else:
            raise Exception('The untie_alphas parameter is badly formed.')

    def get_params(self):
        return [self.alpha]

    def get_output_for(self, input, **kwargs):
        untie_dim = ['x']*input.ndim
        if isinstance(self.untie_alphas, int):
            untie_dim[self.untie_alphas] = 0
            untie_dim = tuple(untie_dim)
            return nonlinearities.LeakyRectify(self.alpha.dimshuffle(untie_dim))(input)
        elif isinstance(self.untie_alphas, list):
            for i, dim in enumerate(self.untie_alphas):
                untie_dim[dim] = i
            untie_dim = tuple(untie_dim)
            return nonlinearities.LeakyRectify(self.alpha.dimshuffle(untie_dim))(input)
        else:
            return nonlinearities.LeakyRectify(self.alpha)(input)
