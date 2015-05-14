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

    alpha : double or None
        The initial "leakiness" of the rectifier which is then learned.
        If argument not provided, initial leakiness will be set to 0.25.

    unique_chan_param : boolean or None
        When true, this argument creates a vector of alphas rather than a
        single theano scalar. Each scalar is applied to a separate channel
        of the previous layer's output. Should only be used if the 
        previous layer is a convolution. If argument not provided, it will
        be set to False.

    Usage
    -------
        >>> from lasagne.layers import InputLayer, DenseLayer, PReLULayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50, nonlinearity=identity)
        >>> l1_prelu = PReLULayer(l1, alpha=0.5)
    """
    def __init__(self, incoming, alpha=0.25, unique_chan_param=False, 
                 **kwargs):
        super(PReLULayer, self).__init__(incoming, **kwargs)
        self.unique_chan_param = unique_chan_param
        if not unique_chan_param:
            self.alpha = self.create_param(alpha, (), name="alpha")
        else:
            self.alpha = self.create_param(np.zeros(incoming.shape[1])+alpha, (incoming.shape[1]), name="W")

    def get_params(self):
        return [self.alpha]
        
    def get_output_for(self, input, **kwargs):
        if input.ndim == 2 or (input.ndim ==4 and not self.unique_chan_param):
            return nonlinearities.LeakyRectify(self.alpha)(input)
        elif input.ndim == 4 and self.unique_chan_param: 
            return nonlinearities.LeakyRectify(self.alpha.dimshuffle('x',0,'x','x'))(input)
        else:
            raise Exception('Incorrect number of dimensions.')
