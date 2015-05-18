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

    untie_alphas : int or None
        When passed this argument creates a vector of alphas rather than a
        single theano scalar. The argument dictates which dimension
        the alphas are shared across. 

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
            self.alpha = self.create_param(alpha, (), name="alpha")
        else:
            self.alpha = self.create_param(np.zeros(incoming.shape[1])+alpha, (incoming.shape[1]), name="W")

    def get_params(self):
        return [self.alpha]
        
    def get_output_for(self, input, **kwargs):
        if self.untie_alphas:
            untie_dim = ['x']*input.ndim
            untie_dim[self.untie_alphas] = 0
            untie_dim = tuple(untie_dim)
            return nonlinearities.LeakyRectify(self.alpha.dimshuffle(untie_dim))(input)
        else:
            return nonlinearities.LeakyRectify(self.alpha)(input)
        '''
        if input.ndim == 2: 
            if self.untie_alphas:
                return nonlinearities.LeakyRectify(self.alpha.dimshuffle('x',0))(input)
            else:
                return nonlinearities.LeakyRectify(self.alpha)(input)
        elif input.ndim == 4:
            if self.untie_alphas: 
                return nonlinearities.LeakyRectify(self.alpha.dimshuffle('x',0,'x','x'))(input)
            else:
                return nonlinearities.LeakyRectify(self.alpha)(input)
        else:
            raise Exception('Incorrect number of dimensions.')
        '''