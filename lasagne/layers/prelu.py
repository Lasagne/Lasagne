import numpy as np
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer

__all__ = [
    "PReLULayer"
]

class PReLULayer(Layer):

    def __init__(self, incoming, alpha=0.25, unique_chan_param=False, 
                 **kwargs):
        super(PReLULayer, self).__init__(incoming, **kwargs)
        self.unique_chan_param = unique_chan_param
        if not unique_chan_param:
            self.alpha = self.create_param(np.float32(alpha), (), name="alpha")
        else:
            self.alpha = self.create_param(np.zeros(incoming.shape[1])+alpha, (incoming.shape[1]), name="W")

    def get_params(self):
        return [self.alpha]
        
    def get_output_for(self, input, **kwargs):
        if input.ndim == 2 or (input.ndim ==4 and not self.unique_chan_param):
            return nonlinearities.LeakyRectify(self.alpha)(input)
        elif input.ndim == 4 and unique_chan_param: 
            return nonlinearities.LeakyRectify(self.alpha.dimshuffle('x',0,'x','x'))(input)
        else:
            raise Exception('Not handling this yet, needs to be dim 2 or 4 and spread across channels')
