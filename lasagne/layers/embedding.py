import numpy as np
import theano.tensor as T

from .. import init
from .base import Layer


__all__ = [
    "EmbeddingLayer"
]


class EmbeddingLayer(Layer):
    """
    lasagne.layers.EmbeddingLayer(incoming, input_size, output_size,
    W=lasagne.init.Normal(), **kwargs)

    A layer for word embeddings. The input should be an integer type
    Tensor variable.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    input_size: int
        The Number of different embeddings. The last embedding will have index
        input_size - 1.

    output_size : int
        The size of each embedding.

    W : Theano shared variable, numpy array or callable
        The embedding matrix.

    Examples
    --------
    >>> from lasagne.layers import EmbeddingLayer, InputLayer, get_output
    >>> import theano
    >>> x = T.imatrix()
    >>> l_in = InputLayer((3, ))
    >>> W = np.arange(3*5).reshape((3, 5)).astype('float32')
    >>> l1 = EmbeddingLayer(l_in, input_size=3, output_size=5, W=W)
    >>> output = get_output(l1, x)
    >>> f = theano.function([x], output)
    >>> x_test = np.array([[0, 2], [1, 2]]).astype('int32')
    >>> f(x_test)
    array([[[  0.,   1.,   2.,   3.,   4.],
            [ 10.,  11.,  12.,  13.,  14.]],
    <BLANKLINE>
           [[  5.,   6.,   7.,   8.,   9.],
            [ 10.,  11.,  12.,  13.,  14.]]], dtype=float32)
    """
    def __init__(self, incoming, input_size, output_size,
                 W=init.Normal(), **kwargs):
        super(EmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input]
