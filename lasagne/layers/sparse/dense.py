import theano.sparse as sparse

from ..dense import DenseLayer as Dense


__all__ = [
    "DenseLayer",
]


class DenseLayer(Dense):
    def get_output_for(self, input, **kwargs):
        """
        Parameters
        ----------
        input : theano sparse matrix
            output from the previous layer
        """
        assert type(input) == sparse.basic.SparseVariable

        activation = sparse.basic.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
