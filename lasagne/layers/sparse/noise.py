import theano
import theano.sparse as sparse

from ..noise import DropoutLayer as Dropout


__all__ = [
    "DropoutLayer",
    "dropout",
]


class DropoutLayer(Dropout):
    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : theano sparse matrix
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            assert type(input) == sparse.basic.SparseVariable

            retain_prob = 1 - self.p
            if self.rescale:
                input = sparse.basic.mul(input, 1/retain_prob)

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=theano.config.floatX)

dropout = DropoutLayer  # shortcut
