import numpy
import pytest
import theano


def test_embedding():
    import numpy as np
    import theano
    import theano.tensor as T
    from lasagne.layers import EmbeddingLayer, InputLayer, helper
    x = T.ivector()
    l_in = InputLayer((3, ))
    l1 = EmbeddingLayer(l_in, input_size=3, output_size=5,
                        W=np.arange(3*5).reshape((3, 5)).astype('float32'))

    # check output shape
    assert helper.get_output_shape(l1, (2, )) == (2, 5)

    output = helper.get_output(l1, x)
    f = theano.function([x], output)
    x_test = np.array([0, 2]).astype('int32')

    output_correct = np.array(
        [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]], dtype='float32')
    np.testing.assert_array_almost_equal(f(x_test), output_correct)
