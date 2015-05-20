import numpy
import pytest
import theano


def test_embedding_2D_input():
    import numpy as np
    import theano
    import theano.tensor as T
    from lasagne.layers import EmbeddingLayer, InputLayer, helper
    x = T.imatrix()
    batch_size = 2
    seq_len = 3
    emb_size = 5
    vocab_size = 3
    l_in = InputLayer((None, seq_len))
    W = np.arange(
        vocab_size*emb_size).reshape((vocab_size, emb_size)).astype('float32')
    l1 = EmbeddingLayer(l_in, input_size=vocab_size, output_size=emb_size,
                        W=W)

    x_test = np.array([[0, 1, 2], [0, 0, 2]], dtype='int32')

    # check output shape
    assert helper.get_output_shape(
        l1, (batch_size, seq_len)) == (batch_size, seq_len, emb_size)

    output = helper.get_output(l1, x)
    f = theano.function([x], output)
    np.testing.assert_array_almost_equal(f(x_test), W[x_test])


def test_embedding_1D_input():
    import numpy as np
    import theano
    import theano.tensor as T
    from lasagne.layers import EmbeddingLayer, InputLayer, helper
    x = T.ivector()
    batch_size = 2
    emb_size = 10
    vocab_size = 3
    l_in = InputLayer((None,))
    W = np.arange(
        vocab_size*emb_size).reshape((vocab_size, emb_size)).astype('float32')
    l1 = EmbeddingLayer(l_in, input_size=vocab_size, output_size=emb_size,
                        W=W)

    x_test = np.array([0, 1, 2], dtype='int32')

    # check output shape
    assert helper.get_output_shape(
        l1, (batch_size, )) == (batch_size, emb_size)

    output = helper.get_output(l1, x)
    f = theano.function([x], output)
    np.testing.assert_array_almost_equal(f(x_test), W[x_test])
