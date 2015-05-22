import pytest

PCT_TOLERANCE = 1E-5


def test_norm_constraint():
    import numpy as np
    import theano
    from lasagne.updates import norm_constraint
    from lasagne.utils import compute_norms

    max_norm = 0.01

    param = theano.shared(
        np.random.randn(100, 200).astype(theano.config.floatX)
    )

    update = norm_constraint(param, max_norm)

    apply_update = theano.function([], [], updates=[(param, update)])
    apply_update()

    assert param.dtype == update.dtype
    assert (np.max(compute_norms(param.get_value())) <=
            max_norm*(1 + PCT_TOLERANCE))


def test_norm_constraint_norm_axes():
    import numpy as np
    import theano
    from lasagne.updates import norm_constraint
    from lasagne.utils import compute_norms

    max_norm = 0.01
    norm_axes = (0, 2)

    param = theano.shared(
        np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)
    )

    update = norm_constraint(param, max_norm, norm_axes=norm_axes)

    apply_update = theano.function([], [], updates=[(param, update)])
    apply_update()

    assert param.dtype == update.dtype
    assert (np.max(compute_norms(param.get_value(), norm_axes=norm_axes)) <=
            max_norm*(1 + PCT_TOLERANCE))


def test_norm_constraint_dim6_raises():
    import numpy as np
    import theano
    from lasagne.updates import norm_constraint

    max_norm = 0.01

    param = theano.shared(
        np.random.randn(1, 2, 3, 4, 5, 6).astype(theano.config.floatX)
    )

    with pytest.raises(ValueError) as excinfo:
        norm_constraint(param, max_norm)
    assert "Unsupported tensor dimensionality" in str(excinfo.value)


def total_norm_constraint():
    import numpy as np
    import theano
    import theano.tensor as T
    from lasagne.updates import total_norm_constraint

    x1 = T.scalar()
    x2 = T.matrix()
    threshold = 5.0
    tensors1 = total_norm_constraint([x1, x2], threshold, return_norm=False)
    tensors2, norm = total_norm_constraint([x1, x2], threshold,
                                           return_norm=True)

    f1 = theano.function([x1, x2], [tensors1[0], tensors1[1]])
    f2 = theano.function([x1, x2], [tensors2[0], tensors2[1],
                                    norm])

    x_test = np.arange(1+9, dtype='float32')
    x1_test = x_test[-1]
    x2_test = x_test[:9].reshape((3, 3))
    x1_out1, x2_out1 = f1(x1_test, x2_test)
    x1_out2, x2_out2, norm = f2(x1_test, x2_test)

    np.testing.assert_array_almost_equal(x1_out1, x1_out2)
    np.testing.assert_array_almost_equal(x2_out1, x2_out2)

    x_out = [float(x1_out1)] + list(x2_out1.flatten())

    np.testing.assert_array_almost_equal(np.linalg.norm(x_test), norm)
    np.testing.assert_array_almost_equal(np.linalg.norm(x_out), threshold)
