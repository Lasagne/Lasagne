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
