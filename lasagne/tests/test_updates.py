import pytest

PCT_TOLERANCE = 1E-5

def test_compute_norms():
    import numpy as np
    import theano
    from lasagne.updates import compute_norms

    array = np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)

    norms = compute_norms(array)

    assert array.dtype == norms.dtype
    assert norms.shape[0] == array.shape[0]

def test_compute_norms_axes():
    import numpy as np
    import theano
    from lasagne.updates import compute_norms

    array = np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)

    norms = compute_norms(array, norm_axes=(0, 2))

    assert array.dtype == norms.dtype
    assert norms.shape == (array.shape[1], array.shape[3])


def test_norm_constraint_abs():
    import numpy as np
    import theano
    from lasagne.updates import norm_constraint, compute_norms

    abs_max = 0.01

    param = theano.shared(
        np.random.randn(100, 200).astype(theano.config.floatX)
    )

    update = norm_constraint(param, abs_max=abs_max)

    apply_update = theano.function([], [], updates=[(param, update)])
    apply_update()

    assert param.dtype == update.dtype
    assert (np.max(compute_norms(param.get_value()))
            <= abs_max*(1 + PCT_TOLERANCE))



def test_norm_constraint_rel():
    import numpy as np
    import theano
    from lasagne.updates import norm_constraint, compute_norms

    rel_max = 1.0

    param = theano.shared(
        np.random.randn(100, 200).astype(theano.config.floatX)
    )

    orig_avg_norm = np.mean(compute_norms(param.get_value()))

    # update: multiply by 5
    update = np.array(5.0, dtype=theano.config.floatX)*param

    update = norm_constraint(update, param, rel_max=rel_max)

    apply_update = theano.function([], [], updates=[(param, update)])
    apply_update()

    assert param.dtype == update.dtype
    assert (np.max(compute_norms(param.get_value()))
            <= rel_max * (1 + PCT_TOLERANCE) * orig_avg_norm)


def test_norm_constraint_norm_axes():
    import numpy as np
    import theano
    from lasagne.updates import norm_constraint, compute_norms

    abs_max = 0.01
    norm_axes = (0, 2)

    param = theano.shared(
        np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)
    )

    update = norm_constraint(param, abs_max=abs_max, norm_axes=norm_axes)

    apply_update = theano.function([], [], updates=[(param, update)])
    apply_update()

    assert param.dtype == update.dtype
    assert (np.max(compute_norms(param.get_value(), norm_axes=norm_axes))
            <= abs_max*(1 + PCT_TOLERANCE))




