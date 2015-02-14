import pytest

PCT_TOLERANCE = 1E-6

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




