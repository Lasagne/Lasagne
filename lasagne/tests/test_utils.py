import pytest


def test_compute_norms():
    import numpy as np
    import theano
    from lasagne.utils import compute_norms

    array = np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)

    norms = compute_norms(array)

    assert array.dtype == norms.dtype
    assert norms.shape[0] == array.shape[0]


def test_compute_norms_axes():
    import numpy as np
    import theano
    from lasagne.utils import compute_norms

    array = np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)

    norms = compute_norms(array, norm_axes=(0, 2))

    assert array.dtype == norms.dtype
    assert norms.shape == (array.shape[1], array.shape[3])


def test_compute_norms_ndim6_raises():
    import numpy as np
    import theano
    from lasagne.utils import compute_norms

    array = np.random.randn(1, 2, 3, 4, 5, 6).astype(theano.config.floatX)

    with pytest.raises(ValueError) as excinfo:
        compute_norms(array)

    assert "Unsupported tensor dimensionality" in str(excinfo.value)
