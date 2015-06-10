from mock import Mock
import pytest
import numpy as np
import theano


def test_shared_empty():
    from lasagne.utils import shared_empty

    X = shared_empty(3)
    assert (np.zeros((1, 1, 1)) == X.eval()).all()


def test_as_theano_expression_fails():
    from lasagne.utils import as_theano_expression
    with pytest.raises(TypeError):
        as_theano_expression({})


def test_one_hot():
    from lasagne.utils import one_hot
    a = np.random.randint(0, 10, 20)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size), a] = 1

    result = one_hot(a).eval()
    assert (result == b).all()


def test_as_tuple_fails():
    from lasagne.utils import as_tuple
    with pytest.raises(ValueError):
        as_tuple([1, 2, 3], 4)


def test_compute_norms():
    from lasagne.utils import compute_norms

    array = np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)

    norms = compute_norms(array)

    assert array.dtype == norms.dtype
    assert norms.shape[0] == array.shape[0]


def test_compute_norms_axes():
    from lasagne.utils import compute_norms

    array = np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)

    norms = compute_norms(array, norm_axes=(0, 2))

    assert array.dtype == norms.dtype
    assert norms.shape == (array.shape[1], array.shape[3])


def test_compute_norms_ndim6_raises():
    from lasagne.utils import compute_norms

    array = np.random.randn(1, 2, 3, 4, 5, 6).astype(theano.config.floatX)

    with pytest.raises(ValueError) as excinfo:
        compute_norms(array)

    assert "Unsupported tensor dimensionality" in str(excinfo.value)


def test_create_param_bad_callable_raises():
    from lasagne.utils import create_param

    with pytest.raises(RuntimeError):
        create_param(lambda x: {}, (1, 2, 3))
    with pytest.raises(RuntimeError):
        create_param(lambda x: np.array(1), (1, 2, 3))


def test_create_param_bad_spec_raises():
    from lasagne.utils import create_param

    with pytest.raises(RuntimeError):
        create_param({}, (1, 2, 3))


def test_create_param_accepts_iterable_shape():
    from lasagne.utils import create_param
    factory = np.empty
    create_param(factory, [2, 3])
    create_param(factory, (x for x in [2, 3]))


def test_create_param_numpy_bad_shape_raises_error():
    from lasagne.utils import create_param

    param = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(RuntimeError):
        create_param(param, (3, 2))


def test_create_param_numpy_returns_shared():
    from lasagne.utils import create_param

    param = np.array([[1, 2, 3], [4, 5, 6]])
    result = create_param(param, (2, 3))
    assert (result.get_value() == param).all()
    assert isinstance(result, type(theano.shared(param)))
    assert (result.get_value() == param).all()


def test_create_param_shared_returns_same():
    from lasagne.utils import create_param

    param = theano.shared(np.array([[1, 2, 3], [4, 5, 6]]))
    result = create_param(param, (2, 3))
    assert result is param


def test_create_param_shared_bad_ndim_raises_error():
    from lasagne.utils import create_param

    param = theano.shared(np.array([[1, 2, 3], [4, 5, 6]]))
    with pytest.raises(RuntimeError):
        create_param(param, (2, 3, 4))


def test_create_param_callable_returns_return_value():
    from lasagne.utils import create_param

    array = np.array([[1, 2, 3], [4, 5, 6]])
    factory = Mock()
    factory.return_value = array
    result = create_param(factory, (2, 3))
    assert (result.get_value() == array).all()
    factory.assert_called_with((2, 3))


def test_nonpositive_dims_raises_value_error():
    from lasagne.utils import create_param
    neg_shape = (-1, -1)
    zero_shape = (0, 0)
    pos_shape = (1, 1)
    spec = np.empty
    with pytest.raises(ValueError):
        create_param(spec, neg_shape)
    with pytest.raises(ValueError):
        create_param(spec, zero_shape)
    create_param(spec, pos_shape)
