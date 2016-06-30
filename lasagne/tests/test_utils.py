from mock import Mock
import pytest
import numpy as np
import theano
import theano.tensor as T


def test_shared_empty():
    from lasagne.utils import shared_empty

    X = shared_empty(3)
    assert (np.zeros((1, 1, 1)) == X.eval()).all()


def test_as_theano_expression_fails():
    from lasagne.utils import as_theano_expression
    with pytest.raises(TypeError):
        as_theano_expression({})


def test_collect_shared_vars():
    from lasagne.utils import collect_shared_vars as collect
    x, y, z = (theano.shared(0, name=n) for n in 'xyz')
    # collecting must not change the order
    assert collect([x, y, z]) == [x, y, z]
    # duplicates should be eliminated
    assert collect([x, y, x, y, y, z]) == [x, y, z]
    # ensure we have left-recursive depth-first search
    assert collect((x + y) + z) == [x, y, z]
    assert collect(x + (y + z)) == [x, y, z]
    # complex expressions and constants should not be included
    assert collect([x**2, y * z * np.ones(10), x + T.matrix()]) == [x, y, z]
    # the result can even be empty
    assert collect([T.matrix() + T.matrix(), T.log(T.matrix())]) == []


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
    with pytest.raises(TypeError):
        as_tuple('asdf', 4, int)


def test_compute_norms():
    from lasagne.utils import compute_norms

    # Test numpy version of compute_norms
    array = np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)

    norms = compute_norms(array)

    assert array.dtype == norms.dtype
    assert norms.shape[0] == array.shape[0]

    # Test theano version of compute_norms
    t_array = theano.shared(array)
    t_norms = compute_norms(t_array)

    # Check if they do not differ much
    assert np.allclose(t_norms.eval(), norms)


def test_compute_norms_axes():
    from lasagne.utils import compute_norms

    # Test numpy versions of compute norms with axes
    array = np.random.randn(10, 20, 30, 40).astype(theano.config.floatX)

    norms = compute_norms(array, norm_axes=(0, 2))

    assert array.dtype == norms.dtype
    assert norms.shape == (array.shape[1], array.shape[3])

    # Test theano version of compute_norms
    t_array = theano.shared(array)
    t_norms = compute_norms(t_array, norm_axes=(0, 2))

    # Check if they do not differ much
    assert np.allclose(t_norms.eval(), norms)


def test_compute_norms_ndim1():
    from lasagne.utils import compute_norms

    # Test numpy versions of compute norms with axes
    array = np.random.randn(10, ).astype(theano.config.floatX)

    norms = compute_norms(array)

    assert array.dtype == norms.dtype
    assert norms.shape == array.shape

    # Check if they do not differ much
    assert np.allclose(norms, abs(array))

    # Test theano version of compute_norms
    t_array = theano.shared(array)
    t_norms = compute_norms(t_array)

    # Check if they do not differ much
    assert np.allclose(t_norms.eval(), norms)


def test_compute_norms_type_raises():
    from lasagne.utils import compute_norms

    array = [[1, 2], [3, 4]]

    with pytest.raises(RuntimeError) as excinfo:
        compute_norms(array)

    assert ("Unsupported type") in str(excinfo.value)


def test_compute_norms_ndim6_raises():
    from lasagne.utils import compute_norms

    array = np.random.randn(1, 2, 3, 4, 5, 6).astype(theano.config.floatX)

    with pytest.raises(ValueError) as excinfo:
        compute_norms(array)

    assert "Unsupported tensor dimensionality" in str(excinfo.value)


def test_create_param_bad_callable_raises():
    from lasagne.utils import create_param

    with pytest.raises(TypeError):
        create_param(lambda x: {}, (1, 2, 3))
    with pytest.raises(ValueError):
        create_param(lambda x: np.array(1), (1, 2, 3))


def test_create_param_bad_spec_raises():
    from lasagne.utils import create_param

    with pytest.raises(TypeError):
        create_param({}, (1, 2, 3))


def test_create_param_accepts_iterable_shape():
    from lasagne.utils import create_param
    factory = np.empty
    create_param(factory, [2, 3])
    create_param(factory, (x for x in [2, 3]))


def test_create_param_numpy_bad_shape_raises_error():
    from lasagne.utils import create_param

    param = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        create_param(param, (2, 3, 4))


def test_create_param_callable_returns_return_value():
    from lasagne.utils import create_param

    array = np.array([[1, 2, 3], [4, 5, 6]])
    factory = Mock()
    factory.return_value = array
    result = create_param(factory, (2, 3))
    assert (result.get_value() == array).all()
    factory.assert_called_with((2, 3))


def test_create_param_callable_returns_shared():
    from lasagne.utils import create_param

    array = np.array([[1, 2, 3], [4, 5, 6]])
    param = theano.shared(array)
    factory = Mock()
    factory.return_value = param
    result = create_param(factory, (2, 3))
    assert (result.get_value() == array).all()
    factory.assert_called_with((2, 3))
    assert result is param


def test_create_param_callable_returns_shared_bad_ndim_raises_error():
    from lasagne.utils import create_param

    array = np.array([[1, 2], [3, 4]])
    param = theano.shared(array)
    factory = Mock()
    factory.return_value = param
    with pytest.raises(ValueError):
        create_param(factory, (2, 3, 4))


def test_create_param_callable_returns_theano_expr():
    from lasagne.utils import create_param

    array = np.array([[1, 2, 3], [4, 5, 6]])
    param = theano.shared(array) * 2
    factory = Mock()
    factory.return_value = param
    result = create_param(factory, (2, 3))
    assert (result.eval() == array * 2).all()
    assert result is param


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


def test_create_param_callable_returns_wrong_type():
    from lasagne.utils import create_param

    param = 'string'
    factory = Mock()
    factory.return_value = param
    with pytest.raises(TypeError):
        create_param(factory, (1, 2))


def test_create_param_retain_ndarray_dtype():
    from lasagne.utils import create_param
    param = np.array([[1, 2, 3], [4, 5, 6]])

    param = param.astype('float64')
    result = create_param(param, (2, 3))
    assert (result.dtype == param.dtype)

    param = param.astype('int16')
    result = create_param(param, (2, 3))
    assert (result.dtype == param.dtype)


def test_create_param_broadcast_pattern():
    from lasagne.utils import create_param
    for shape in (10, 1, 20), (1, 2), (3, 1), (2, 3):
        bcast = tuple(s == 1 for s in shape)
        assert create_param(np.zeros, shape).broadcastable == bcast
        assert create_param(np.zeros(shape, np.float32),
                            shape).broadcastable == bcast


def test_unroll_scan():
    from lasagne.utils import unroll_scan
    k = 2
    a = T.scalar("a")

    result = unroll_scan(
        fn=lambda step, prior_result, a: prior_result * a,
        sequences=T.arange(k), outputs_info=[1.], non_sequences=[a], n_steps=k)
    final_result = result[-1]
    power = theano.function(inputs=[a], outputs=final_result)

    assert np.all(power(10) == [10, 100])

    b = T.scalar("b")

    def mul_div(step, previous_mul, previous_div, mul, div):
            return previous_mul*mul, previous_div/div

    result = unroll_scan(
        fn=mul_div, sequences=T.arange(k), outputs_info=[1., 1.],
        non_sequences=[a, b], n_steps=k)
    power = theano.function(inputs=[a, b], outputs=result)
    assert np.allclose(power(10, 10), [[10, 100], [.1, .01]])
