import pytest
import numpy as np
import theano
import theano.tensor as T
import lasagne

PCT_TOLERANCE = 1E-5


class TestUpdateFunctions(object):
    # These tests compare results on a toy problem to values
    # calculated by the torch.optim package, using this script:
    # https://gist.github.com/ebenolson/931e879ed38f257253d2
    torch_values = {'sgd': [0.81707280688755,
                            0.6648326359915,
                            0.5386151140949],
                    'momentum': [0.6848486952183,
                                 0.44803321781003,
                                 0.27431190123502],
                    'nesterov_momentum': [0.67466543592725,
                                          0.44108468114241,
                                          0.2769002108997],
                    'adagrad': [0.55373120047759,
                                0.55373120041518,
                                0.55373120039438],
                    'rmsprop': [0.83205403985348,
                                0.83205322744821,
                                0.83205295664444],
                    'adadelta': [0.95453237704725,
                                 0.9545237471374,
                                 0.95452214847397],
                    'adam': [0.90034972009036,
                             0.90034967993061,
                             0.90034966654402],
                    }

    def f(self, X):
        return ([0.1, 0.2, 0.3] * X**2).sum()

    @pytest.mark.parametrize('method, kwargs', [
        ['sgd', {'learning_rate': 0.1}],
        ['momentum', {'learning_rate': 0.1, 'momentum': 0.5}],
        ['nesterov_momentum', {'learning_rate': 0.1, 'momentum': 0.5}],
        ['adagrad', {'learning_rate': 0.1}],
        ['rmsprop', {'learning_rate': 0.01}],
        ['adadelta', {}],
        ['adam', {'learning_rate': 0.01}],
        ])
    def test_updates(self, method, kwargs):
        A = theano.shared(lasagne.utils.floatX([1, 1, 1]))
        B = theano.shared(lasagne.utils.floatX([1, 1, 1]))
        update_func = getattr(lasagne.updates, method)
        updates = update_func(self.f(A) + self.f(B),
                              [A, B],
                              **kwargs)
        do_update = theano.function([], [], updates=updates)

        for _ in range(10):
            do_update()

        assert np.allclose(A.get_value(), B.get_value())
        assert np.allclose(A.get_value(), self.torch_values[method])


def test_get_or_compute_grads_raises():

    from lasagne.updates import get_or_compute_grads

    A = T.scalar()
    B = T.scalar()
    loss = A + B
    grads = get_or_compute_grads(loss, [A, B])

    assert get_or_compute_grads(grads, [A, B]) is grads

    with pytest.raises(ValueError):
        get_or_compute_grads(grads, [A])


@pytest.mark.parametrize('ndim', [2, 3])
def test_norm_constraint(ndim):
    import numpy as np
    import theano
    from lasagne.updates import norm_constraint
    from lasagne.utils import compute_norms

    max_norm = 0.01

    param = theano.shared(
        np.random.randn(*((25,) * ndim)).astype(theano.config.floatX)
    )

    update = norm_constraint(param, max_norm)

    apply_update = theano.function([], [], updates=[(param, update)])
    apply_update()

    assert param.dtype == update.dtype
    assert (np.max(compute_norms(param.get_value())) <=
            max_norm * (1 + PCT_TOLERANCE))


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


def test_total_norm_constraint():
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
