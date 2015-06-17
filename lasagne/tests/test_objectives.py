import numpy as np
import theano
import pytest


def test_binary_crossentropy():
    # symbolic version
    from lasagne.objectives import binary_crossentropy
    p, t = theano.tensor.matrices('pt')
    c = binary_crossentropy(p, t)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(10, 20).astype(floatX)
    targets = np.random.rand(10, 20).astype(floatX)
    crossent = (- targets * np.log(predictions) -
                (1-targets) * np.log(1-predictions))
    # compare
    assert np.allclose(crossent, c.eval({p: predictions, t: targets}))


def test_categorical_crossentropy():
    # symbolic version
    from lasagne.objectives import categorical_crossentropy
    p, t = theano.tensor.matrices('pt')
    c = categorical_crossentropy(p, t)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(10, 20).astype(floatX)
    predictions /= predictions.sum(axis=1, keepdims=True)
    targets = np.random.rand(10, 20).astype(floatX)
    targets /= targets.sum(axis=1, keepdims=True)
    crossent = -(targets * np.log(predictions)).sum(axis=-1)
    # compare
    assert np.allclose(crossent, c.eval({p: predictions, t: targets}))


def test_categorical_crossentropy_onehot():
    # symbolic version
    from lasagne.objectives import categorical_crossentropy
    p = theano.tensor.matrix('p')
    t = theano.tensor.ivector('t')  # correct class per item
    c = categorical_crossentropy(p, t)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(10, 20).astype(floatX)
    predictions /= predictions.sum(axis=1, keepdims=True)
    targets = np.random.randint(20, size=10).astype(np.uint8)
    crossent = -np.log(predictions[np.arange(10), targets])
    # compare
    assert np.allclose(crossent, c.eval({p: predictions, t: targets}))


def test_squared_error():
    # symbolic version
    from lasagne.objectives import squared_error
    a, b = theano.tensor.matrices('ab')
    c = squared_error(a, b)
    # numeric version
    floatX = theano.config.floatX
    x = np.random.randn(10, 20).astype(floatX)
    y = np.random.randn(10, 20).astype(floatX)
    z = (x - y)**2
    # compare
    assert np.allclose(z, c.eval({a: x, b: y}))


def test_aggregate_mean():
    from lasagne.objectives import aggregate
    x = theano.tensor.matrix('x')
    assert theano.gof.graph.is_same_graph(aggregate(x), x.mean())
    assert theano.gof.graph.is_same_graph(aggregate(x, mode='mean'), x.mean())


def test_aggregate_sum():
    from lasagne.objectives import aggregate
    x = theano.tensor.matrix('x')
    assert theano.gof.graph.is_same_graph(aggregate(x, mode='sum'), x.sum())


def test_aggregate_weighted_mean():
    from lasagne.objectives import aggregate
    x = theano.tensor.matrix('x')
    w = theano.tensor.matrix('w')
    assert theano.gof.graph.is_same_graph(aggregate(x, w), (x * w).mean())
    assert theano.gof.graph.is_same_graph(aggregate(x, w, mode='mean'),
                                          (x * w).mean())


def test_aggregate_weighted_sum():
    from lasagne.objectives import aggregate
    x = theano.tensor.matrix('x')
    w = theano.tensor.matrix('w')
    assert theano.gof.graph.is_same_graph(aggregate(x, w, mode='sum'),
                                          (x * w).sum())


def test_aggregate_weighted_normalized_sum():
    from lasagne.objectives import aggregate
    x = theano.tensor.matrix('x')
    w = theano.tensor.matrix('w')
    assert theano.gof.graph.is_same_graph(aggregate(x, w, 'normalized_sum'),
                                          (x * w).sum() / w.sum())


def test_aggregate_invalid():
    from lasagne.objectives import aggregate
    with pytest.raises(ValueError) as exc:
        aggregate(theano.tensor.matrix(), mode='asdf')
    assert 'mode must be' in exc.value.args[0]
    with pytest.raises(ValueError) as exc:
        aggregate(theano.tensor.matrix(), mode='normalized_sum')
    assert 'require weights' in exc.value.args[0]
