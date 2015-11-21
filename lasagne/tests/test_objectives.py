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


def test_binary_hinge_loss():
    from lasagne.objectives import binary_hinge_loss
    from lasagne.nonlinearities import rectify
    p = theano.tensor.vector('p')
    t = theano.tensor.ivector('t')
    c = binary_hinge_loss(p, t)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(10).astype(floatX)
    targets = np.random.random_integers(0, 1, (10,)).astype("int8")
    hinge = rectify(1 - predictions * (2 * targets - 1))
    # compare
    assert np.allclose(hinge, c.eval({p: predictions, t: targets}))


def test_binary_hinge_loss_not_binary_targets():
    from lasagne.objectives import binary_hinge_loss
    from lasagne.nonlinearities import rectify
    p = theano.tensor.vector('p')
    t = theano.tensor.ivector('t')
    c = binary_hinge_loss(p, t, binary=False)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(10, ).astype(floatX)
    targets = np.random.random_integers(0, 1, (10, )).astype("int8")
    targets = 2 * targets - 1
    hinge = rectify(1 - predictions * targets)
    # compare
    assert np.allclose(hinge, c.eval({p: predictions, t: targets}))


def test_multiclass_hinge_loss():
    from lasagne.objectives import multiclass_hinge_loss
    from lasagne.nonlinearities import rectify
    p = theano.tensor.matrix('p')
    t = theano.tensor.ivector('t')
    c = multiclass_hinge_loss(p, t)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(10, 20).astype(floatX)
    targets = np.random.random_integers(0, 19, (10,)).astype("int8")
    one_hot = np.zeros((10, 20))
    one_hot[np.arange(10), targets] = 1
    correct = predictions[one_hot > 0]
    rest = predictions[one_hot < 1].reshape((10, 19))
    rest = np.max(rest, axis=1)
    hinge = rectify(1 + rest - correct)
    # compare
    assert np.allclose(hinge, c.eval({p: predictions, t: targets}))


def test_multiclass_hinge_loss_invalid():
    from lasagne.objectives import multiclass_hinge_loss
    with pytest.raises(TypeError) as exc:
        multiclass_hinge_loss(theano.tensor.vector(),
                              theano.tensor.matrix())
    assert 'rank mismatch' in exc.value.args[0]


def test_binary_accuracy():
    from lasagne.objectives import binary_accuracy
    p = theano.tensor.vector('p')
    t = theano.tensor.ivector('t')
    c = binary_accuracy(p, t)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(10, ).astype(floatX) > 0.5
    targets = np.random.random_integers(0, 1, (10,)).astype("int8")
    accuracy = predictions == targets
    # compare
    assert np.allclose(accuracy, c.eval({p: predictions, t: targets}))


def test_categorical_accuracy():
    from lasagne.objectives import categorical_accuracy
    p = theano.tensor.matrix('p')
    t = theano.tensor.ivector('t')
    c = categorical_accuracy(p, t)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(100, 5).astype(floatX)
    cls_predictions = np.argmax(predictions, axis=1)
    targets = np.random.random_integers(0, 4, (100,)).astype("int8")
    accuracy = cls_predictions == targets
    # compare
    assert np.allclose(accuracy, c.eval({p: predictions, t: targets}))
    one_hot = np.zeros((100, 5)).astype("int8")
    one_hot[np.arange(100), targets] = 1
    t = theano.tensor.imatrix('t')
    c = categorical_accuracy(p, t)
    assert np.allclose(accuracy, c.eval({p: predictions, t: one_hot}))


def test_categorical_accuracy_top_k():
    from lasagne.objectives import categorical_accuracy
    p = theano.tensor.matrix('p')
    t = theano.tensor.ivector('t')
    top_k = 4
    c = categorical_accuracy(p, t, top_k=top_k)
    # numeric version
    floatX = theano.config.floatX
    predictions = np.random.rand(10, 20).astype(floatX)
    cls_predictions = np.argsort(predictions, axis=1).astype("int8")
    # (construct targets such that top-1 to top-10 predictions are in there)
    targets = cls_predictions[np.arange(10), -np.random.permutation(10)]
    top_predictions = cls_predictions[:, -top_k:]
    accuracy = np.any(top_predictions == targets[:, np.newaxis], axis=1)
    # compare
    assert np.allclose(accuracy, c.eval({p: predictions, t: targets}))
    one_hot = np.zeros((10, 20)).astype("int8")
    one_hot[np.arange(10), targets] = 1
    t = theano.tensor.imatrix('t')
    c = categorical_accuracy(p, t, top_k=top_k)
    assert np.allclose(accuracy, c.eval({p: predictions, t: one_hot}))


def test_categorial_accuracy_invalid():
    from lasagne.objectives import categorical_accuracy
    with pytest.raises(TypeError) as exc:
        categorical_accuracy(theano.tensor.vector(),
                             theano.tensor.matrix())
    assert 'rank mismatch' in exc.value.args[0]
