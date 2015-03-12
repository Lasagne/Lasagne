import numpy as np


def test_rectify():
    from lasagne.nonlinearities import rectify
    assert [rectify(x) for x in (-1, 0, 1, 2)] == [0, 0, 1, 2]


def test_leaky_rectify():
    from lasagne.nonlinearities import LeakyRectify
    result = LeakyRectify(0.1)(np.array([-1, 0, 1, 2])).eval()
    assert np.allclose(result, [-.1, 0, 1, 2])


def test_linear():
    from lasagne.nonlinearities import linear
    assert [linear(x) for x in (-1, 0, 1, 2)] == [-1, 0, 1, 2]
