from __future__ import absolute_import
def test_rectify():
    from lasagne.nonlinearities import rectify
    assert [rectify(x) for x in (-1, 0, 1, 2)] == [0, 0, 1, 2]


def test_linear():
    from lasagne.nonlinearities import linear
    assert [linear(x) for x in (-1, 0, 1, 2)] == [-1, 0, 1, 2]
