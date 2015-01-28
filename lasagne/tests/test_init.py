from __future__ import absolute_import
import pytest


def test_shape():
    from lasagne.init import Initializer

    # Assert that all `Initializer` sublasses return the shape that
    # we've asked for in `sample`:
    for klass in Initializer.__subclasses__():
        assert klass().sample((12, 23)).shape == (12, 23)


def test_normal():
    from lasagne.init import Normal

    sample = Normal().sample((100, 200))
    assert -0.001 < sample.mean() < 0.001


def test_constant():
    from lasagne.init import Constant

    sample = Constant(1.0).sample((10, 20))
    assert (sample == 1.0).all()


def test_sparse():
    from lasagne.init import Sparse

    sample = Sparse(sparsity=0.5).sample((10, 20))
    assert (sample == 0.0).sum() == (sample != 0.0).sum()
    assert (sample == 0.0).sum() == (10 * 20) / 2


def test_uniform_glorot():
    from lasagne.init import Uniform

    sample = Uniform().sample((150, 450))
    assert -0.11 < sample.min() < -0.09
    assert 0.09 < sample.max() < 0.11


def test_uniform_glorot_receptive_field():
    from lasagne.init import Uniform

    sample = Uniform().sample((150, 150, 2))
    assert -0.11 < sample.min() < -0.09
    assert 0.09 < sample.max() < 0.11


def test_uniform_range_as_number():
    from lasagne.init import Uniform

    sample = Uniform(1.0).sample((300, 400))
    assert sample.shape == (300, 400)
    assert -1.1 < sample.min() < -0.9
    assert 0.9 < sample.max() < 1.1


def test_uniform_range_as_range():
    from lasagne.init import Uniform

    sample = Uniform((0.0, 1.0)).sample((300, 400))
    assert sample.shape == (300, 400)
    assert -0.1 < sample.min() < 0.1
    assert 0.9 < sample.max() < 1.1


def test_orthogonal():
    import numpy as np
    from lasagne.init import Orthogonal

    sample = Orthogonal().sample((100, 200))
    assert np.allclose(np.dot(sample, sample.T), np.eye(100), atol=1e-6)

    sample = Orthogonal().sample((200, 100))
    assert np.allclose(np.dot(sample.T, sample), np.eye(100), atol=1e-6)


def test_orthogonal_gain():
    import numpy as np
    from lasagne.init import Orthogonal

    gain = 2
    sample = Orthogonal(gain).sample((100, 200))
    assert np.allclose(np.dot(sample, sample.T), gain * gain * np.eye(100), atol=1e-6)


def test_orthogonal_multi():
    import numpy as np
    from lasagne.init import Orthogonal

    sample = Orthogonal().sample((100, 50, 80))
    sample = sample.reshape(100, 50*80)
    assert np.allclose(np.dot(sample, sample.T), np.eye(100), atol=1e-6)


def test_orthogonal_1d_not_supported():
    import numpy as np
    from lasagne.init import Orthogonal

    with pytest.raises(RuntimeError):
        Orthogonal().sample((100,))
