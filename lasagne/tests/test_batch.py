import pytest
import numpy as np

def test_is_arraylike():
    from lasagne import batch

    class ArrayLike (object):
        def __len__(self):
            return 10

        def __getitem__(self, item):
            return list(range(10))[item]

    assert batch.is_arraylike([1,2,3])
    assert batch.is_arraylike((1,2,3))
    assert batch.is_arraylike(np.arange(3))
    assert batch.is_arraylike(ArrayLike())
    assert not batch.is_arraylike(1)
    assert not batch.is_arraylike((x for x in range(3)))
