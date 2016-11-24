import pytest
import numpy as np


class ArrayLike (object):
    # Array-like helper class
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, item):
        return self.arr[item]


class WrappedList (object):
    # Helper class designed to look like a sequence, for testing
    # is_sequence_of_arraylikes
    def __init__(self, xs):
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item]


def test_is_arraylike():
    from lasagne import batch

    assert batch.is_arraylike(np.arange(3))
    assert batch.is_arraylike(ArrayLike(np.arange(4)))
    assert not batch.is_arraylike(1)
    assert not batch.is_arraylike((x for x in range(3)))


def test_is_sequence_of_arraylikes():
    from lasagne import batch

    assert batch.is_sequence_of_arraylike([np.arange(3), np.arange(4)])
    assert batch.is_sequence_of_arraylike([np.arange(3),
                                           ArrayLike(np.arange(4))])
    assert not batch.is_sequence_of_arraylike([np.arange(3), 4])
    assert not batch.is_sequence_of_arraylike(
            WrappedList([np.arange(3), ArrayLike(np.arange(4))]))


def test_length_of_arraylikes_in_sequence():
    from lasagne import batch

    a3a = np.arange(3)
    a3b = np.arange(6).reshape((3, 2))
    a10 = np.arange(10)

    assert batch.length_of_arraylikes_in_sequence([a3a]) == 3
    assert batch.length_of_arraylikes_in_sequence([a10]) == 10
    assert batch.length_of_arraylikes_in_sequence([a3a, a3b]) == 3
    with pytest.raises(ValueError):
        batch.length_of_arraylikes_in_sequence([a3a, a10])


def test_dataset_length():
    from lasagne import batch

    a3a = np.arange(3)
    a3b = np.arange(6).reshape((3, 2))
    a10 = np.arange(10)

    def callable_ds(batchsize, shuffle_rng=None):
        for i in range(0, batchsize * 10, batchsize):
            yield [np.arange(i, i + batchsize)]

    assert batch.dataset_length([a3a]) == 3
    assert batch.dataset_length([a10]) == 10
    assert batch.dataset_length([a3a, a3b]) == 3
    assert batch.dataset_length(callable_ds) is None


def test_arraylikes_batch_iterator():
    from lasagne import batch

    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))

    # Three in-order batches
    batches = list(batch.arraylikes_batch_iterator([X, Y], batchsize=15))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[:15]).all()
    assert (batches[0][1] == Y[:15]).all()
    assert (batches[1][0] == X[15:30]).all()
    assert (batches[1][1] == Y[15:30]).all()
    assert (batches[2][0] == X[30:]).all()
    assert (batches[2][1] == Y[30:]).all()

    # Three shuffled batches
    batches = list(batch.arraylikes_batch_iterator(
            [X, Y], batchsize=15, shuffle_rng=np.random.RandomState(12345)))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[order[:15]]).all()
    assert (batches[0][1] == Y[order[:15]]).all()
    assert (batches[1][0] == X[order[15:30]]).all()
    assert (batches[1][1] == Y[order[15:30]]).all()
    assert (batches[2][0] == X[order[30:]]).all()
    assert (batches[2][1] == Y[order[30:]]).all()


def test_batch_iterator():
    from lasagne import batch

    # Data to extract batches from
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))

    # Helper functions to check
    def check_in_order_batches(batches):
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[:15]).all()
        assert (batches[0][1] == Y[:15]).all()
        assert (batches[1][0] == X[15:30]).all()
        assert (batches[1][1] == Y[15:30]).all()
        assert (batches[2][0] == X[30:]).all()
        assert (batches[2][1] == Y[30:]).all()

    def check_shuffled_batches(batches, order_seed=12345):
        # Get the expected order
        order = np.random.RandomState(order_seed).permutation(45)
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[order[:15]]).all()
        assert (batches[0][1] == Y[order[:15]]).all()
        assert (batches[1][0] == X[order[15:30]]).all()
        assert (batches[1][1] == Y[order[15:30]]).all()
        assert (batches[2][0] == X[order[30:]]).all()
        assert (batches[2][1] == Y[order[30:]]).all()

    #
    # Test sequence of array-likes protocol
    #

    # Three in-order batches
    batches = list(batch.batch_iterator([X, Y], batchsize=15))
    check_in_order_batches(batches)

    # Three shuffled batches
    batches = list(batch.batch_iterator(
            [X, Y], batchsize=15, shuffle_rng=np.random.RandomState(12345)))
    check_shuffled_batches(batches, 12345)

    #
    # Test `batch_iterator` method protocol
    #

    class HasBatchIterator (object):
        # Helper class to test `batch_iterator` method protocol
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def batch_iterator(self, batchsize, shuffle_rng=None):
            # Make `batch.arraylikes_batch_iterator` do the work :)
            return batch.arraylikes_batch_iterator(
                    [self.X, self.Y], batchsize, shuffle_rng=shuffle_rng)

    # Three in-order batches
    batches = list(batch.batch_iterator(HasBatchIterator(X, Y), batchsize=15))
    check_in_order_batches(batches)

    # Three shuffled batches
    batches = list(batch.batch_iterator(
            HasBatchIterator(X, Y), batchsize=15,
            shuffle_rng=np.random.RandomState(12345)))
    check_shuffled_batches(batches)

    #
    # Test callable
    #

    def make_batch_iterator(batchsize, shuffle_rng=None):
        # Helper function to test `callable` protocol
        # Make `batch.arraylikes_batch_iterator` do the work :)
        return batch.arraylikes_batch_iterator(
                [X, Y], batchsize, shuffle_rng=shuffle_rng)

    # Three in-order batches
    batches = list(batch.batch_iterator(make_batch_iterator, batchsize=15))
    check_in_order_batches(batches)

    # Three shuffled batches
    batches = list(batch.batch_iterator(
            make_batch_iterator, batchsize=15,
            shuffle_rng=np.random.RandomState(12345)))
    check_shuffled_batches(batches)

    #
    # Test invalid type
    #

    with pytest.raises(TypeError):
        batch.batch_iterator(1, batchsize=15)
