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


class HasBatchIterator (object):
    # Helper class to test `batch_iterator` method protocol
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def batch_iterator(self, batchsize, shuffle_rng=None):
        from lasagne import batch
        # Make `batch.arraylikes_batch_iterator` do the work :)
        return batch.arraylikes_batch_iterator(
                [self.X, self.Y], batchsize, shuffle_rng=shuffle_rng)


class HasCircularBatchIterator (object):
    # Helper class to test `circular_batch_iterator` method protocol
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def circular_batch_iterator(self, batchsize, shuffle_rng=None):
        from lasagne import batch
        # Make `batch.arraylikes_batch_iterator` do the work :)
        return batch.circular_arraylikes_batch_iterator(
                [self.X, self.Y], batchsize, shuffle_rng=shuffle_rng)


# Helper function to test the callable protocol
def make_batch_iterator_callable(X, Y):
    from lasagne import batch

    def batch_iterator(batchsize, shuffle_rng=None):
        # Make `batch.arraylikes_batch_iterator` do the work :)
        return batch.arraylikes_batch_iterator(
                [X, Y], batchsize, shuffle_rng=shuffle_rng)
    return batch_iterator


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


def test_is_dataset():
    from lasagne import batch

    obj_with_batch_it = HasBatchIterator(np.arange(3), np.arange(3))
    obj_with_circ_batch_it = HasCircularBatchIterator(np.arange(3),
                                                      np.arange(3))
    callable_it = make_batch_iterator_callable(np.arange(3), np.arange(3))

    assert batch.is_dataset([np.arange(3), np.arange(4)])
    assert batch.is_dataset([np.arange(3), np.arange(4)], circular=True)
    assert batch.is_dataset([np.arange(3), ArrayLike(np.arange(4))])
    assert batch.is_dataset([np.arange(3), ArrayLike(np.arange(4))],
                            circular=True)
    assert not batch.is_dataset([np.arange(3), 4])
    assert not batch.is_dataset(
        WrappedList([np.arange(3), ArrayLike(np.arange(4))]))

    assert batch.is_dataset(obj_with_batch_it, circular=False)
    assert not batch.is_dataset(obj_with_batch_it, circular=True)
    assert not batch.is_dataset(obj_with_circ_batch_it, circular=False)
    assert batch.is_dataset(obj_with_circ_batch_it, circular=True)
    assert batch.is_dataset(callable_it, circular=False)
    assert batch.is_dataset(callable_it, circular=True)

    it = batch.batch_iterator([np.arange(3), np.arange(4)], 3)
    assert batch.is_dataset(it, circular=False)
    assert batch.is_dataset(it, circular=True)


def test_is_sequence_of_datasets():
    from lasagne import batch

    seq_of_arrays1 = [np.arange(3), np.arange(4)]
    seq_of_arrays2 = [np.arange(3) * 2, np.arange(4) * 2]
    obj_with_batch_it = HasBatchIterator(np.arange(3), np.arange(3))
    obj_with_circ_batch_it = HasCircularBatchIterator(np.arange(3),
                                                      np.arange(3))
    callable_it = make_batch_iterator_callable(np.arange(3), np.arange(3))
    it = batch.batch_iterator([np.arange(3), np.arange(4)], 3)

    # Two lists of arrays
    assert batch.is_sequence_of_datasets([seq_of_arrays1, seq_of_arrays2])
    # List of arrays and object with `batch_iterator` method
    assert batch.is_sequence_of_datasets(
        [seq_of_arrays1, obj_with_batch_it], circular=False)
    assert not batch.is_sequence_of_datasets(
        [seq_of_arrays1, obj_with_batch_it], circular=True)
    assert not batch.is_sequence_of_datasets(
        [seq_of_arrays1, obj_with_circ_batch_it], circular=False)
    assert batch.is_sequence_of_datasets(
        [seq_of_arrays1, obj_with_circ_batch_it], circular=True)
    assert batch.is_sequence_of_datasets([seq_of_arrays1, callable_it])
    assert batch.is_sequence_of_datasets([seq_of_arrays1, it])
    # Not a data set
    assert not batch.is_sequence_of_datasets([seq_of_arrays1, np.arange(3)])


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


def test_circular_arraylikes_batch_iterator():
    from lasagne import batch

    X = np.arange(50)
    Y = np.arange(100).reshape((50, 2))

    # Five in-order batches
    inorder_iter = batch.circular_arraylikes_batch_iterator(
        [X, Y], batchsize=20)
    batches = [inorder_iter.next() for i in range(5)]
    # Five batches
    assert len(batches) == 5
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    assert len(batches[3]) == 2
    assert len(batches[4]) == 2
    # Verify values
    assert (batches[0][0] == X[:20]).all()
    assert (batches[0][1] == Y[:20]).all()
    assert (batches[1][0] == X[20:40]).all()
    assert (batches[1][1] == Y[20:40]).all()
    assert (batches[2][0] == np.append(X[40:50], X[0:10], axis=0)).all()
    assert (batches[2][1] == np.append(Y[40:50], Y[0:10], axis=0)).all()
    assert (batches[3][0] == X[10:30]).all()
    assert (batches[3][1] == Y[10:30]).all()
    assert (batches[4][0] == X[30:50]).all()
    assert (batches[4][1] == Y[30:50]).all()

    # Five shuffled batches
    shuffled_iter = batch.circular_arraylikes_batch_iterator(
        [X, Y], batchsize=20, shuffle_rng=np.random.RandomState(12345))
    batches = [shuffled_iter.next() for i in range(5)]
    # Get the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.append(order_shuffle_rng.permutation(50),
                      order_shuffle_rng.permutation(50), axis=0)
    # Five batches
    assert len(batches) == 5
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    assert len(batches[3]) == 2
    assert len(batches[4]) == 2
    # Verify values
    assert (batches[0][0] == X[order[:20]]).all()
    assert (batches[0][1] == Y[order[:20]]).all()
    assert (batches[1][0] == X[order[20:40]]).all()
    assert (batches[1][1] == Y[order[20:40]]).all()
    assert (batches[2][0] == X[order[40:60]]).all()
    assert (batches[2][1] == Y[order[40:60]]).all()
    assert (batches[3][0] == X[order[60:80]]).all()
    assert (batches[3][1] == Y[order[60:80]]).all()
    assert (batches[4][0] == X[order[80:]]).all()
    assert (batches[4][1] == Y[order[80:]]).all()


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

    # Three in-order batches
    batches = list(batch.batch_iterator(make_batch_iterator_callable(X, Y),
                                        batchsize=15))
    check_in_order_batches(batches)

    # Three shuffled batches
    batches = list(batch.batch_iterator(
            make_batch_iterator_callable(X, Y), batchsize=15,
            shuffle_rng=np.random.RandomState(12345)))
    check_shuffled_batches(batches)

    #
    # Test iterator
    #

    # Re-use the function defined above to create the iterator
    in_order_batch_iter = make_batch_iterator_callable(X, Y)(15)
    batches = list(batch.batch_iterator(in_order_batch_iter, batchsize=15,
                                        restartable=False))
    check_in_order_batches(batches)

    # Three shuffled batches
    shuffled_batch_iter = make_batch_iterator_callable(X, Y)(
        15, shuffle_rng=np.random.RandomState(12345))
    batches = list(batch.batch_iterator(shuffled_batch_iter, batchsize=15,
                                        restartable=False))
    check_shuffled_batches(batches)

    #
    # Test invalid type
    #

    with pytest.raises(TypeError):
        batch.batch_iterator(1, batchsize=15)

    #
    # Passing an iterator should raise `TypeError` if `restartable` argument
    # is true
    #

    in_order_batch_iter = make_batch_iterator_callable(X, Y)(15)
    with pytest.raises(TypeError):
        batch.batch_iterator(in_order_batch_iter, batchsize=15,
                             restartable=True)


def test_batch_map():
    from lasagne import batch

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return [batch_X + 2, (batch_Y**2).sum(axis=1)]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    [x, y] = batch.batch_map(batch_func, [X, Y], 5,
                             progress_iter_func=progress_iter_func)

    assert np.allclose(x, X + 2)
    assert np.allclose(y, (Y**2).sum(axis=1))

    #
    # Single return value
    #
    def batch_func_single(batch_X, batch_Y):
        return batch_X + 2

    [x] = batch.batch_map(batch_func_single, [X, Y], 5)

    assert np.allclose(x, X + 2)

    #
    # Batch function that returns no results
    #
    def batch_func_no_results(batch_X, batch_Y):
        return None

    res = batch.batch_map(batch_func_no_results, [X, Y], 5,
                          progress_iter_func=progress_iter_func)

    assert res is None

    #
    # Invalid return value
    #
    def batch_func_invalid(batch_X, batch_Y):
        return 'Should not return a string'

    with pytest.raises(TypeError):
        batch.batch_map(batch_func_invalid, [X, Y], 5)

    #
    # Prepend arguments to batch function
    #
    def batch_func_prepend(a, b, batch_X, batch_Y):
        assert a == 42
        assert b == 3.14
        return [batch_X + 2, (batch_Y**2).sum(axis=1)]

    [x, y] = batch.batch_map(batch_func_prepend, [X, Y], 5,
                             progress_iter_func=progress_iter_func,
                             prepend_args=(42, 3.14))

    assert np.allclose(x, X + 2)
    assert np.allclose(y, (Y**2).sum(axis=1))


def test_batch_map_callable():
    # Get data from callable that creates iterator rather than list of arrays
    from lasagne import batch

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))

    def dataset(batchsize, shuffle_rng=None):
        return batch.batch_iterator([X, Y], batchsize, shuffle_rng=shuffle_rng)

    def batch_func(batch_X, batch_Y):
        return [batch_X + 2, (batch_Y**2).sum(axis=1)]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total is None
        # not leave
        assert not leave
        return iterator

    [x, y] = batch.batch_map(batch_func, dataset, 5,
                             progress_iter_func=progress_iter_func)

    assert np.allclose(x, X + 2)
    assert np.allclose(y, (Y**2).sum(axis=1))


def test_mean_batch_map_in_order():
    from lasagne import batch

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return [batch_X.sum(), (batch_Y**2).sum(axis=1).sum()]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    [x, y] = batch.mean_batch_map(batch_func, [X, Y], 5,
                                  progress_iter_func=progress_iter_func,
                                  sum_axis=None)

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Single return value
    #
    def batch_func_single(batch_X, batch_Y):
        return batch_X.sum()

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    [x] = batch.mean_batch_map(batch_func_single, [X, Y], 5,
                               progress_iter_func=progress_iter_func,
                               sum_axis=None)

    assert np.allclose(x, X.mean())

    #
    # Batch function that returns no results
    #
    def batch_func_no_results(batch_X, batch_Y):
        return None

    res = batch.mean_batch_map(batch_func_no_results, [X, Y], 5,
                               progress_iter_func=progress_iter_func,
                               sum_axis=None)

    assert res is None

    #
    # Invalid return value
    #
    def batch_func_invalid(batch_X, batch_Y):
        return 'Should not return a string'

    with pytest.raises(TypeError):
        batch.mean_batch_map(batch_func_invalid, [X, Y], 5)

    #
    # Prepend arguments to batch function
    #
    def batch_func_prepend(a, b, batch_X, batch_Y):
        assert a == 42
        assert b == 3.14
        return [batch_X.sum(), (batch_Y**2).sum(axis=1).sum()]

    [x, y] = batch.mean_batch_map(batch_func_prepend, [X, Y], 5,
                                  progress_iter_func=progress_iter_func,
                                  sum_axis=None, prepend_args=(42, 3.14))

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())


def test_mean_batch_map_in_order_per_sample_func():
    # Test `mean_batch_map` where the batch function returns per-sample
    # results
    from lasagne import batch

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return [batch_X + 2, (batch_Y**2).sum(axis=1)]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    [x, y] = batch.mean_batch_map(batch_func, [X, Y], 5,
                                  progress_iter_func=progress_iter_func,
                                  sum_axis=0)

    assert np.allclose(x, X.mean() + 2.0)
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Single return value
    #
    def batch_func_single(batch_X, batch_Y):
        return batch_X + 2

    [x] = batch.mean_batch_map(batch_func_single, [X, Y], 5, sum_axis=0)

    assert np.allclose(x, X.mean() + 2.0)
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Batch function that returns no results
    #
    def batch_func_no_results(batch_X, batch_Y):
        return None

    res = batch.mean_batch_map(batch_func_no_results, [X, Y], 5,
                               progress_iter_func=progress_iter_func,
                               sum_axis=0)

    assert res is None


def test_mean_batch_map_in_order_callable():
    # Get data from callable
    from lasagne import batch

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))

    def dataset(batchsize, shuffle_rng=None):
        return batch.batch_iterator([X, Y], batchsize, shuffle_rng=shuffle_rng)

    # Apply in order
    def batch_func(batch_X, batch_Y):
        return [batch_X.sum(), (batch_Y**2).sum(axis=1).sum()]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total is None
        # not leave
        assert not leave
        return iterator

    [x, y] = batch.mean_batch_map(batch_func, dataset, 5,
                                  progress_iter_func=progress_iter_func)

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())


def test_mean_batch_map_shuffled():
    from lasagne import batch

    # Data to extract batches from
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))

    # Array to keep track of which samples have been used
    used = np.zeros(X.shape, dtype=bool)

    def batch_func(batch_X, batch_Y):
        used[batch_X] = True
        return [batch_X.sum(), (batch_Y**2).sum(axis=1).sum()]

    [x, y] = batch.mean_batch_map(batch_func, [X, Y], 5)

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())
    assert used.all()
