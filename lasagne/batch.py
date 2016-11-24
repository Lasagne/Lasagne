import numpy as np


def is_arraylike(x):
    """
    Determine if `x` is array-like. `x` is index-able if it provides the
    `__len__` and `__getitem__` methods. Note that `__getitem__` should
    accept 1D NumPy integer arrays as an index

    Parameters
    ----------
    x: any
        The value to test for being array-like

    Returns
    -------
    bool
        `True` if array-like, `False` if not
    """
    return hasattr(x, '__len__') and hasattr(x, '__getitem__')


def is_sequence_of_arraylike(xs):
    """
    Determine if `x` is a sequence of array-like values. For definition of
    array-like see :func:`is_arraylike`. Tests the sequence by checking each
    value to see if it is array-like. Note that the containing sequence should
    be either a tuple or a list.

    Parameters
    ----------
    xs: tuple or list
        The sequence to test

    Returns
    -------
    bool
        `True` if tuple or list and all elements are array-like, `False`
        otherwise
    """
    if isinstance(xs, (tuple, list)):
        for x in xs:
            if not is_arraylike(x):
                return False
        return True
    return False


def length_of_arraylikes_in_sequence(xs):
    """
    Determine the length of the array-like elements in the sequence `xs`.
    `ValueError` is raised if the elements do not all have the same length.

    Parameters
    ----------
    xs: tuple or list
        Sequence of array-like elements.

    Returns
    -------
    int
        The length of the array-like elements in `xs`

    Raises
    ------
    ValueError
        If the lengths of the elements are not the same
    """
    N = len(xs[0])
    # Ensure remainder are consistent
    for i, d1 in enumerate(xs[1:]):
        if len(d1) != N:
            raise ValueError('Index-ables have inconsistent length; element '
                             '0 has length {}, while element {} has length '
                             '{}'.format(N, i+1, len(d1)))
    return N


def dataset_length(dataset):
    """
    Determine the length of the data set (number of samples).
    If `dataset` is ` sequence of array-like elements it is their length. If
    not, the length cannot be determined and `None` will be returned
    :param dataset: a data set; see the `batch_iterator` function
    :return: the length of the data set or `None`
    """
    if is_sequence_of_arraylike(dataset):
        return length_of_arraylikes_in_sequence(dataset)
    else:
        return None


def arraylikes_batch_iterator(dataset, batchsize,
                              shuffle_rng=None):
    """
    Create an iterator that generates mini-batches extracted from the
    sequence of array-likes `dataset`. The batches will have `batchsize`
    elements. If `shuffle_rng` is `None`, elements will be extracted in
    order. If it is not `None`, it will be used to randomise the order in
    which elements are extracted from `dataset`.

    The generated mini-batches take the form `[batch_x, batch_y, ...]`
    where `batch_x`, `batch_y`, etc. are extracted from each array-like in
    `dataset`.

    Parameters
    ----------
    dataset: tuple or list
        Sequence of array-like elements.
    batchsize: int
        Mini-batch size
    shuffle_rng: `np.random.RandomState` or `None`
        Used to randomise element order. If `None`, elements will be extracted
        in order.

    Returns
    -------
    iterator
        An iterator that generates items of type `[batch_x, batch_y, ...]`
        where `batch_x`, `batch_y`, etc are themselves arrays.
    """
    N = length_of_arraylikes_in_sequence(dataset)
    if shuffle_rng is not None:
        indices = shuffle_rng.permutation(N)
        for start_idx in range(0, N, batchsize):
            excerpt = indices[start_idx:start_idx + batchsize]
            yield [d[excerpt] for d in dataset]
    else:
        for start_idx in range(0, N, batchsize):
            yield [d[start_idx:start_idx+batchsize] for d in dataset]


def batch_iterator(dataset, batchsize, shuffle_rng=None):
    """
    Create an iterator that will iterate over the data in `dataset` in
    mini-batches consisting of `batchsize` samples, with their order shuffled
    using the random number generate `shuffle_rng` if supplied or in-order if
    not.

    The data in `dataset` must take the form of either:

    - a sequence of array-likes (see :func:`is_arraylike`) (e.g. NumPy
        arrays) - one for each variable (input/target/etc) - where each
        array-like contains an entry for each sample in the complete dataset.
        The use of array-like allows the use of NumPy arrays or other objects
        that support `__len__` and `__getitem__`:

    >>> # 5000 samples, 3 channel 24x24 images
    >>> train_X = np.random.normal(size=(5000,3,24,24))
    >>> # 5000 samples, classes
    >>> train_y = np.random.randint(0, 5, size=(5000,))
    >>> shuffle_rng = np.random.RandomState(12345)
    >>> batches = batch_iterator([train_X, train_y], batchsize=128,
    ...                          shuffle_rng=shuffle_rng)

    - an object that has the method
        `dataset.batch_iterator(batchsize, shuffle_rng=None) -> iterator` or a
        callable of the form
        `dataset(batchsize, shuffle_rng=None) -> iterator` that returns an
        iterator, where the iterator generates mini-batches,
        where each mini-batch is a list of numpy arrays:

    >>> def make_iterator(X, y):
    ...     def iter_minibatches(batchsize, shuffle_rng=None):
    ...         indices = np.arange(X.shape[0])
    ...         if shuffle_rng is not None:
    ...             shuffle_rng.shuffle(indices)
    ...         for i in range(0, indices.shape[0], batchsize):
    ...             batch_ndx = indices[i:i+batchsize]
    ...             batch_X = X[batch_ndx]
    ...             batch_y = y[batch_ndx]
    ...             yield [batch_X, batch_y]
    ...     return iter_minibatches
    >>> shuffle_rng = np.random.RandomState(12345)
    >>> batches = batch_iterator(make_iterator(train_X, train_y),
    ...                          batchsize=128, shuffle_rng=shuffle_rng)

    Parameters
    ----------
    dataset: a tuple/list of array-likes, or an object with a `batch_iterator`
        method or a callable.
        The dataset to draw mini-batches from
    batchsize: int
        Mini-batch size
    shuffle_rng: `np.random.RandomState` or `None`
        Used to randomise element order. If `None`, elements will be extracted
        in order.

    Returns
    -------
    iterator
        An iterator that generates items of type `[batch_x, batch_y, ...]`
        where `batch_x`, `batch_y`, etc are themselves arrays.
    """
    if is_sequence_of_arraylike(dataset):
        # First, try sequence of array-likes; likely the most common dataset
        # type. Furthermore, using the array-like interface is preferable to
        # using `batch_iterator` method
        return arraylikes_batch_iterator(
                dataset, batchsize, shuffle_rng=shuffle_rng)
    elif hasattr(dataset, 'batch_iterator'):
        # Next, try `batch_iterator` method
        return dataset.batch_iterator(batchsize, shuffle_rng=shuffle_rng)
    elif callable(dataset):
        # Now try callable; basically the same as `batch_iterator`
        return dataset(batchsize, shuffle_rng=shuffle_rng)
    else:
        # Don't know how to handle this
        raise TypeError('dataset should either: be a sequence of array-likes; '
                        'have a `batch_iterator` method; or be or a callable, '
                        'don\'t know how to handle {}'.format(type(dataset)))
