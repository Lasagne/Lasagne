import collections, itertools
import numpy as np


def is_arraylike(x):
    """
    Determine if `x` is array-like. `x` is index-able if it provides the
    `__len__` and `__getitem__` methods. Note that `__getitem__` should
    accept 1D NumPy integer arrays as indices.

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


def is_sequence_of_arraylike(data):
    """
    Determine if `data` is a sequence of array-like values. For definition of
    array-like see :func:`is_arraylike`. Tests the sequence by checking each
    value to see if it is array-like. Note that the containing sequence should
    be either a tuple or a list.

    Parameters
    ----------
    data: tuple or list
        The sequence to test

    Returns
    -------
    bool
        `True` if tuple or list and all elements are array-like, `False`
        otherwise
    """
    if isinstance(data, (tuple, list)):
        for x in data:
            if not is_arraylike(x):
                return False
        return True
    return False


def is_dataset(data, circular=False):
    """
    Determine if `data` can be used as a data set. It can be used as a
    data set if:
    - it has a `batch_iterator` method if `circular` is `False`, or
    - it has a `circular_batch_iterator` method if `circular` is `True`, or
    - it is a callable, or
    - `restartable` is `False` and it is an iterator, or
    - it is a sequence of array-likes (see :func:`is_sequence_of_arraylike`)

    Parameters
    ----------
    data:
        Object to test
    circular: bool
        If `False`, will look for the `batch_iterator`
        method, if `True` will look for `circular_batch_iterator`

    Returns
    -------
    bool
        `True` if `dataset can be used as a data set, `False` otherwise
    """
    iter_method_name = 'circular_batch_iterator' if circular \
        else 'batch_iterator'

    return hasattr(data, iter_method_name) or callable(data) or \
           isinstance(data, collections.Iterator) or \
           is_sequence_of_arraylike(data)


def is_sequence_of_datasets(data, circular=False):
    """
    Determine if `xs` is a sequence of data sets. For definition of a
    data set see :func:`is_dataset`. Tests the sequence by checking each
    value to see if it can be used as a data set. Note that the containing
    sequence should be either a tuple or a list.

    Parameters
    ----------
    data: tuple or list
        The sequence to test
    circular: bool
        If `False`, will look for the `batch_iterator`
        method, if `True` will look for `circular_batch_iterator`

    Returns
    -------
    bool
        `True` if tuple or list and all elements can be used as data sets,
        `False` otherwise
    """
    if isinstance(data, (tuple, list)):
        for x in data:
            if not is_dataset(x, circular=circular):
                return False
        return True
    return False


def length_of_arraylikes_in_sequence(data):
    """
    Determine the length of the array-like elements in the sequence `data`.
    `ValueError` is raised if the elements do not all have the same length.

    Parameters
    ----------
    data: tuple or list
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
    N = len(data[0])
    # Ensure remainder are consistent
    for i, d1 in enumerate(data[1:]):
        if len(d1) != N:
            raise ValueError('Index-ables have inconsistent length; element '
                             '0 has length {}, while element {} has length '
                             '{}'.format(N, i+1, len(d1)))
    return N


def dataset_length(data):
    """
    Determine the length of the data set (number of samples).
    If `data` is ` sequence of array-like elements it is their length. If
    not, the length cannot be determined and `None` will be returned
    :param data: a data set; see the `batch_iterator` function
    :return: the length of the data set or `None`
    """
    if is_sequence_of_arraylike(data):
        return length_of_arraylikes_in_sequence(data)
    else:
        return None


def arraylikes_batch_iterator(data, batchsize,
                              shuffle_rng=None):
    """
    Create an iterator that generates mini-batches extracted from the
    sequence of array-likes `data`. The batches will have `batchsize`
    elements. If `shuffle_rng` is `None`, elements will be extracted in
    order. If it is not `None`, it will be used to randomise the order in
    which elements are extracted from `dataset`.

    The generated mini-batches take the form `[batch_x, batch_y, ...]`
    where `batch_x`, `batch_y`, etc. are extracted from each array-like in
    `dataset`.

    Parameters
    ----------
    data: tuple or list
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
    N = length_of_arraylikes_in_sequence(data)
    if shuffle_rng is not None:
        indices = shuffle_rng.permutation(N)
        for start_idx in range(0, N, batchsize):
            excerpt = indices[start_idx:start_idx + batchsize]
            yield [d[excerpt] for d in data]
    else:
        for start_idx in range(0, N, batchsize):
            yield [d[start_idx:start_idx+batchsize] for d in data]


def circular_arraylikes_batch_iterator(data, batchsize,
                                       shuffle_rng=None):
    """
    Create an iterator that generates an infinite sequence of mini-batches
    extracted from the sequence of array-likes `data`. The batches will
    have `batchsize` elements. If `shuffle_rng` is `None`, elements will
    be extracted in order. If it is not `None`, it will be used to
    randomise the order in which elements are extracted from `data`.
    Once the supply of elements from `data` are exhausted, it will start
    from the beginning, or from a random position if shuffling is used.

    The generated mini-batches take the form `[batch_x, batch_y, ...]`
    where `batch_x`, `batch_y`, etc. are extracted from each array-like in
    `data`.

    Parameters
    ----------
    data: tuple or list
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
    N = length_of_arraylikes_in_sequence(data)
    if shuffle_rng is not None:
        indices = shuffle_rng.permutation(N)
        i = 0
        while True:
            j = i + batchsize
            if j <= N:
                # Within size of data
                batch_ndx = indices[i:j]
                i = j
            else:
                # Wrap over
                # Compute number of elements required to make up the batch
                k = batchsize - (N - i)
                # Get available indices
                batch_ndx = indices[i:N]
                # Re-populate indices
                indices = shuffle_rng.permutation(N)
                # Get remaining indices and append
                batch_ndx = np.append(batch_ndx, indices[:k], axis=0)
                i = k
            yield [d[batch_ndx] for d in data]
    else:
        i = 0
        while True:
            j = i + batchsize
            if j <= N:
                # Within size of data
                yield [d[i:j] for d in data]
                i = j
            else:
                # Wrap over
                # Compute number of elements required to make up the batch
                k = batchsize - (N - i)
                yield [np.append(d[i:N], d[:k], axis=0) for d in data]
                i = k


def _nested_batch_iterator(iterators):
    for batch in itertools.izip(*iterators):
        yield sum(batch, [])

def batch_iterator(data, batchsize, shuffle_rng=None, restartable=False):
    """
    Create an iterator that will iterate over the data in `data` in
    mini-batches consisting of `batchsize` samples, with their order shuffled
    using the random number generate `shuffle_rng` if supplied or in-order if
    not.

    The data in `data` must take the form of either:

    - a sequence of array-likes (see :func:`is_arraylike`) (e.g. NumPy
        arrays) - one for each variable (input/target/etc) - where each
        array-like contains an entry for each sample in the complete data set.
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
        `data.batch_iterator(batchsize, shuffle_rng=None) -> iterator` or a
        callable of the form
        `data(batchsize, shuffle_rng=None) -> iterator` that returns an
        iterator, where the iterator generates mini-batches,
        where each mini-batch is a list of numpy arrays:

    >>> def make_batch_iterator(X, y):
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
    >>> batches = batch_iterator(make_batch_iterator(train_X, train_y),
    ...                          batchsize=128, shuffle_rng=shuffle_rng)

    - an iterator; note that the `restartable` argument must be `False`
        otherwise `TypeError` will be raised. Also note tahtn the `batchsize`
        and `shuffle_rng` arguments will be ignored as they cannot be passed
        to the iterator as construction time as it has already been built

    >>> X = np.random.normal(size=(2048, 256))
    >>> y = np.random.randint(low=0, high=10, size=(2048,))
    >>> def make_iter():
    ...     indices = np.arange(X.shape[0])
    ...     np.random.shuffle(indices)
    ...     for i in range(0, indices.shape[0], 128):
    ...         batch_ndx = indices[i:i+128]
    ...         batch_X = X[batch_ndx]
    ...         batch_y = y[batch_ndx]
    ...         yield [batch_X, batch_y]
    >>> batch_iter = make_iter()
    >>> batches = batch_iterator(batch_iter, batchsize=42, restartable=False)

    Parameters
    ----------
    data: a tuple/list of array-likes, or an object with a `batch_iterator`
        method or a callable, or a tuple/list of data sets.
        The data set to draw mini-batches from
    batchsize: int
        Mini-batch size
    shuffle_rng: `np.random.RandomState` or `None`
        Used to randomise element order. If `None`, elements will be extracted
        in order.
    restartable: bool (default=False)
        If `True`, require that the data set `data` should be re-startable;
        e.g. passing a plain iterator as `data` will result in a
        `TypeError` as it cannot be restarted.

    Returns
    -------
    iterator
        An iterator that generates items of type `[batch_x, batch_y, ...]`
        where `batch_x`, `batch_y`, etc are themselves arrays.
    """
    if is_sequence_of_arraylike(data):
        # First, try sequence of array-likes; likely the most common data
        # type. Furthermore, using the array-like interface is preferable to
        # using `batch_iterator` method
        return arraylikes_batch_iterator(
                data, batchsize, shuffle_rng=shuffle_rng)
    elif hasattr(data, 'batch_iterator'):
        # Next, try `batch_iterator` method
        return data.batch_iterator(batchsize, shuffle_rng=shuffle_rng)
    elif callable(data):
        # Now try callable; basically the same as `batch_iterator`
        return data(batchsize, shuffle_rng=shuffle_rng)
    elif not restartable and isinstance(data, collections.Iterator):
        return data
    elif is_sequence_of_datasets(data):
        iterators = [batch_iterator(x, batchsize, shuffle_rng, restartable)
                     for x in data]
        return _nested_batch_iterator(iterators)
    else:
        # Don't know how to handle this
        raise TypeError('data should either: be a sequence of array-likes; '
                        'have a `batch_iterator` method; or be or a callable, '
                        'don\'t know how to handle {}'.format(type(data)))


def batch_map(func, data, batchsize, restartable=False,
              progress_iter_func=None, prepend_args=None):
    """
    Apply a function to all the samples in a data set by breaking the data
    set into mini-batches and applying the function to each mini-batch.
    Note that samples are processed in-order; to process samples in
    random order, use the `mean_batch_apply` function.
    Returns the per-sample results as a list of arrays.

    The function `func` should return the result for each sample in the
    mini-batch as an array. To return multiple results (e.g. loss and errors)
    return a list of arrays (e.g. `[loss_array, error_array]`)

    `data` must must either be a sequence of array-likes, an object with a
    `batch_iterator` method or a callable; see :func:`batch.batch_iterator`

    Parameters
    ----------
    func: callable `func(*batch) -> results`
        The function to call on each mini-batch. Note that the results
    data: data set
        The data to draw mini-batches from
    batchsize: int
        The number of samples per mini-batch
    restartable: bool (default=False)
        If `True`, require that the data-set `data` should be re-startable;
        e.g. passing a plain iterator as `data` will result in a
        `TypeError` as it cannot be restarted.
    progress_iter_func: [optional] callable
        `progress_iter_func(iterator, total=total, leave=leave)`
        A `tqdm` style function that will be passed the iterator that
        generates training batches along with the total number of batches
        and `False` for the `leave` parameter. By passing either
        `tqdm.tqdm` or `tqdm.tqdm_notebook` as this argument you can have
        the training loop display a progress bar.
    prepend_args: [optional] tuple
        Arguments to prepend to the arguments passed to `func`

    Returns
    -------
    list
        The per-sample sum of the results of the function `func` e.g.
        `[batch_A, batch_B, ...]`
        Returns an empty list if there were 0 samples in the data set.
    """
    # Accumulator for results and number of samples
    results = []

    # Create the iterator that will generate mini-batches
    batch_iter = batch_iterator(data, batchsize, restartable=restartable)

    # If `progress_iter_func` is not `None`, apply it
    if progress_iter_func is not None:
        n_samples = dataset_length(data)
        if n_samples is not None:
            n_batches = n_samples // batchsize
            if (n_samples % batchsize) > 0:
                n_batches += 1
        else:
            n_batches = None
        batch_iter = progress_iter_func(batch_iter, total=n_batches,
                                        leave=False)

    # Apply `func` to each batch
    for batch_i, batch in enumerate(batch_iter):
        # Apply on batch and check the type of the results
        if prepend_args is not None:
            batch_results = func(*(prepend_args + tuple(batch)))
        else:
            batch_results = func(*batch)
        if batch_results is None:
            pass
        elif isinstance(batch_results, np.ndarray):
            batch_results = [batch_results]
        elif isinstance(batch_results, list):
            pass
        else:
            raise TypeError(
                    'Batch function should return a list of results, a '
                    'single result as a NumPy array or float, or None, '
                    'not {}'.format(type(batch_results)))

        # Accumulate training results
        if batch_results is not None:
            results.append(batch_results)

    # Concatenate result arrays
    if len(results) > 0:
        results = zip(*results)
        results = [np.concatenate(list(r), axis=0) for r in results]
        return results
    else:
        return None


def mean_batch_map(func, data, batchsize, shuffle_rng=None, restartable=False,
                   progress_iter_func=None, sum_axis=None,
                   prepend_args=None):
    """
    Apply a function to all the samples in a data set by breaking the data
    set into mini-batches and applying the function to each mini-batch.
    Returns the across-samples mean of the results returned by `func`

    The `sum_axis` arguments tells `mean_batch_map` how to process the
    results of `func` before accumulating them:
    - If `sum_axis` is `None`, `func` should return the
    across-samples SUM of the  results of operating on the mini-batch the
    sum of the values for the samples, e.g. for loss and error it should
    return `[sum([loss0, loss1, ... lossN]), sum([err0, err1, ... errN])]`
    `mean_batch_apply` will accumulate these values and divide them by the
    number of samples in the data set at the end, returning the mean values
    for the complete data set.
    - Otherwise, `sum_axis` should specify the axis or axes over which
    the the batch results should be summed, e.g. if `func` returns a
    per-sample loss and error in two arrays
    `[[loss0, loss1, ... lossN], [err0, err1, ... errN]`, give `sum_axis`
    a value of `0` to sum over axis 0 to get the per-batch loss and error.
    `mean_batch_map` will accumulate these and divide by the number of samples
    at the end to get the mean.

    `data` must must either be a sequence of array-likes, an object with a
    `batch_iterator` method or a callable; see :func:`batch.batch_iterator`

    Parameters
    ----------
    func: callable `func(*batch) -> results`
        The function to call on each mini-batch. Note that the results
    data: dataset
        The data to draw mini-batches from
    batchsize: int
        The number of samples per mini-batch
    shuffle_rng: `None` or a `np.random.RandomState`
        A random number generator used to shuffle the order of samples. If one
        is not provided samples will be processed in-order (e.g.
        during validation and test).
    restartable: bool (default=False)
        If `True`, require that the data-set `data` should be re-startable;
        e.g. passing a plain iterator as `data` will result in a
        `TypeError` as it cannot be restarted.
    progress_iter_func: [optional] callable
        `progress_iter_func(iterator, total=total, leave=leave)`
        A `tqdm` style function that will be passed the iterator that
        generates training batches along with the total number of batches
        and `False` for the `leave` parameter. By passing either
        `tqdm.tqdm` or `tqdm.tqdm_notebook` as this argument you can have
        the training loop display a progress bar.
    sum_axis: (default=`None`) int, tuple of ints or None
        If an integer or a tuple of integers, the results returned by `func`
        will be summed across this axis / these axes before being accumulated;
        e.g. if `func` returns an array of per-sample losses, with axis 0
        being the sample dimension, passing a value of `0` as `sum_axis`
        will cause these results to be summed along axis 0 to get the
        per-batch sum before accumulating the losses. The total summed loss
        will be divided by the number of samples at the end in order to
        compute the mean loss.
    prepend_args: [optional] tuple
        Arguments to prepend to the arguments passed to `func`

    Returns
    -------
    list
        The sum of the results of the function `fn` divided by the number of
        samples processed, e.g.
        `[sum(outA_per_batch) / n_samples,
          sum(outB_per_batch) / n_samples,
          ...]`
    """
    # Accumulator for results and number of samples
    results_accum = None
    n_samples_accum = 0

    # Create the iterator that will generate mini-batches
    batch_iter = batch_iterator(data, batchsize, shuffle_rng=shuffle_rng,
                                restartable=restartable)

    # If `progress_iter_func` is not `None`, apply it
    if progress_iter_func is not None:
        n_samples = dataset_length(data)
        if n_samples is not None:
            n_batches = n_samples // batchsize
            if (n_samples % batchsize) > 0:
                n_batches += 1
        else:
            n_batches = None
        batch_iter = progress_iter_func(batch_iter, total=n_batches,
                                        leave=False)

    # Train on each batch
    for batch_i, batch in enumerate(batch_iter):
        # Get number of samples in batch; can vary
        batch_n = batch[0].shape[0]

        # Apply on batch and check the type of the results
        if prepend_args is not None:
            batch_results = func(*(prepend_args + tuple(batch)))
        else:
            batch_results = func(*batch)
        if batch_results is None:
            pass
        elif isinstance(batch_results, (np.ndarray, float)):
            batch_results = [batch_results]
        elif isinstance(batch_results, list):
            pass
        else:
            raise TypeError(
                    'Batch function should return a list of results, a '
                    'single result as a NumPy array or float, or None, '
                    'not {}'.format(type(batch_results)))

        # Accumulate results and number of samples
        if results_accum is None:
            # Initialise the accumulator to the batch results if `func`
            # returns summed results or if it returned None;
            # don't attempt to iterate over None and sum each item
            if sum_axis is None or batch_results is None:
                results_accum = batch_results
            else:
                results_accum = [br.sum(axis=sum_axis) for br in batch_results]
        else:
            if batch_results is not None:
                for i in range(len(results_accum)):
                    br = batch_results[i]
                    if sum_axis is not None:
                        br = br.sum(axis=sum_axis)
                    results_accum[i] += br
        n_samples_accum += batch_n

    # Divide by the number of training examples used to compute mean
    if results_accum is not None:
        results_accum = [np.array(r).astype(float) / n_samples_accum
                         for r in results_accum]

    return results_accum
