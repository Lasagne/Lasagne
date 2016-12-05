import collections
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


def batch_iterator(dataset, batchsize, shuffle_rng=None, restartable=False):
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
    dataset: a tuple/list of array-likes, or an object with a `batch_iterator`
        method or a callable.
        The dataset to draw mini-batches from
    batchsize: int
        Mini-batch size
    shuffle_rng: `np.random.RandomState` or `None`
        Used to randomise element order. If `None`, elements will be extracted
        in order.
    restartable: bool (default=False)
        If `True`, require that the data-set `dataset` should be re-startable;
        e.g. passing a plain iterator as `dataset` will result in a
        `TypeError` as it cannot be restarted.

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
    elif not restartable and isinstance(dataset, collections.Iterator):
        return dataset
    else:
        # Don't know how to handle this
        raise TypeError('dataset should either: be a sequence of array-likes; '
                        'have a `batch_iterator` method; or be or a callable, '
                        'don\'t know how to handle {}'.format(type(dataset)))


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

        # Accumulate training results and number of examples
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
