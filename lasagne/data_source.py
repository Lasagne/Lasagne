"""
Data source module.

Defines types for iterating over data in mini-batches to be passed to neural
network training functions.
"""

import six
import numpy as np
from .random import get_rng


def _num_batches(n, batch_size):
    """Compute the number of mini-batches required to cover a data set of
    size `n` using batches of size `batch_size`.

    Parameters
    ----------
    n: int
        the number of samples in the data set
    batch_size: int
        the mini-batch size

    Returns
    -------
    int: the number of batches required
    """
    b = n // batch_size
    if n % batch_size > 0:
        b += 1
    return b


def _length_of_batch(batch):
    """Get the number of samples in the mini-batch `batch`.

    `batch` can be:
    - a NumPy array, in which case `len(batch)` (size of first axis) will
      be returned
    - a list or tuple, in which case `_length_of_batch` will be invoked
      (recursively) on the first element

    As a consequence, mini-batches can be structured; lists and tuples can
    be nested arbitrarily deep.

    Parameters
    ----------
    batch: list, tuple or NumPy array
        a mini-batch

    Returns
    -------
    int: the number of samples in the mini-batch
    """
    if isinstance(batch, (list, tuple)):
        return _length_of_batch(batch[0])
    else:
        return len(batch)


def _trim_batch(batch, length):
    """Trim the mini-batch `batch` to the size `length`.

    `batch` can be:
    - a NumPy array, in which case it's first axis will be trimmed to size
      `length`
    - a list or tuple, in which case `_trim_batch` applied recursively to
      each element and the resulting list returned

    As a consequence, mini-batches can be structured; lists and tuples can
    be nested arbitrarily deep.

    Parameters
    ----------
    batch: list, tuple or NumPy array
        the mini-batch to trim
    length: int
        the size to which `batch` is to be trimmed

    Returns
    -------
    list or NumPy array of same structure as `batch`
    The trimmed mini-batch
    """
    if isinstance(batch, (list, tuple)):
        return [_trim_batch(b, length) for b in batch]
    else:
        return batch[:length]


class AbstractDataSource (object):
    """Data source abstract base class
    """
    def num_samples(self, **kwargs):
        """
        Get the number of samples in this data source. Returns

        Returns
        -------
        int, `np.inf` or `None`.
            An int if the number of samples is known, `np.inf` if it is
            infinite or `None` if the number of samples is unknown.
        """
        raise NotImplementedError

    def batch_iterator(self, batch_size, **kwargs):
        """
        Return an iterator that generates mini-batches extracted from `self`.

        Parameters
        ----------
        batch_size: int
            Mini-batch size

        Returns
        -------
            An iterator that yields mini-batches.
        """
        raise NotImplementedError

    def with_params(self, **settings):
        """
        Convenience function for constructing an `ApplyParamsDataSource`
        instance that wraps `self` and applies settings when invoking the
        `batch_iterator` and `num_samples` methods.

        Parameters
        ----------
        settings: keyword arguments
            Settings that are to be applied

        Returns
        -------
        ApplyParamsDataSource
            Wrapped data source

        Examples
        --------
        >>> X = np.random.normal(size=(7, 10))
        >>> ds = ArrayDataSource([X])

        >>> inf_ds = ds.with_params(epochs=-1)
        >>> batch_iter = inf_ds.batch_iterator(15)

        Is equivalent to:

        >>> batch_iter = ds.batch_iterator(15, epochs=-1)

        Given that the above increases the amount of code, the main use of
        the `with_params` method is to apply parameters to a data source when
        combining it with others in a `CompositeDataSource` instance.
        """
        return ApplyParamsDataSource(self, **settings)

    def batch_map(self, func, batch_size, progress_iter_func=None,
                  n_batches=None, prepend_args=None, **kwargs):
        """A batch oriented implementation of `map`.
        Applies a function to all the samples in this data source by breaking
        the data into mini-batches and applying the function to each
        mini-batch.
        Returns the per-sample results.

        This method is a wrapper around the :func:`batch_map` function;
        please see its documentation for more information and examples.

        The function `func` should return the result for each sample in the
        mini-batch as an array. To return multiple results (e.g. loss and
        errors) return a list of arrays (e.g. `[loss_array, error_array]`)

        Parameters
        ----------
        func: callable `func(*batch) -> results`
            The function to call on each mini-batch. Note that the results
            must be `None`, a list or a NumPy array
        batch_size: int
            The mini-batch size
        progress_iter_func: [optional] callable
            `progress_iter_func(iterator, total=total, leave=leave)`
            A `tqdm` style function that will be passed the iterator that
            generates training batches along with the total number of batches
            and `False` for the `leave` parameter. By passing either
            `tqdm.tqdm` or `tqdm.tqdm_notebook` as this argument you can have
            the training loop display a progress bar.
        n_batches: [optional] integer that specifies the number of mini-batches
            to process before returning
        prepend_args: [optional] tuple
            Arguments to prepend to the arguments passed to `func`

        Returns
        -------
        list
            The per-sample sum of the results of the function `func` e.g.
            `[batch_A, batch_B, ...]`
            Returns an empty list if there were 0 samples in the data set.

        Examples
        --------
        Define a function to apply to samples:
        >>> def sqr_sum(x):
        ...     return (x ** 2).sum(axis=1)

        Construct data to process and create a data source:
        >>> X = np.random.normal(size=(7, 10))
        >>> ds = ArrayDataSource([X])

        Apply the function defined above:
        >>> X_sqr_sum = ds.batch_map(sqr_sum, batch_size=5)
        >>> assert (X_sqr_sum[0] == (X ** 2).sum(axis=1)).all()
        """
        if n_batches is None:
            n = self.num_samples(**kwargs)
            if n == np.inf:
                raise ValueError('Data set has infinite size but no n_batches '
                                 'limit specified')
            elif n is not None:
                n_batches = _num_batches(n, batch_size)
        batch_iter = self.batch_iterator(batch_size, **kwargs)
        return batch_map(func, batch_iter, progress_iter_func, n_batches,
                         prepend_args)

    def mean_batch_map(self, func, batch_size, progress_iter_func=None,
                       sum_axis=None, n_batches=None, prepend_args=None,
                       **kwargs):
        """
        Apply a function to all the samples in this data source by breaking
        the data into mini-batches and applying the function to each
        mini-batch.
        Returns the across-samples mean of the results returned by `func`

        This method is a wrapper around the :func:`mean_batch_map` function;
        please see its documentation for more information and examples.

        The `sum_axis` arguments tells `mean_batch_map` how to process the
        results of `func` before accumulating them:
        - If `sum_axis` is `None`, `func` should return the
        across-samples SUM of the  results of operating on the mini-batch the
        sum of the values for the samples, e.g. for loss and error it should
        return
        `[sum([loss0, loss1, ... lossN]), sum([err0, err1, ... errN])]`
        - Otherwise, `sum_axis` should specify the axis or axes over which
        the the batch results should be summed, e.g. if `func` returns a
        per-sample loss and error in two arrays
        `[[loss0, loss1, ... lossN], [err0, err1, ... errN]`, give `sum_axis`
        a value of `0` to sum over axis 0 to get the per-batch loss and
        error.
        These esults will be accumulated and divided by the number of samples
        at the end to get the mean.

        Parameters
        ----------
        func: callable `func(*batch) -> results`
            The function to call on each mini-batch. Note that the results
            must be `None`, a list or a NumPy array
        batch_size: int
            The mini-batch size
        progress_iter_func: [optional] callable
            `progress_iter_func(iterator, total=total, leave=leave)`
            A `tqdm` style function that will be passed the iterator that
            generates training batches along with the total number of batches
            and `False` for the `leave` parameter. By passing either
            `tqdm.tqdm` or `tqdm.tqdm_notebook` as this argument you can have
            the training loop display a progress bar.
        sum_axis: (default=`None`) int, tuple of ints or None
            If an integer or a tuple of integers, the results returned by
            `func` will be summed across this axis / these axes before being
            accumulated; e.g. if `func` returns an array of per-sample
            losses, with axis 0 being the sample dimension, passing a value
            of `0` as `sum_axis` will cause these results to be summed along
            axis 0 to get the per-batch sum before accumulating the losses.
            The total summed loss will be divided by the number of samples at
            the end in order to compute the mean loss.
        n_batches: [optional] integer that specifies the number of
            mini-batches to process before returning
        prepend_args: [optional] tuple
            Arguments to prepend to the arguments passed to `func`

        Returns
        -------
        list
            The sum of the results of the function `fn` divided by the number
            of samples processed, e.g.
            `[sum(outA_per_batch) / n_samples,
              sum(outB_per_batch) / n_samples,
              ...]`

        Examples
        --------
        The following examples will demonstrate the use of `mean_batch_map`
        to compute binary cross entropy loss over a data set.
        A few variants will be demonstrated:
        - the default behaviour in which the function being applied should
          return the sum over the batch sample axis
        - applying only to a limited number of batches; useful in cases where
          the data source will generate an infinite number of samples
        - having the function return per sample results and maving
          `mean_batch_map` perform the sum operation. This is easier to
          understand but less efficient as a Theano function would have to
          move more data back from the GPU.

        Define a function to compute the per-sample binary cross entropy
        loss:
        >>> def binary_crossentropy_loss(pred, target):
        ...     e = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
        ...     return e.mean(axis=1)

        Now define a function that computes the *SUM* of the binary cross
        entropy losses over the sample axis (axis 0), as the default
        behaviour of `mean_batch_map` will sum them up and divide by the
        number of samples at the end:
        >>> def binary_crossentropy_loss_sum(pred, target):
        ...     return binary_crossentropy_loss(pred, target).sum()

        Construct prediction and target data
        >>> pred = np.random.uniform(0.0, 1.0, size=(15, 10))
        >>> tgt = np.random.uniform(0.0, 1.0, size=(15, 10))
        >>> ds = ArrayDataSource([pred, tgt])

        Apply the loss sum function defined above:
        >>> loss = ds.mean_batch_map(binary_crossentropy_loss_sum,
        ...                          batch_size=5)
        >>> assert np.allclose(
        ...     loss, binary_crossentropy_loss(pred, tgt).mean())
        """
        if n_batches is None:
            n = self.num_samples(**kwargs)
            if n == np.inf:
                raise ValueError('Data set has infinite size but no n_batches '
                                 'limit specified')
            elif n is not None:
                n_batches = _num_batches(n, batch_size)
        batch_iter = self.batch_iterator(batch_size, **kwargs)
        return mean_batch_map(func, batch_iter, progress_iter_func,
                              sum_axis, n_batches, prepend_args)

    @staticmethod
    def _get_shuffle_rng(shuffle):
        if shuffle is False:
            return None
        elif shuffle is True:
            return get_rng()
        else:
            return shuffle


class ApplyParamsDataSource (AbstractDataSource):
    """Apply parameters to wrapped data source.

    Invokations to the `batch_iterator` and `num_samples` methods will be
    passed on to the underlying data source, but with the additional keyword
    arguments that are passed to the constructor.

    Parameters
    ----------
    datasource: AbstractDataSource
        The data source to wrap
    settings: keyword arguments
        The additional keyword arguments to pass when invoking the
        `batch_iterator` and `num_samples` methods.

    Examples
    --------
    >>> X = np.random.normal(size=(7, 10))
    >>> ds = ArrayDataSource([X])

    Using `ApplyParamsDataSource` or the `with_params` method to add
    parameters:

    >>> inf_ds = ApplyParamsDataSource(ds, epochs=-1)
    >>> batch_iter = inf_ds.batch_iterator(15)

    Is equivalent to:

    >>> batch_iter = ds.batch_iterator(15, epochs=-1)

    Using the `with_params` method
    ------------------------------
    The `with_params` convenience method (availble on all data source classes)
    simplifies the construction of `ApplyParamsDataSource` instances.


    Using `with_params`:

    >>> inf_ds = ds.with_params(epochs=-1)

    Is equivalent to:

    >>> inf_ds = ApplyParamsDataSource(ds, epochs=-1)
    """
    def __init__(self, datasource, **params):
        self.datasource = datasource
        self.params = params

    def num_samples(self, **kwargs):
        """
        Get the number of samples in this data source.

        Returns
        -------
        int, `np.inf` or `None`.
            An int if the number of samples is known, `np.inf` if it is
            infinite or `None` if the number of samples is unknown.
        """
        kwargs.update(self.params)
        return self.datasource.num_samples(**kwargs)

    def batch_iterator(self, batch_size, **kwargs):
        """
        Return an iterator that generates mini-batches extracted from `self`.

        Parameters
        ----------
        batch_size: int
            Mini-batch size

        Returns
        -------
            An iterator that yields mini-batches.
        """
        kwargs.update(self.params)
        return self.datasource.batch_iterator(batch_size, **kwargs)


class ArrayDataSource (AbstractDataSource):
    """A data source whose data comes from NumPy arrays (or array-like
    objects.

    The arrays can either be NumPy arrays, or array-like objects that
    implement `__len__` and `__getitem__`. `__getitem__` must accept
    integer indices, slices or index arrays (a 1D array of integers that are
    the indices of samples to retrieve).

    Parameters
    ----------
    data: list
        A list of arrays from which data is drawn.

    Examples:
    Create a data set of size 12, where each input sample is a 7-element
    vector and ground classifications are integers:
    >>> X = np.random.normal(size=(12,7))
    >>> y = np.random.randint(0, 10, size=(12,))
    >>> ds = ArrayDataSource([X, y])

    Iterate over data, drawing 5-element mini-batches, in order:
    >>> for batch_X, batch_y in ds.batch_iterator(5):
    ...     # Perform operations on batch_X and batch_y
    ...     pass

    Iterate over data, drawing 5-element mini-batches, shuffled randomly
    using a RandomState:
    >>> rng = np.random.RandomState(12345)
    >>> for batch_X, batch_y in ds.batch_iterator(5, shuffle=rng):
    ...     # Perform operations on batch_X and batch_y
    ...     pass

    Iterate over data, drawing 5-element mini-batches, shuffled randomly
    using Lasagne's default random number generator:
    >>> for batch_X, batch_y in ds.batch_iterator(5, shuffle=True):
    ...     # Perform operations on batch_X and batch_y
    ...     pass

    Randomly choose 5 samples and use only those:
    >>> dsi = ArrayDataSource([X, y], indices=np.random.permutation(10)[:5])
    >>> for batch_X, batch_y in dsi.batch_iterator(5):
    ...     # Perform operations on batch_X and batch_y
    ...     pass

    The `epochs` parameter will cause the iterator to walk over the data
    a specified number of times:
    >>> for batch_X, batch_y in ds.batch_iterator(5, shuffle=rng, epochs=10):
    ...     # Perform operations on batch_X and batch_y
    ...     break

    If it is given the value `-1`, the iterator will repeat infinitely:
    >>> for batch_X, batch_y in ds.batch_iterator(5, shuffle=rng, epochs=-1):
    ...     # Perform operations on batch_X and batch_y
    ...     break
    """
    def __init__(self, data, indices=None):
        if not isinstance(data, list):
            raise TypeError('data must be a list of array-like objects, not '
                            'a {}'.format(type(data)))
        self.data = data
        self.indices = indices
        if self.indices is not None:
            self.length = len(self.indices)
        else:
            # Get the length from the first array
            self.length = len(data[0])
            # Ensure that rest of the arrays have the same length
            for i, d1 in enumerate(data[1:]):
                if len(d1) != self.length:
                    raise ValueError(
                        'Arrays have inconsistent length; array 0 has '
                        'length {}, while array {} has length {}'.format(
                            self.length, i+1, len(d1)))

    def num_samples(self, epochs=1, **kwargs):
        """
        Get the number of samples in this data source.

        Parameters
        ----------
        epochs: int
            The number of repetitions, or `-1` for infinite, in which case
            `np.inf` will be returned. A value of 0 or a negative value that
            is not -1 will cause `ValueError` to be raised.

        Returns
        -------
        int or `np.inf`
            If `epochs` is `-1`, `np.inf`.
            Otherwise, the length of the data set multiplied by the value of
            `epochs. The length of the data set is the length of the
            indices array - if one was provided to the constructor - or the
            length of the arrays passed in the list `data`.
        """
        if epochs == 0 or epochs < -1:
            raise ValueError('Invalid number of epochs; should be >= 1 or '
                             '-1, not {}'.format(epochs))
        if epochs == -1:
            return np.inf
        else:
            return self.length * epochs

    def batch_iterator(self, batch_size, shuffle=None, epochs=1,
                       **kwargs):
        """
        Create an iterator that generates mini-batches extracted from the
        arrays that make up this dataset. The batches will have `batchsize`
        elements, with the exception of the final batch which will have less
        if there are insufficient elements left to make a complete batch.

        If `shuffle` is `None` or `False` elements will be extracted in
        order. If it is a `numpy.random.RandomState`, it will be used to
        randomise the order in which elements are extracted from the data.
        If it is `True`, Lasagne's default random number generator will be
        use to shuffle elements.

        `epochs` controls the number of repetitions; e.g. a value of `2` will
        cause the iterator to walk the data twice before terminating. A value
        of `-1` will result in an infinite number of repetitions. If
        shuffling is used a different permutation of the elements in the data
        set will be used for each repetition.

        If an array of indices was provided to the constructor, the subset of
        samples identified in that array is used, rather than the complete
        set of samples.

        The generated mini-batches take the form `[batch_x, batch_y, ...]`
        where `batch_x`, `batch_y`, etc. are extracted from each array that
        is a part of `self`.

        Parameters
        ----------
        batch_size: int
            Mini-batch size
        shuffle: `numpy.random.RandomState` or `True` or `None`
            Used to randomise element order. If `None`, elements will be
            extracted in order. If it is a `RandomState` instance, that
            RNG will be used to shuffle elements. If it is `True`, Lasagne's
            default RNG will be used.
        epochs: int (default=1)
            The number of repetitions, or `-1` for infinite. A value of 0 or
            a negative value that is not -1 will cause `ValueError` to be
            raised.

        Returns
        -------
        iterator
            An iterator that generates items of type `[batch_x, batch_y, ...]`
            where `batch_x`, `batch_y`, etc are themselves arrays.
        """
        if epochs == 0 or epochs < -1:
            raise ValueError('Invalid number of epochs; should be >= 1 or '
                             '-1, not {}'.format(epochs))
        shuffle = self._get_shuffle_rng(shuffle)
        if epochs == 1:
            if shuffle is not None:
                if self.indices is not None:
                    indices = shuffle.permutation(self.indices)
                else:
                    indices = shuffle.permutation(self.length)
                for i in range(0, self.length, batch_size):
                    excerpt = indices[i:i + batch_size]
                    yield [d[excerpt] for d in self.data]
            else:
                if self.indices is not None:
                    for i in range(0, self.length, batch_size):
                        batch_ndx = self.indices[i:i + batch_size]
                        yield [d[batch_ndx] for d in self.data]
                else:
                    for i in range(0, self.length, batch_size):
                        yield [d[i:i + batch_size] for d in self.data]
        else:
            if shuffle is not None:
                if self.indices is not None:
                    indices = shuffle.permutation(self.indices)
                else:
                    indices = shuffle.permutation(self.length)
                i = 0
                while True:
                    j = i + batch_size
                    if j <= self.length:
                        # Within size of data
                        batch_ndx = indices[i:j]
                        i = j
                    else:
                        # Reduce the number of remaining epochs
                        epochs = epochs - 1 if epochs != -1 else -1
                        if epochs == 0:
                            # Finished; emit remaining elements
                            if i < self.length:
                                batch_ndx = indices[i:self.length]
                                yield [d[batch_ndx] for d in self.data]
                            break

                        # Wrap over
                        # Compute number of elements required to make up the
                        # batch
                        k = batch_size - (self.length - i)
                        # Get available indices
                        batch_ndx = indices[i:self.length]
                        if self.indices is not None:
                            indices = shuffle.permutation(self.indices)
                        else:
                            indices = shuffle.permutation(self.length)
                        # Get remaining indices and append
                        batch_ndx = np.append(batch_ndx, indices[:k], axis=0)
                        i = k
                    yield [d[batch_ndx] for d in self.data]
            else:
                if self.indices is not None:
                    i = 0
                    while True:
                        j = i + batch_size
                        if j <= self.length:
                            # Within size of data
                            batch_ndx = self.indices[i:j]
                            i = j
                        else:
                            # Reduce the number of remaining epochs
                            epochs = epochs - 1 if epochs != -1 else -1
                            if epochs == 0:
                                # Finished; emit remaining elements
                                if i < self.length:
                                    batch_ndx = self.indices[i:self.length]
                                    yield [d[batch_ndx] for d in self.data]
                                break

                            # Wrap over
                            # Compute number of elements required to make up
                            # the batch
                            k = batch_size - (self.length - i)
                            batch_ndx = np.append(self.indices[i:self.length],
                                                  self.indices[:k],
                                                  axis=0)
                            i = k
                        yield [d[batch_ndx] for d in self.data]
                else:
                    i = 0
                    while True:
                        j = i + batch_size
                        if j <= self.length:
                            # Within size of data
                            yield [d[i:j] for d in self.data]
                            i = j
                        else:
                            # Reduce the number of remaining epochs
                            epochs = epochs - 1 if epochs != -1 else -1
                            if epochs == 0:
                                # Finished; emit remaining elements
                                if i < self.length:
                                    yield [d[i:self.length] for d in self.data]
                                break

                            # Wrap over
                            # Compute number of elements required to make up
                            # the batch
                            k = batch_size - (self.length - i)
                            yield [np.append(d[i:self.length], d[:k], axis=0)
                                   for d in self.data]
                            i = k


class CallableDataSource (AbstractDataSource):
    """A data source that calls functions to generate a batch iterator
    or get the number of samples.

    Parameters
    ----------
    batch_iterator_fn: callable `fn(batch_size, **kwargs) -> iterator`
        Callable function that returns an iterator that yields mini-batches.
        Its first argument is the mini-batch size, with keyword arguments
        providing settings. The `batch_iterator` will invoke this function
        and return its return value.
    num_samples_fn: [optional] None, or callable or int or np.inf
        If None, an int or np.inf, this value will be returned by the
        `num_samples`. If its a callable, `num_samples` will call it, passing
        the keyword arguments and return its return value.

    Examples
    --------
    Data to iterate over:
    >>> X = np.random.normal(size=(7, 10))

    Function to build batch iterator:
    >>> def make_batch_iterator(batch_size):
    ...     for i in range(0, X.shape[0], batch_size):
    ...         yield [X[i:i + batch_size]]

    Data source acquiring batches from the `make_batch_iterator` function:
    >>> ds = CallableDataSource(make_batch_iterator)
    >>> ds.num_samples() is None
    True

    Iterate over batches
    >>> for (batch_X,) in ds.batch_iterator(5):
    ...     break

    We can also provide a function that computes the number of samples:
    >>> def num_samples_fn():
    ...     return X.shape[0]

    >>> ds = CallableDataSource(make_batch_iterator, num_samples_fn)
    >>> ds.num_samples()
    7

    Or, we could provide the number of samples:
    >>> ds = CallableDataSource(make_batch_iterator, 7)
    >>> ds.num_samples()
    7
    """
    def __init__(self, batch_iterator_fn, num_samples_fn=None):
        if num_samples_fn is not None:
            if not callable(num_samples_fn) and \
                    not isinstance(num_samples_fn, six.integer_types) and \
                    num_samples_fn != np.inf:
                raise TypeError(
                    'num_samples_fn should be None, a callable, an int, or '
                    'np.inf, not {}'.format(num_samples_fn)
                )
        self.batch_iterator_fn = batch_iterator_fn
        self.num_samples_fn = num_samples_fn

    def num_samples(self, **kwargs):
        """
        Get the number of samples in this data source.

        Returns
        -------
        int, `np.inf` or `None`.
            An int if the number of samples is known, `np.inf` if it is
            infinite or `None` if the number of samples is unknown.
        """
        if self.num_samples_fn is None:
            return None
        elif callable(self.num_samples_fn):
            return self.num_samples_fn(**kwargs)
        else:
            return self.num_samples_fn

    def batch_iterator(self, batch_size, **kwargs):
        """
        Return an iterator that generates mini-batches extracted from `self`.

        Parameters
        ----------
        batch_size: int
            Mini-batch size

        Returns
        -------
            An iterator that yields mini-batches.
        """
        return self.batch_iterator_fn(batch_size, **kwargs)


class IteratorDataSource (AbstractDataSource):
    """A data source that wraps an iterator.

    Note that all parameters passed to the `batch_iterator` method -
    including `batch_size` - are ignored, as there is no way of passing them
    to an already constructed iterator.

    Parameters
    ----------
    batch_iter: iterator
        This iterator will be returned (as-is) by the `batch_iterator` method.
    n_samples: [optional] None, or int or np.inf
        The number of samples in this data source. `None` for unknown (the
        default), an int for a known number of samples, or `np.inf`
        for an infinite data source.

    Examples
    --------
    Data to iterate over:
    >>> X = np.random.normal(size=(7, 10))

    Function to build batch iterator:
    >>> def make_batch_iterator(batch_size):
    ...     for i in range(0, X.shape[0], batch_size):
    ...         yield [X[i:i + batch_size]]

    Build batch iterator:
    >>> batch_iter = make_batch_iterator(5)

    Data source acquiring batches from the `make_batch_iterator` function:
    >>> ds = IteratorDataSource(batch_iter)
    >>> ds.num_samples() is None
    True

    Iterate over batches. Note that the batch size of 3 will be *ignored*
    as the iterator was constructed above with a batch size of 5.
    >>> for (batch_X,) in ds.batch_iterator(3):
    ...     break

    We can provide the number of samples:
    >>> ds = IteratorDataSource(batch_iter, 7)
    >>> ds.num_samples()
    7
    """
    def __init__(self, batch_iter, n_samples=None):
        if n_samples is not None and \
                not isinstance(n_samples, six.integer_types) and \
                n_samples != np.inf:
            raise TypeError('n_samples should be None, an int, or np.inf, '
                            'not {}'.format(n_samples))
        self.batch_iter = batch_iter
        self.n_samples = n_samples

    def num_samples(self, **kwargs):
        """
        Get the number of samples in this data source.

        Returns
        -------
        int, `np.inf` or `None`.
            An int if the number of samples is known, `np.inf` if it is
            infinite or `None` if the number of samples is unknown.
        """
        return self.n_samples

    def batch_iterator(self, batch_size, **kwargs):
        """
        Return an iterator that generates mini-batches extracted from `self`.

        Parameters
        ----------
        batch_size: int
            Mini-batch size

        Returns
        -------
            An iterator that yields mini-batches.
        """
        return self.batch_iter


class CompositeDataSource (AbstractDataSource):
    """A data source that is the combination of a number of member
    data sources. Samples are drawn from every member source to form a
    mini-batch.

    A common use of `CompositeDataSource` would be in a semi-supervised
    learning scenario, in which our data set consists of labeled and
    unlabeled samples, where the second outnumber the first by a large
    multiple. We often need to train on both simultaneously.
    `CompositeDataSource` allows us to iterate over the smaller labeled
    set repeatedly, while iterating over the unlabeled samples once.

    Create 10 labeled samples:
    >>> lab_X = np.random.normal(size=(10, 10))
    >>> lab_y = np.random.randint(0, 10, size=(10,))

    Create 33 unlabeled samples:
    >>> unlab_X = np.random.normal(size=(33, 10))

    Array data sources for labeled and unlabeled samples:
    >>> lab_ds = ArrayDataSource([lab_X, lab_y])
    >>> unlab_ds = ArrayDataSource([unlab_X])

    Create a data source that iterates repeatedly over the labeled samples
    and once over the unlabeled samples (note the use of the `with_params`
    method to apply the `epochs` parameter to the labeled samples):
    >>> semi_ds = CompositeDataSource([
    ...     lab_ds.with_params(epochs=-1), unlab_ds
    ... ])

    When we iterate over them, we get batches of the form
    `[[batch_lab_X, batch_lab_y], [batch_unlab_X]]`:
    >>> for batch in semi_ds.batch_iterator(batch_size=5):
    ...     # Normally we would pass the batch to a training function, but
    ...     # we're just going to check its shape here:
    ...     assert len(batch) == 3
    ...     assert batch[0].shape == (5, 10)
    ...     assert batch[1].shape == (5,)
    ...     assert batch[2].shape == (5, 10)
    ...     break

    Alternatively, if you want structured mini-batches that have the same
    nesting structure as the composite data soruce:
    >>> semi_flat_ds = CompositeDataSource([
    ...     lab_ds.with_params(epochs=-1), unlab_ds
    ... ], flatten=False)

    >>> for batch in semi_flat_ds.batch_iterator(batch_size=5):
    ...     # Check the shape
    ...     assert len(batch) == 2
    ...     assert len(batch[0]) == 2
    ...     assert batch[0][0].shape == (5, 10)
    ...     assert batch[0][1].shape == (5,)
    ...     assert len(batch[1]) == 1
    ...     assert batch[1][0].shape == (5, 10)
    ...     break
    """
    def __init__(self, datasets, flatten=True):
        self.datasets = datasets
        self.flatten = flatten

    def num_samples(self, **kwargs):
        return min([d.num_samples(**kwargs) for d in self.datasets])

    def batch_iterator(self, batch_size, flatten=None, **kwargs):
        iterators = [d.batch_iterator(batch_size, flatten=flatten, **kwargs)
                     for d in self.datasets]

        for batch in six.moves.zip(*iterators):
            # Get the lengths of all the sub-batches
            sub_lens = [_length_of_batch(sub_batch) for sub_batch in batch]
            # Get the minimum length
            trim_len = min(sub_lens)
            # If its not the same as the maximum length, we need to trim
            if trim_len != max(sub_lens):
                batch = _trim_batch(batch, trim_len)

            if flatten is None:
                flatten = self.flatten

            if flatten:
                yield sum(batch, [])
            else:
                yield list(batch)


def batch_map(func, batch_iter, progress_iter_func=None,
              batch_limit=None, prepend_args=None):
    """
    Apply a function to all the samples that are accessed as mini-batches
    obtained from an iterator.
    Returns the per-sample results.

    The function `func` should return the result for each sample in the
    mini-batch as an array. To return multiple results (e.g. loss and errors)
    return a list of arrays (e.g. `[loss_array, error_array]`)

    `batch_iter` must be an iterator that generates mini-batches that
    contain samples

    Parameters
    ----------
    func: callable `func(*batch) -> results`
        The function to call on each mini-batch. Note that the results
        must be `None`, a list or a NumPy array
    batch_iter: data set iterator
        Iterator that generates mini-batches of data
    progress_iter_func: [optional] callable
        `progress_iter_func(iterator, total=total, leave=leave)`
        A `tqdm` style function that will be passed the iterator that
        generates training batches along with the total number of batches
        and `False` for the `leave` parameter. By passing either
        `tqdm.tqdm` or `tqdm.tqdm_notebook` as this argument you can have
        the training loop display a progress bar.
    batch_limit: [optional] integer
        Process at most this number of batches before returning.
    prepend_args: [optional] tuple
        Arguments to prepend to the arguments passed to `func`

    Returns
    -------
    list
        The per-sample sum of the results of the function `func` e.g.
        `[batch_A, batch_B, ...]`
        Returns an empty list if there were 0 samples in the data set.

    Examples
    --------
    In these examples we will demonstrate the use of `batch_map` to apply
    a function (e.g. a Theano function that runs on the GPU) to samples
    in a data set. We construct an iterator that generates mini-batches from
    the data set and pass it to `batch_map` along with the function that
    we wish to apply. The function will receive the batches and process them.

    Define a function to apply to samples:
    >>> def sqr_sum(x):
    ...     # Ensure that we receive batches of the expected size:
    ...     assert x.shape[0] in {5, 2}
    ...     return (x ** 2).sum(axis=1)

    Construct data to process and create a data source:
    >>> X = np.random.normal(size=(7, 10))
    >>> ds = ArrayDataSource([X])

    Apply the function defined above:
    >>> batch_iter = ds.batch_iterator(batch_size=5)
    >>> X_sqr_sum = batch_map(sqr_sum, batch_iter)
    >>> assert np.allclose(X_sqr_sum[0], (X ** 2).sum(axis=1))

    There are also cases where we wish to limit the number of batches that
    will be processed:
    - when the iterator generates an infinite number of samples
    - when the data set is huge and we wish to show results as we go
    Use the `batch_limit` argument to limit the number of batches to process:
    >>> X_large = np.random.normal(size=(100, 10))
    >>> ds_large = ArrayDataSource([X_large])
    >>> iter_large = ds_large.batch_iterator(batch_size=5)
    >>> for i in range(10):
    ...     partial_result = batch_map(sqr_sum, iter_large, batch_limit=2)
    ...     # Should have 10 samples per partial result
    ...     assert partial_result[0].shape[0] == 10
    ...     j = i * 10
    ...     assert np.allclose(partial_result[0],
    ...                        (X_large[j:j + 10]**2).sum(axis=1))
    """
    # Accumulator for results and number of samples
    results = []

    # If `progress_iter_func` is not `None`, apply it
    if progress_iter_func is not None:
        batch_iter = progress_iter_func(batch_iter, total=batch_limit,
                                        leave=False)

    # Apply `func` to each batch
    n_processed = 0
    for batch in batch_iter:
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
                    'single result as a NumPy array, or None, '
                    'not {}'.format(type(batch_results)))

        # Accumulate training results
        if batch_results is not None:
            results.append(batch_results)

        n_processed += 1
        if batch_limit is not None and n_processed >= batch_limit:
            break

    # Concatenate result arrays
    if len(results) > 0:
        results = zip(*results)
        results = [np.concatenate(list(r), axis=0) for r in results]
        return results
    else:
        return None


def mean_batch_map(func, batch_iter, progress_iter_func=None, sum_axis=None,
                   batch_limit=None, prepend_args=None):
    """
    Apply a function to all the samples that are accessed as mini-batches
    obtained from an iterator.
    Returns the across-samples mean of the results returned by `func`

    The `sum_axis` arguments tells `mean_batch_map` how to process the
    results of `func` before accumulating them:
    - If `sum_axis` is `None`, `func` should return the
    across-samples SUM of the  results of operating on the mini-batch the
    sum of the values for the samples, e.g. for loss and error it should
    return `[sum([loss0, loss1, ... lossN]), sum([err0, err1, ... errN])]`
    - Otherwise, `sum_axis` should specify the axis or axes over which
    the the batch results should be summed, e.g. if `func` returns a
    per-sample loss and error in two arrays
    `[[loss0, loss1, ... lossN], [err0, err1, ... errN]`, give `sum_axis`
    a value of `0` to sum over axis 0 to get the per-batch loss and error.
    These esults will be accumulated and divided by the number of samples
    at the end to get the mean.

    Parameters
    ----------
    func: callable `func(*batch) -> results`
        The function to call on each mini-batch. Note that the results
        must be `None`, a list or a NumPy array
    batch_iter: data set iterator
        Iterator that generates mini-batches of data
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
    batch_limit: [optional] integer that specifies the number of mini-batches
        to process before returning
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

    Examples
    --------
    The following examples will demonstrate the use of `mean_batch_map`
    to compute binary cross entropy loss over a data set.
    A few variants will be demonstrated:
    - the default behaviour in which the function being applied should
      return the sum over the batch sample axis
    - having the function return per sample results and maving
      `mean_batch_map` perform the sum operation. This is easier to
      understand but less efficient as a Theano function would have to
      move more data back from the GPU.
    - limiting the number of batches that will be processed in order to get
      partial results when dealing with a large data set

    Define a function to compute the per-sample binary cross entropy
    loss:
    >>> def binary_crossentropy_loss(pred, target):
    ...     e = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
    ...     return e.mean(axis=1)

    Now define a function that computes the *SUM* of the binary cross
    entropy losses over the sample axis (axis 0), as the default
    behaviour of `mean_batch_map` will sum them up and divide by the
    number of samples at the end:
    >>> def binary_crossentropy_loss_sum(pred, target):
    ...     return binary_crossentropy_loss(pred, target).sum()

    Construct prediction and target data
    >>> pred = np.random.uniform(0.1, 0.9, size=(7, 10))
    >>> tgt = np.random.uniform(0.1, 0.9, size=(7, 10))
    >>> ds = ArrayDataSource([pred, tgt])

    Apply the loss sum function defined above:
    >>> batch_iter = ds.batch_iterator(batch_size=5)
    >>> loss = mean_batch_map(binary_crossentropy_loss_sum, batch_iter)
    >>> assert np.allclose(
    ...     loss, binary_crossentropy_loss(pred, tgt).mean())

    Have `mean_batch_map` sum over axis 0:
    >>> batch_iter = ds.batch_iterator(batch_size=5)
    >>> loss = mean_batch_map(binary_crossentropy_loss, batch_iter,
    ...                       sum_axis=0)
    >>> assert np.allclose(
    ...     loss, binary_crossentropy_loss(pred, tgt).mean())

    Construct a large data set and use `batch
    >>> pred_large = np.random.uniform(0.1, 0.9, size=(100, 10))
    >>> tgt_large = np.random.uniform(0.1, 0.9, size=(100, 10))
    >>> ds_large = ArrayDataSource([pred_large, tgt_large])
    >>> iter_large = ds_large.batch_iterator(batch_size=5)
    >>> for i in range(10):
    ...     partial_loss = mean_batch_map(binary_crossentropy_loss_sum,
    ...                                   iter_large, batch_limit=2)
    ...     j = i * 10
    ...     assert np.allclose(
    ...         partial_loss, binary_crossentropy_loss(
    ...             pred_large[j:j + 10], tgt_large[j:j + 10]).mean())
    """
    # Accumulator for results and number of samples
    results_accum = None
    n_samples_accum = 0

    # If `progress_iter_func` is not `None`, apply it
    if progress_iter_func is not None:
        batch_iter = progress_iter_func(batch_iter, total=batch_limit,
                                        leave=False)

    # Train on each batch
    n_processed = 0
    for batch in batch_iter:
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

        n_processed += 1
        if batch_limit is not None and n_processed >= batch_limit:
            break

    # Divide by the number of training examples used to compute mean
    if results_accum is not None:
        results_accum = [np.array(r).astype(float) / n_samples_accum
                         for r in results_accum]

    return results_accum
