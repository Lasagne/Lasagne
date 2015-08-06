import numpy as np

import theano
import theano.tensor as T


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.

    Parameters
    ----------
    arr : array_like
        The data to be converted.

    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


def shared_empty(dim=2, dtype=None):
    """Creates empty Theano shared variable.

    Shortcut to create an empty Theano shared variable with
    the specified number of dimensions.

    Parameters
    ----------
    dim : int, optional
        The number of dimensions for the empty variable, defaults to 2.
    dtype : a numpy data-type, optional
        The desired dtype for the variable. Defaults to the Theano
        ``floatX`` dtype.

    Returns
    -------
    Theano shared variable
        An empty Theano shared variable of dtype ``dtype`` with
        `dim` dimensions.
    """
    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))


def as_theano_expression(input):
    """Wrap as Theano expression.

    Wraps the given input as a Theano constant if it is not
    a valid Theano expression already. Useful to transparently
    handle numpy arrays and Python scalars, for example.

    Parameters
    ----------
    input : number, numpy array or Theano expression
        Expression to be converted to a Theano constant.

    Returns
    -------
    Theano symbolic constant
        Theano constant version of `input`.
    """
    if isinstance(input, theano.gof.Variable):
        return input
    else:
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))


def one_hot(x, m=None):
    """One-hot representation of integer vector.

    Given a vector of integers from 0 to m-1, returns a matrix
    with a one-hot representation, where each row corresponds
    to an element of x.

    Parameters
    ----------
    x : integer vector
        The integer vector to convert to a one-hot representation.
    m : int, optional
        The number of different columns for the one-hot representation. This
        needs to be strictly greater than the maximum value of `x`.
        Defaults to ``max(x) + 1``.

    Returns
    -------
    Theano tensor variable
        A Theano tensor variable of shape (``n``, `m`), where ``n`` is the
        length of `x`, with the one-hot representation of `x`.

    Notes
    -----
    If your integer vector represents target class memberships, and you wish to
    compute the cross-entropy between predictions and the target class
    memberships, then there is no need to use this function, since the function
    :func:`lasagne.objectives.categorical_crossentropy()` can compute the
    cross-entropy from the integer vector directly.

    """
    if m is None:
        m = T.cast(T.max(x) + 1, 'int32')

    return T.eye(m)[T.cast(x, 'int32')]


def unique(l):
    """Filters duplicates of iterable.

    Create a new list from l with duplicate entries removed,
    while preserving the original order.

    Parameters
    ----------
    l : iterable
        Input iterable to filter of duplicates.

    Returns
    -------
    list
        A list of elements of `l` without duplicates and in the same order.
    """
    new_list = []
    for el in l:
        if el not in new_list:
            new_list.append(el)

    return new_list


def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).

    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements

    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.

    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X


def compute_norms(array, norm_axes=None):
    """ Compute incoming weight vector norms.

    Parameters
    ----------
    array : ndarray
        Weight array.
    norm_axes : sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `array`. When this is not specified and `array` is a 2D array,
        this is set to `(0,)`. If `array` is a 3D, 4D or 5D array, it is
        set to a tuple listing all axes but axis 0. The former default is
        useful for working with dense layers, the latter is useful for 1D,
        2D and 3D convolutional layers.
        (Optional)

    Returns
    -------
    norms : 1D array
        1D array of incoming weight vector norms.

    Examples
    --------
    >>> array = np.random.randn(100, 200)
    >>> norms = compute_norms(array)
    >>> norms.shape
    (200,)

    >>> norms = compute_norms(array, norm_axes=(1,))
    >>> norms.shape
    (100,)
    """
    ndim = array.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(array.ndim)
        )

    norms = np.sqrt(np.sum(array**2, axis=sum_over))

    return norms


def create_param(spec, shape, name=None):
    """
    Helper method to create Theano shared variables for layer parameters
    and to initialize them.

    Parameters
    ----------
    spec : numpy array, Theano shared variable, or callable
        Either of the following:

        * a numpy array with the initial parameter values
        * a Theano shared variable representing the parameters
        * a function or callable that takes the desired shape of
          the parameter array as its single argument and returns
          a numpy array.

    shape : iterable of int
        a tuple or other iterable of integers representing the desired
        shape of the parameter array.

    name : string, optional
        If a new variable is created, the name to give to the parameter
        variable. This is ignored if `spec` is already a Theano shared
        variable.

    Returns
    -------
    Theano shared variable
        a Theano shared variable representing layer parameters. If a
        numpy array was provided, the variable is initialized to
        contain this array. If a shared variable was provided, it is
        simply returned. If a callable was provided, it is called, and
        its output is used to initialize the variable.

    Notes
    -----
    This function is called by :meth:`Layer.add_param()` in the constructor
    of most :class:`Layer` subclasses. This enables those layers to
    support initialization with numpy arrays, existing Theano shared
    variables, and callables for generating initial parameter values.
    """
    shape = tuple(shape)  # convert to tuple if needed
    if any(d <= 0 for d in shape):
        raise ValueError((
            "Cannot create param with a non-positive shape dimension. "
            "Tried to create param with shape=%r, name=%r") % (shape, name))

    if isinstance(spec, theano.compile.SharedVariable):
        # We cannot check the shape here, the shared variable might not be
        # initialized correctly yet. We can check the dimensionality
        # though. Note that we cannot assign a name here. We could assign
        # to the `name` attribute of the shared variable, but we shouldn't
        # because the user may have already named the variable and we don't
        # want to override this.
        if spec.ndim != len(shape):
            raise RuntimeError("shared variable has %d dimensions, "
                               "should be %d" % (spec.ndim, len(shape)))
        return spec

    elif isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise RuntimeError("parameter array has shape %s, should be "
                               "%s" % (spec.shape, shape))
        return theano.shared(spec, name=name)

    elif hasattr(spec, '__call__'):
        arr = spec(shape)
        try:
            arr = floatX(arr)
        except Exception:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return an "
                               "array-like value")
        if arr.shape != shape:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return a value "
                               "with the correct shape")
        return theano.shared(arr, name=name)

    else:
        raise RuntimeError("cannot initialize parameters: 'spec' is not "
                           "a numpy array, a Theano shared variable, or a "
                           "callable")


def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps,
                go_backwards=False):
        """
        Helper function to unroll for loops. Can be used to unroll theano.scan.
        The parameter names are identical to theano.scan, please refer to here
        for more information.

        Note that this function does not support the truncate_gradient
        setting from theano.scan.

        Parameters
        ----------

        fn : function
            Function that defines calculations at each step.

        sequences : TensorVariable or list of TensorVariables
            List of TensorVariable with sequence data. The function iterates
            over the first dimension of each TensorVariable.

        outputs_info : list of TensorVariables
            List of tensors specifying the initial values for each recurrent
            value.

        non_sequences: list of TensorVariables
            List of theano.shared variables that are used in the step function.

        n_steps: int
            Number of steps to unroll.

        go_backwards: bool
            If true the recursion starts at sequences[-1] and iterates
            backwards.

        Returns
        -------
        List of TensorVariables. Each element in the list gives the recurrent
        values at each time step.

        """
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]

        # When backwards reverse the recursion direction
        counter = range(n_steps)
        if go_backwards:
            counter = counter[::-1]

        output = []
        prev_vals = outputs_info
        for i in counter:
            step_input = [s[i] for s in sequences] + prev_vals + non_sequences
            out_ = fn(*step_input)
            # The returned values from step can be either a TensorVariable,
            # a list, or a tuple.  Below, we force it to always be a list.
            if isinstance(out_, T.TensorVariable):
                out_ = [out_]
            if isinstance(out_, tuple):
                out_ = list(out_)
            output.append(out_)

            prev_vals = output[-1]

        # iterate over each scan output and convert it to same format as scan:
        # [[output11, output12,...output1n],
        # [output21, output22,...output2n],...]
        output_scan = []
        for i in range(len(output[0])):
            l = map(lambda x: x[i], output)
            output_scan.append(T.stack(*l))

        return output_scan
