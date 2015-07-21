# -*- coding: utf-8 -*-
"""
Layers to construct recurrent networks. Recurrent layers can be used similarly
to feed-forward layers except that the input shape is expected to be
``(batch_size, sequence_length, num_inputs)``. The input is allowed to have
more than three dimensions in which case dimensions trailing the third
dimension are flattened.

The following recurrent layers are implemented:

.. autosummary::
    :nosignatures:

    CustomRecurrentLayer
    RecurrentLayer
    LSTMLayer
    GRULayer

For recurrent layers with gates we use a helper class to set up the parameters
in each gate:

.. autosummary::
    :nosignatures:

    Gate

Please refer to that class if you need to modify initial conditions of gates.

Recurrent layers and feed-forward layers can be combined in the same network
by using a few reshape operations; please refer to the example below.

Examples
--------
The following example demonstrates how recurrent layers can be easily mixed
with feed-forward layers using :class:`ReshapeLayer` and how to build a
network with variable batch size and number of time steps.

>>> from lasagne.layers import *
>>> num_inputs, num_units, num_classes = 10, 12, 5
>>> # By setting the first two dimensions as None, we are allowing them to vary
>>> # They correspond to batch size and sequence length, so we will be able to
>>> # feed in batches of varying size with sequences of varying length.
>>> l_inp = InputLayer((None, None, num_inputs))
>>> # We can retrieve symbolic references to the input variable's shape, which
>>> # we will later use in reshape layers.
>>> batchsize, seqlen, _ = l_inp.input_var.shape
>>> l_lstm = LSTMLayer(l_inp, num_units=num_units)
>>> # In order to connect a recurrent layer to a dense layer, we need to
>>> # flatten the first two dimensions (our "sample dimensions"); this will
>>> # cause each time step of each sequence to be processed independently
>>> l_shp = ReshapeLayer(l_lstm, (-1, num_units))
>>> l_dense = DenseLayer(l_shp, num_units=num_classes)
>>> # To reshape back to our original shape, we can use the symbolic shape
>>> # variables we retrieved above.
>>> l_out = ReshapeLayer(l_dense, (batchsize, seqlen, num_classes))
"""
import numpy as np
import theano
import theano.tensor as T
from .. import nonlinearities
from .. import init
from ..utils import unroll_scan

from .base import Layer
from .input import InputLayer
from .dense import DenseLayer
from . import helper

__all__ = [
    "CustomRecurrentLayer",
    "RecurrentLayer",
    "Gate",
    "LSTMLayer",
    "GRULayer"
]


class CustomRecurrentLayer(Layer):
    """
    lasagne.layers.recurrent.CustomRecurrentLayer(incoming, input_to_hidden,
    hidden_to_hidden, nonlinearity=lasagne.nonlinearities.rectify,
    hid_init=lasagne.init.Constant(0.), backwards=False,
    learn_init=False, gradient_steps=-1, grad_clipping=False,
    unroll_scan=False, precompute_input=True, **kwargs)

    A layer which implements a recurrent connection.

    This layer allows you to specify custom input-to-hidden and
    hidden-to-hidden connections by instantiating :class:`lasagne.layers.Layer`
    instances and passing them on initialization.  Note that these connections
    can consist of multiple layers chained together.  The output shape for the
    provided input-to-hidden and hidden-to-hidden connections must be the same.
    If you are looking for a standard, densely-connected recurrent layer,
    please see :class:`RecurrentLayer`.  The output is computed by

    .. math ::
        h_t = \sigma(f_i(x_t) + f_h(h_{t-1}))

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    input_to_hidden : :class:`lasagne.layers.Layer`
        :class:`lasagne.layers.Layer` instance which connects input to the
        hidden state (:math:`f_i`).  This layer may be connected to a chain of
        layers, which must end in a :class:`lasagne.layers.InputLayer` with the
        same input shape as `incoming`.
    hidden_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects the previous hidden state to the new state
        (:math:`f_h`).  This layer may be connected to a chain of layers, which
        must end in a :class:`lasagne.layers.InputLayer` with the same input
        shape as `hidden_to_hidden`'s output shape.
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode, `learn_init` is
        ignored.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then the TensorVariable is used and
        `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    Examples
    --------

    The following example constructs a simple `CustomRecurrentLayer` which
    has dense input-to-hidden and hidden-to-hidden connections.

    >>> import lasagne
    >>> n_batch, n_steps, n_in = (2, 3, 4)
    >>> n_hid = 5
    >>> l_in = lasagne.layers.InputLayer((n_batch, n_steps, n_in))
    >>> l_in_hid = lasagne.layers.DenseLayer(
    ...     lasagne.layers.InputLayer((None, n_in)), n_hid)
    >>> l_hid_hid = lasagne.layers.DenseLayer(
    ...     lasagne.layers.InputLayer((None, n_hid)), n_hid)
    >>> l_rec = lasagne.layers.CustomRecurrentLayer(l_in, l_in_hid, l_hid_hid)


    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 **kwargs):

        super(CustomRecurrentLayer, self).__init__(incoming, **kwargs)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Check that output shapes match
        if input_to_hidden.output_shape != hidden_to_hidden.output_shape:
            raise ValueError("The output shape for input_to_hidden and "
                             "input_to_hidden must be equal, but "
                             "input_to_hidden.output_shape={} and "
                             "hidden_to_hidden.output_shape={}".format(
                                 input_to_hidden.output_shape,
                                 hidden_to_hidden.output_shape))

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Get the input dimensionality and number of units based on the
        # expected output of the input-to-hidden layer
        self.num_inputs = np.prod(self.input_shape[2:])
        self.num_units = input_to_hidden.output_shape[-1]

        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(CustomRecurrentLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        return params

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, input, mask=None, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        input : theano.TensorType
            Symbolic input variable.
        mask : theano.TensorType
            Theano variable denoting whether each time step in each
            sequence in the batch is part of the sequence or not.  If ``None``,
            then it is assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape ``(n_batch, n_time_steps)`` where
            ``mask[i, j] = 1`` when ``j <= (length of sequence i)`` and
            ``mask[i, j] = 0`` when ``j > (length of sequence i)``.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        if self.precompute_input:
            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, batch_size, num_inputs) to
            # (seq_len*batch_size, num_inputs)
            input = T.reshape(input,
                              (seq_len*num_batch, -1))
            input = helper.get_output(
                self.input_to_hidden, input, **kwargs)

            # Reshape back to (seq_len, batch_size, num_units)
            input = T.reshape(input, (seq_len, num_batch, -1))

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        # When we are not precomputing the input, we also need to pass the
        # input-to-hidden parameters to step
        if not self.precompute_input:
            non_seqs += helper.get_all_params(self.input_to_hidden)

        # Create single recurrent computation step function
        def step(input_n, hid_previous, *args):
            # Compute the hidden-to-hidden activation
            hid_pre = helper.get_output(self.hidden_to_hidden, hid_previous)

            # If the dot product is precomputed then add it, otherwise
            # calculate the input_to_hidden values and add them
            if self.precompute_input:
                hid_pre += input_n
            else:
                hid_pre += helper.get_output(self.input_to_hidden, input_n)

            # Clip gradients
            if self.grad_clipping is not False:
                hid_pre = theano.gradient.grad_clip(
                    hid_pre, -self.grad_clipping, self.grad_clipping)

            return self.nonlinearity(hid_pre)

        def step_masked(input_n, mask_n, hid_previous, *args):
            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            hid = step(input_n, hid_previous, *args)
            hid_out = hid*mask_n + hid_previous*(1 - mask_n)
            return [hid_out]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        # When hid_init is provided as a TensorVariable, use it as-is
        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        if self.unroll_scan:
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=self.input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1, :]

        self.hid_out = hid_out
        return hid_out


class RecurrentLayer(CustomRecurrentLayer):
    """
    lasagne.layers.recurrent.RecurrentLayer(incoming, num_units,
    W_in_to_hid=lasagne.init.Uniform(), W_hid_to_hid=lasagne.init.Uniform(),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=False, unroll_scan=False,
    precompute_input=True, **kwargs)

    Dense recurrent neural network (RNN) layer

    A "vanilla" RNN layer, which has dense input-to-hidden and
    hidden-to-hidden connections.  The output is computed as

    .. math ::
        h_t = \sigma(x_t W_x + h_{t-1} W_h + b)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    W_in_to_hid : Theano shared variable, numpy array or callable
        Initializer for input-to-hidden weight matrix (:math:`W_x`).
    W_hid_to_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden weight matrix (:math:`W_h`).
    b : Theano shared variable, numpy array, callable or None
        Initializer for bias vector (:math:`b`). If None is provided there will
        be no bias.
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode `learn_init` is
        ignored.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 **kwargs):
        input_shape = helper.get_output_shape(incoming)
        num_batch = input_shape[0]
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((num_batch,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None, **kwargs)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((num_batch, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None, **kwargs)

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = in_to_hid.W
        self.W_hid_to_hid = hid_to_hid.W
        self.b = in_to_hid.b

        # Just use the CustomRecurrentLayer with the DenseLayers we created
        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, **kwargs)


class Gate(object):
    """
    lasagne.layers.recurrent.Gate(W_in=lasagne.init.Normal(0.1),
    W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)

    Simple class to hold the parameters for a gate connection.  We define
    a gate loosely as something which compute the linear mix of two inputs,
    optionally computes an element-wise product with a third, adds a bias, and
    applies a nonlinearity.

    Parameters
    ----------
    W_in : Theano shared variable, numpy array or callable
        Initializer for input-to-gate weight matrix.
    W_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-gate weight matrix.
    W_cell : Theano shared variable, numpy array, callable, or None
        Initializer for cell-to-gate weight vector.  If None, no cell-to-gate
        weight vector will be stored.
    b : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector.
    nonlinearity : callable or None
        The nonlinearity that is applied to the input gate activation. If None
        is provided, no nonlinearity will be applied.

    Examples
    --------
    For :class:`LSTMLayer` the bias of the forget gate is often initialized to
    a large positive value to encourage the layer initially remember the cell
    value, see e.g. [1]_ page 15.

    >>> import lasagne
    >>> forget_gate = Gate(b=lasagne.init.Constant(5.0))
    >>> l_lstm = LSTMLayer((10, 20, 30), num_units=10,
    ...                    forgetgate=forget_gate)

    References
    ----------
    .. [1] Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to
           forget: Continual prediction with LSTM." Neural computation 12.10
           (2000): 2451-2471.

    """
    def __init__(self, W_in=init.Normal(0.1), W_hid=init.Normal(0.1),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity


class LSTMLayer(Layer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity_out=lasagne.nonlinearities.tanh,
    cell_init=b=lasagne.init.Constant(0.),
    hid_init=b=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=False, unroll_scan=False,
    precompute_input=True, **kwargs)

    A long short-term memory (LSTM) layer.

    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by

    .. math ::

        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t\sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity_out : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `cell_init` (:math:`c_0`). In this mode `learn_init` is
        ignored for the cell state.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode `learn_init` is
        ignored for the hidden state.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` or
        `cell_init` are TensorVariables then the TensorVariable is used and
        `learn_init` is ignored for that initial state.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `W_cell_to_ingate`, `W_cell_to_forgetgate` and
        `W_cell_to_outgate` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping: False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity_out=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 **kwargs):

        # Initialize parent layer
        super(LSTMLayer, self).__init__(incoming, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity_out is None:
            self.nonlinearity_out = nonlinearities.identity
        else:
            self.nonlinearity_out = nonlinearity_out

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(self.input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        self.W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        self.W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        self.b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, T.TensorVariable):
            if cell_init.ndim != 2:
                raise ValueError(
                    "When cell_init is provided as a TensorVariable, it should"
                    " have 2 dimensions and have shape (num_batch, num_units)")
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, input, mask=None, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        input : theano.TensorType
            Symbolic input variable.
        mask : theano.TensorType
            Theano variable denoting whether each time step in each
            sequence in the batch is part of the sequence or not.  If ``None``,
            then it is assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape ``(n_batch, n_time_steps)`` where
            ``mask[i, j] = 1`` when ``j <= (length of sequence i)`` and
            ``mask[i, j] = 0`` when ``j > (length of sequence i)``.

        Returns
        -------
        layer_output : theano.TensorType
            Symblic output variable.
        """
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, self.W_in_stacked) + self.b_stacked

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, W_hid_stacked,
                 W_cell_to_ingate, W_cell_to_forgetgate,
                 W_cell_to_outgate, W_in_stacked, b_stacked):

            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping is not False:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*W_cell_to_ingate
                forgetgate += cell_previous*W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*W_cell_to_outgate

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity_out(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous,
                        W_hid_stacked, W_cell_to_ingate, W_cell_to_forgetgate,
                        W_cell_to_outgate, W_in_stacked, b_stacked):

            cell, hid = step(input_n, cell_previous, hid_previous,
                             W_hid_stacked, W_cell_to_ingate,
                             W_cell_to_forgetgate, W_cell_to_outgate,
                             W_in_stacked, b_stacked)

            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            not_mask = 1 - mask_n
            cell = cell*mask_n + cell_previous*not_mask
            hid = hid*mask_n + hid_previous*not_mask

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if isinstance(self.cell_init, T.TensorVariable):
            cell_init = self.cell_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [self.W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]
        # theano.scan only allows for positional arguments, so when
        # self.peepholes is False, we need to supply fake placeholder arguments
        # for the three peephole matrices.
        else:
            non_seqs += [(), (), ()]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [self.W_in_stacked, self.b_stacked]
        # As above, when we aren't providing these parameters, we need to
        # supply placehold arguments
        else:
            non_seqs += [(), ()]

        if self.unroll_scan:
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=self.input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, 2)
        cell_out = cell_out.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1]
            cell_out = cell_out[:, ::-1]

        self.hid_out = hid_out
        self.cell_out = cell_out

        return hid_out


class GRULayer(Layer):
    r"""
    lasagne.layers.recurrent.GRULayer(incoming, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), learn_init=True, backwards=False,
    gradient_steps=-1, grad_clipping=False, unroll_scan=False,
    precompute_input=True, **kwargs)

    Gated Recurrent Unit (GRU) Layer

    Implements the updates proposed in [1]_, which computes the output by

    .. math ::
        r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
        u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
        c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode, `learn_init` is
        ignored.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then the TensorVariable is used and
        `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    References
    ----------
    .. [1] Cho, Kyunghyun, et al: On the properties of neural
       machine translation: Encoder-decoder approaches.
       arXiv preprint arXiv:1409.1259 (2014).
    .. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
       Recurrent Neural Networks on Sequence Modeling.
       arXiv preprint arXiv:1412.3555 (2014).
    .. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).

    Notes
    -----
    An alternate update for the candidate hidden state is proposed in [2]_:

    .. math::
        c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\

    We use the formulation from [1]_ because it allows us to do all matrix
    operations in a single dot product.
    """
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 learn_init=True,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 **kwargs):

        # Initialize parent layer
        super(GRULayer, self).__init__(incoming, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        if unroll_scan and self.input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        self.W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        self.W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        self.b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, input, mask=None, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        input : theano.TensorType
            Symbolic input variable
        mask : theano.TensorType
            Theano variable denoting whether each time step in each
            sequence in the batch is part of the sequence or not.  If None,
            then it is assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape (n_batch, n_time_steps) where
            ``mask[i, j] = 1`` when ``j <= (length of sequence i)`` and
            ``mask[i, j] = 0`` when ``j > (length of sequence i)``.
        """

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, self.W_in_stacked) + self.b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                 b_stacked):
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping is not False:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update
            return hid

        def step_masked(input_n, mask_n, hid_previous, W_hid_stacked,
                        W_in_stacked, b_stacked):

            hid = step(input_n, hid_previous, W_hid_stacked, W_in_stacked,
                       b_stacked)

            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [self.W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [self.W_in_stacked, self.b_stacked]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        else:
            non_seqs += [(), ()]

        if self.unroll_scan:
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=self.input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1, :]

        self.hid_out = hid_out

        return hid_out
