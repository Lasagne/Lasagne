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

Recurrent layers and feed-forward layers can be combined in the same network
by using a few reshape operations; please refer to the recurrent examples for
further explanations.

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
    "LSTMLayer",
    "GRULayer"
]


class CustomRecurrentLayer(Layer):
    """A layer which implements a recurrent connection.

    This layer allows you to specify custom input-to-hidden and
    hidden-to-hidden connections by instantiating layer instances and passing
    them on initialization.  The output shape for the provided layers must be
    the same.  If you are looking for a standard, densely-connected recurrent
    layer, please see :class:`RecurrentLayer`.  The output is computed
    by

    .. math ::
        h_t = \sigma(f_i(x_t) + f_h(h_{t-1}))

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    input_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects input to the hidden state (:math:`f_i`).
    hidden_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects the previous hidden state to the new state
        (:math:`f_h`).
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
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored.
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

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
                 precompute_input=True):

        super(CustomRecurrentLayer, self).__init__(incoming)

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

        def clone_and_compute_output(network, network_input, new_params):
            # Gets the output from the given network and calculate the output.
            # The original parameters are replaced with the parameters given
            # in new_params. Allows for use of the strict setting in scan.
            original_hid_pre = helper.get_output(
                network, network_input, **kwargs)
            original_params = helper.get_all_params(network)
            new_hid_pre = theano.clone(
                original_hid_pre,
                replace=dict(zip(original_params, new_params)))
            return new_hid_pre

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        # This will allow us to keep track of how long the list of parameters
        # are in hidden_to_hidden
        num_params_hid_to_hid = len(non_seqs)
        # When we are not precomputing the input, we also need to pass the
        # input-to-hidden parameters to step
        if not self.precompute_input:
            non_seqs += helper.get_all_params(self.input_to_hidden)

        # Create single recurrent computation step function
        def step(input_n, hid_previous, *args):
            # For optimization reasons we need to replace the calculation
            # performed by hidden_to_hidden with weight values that scan
            # knows. The weights are given in args. We use theano.clone to
            # replace the relevant variables. This allows us to use
            # strict=True when calling theano.scan(...)
            hid_pre = clone_and_compute_output(
                self.hidden_to_hidden, hid_previous,
                args[:num_params_hid_to_hid])

            # if the dot product is precomputed then add it otherwise
            # calculate the input_to_hidden values and add them
            if self.precompute_input:
                hid_pre += input_n
            else:
                hid_pre += clone_and_compute_output(
                    self.input_to_hidden, input_n,
                    args[num_params_hid_to_hid:])

            # clip gradients
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
    """Dense recurrent neural network (RNN) layer

    A "vanilla" RNN layer, which has dense input-to-hidden and
    hidden-to-hidden connections.  The output is computed as

    .. math ::
        h_t = \sigma(W_x x_t + W_h h_{t-1} + b)

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
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored.
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
                 precompute_input=True):
        input_shape = helper.get_output_shape(incoming)
        num_batch = input_shape[0]
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((num_batch,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((num_batch, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None)

        # Just use the CustomRecurrentLayer with the DenseLayers we created
        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input)


class LSTMLayer(Layer):
    r"""A long short-term memory (LSTM) layer.

    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by

    .. math ::

        i_t &= \sigma_i(W_{xi}x_t + W_{hi}h_{t-1} + w_{ci}\odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(W_{xf}x_t + W_{hf}h_{t-1}
               + w_{cf}\odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t\sigma_c(W_{xc}x_t + W_{hc} h_{t-1} + b_c)\\
        o_t &= \sigma_o(W_{xo}x_t + W_{ho}h_{t-1} + w_{co}\odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    W_in_to_ingate : Theano shared variable, numpy array or callable
        Initializer for input-to-input gate weight matrix (:math:`W_{xi}`).
    W_hid_to_ingate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-input gate weight matrix (:math:`W_{hi}`).
    W_cell_to_ingate : Theano shared variable, numpy array or callable
        Initializer for cell-to-input gate weight vector (:math:`w_{ci}`).
    b_ingate : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector (:math:`b_i`).
    nonlinearity_ingate : callable or None
        The nonlinearity that is applied to the input gate activation
        (:math:`\sigma_i`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for input-to-forget gate weight matrix (:math:`W_{xf}`).
    W_hid_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-forget gate weight matrix (:math:`W_{hf}`).
    W_cell_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for cell-to-forget gate weight vector (:math:`w_{cf}`).
    b_forgetgate : Theano shared variable, numpy array or callable
        Initializer for forget gate bias vector (:math:`b_f`).
    nonlinearity_forgetgate : callable or None
        The nonlinearity that is applied to the forget gate activation
        (:math:`\sigma_f`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_cell : Theano shared variable, numpy array or callable
        Initializer for input-to-cell weight matrix (:math:`W_{ic}`).
    W_hid_to_cell : Theano shared variable, numpy array or callable
        Initializer for hidden-to-cell weight matrix (:math:`W_{hc}`).
    b_cell : Theano shared variable, numpy array or callable
        Initializer for cell bias vector (:math:`b_c`).
    nonlinearity_cell : callable or None
        The nonlinearity that is applied to the cell activation
        (;math:`\sigma_c`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_outgate : Theano shared variable, numpy array or callable
        Initializer for input-to-output gate weight matrix (:math:`W_{io}`).
    W_hid_to_outgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-output gate weight matrix (:math:`W_{ho}`).
    W_cell_to_outgate : Theano shared variable, numpy array or callable
        Initializer for cell-to-output gate weight vector (:math:`w_{co}`).
    b_outgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-input gate weight matrix (:math:`b_o`).
    nonlinearity_outgate : callable or None
        The nonlinearity that is applied to the output gate activation
        (:math:`\sigma_o`). If None is provided, no nonlinearity will be
        applied.
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
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored.
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
                 W_in_to_ingate=init.Normal(0.1),
                 W_hid_to_ingate=init.Normal(0.1),
                 W_cell_to_ingate=init.Normal(0.1),
                 b_ingate=init.Normal(0.1),
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 W_in_to_forgetgate=init.Normal(0.1),
                 W_hid_to_forgetgate=init.Normal(0.1),
                 W_cell_to_forgetgate=init.Normal(0.1),
                 b_forgetgate=init.Normal(0.1),
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 W_in_to_cell=init.Normal(0.1),
                 W_hid_to_cell=init.Normal(0.1),
                 b_cell=init.Normal(0.1),
                 nonlinearity_cell=nonlinearities.tanh,
                 W_in_to_outgate=init.Normal(0.1),
                 W_hid_to_outgate=init.Normal(0.1),
                 W_cell_to_outgate=init.Normal(0.1),
                 b_outgate=init.Normal(0.1),
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearity_out=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True):

        # Initialize parent layer
        super(LSTMLayer, self).__init__(incoming)

        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_ingate is None:
            self.nonlinearity_ingate = nonlinearities.identity
        else:
            self.nonlinearity_ingate = nonlinearity_ingate

        if nonlinearity_forgetgate is None:
            self.nonlinearity_forgetgate = nonlinearities.identity
        else:
            self.nonlinearity_forgetgate = nonlinearity_forgetgate

        if nonlinearity_cell is None:
            self.nonlinearity_cell = nonlinearities.identity
        else:
            self.nonlinearity_cell = nonlinearity_cell

        if nonlinearity_outgate is None:
            self.nonlinearity_outgate = nonlinearities.identity
        else:
            self.nonlinearity_outgate = nonlinearity_outgate

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

        num_inputs = np.prod(self.input_shape[2:])

        # Initialize parameters using the supplied args
        self.W_in_to_ingate = self.add_param(
            W_in_to_ingate, (num_inputs, num_units), name="W_in_to_ingate")

        self.W_hid_to_ingate = self.add_param(
            W_hid_to_ingate, (num_units, num_units), name="W_hid_to_ingate")

        self.b_ingate = self.add_param(
            b_ingate, (num_units,), name="b_ingate", regularizable=False)

        self.W_in_to_forgetgate = self.add_param(
            W_in_to_forgetgate, (num_inputs, num_units),
            name="W_in_to_forgetgate")

        self.W_hid_to_forgetgate = self.add_param(
            W_hid_to_forgetgate, (num_units, num_units),
            name="W_hid_to_forgetgate")

        self.b_forgetgate = self.add_param(
            b_forgetgate, (num_units,), name="b_forgetgate",
            regularizable=False)

        self.W_in_to_cell = self.add_param(
            W_in_to_cell, (num_inputs, num_units), name="W_in_to_cell")

        self.W_hid_to_cell = self.add_param(
            W_hid_to_cell, (num_units, num_units), name="W_hid_to_cell")

        self.b_cell = self.add_param(
            b_cell, (num_units,), name="b_cell", regularizable=False)

        self.W_in_to_outgate = self.add_param(
            W_in_to_outgate, (num_inputs, num_units), name="W_in_to_outgate")

        self.W_hid_to_outgate = self.add_param(
            W_hid_to_outgate, (num_units, num_units), name="W_hid_to_outgate")

        self.b_outgate = self.add_param(
            b_outgate, (num_units,), name="b_outgate", regularizable=False)

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
                W_cell_to_ingate, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                W_cell_to_forgetgate, (num_units, ),
                name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                W_cell_to_outgate, (num_units, ), name="W_cell_to_outgate")

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
    r"""Gated Recurrent Unit (GRU) Layer

    Implements the updates proposed in [1]_, which computes the output by

    .. math ::
        r_t &= \sigma_r(W_{xr} x_t + W_{hr} h_{t - 1} + b_r)\\
        u_t &= \sigma_u(W_{xu} x_t + W_{hu} h_{t - 1} + b_u)\\
        c_t &= \sigma_c(W_{xc} x_t + r_t \odot (W_{hc} h_{t - 1}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    W_in_to_resetgate : Theano shared variable, numpy array or callable
        Initializer for input-to-reset gate weight matrix (:math:`W_{xr}`).
    W_hid_to_resetgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-reset gate weight matrix (:math:`W_{hr}`).
    b_resetgate : Theano shared variable, numpy array or callable
        Initializer for the reset gate bias vector (:math:`b_r`).
    W_in_to_updategate : Theano shared variable, numpy array or callable
        Initializer for input-to-update gate weight matrix (:math:`W_{xu}`).
    W_hid_to_updategate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-update gate weight matrix (:math:`W_{hu}`).
    b_updategate : Theano shared variable, numpy array or callable
        Initializer for the update gate bias vector (:math:`b_u`).
    W_in_to_hidden_update : Theano shared variable, numpy array or callable
        Initializer for input-to-hidden update weight matrix (:math:`W_{xc}`).
    W_hid_to_hidden_update : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden update weight matrix (:math:`W_{hc}`).
    b_hidden_update : Theano shared variable, numpy array or callable
        Initializer for the hidden update bias vector (:math:`b_c`}.
    nonlinearity_resetgate : callable or None
        The nonlinearity that is applied to the reset gate activation
        (:math:`\sigma_r`). If None is provided, no nonlinearity will be
        applied.
    nonlinearity_updategate : callable or None
        The nonlinearity that is applied to the update gate activation
        (:math:`\sigma_u`). If None is provided, no nonlinearity will be
        applied.
    nonlinearity_hid : callable or None
        The nonlinearity that is applied to the hidden update activation
        (:math:`\sigma_c`). If None is provided, no nonlinearity will be
        applied.
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
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored.
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
        c_t &= \sigma_c(W_{ic} x_t + W_{hc}(r_t \odot h_{t - 1}) + b_c)\\

    We use the formulation from [1]_ because it allows us to do all matrix
    operations in a single dot product.
    """
    def __init__(self, incoming, num_units,
                 W_in_to_resetgate=init.Normal(0.1),
                 W_hid_to_resetgate=init.Normal(0.1),
                 b_resetgate=init.Normal(0.1),
                 W_in_to_updategate=init.Normal(0.1),
                 W_hid_to_updategate=init.Normal(0.1),
                 b_updategate=init.Normal(0.1),
                 W_in_to_hidden_update=init.Normal(0.1),
                 W_hid_to_hidden_update=init.Normal(0.1),
                 b_hidden_update=init.Constant(0.),
                 nonlinearity_resetgate=nonlinearities.sigmoid,
                 nonlinearity_updategate=nonlinearities.sigmoid,
                 nonlinearity_hid=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 learn_init=True,
                 backwards=False,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True):

        # Initialize parent layer
        super(GRULayer, self).__init__(incoming)
        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_resetgate is None:
            self.nonlinearity_resetgate = nonlinearities.identity
        else:
            self.nonlinearity_resetgate = nonlinearity_resetgate

        if nonlinearity_updategate is None:
            self.nonlinearity_updategate = nonlinearities.identity
        else:
            self.nonlinearity_updategate = nonlinearity_updategate

        if nonlinearity_hid is None:
            self.nonlinearity_hid = nonlinearities.identity
        else:
            self.nonlinearity_hid = nonlinearity_hid

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

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(self.input_shape[2:])

        self.W_in_to_updategate = self.add_param(
            W_in_to_updategate, (num_inputs, num_units),
            name="W_in_to_updategate")

        self.W_hid_to_updategate = self.add_param(
            W_hid_to_updategate, (num_units, num_units),
            name="W_hid_to_updategate")

        self.b_updategate = self.add_param(
            b_updategate, (num_units,),
            name="b_updategate", regularizable=False)

        self.W_in_to_resetgate = self.add_param(
            W_in_to_resetgate, (num_inputs, num_units),
            name="W_in_to_resetgate")

        self.W_hid_to_resetgate = self.add_param(
            W_hid_to_resetgate, (num_units, num_units),
            name="W_hid_to_resetgate")

        self.b_resetgate = self.add_param(
            b_resetgate, (num_units,),
            name="b_resetgate", regularizable=False)

        self.W_in_to_hidden_update = self.add_param(
            W_in_to_hidden_update, (num_inputs, num_units),
            name="W_in_to_hidden_update")

        self.W_hid_to_hidden_update = self.add_param(
            W_hid_to_hidden_update, (num_units, num_units),
            name="W_hid_to_hidden_update")

        self.b_hidden_update = self.add_param(
            b_hidden_update, (num_units,), name="b_hidden_update",
            regularizable=False)

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

    def get_output_for(self, input, mask=None, *args, **kwargs):
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
