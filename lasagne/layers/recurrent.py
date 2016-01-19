# -*- coding: utf-8 -*-
"""
Layers to construct recurrent networks. Recurrent layers can be used similarly
to feed-forward layers except that the input shape is expected to be
``(batch_size, sequence_length, num_inputs)``.   The CustomRecurrentLayer can
also support more than one "feature" dimension (e.g. using convolutional
connections), but for all other layers, dimensions trailing the third
dimension are flattened.

The following recurrent layers are implemented:

.. currentmodule:: lasagne.layers

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

from .base import MergeLayer, Layer
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


def get_cell_shape(incoming, cell=True):
    if isinstance(incoming, Layer):
        incoming = incoming.output_shape
    # We will be passing the input at each time step to the dense layer,
    # so we need to remove the second dimension (the time dimension)
    return incoming if cell else (incoming[0],) + incoming[2:]


class RecurrentContainerLayer(MergeLayer):
    def __init__(self, incoming, cell,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.has_mask = mask_input is not None
        if self.has_mask:
            incomings.append(mask_input)
        for name, init in cell.inits.items():
            if isinstance(init, Layer):
                incomings.append(init)

        super(RecurrentContainerLayer, self).__init__(incomings, **kwargs)

        self.cell = cell
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Check that params are valid as early as possible
        helper.get_all_params(
            self.cell, step=True, precompute_input=self.precompute_input)

        # Precompute shape
        if precompute_input:
            input_shape = cell.precompute_shape_for(input_shape)

        # Initialize states
        self.inits = {}
        for name, init in cell.inits.items():
            if isinstance(init, Layer):
                self.inits[name] = init
            else:
                self.inits[name] = self.add_param(
                    init, (1,) + cell.get_output_shape_for(
                        get_cell_shape(input_shape))[name][1:],
                    name='hid_init', trainable=learn_init,
                    regularizable=False)

    def get_params(self, **tags):
        for tag in ('step', 'step_only', 'precompute_input'):
            tags.pop(tag, None)
        # Get all parameters from this layer, the master layer
        params = super(RecurrentContainerLayer, self).get_params(**tags)
        # Combine with all parameters from the cells
        params += helper.get_all_params(self.cell, **tags)
        return params

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        if self.precompute_input:
            input_shape = self.cell.precompute_shape_for(input_shape)
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return (input_shape[0],) + self.cell.get_output_shape_for(
                get_cell_shape(input_shape))['output'][1:]
        # Otherwise, the shape will be (n_batch, n_steps, trailing_dims...)
        else:
            return ((input_shape[0], input_shape[1]) +
                    self.cell.get_output_shape_for(get_cell_shape(input_shape))
                    ['output'][1:])

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = inputs[1] if self.has_mask else None

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]

        if self.precompute_input:
            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning.
            input = self.cell.precompute_for(input, **kwargs)

        # Pass the cell params to step
        non_seqs = helper.get_all_params(
            self.cell, step=True, precompute_input=self.precompute_input)

        # Create single recurrent computation step function
        def step(*args):
            inputs_n = {'input': args[0]}
            for i, name in enumerate(self.cell.inits):
                inputs_n[name] = args[i + 1]
            cell_outs_n = self.cell.get_output_for(
                inputs_n, precompute_input=self.precompute_input, **kwargs)
            return [cell_outs_n[name] for name in self.cell.inits]

        def step_masked(*args):
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            input_n, mask_n = args[0], args[1]
            outs = step(input_n, *args[2:])
            for i, out in enumerate(outs):
                outs[i] = T.switch(mask_n, out, args[i + 2])
            return outs

        if self.has_mask:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        inits = []
        ones = T.ones((num_batch, 1))
        i = 2 if self.has_mask else 1
        for name in self.cell.inits:
            init = self.inits[name]
            if isinstance(init, Layer):
                inits.append(inputs[i])
                i += 1
            else:
                # The code below simply repeats self.hid_init num_batch times
                # in its first dimension.  Turns out using a dot product and a
                # dimshuffle is faster than T.repeat.
                dot_dims = list(range(1, init.ndim - 1)) + [0, init.ndim - 1]
                inits.append(T.dot(ones, init.dimshuffle(dot_dims)))

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=inits,
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=inits,
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]
            if len(inits) == 1:
                out = [out]

        # retrieve the output state when there are multiple states per step
        out = out[list(self.cell.inits).index('output')]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            out = out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            out = out.dimshuffle(1, 0, *range(2, out.ndim))

            # if scan is backward reverse the output
            if self.backwards:
                out = out[:, ::-1]

        return out


class CellLayer(Layer):
    def add_param(self, spec, shape, name=None, **tags):
        tags['step'] = tags.get('step', True)
        tags['step_only'] = tags.get('step_only', False)
        tags['precompute_input'] = tags.get('precompute_input', True)
        return super(CellLayer, self).add_param(spec, shape, name, **tags)

    def get_params(self, **tags):
        if not tags.get('step', False):
            tags['step_only'] = tags.get('step_only', False)
        if not tags.get('precompute_input', True):
            tags.pop('precompute_input')
        return super(CellLayer, self).get_params(**tags)

    def precompute_shape_for(self, input_shape):
        return input_shape

    def precompute_for(self, input):
        return input


class CustomRecurrentCell(CellLayer):
    """
    lasagne.layers.recurrent.CustomRecurrentLayer(incoming, input_to_hidden,
    hidden_to_hidden, nonlinearity=lasagne.nonlinearities.rectify,
    hid_init=lasagne.init.Constant(0.), backwards=False,
    learn_init=False, gradient_steps=-1, grad_clipping=0,
    unroll_scan=False, precompute_input=True, mask_input=None,
    only_return_final=False, **kwargs)

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
        same input shape as `incoming`, except for the first dimension: When
        ``precompute_input == True`` (the default), it must be
        ``incoming.output_shape[0]*incoming.output_shape[1]`` or ``None``; when
        ``precompute_input == False``, it must be ``incoming.output_shape[0]``
        or ``None``.
    hidden_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects the previous hidden state to the new state
        (:math:`f_h`).  This layer may be connected to a chain of layers, which
        must end in a :class:`lasagne.layers.InputLayer` with the same input
        shape as `hidden_to_hidden`'s output shape.
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
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
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

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

    The CustomRecurrentLayer can also support "convolutional recurrence", as is
    demonstrated below.

    >>> n_batch, n_steps, n_channels, width, height = (2, 3, 4, 5, 6)
    >>> n_out_filters = 7
    >>> filter_shape = (3, 3)
    >>> l_in = lasagne.layers.InputLayer(
    ...     (n_batch, n_steps, n_channels, width, height))
    >>> l_in_to_hid = lasagne.layers.Conv2DLayer(
    ...     lasagne.layers.InputLayer((None, n_channels, width, height)),
    ...     n_out_filters, filter_shape, pad='same')
    >>> l_hid_to_hid = lasagne.layers.Conv2DLayer(
    ...     lasagne.layers.InputLayer(l_in_to_hid.output_shape),
    ...     n_out_filters, filter_shape, pad='same')
    >>> l_rec = lasagne.layers.CustomRecurrentLayer(
    ...     l_in, l_in_to_hid, l_hid_to_hid)

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 grad_clipping=0,
                 **kwargs):
        super(CustomRecurrentCell, self).__init__(incoming, **kwargs)
        input_to_hidden_in_layers = \
            [layer for layer in helper.get_all_layers(input_to_hidden)
             if isinstance(layer, InputLayer)]
        if len(input_to_hidden_in_layers) != 1:
            raise ValueError(
                '`input_to_hidden` must have exactly one InputLayer, but it '
                'has {}'.format(len(input_to_hidden_in_layers)))

        hidden_to_hidden_in_lyrs = \
            [layer for layer in helper.get_all_layers(hidden_to_hidden)
             if isinstance(layer, InputLayer)]
        if len(hidden_to_hidden_in_lyrs) != 1:
            raise ValueError(
                '`hidden_to_hidden` must have exactly one InputLayer, but it '
                'has {}'.format(len(hidden_to_hidden_in_lyrs)))
        self.hidden_to_hidden_in_layer = hidden_to_hidden_in_lyrs[0]

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.inits = {'output': hid_init}
        self.grad_clipping = grad_clipping
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

    def get_params(self, **tags):
        step = tags.pop('step', False)
        tags.pop('step_only', None)
        precompute_input = tags.pop('precompute_input', False)

        # Check that input_to_hidden and hidden_to_hidden output shapes match,
        # but don't check a dimension if it's None for either shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(self.input_to_hidden.output_shape[1:],
                                     self.hidden_to_hidden.output_shape[1:])):
            raise ValueError("The output shape for input_to_hidden and "
                             "hidden_to_hidden must be equal after the first "
                             "dimension, but input_to_hidden.output_shape={} "
                             "and hidden_to_hidden.output_shape={}".format(
                                 self.input_to_hidden.output_shape,
                                 self.hidden_to_hidden.output_shape))

        # Check that input_to_hidden's output shape is the same as
        # hidden_to_hidden's input shape but don't check a dimension if it's
        # None for either shape
        h_to_h_input_shape = self.hidden_to_hidden_in_layer.output_shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(self.input_to_hidden.output_shape[1:],
                                     h_to_h_input_shape[1:])):
            raise ValueError(
                "The output shape for input_to_hidden must be equal to the "
                "input shape of hidden_to_hidden after the first dimension, "
                "but input_to_hidden.output_shape={} and "
                "hidden_to_hidden:input_layer.shape={}".format(
                    self.input_to_hidden.output_shape, h_to_h_input_shape))

        # Check that the first dimension of input_to_hidden and
        # hidden_to_hidden's outputs match when we won't precompute the input
        # dot product
        if (not precompute_input and
                self.input_to_hidden.output_shape[0] is not None and
                self.hidden_to_hidden.output_shape[0] is not None and
                (self.input_to_hidden.output_shape[0] !=
                 self.hidden_to_hidden.output_shape[0])):
            raise ValueError(
                'When precompute_input == False, '
                'input_to_hidden.output_shape[0] must equal '
                'hidden_to_hidden.output_shape[0] but '
                'input_to_hidden.output_shape[0] = {} and '
                'hidden_to_hidden.output_shape[0] = {}'.format(
                    self.input_to_hidden.output_shape[0],
                    self.hidden_to_hidden.output_shape[0]))

        params = helper.get_all_params(self.hidden_to_hidden, **tags)
        if not (step and precompute_input):
            params += helper.get_all_params(self.input_to_hidden, **tags)
        return params

    def precompute_shape_for(self, input_shape):
        # Check that the input_to_hidden connection can appropriately handle
        # a first dimension of input_shape[0]*input_shape[1] when we will
        # precompute the input dot product
        if (self.input_to_hidden.output_shape[0] is not None and
                input_shape[0] is not None and
                input_shape[1] is not None and
                (self.input_to_hidden.output_shape[0] !=
                 input_shape[0]*input_shape[1])):
            raise ValueError(
                'When precompute_input == True, '
                'input_to_hidden.output_shape[0] must equal '
                'incoming.output_shape[0]*incoming.output_shape[1] '
                '(i.e. batch_size*sequence_length) or be None but '
                'input_to_hidden.output_shape[0] = {} and '
                'incoming.output_shape[0]*incoming.output_shape[1] = '
                '{}'.format(self.input_to_hidden.output_shape[0],
                            input_shape[0]*input_shape[1]))
        return input_shape

    def get_output_shape_for(self, input_shape):
        return {'output': self.hidden_to_hidden.output_shape}

    def precompute_for(self, input, **kwargs):
        seq_len, num_batch = input.shape[0], input.shape[1]

        # Because the input is given for all time steps, we can precompute
        # the inputs to hidden before scanning. First we need to reshape
        # from (seq_len, batch_size, trailing dimensions...) to
        # (seq_len*batch_size, trailing dimensions...)
        # This strange use of a generator in a tuple was because
        # input.shape[2:] was raising a Theano error
        trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
        input = T.reshape(input, (seq_len*num_batch,) + trailing_dims)
        input = helper.get_output(
            self.input_to_hidden, input, **kwargs)

        # Reshape back to (seq_len, batch_size, trailing dimensions...)
        trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
        return T.reshape(input, (seq_len, num_batch) + trailing_dims)

    def get_output_for(self, inputs, precompute_input=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        input, hidden = inputs['input'], inputs['output']

        # Compute the hidden-to-hidden activation
        hid_pre = helper.get_output(
            self.hidden_to_hidden, hidden, **kwargs)

        # If the dot product is precomputed then add it, otherwise
        # calculate the input_to_hidden values and add them
        if precompute_input:
            hid_pre += input
        else:
            hid_pre += helper.get_output(
                self.input_to_hidden, input, **kwargs)

        # Clip gradients
        if self.grad_clipping:
            hid_pre = theano.gradient.grad_clip(
                hid_pre, -self.grad_clipping, self.grad_clipping)

        return {'output': self.nonlinearity(hid_pre)}


class CustomRecurrentLayer(RecurrentContainerLayer):
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 grad_clipping=0,
                 **kwargs):
        cell_kwargs = {'name': kwargs['name']} if 'name' in kwargs else {}
        cell = CustomRecurrentCell(
            get_cell_shape(incoming, cell=False), input_to_hidden,
            hidden_to_hidden, nonlinearity, hid_init, grad_clipping, **cell_kwargs)
        super(CustomRecurrentLayer, self).__init__(incoming, cell, **kwargs)


class DenseRecurrentCell(CustomRecurrentCell):
    """
    lasagne.layers.recurrent.RecurrentLayer(incoming, num_units,
    W_in_to_hid=lasagne.init.Uniform(), W_hid_to_hid=lasagne.init.Uniform(),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

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
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
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
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

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
                 grad_clipping=0,
                 **kwargs):
        input_shape = get_cell_shape(incoming)

        # Retrieve the supplied name, if it exists; otherwise use ''
        if 'name' in kwargs:
            basename = kwargs['name'] + '.'
            # Create a separate version of kwargs for the contained layers
            # which does not include 'name'
            layer_kwargs = dict((key, arg) for key, arg in kwargs.items()
                                if key != 'name')
        else:
            basename = ''
            layer_kwargs = kwargs

        in_to_hid = DenseLayer(InputLayer((None,) + input_shape[1:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None,
                               name=basename + 'input_to_hidden',
                               **layer_kwargs)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((None, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None,
                                name=basename + 'hidden_to_hidden',
                                **layer_kwargs)

        super(DenseRecurrentCell, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity, hid_init,
            grad_clipping, **kwargs)


class RecurrentLayer(RecurrentContainerLayer):
    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 grad_clipping=0,
                 **kwargs):
        cell_kwargs = {'name': kwargs['name']} if 'name' in kwargs else {}
        cell = DenseRecurrentCell(
            get_cell_shape(incoming, cell=False), num_units, W_in_to_hid,
            W_hid_to_hid, b, nonlinearity, hid_init, grad_clipping,
            **cell_kwargs)

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = cell.input_to_hidden.W
        self.W_hid_to_hid = cell.hidden_to_hidden.W
        self.b = cell.input_to_hidden.b

        super(RecurrentLayer, self).__init__(incoming, cell, **kwargs)


class Gate(object):
    """
    lasagne.layers.recurrent.Gate(W_in=lasagne.init.Normal(0.1),
    W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)

    Simple class to hold the parameters for a gate connection.  We define
    a gate loosely as something which computes the linear mix of two inputs,
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
                 nonlinearity=nonlinearities.sigmoid, name=''):
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
        self.name = name

    def add_params_to(self, layer, num_inputs, num_units, **tags):
        """ Convenience function for adding layer parameters from a Gate
        instance. """
        return (layer.add_param(self.W_in, (num_inputs, num_units),
                                name='W_in_to_{}'.format(self.name), **tags),
                layer.add_param(self.W_hid, (num_units, num_units),
                                name='W_hid_to_{}'.format(self.name), **tags),
                layer.add_param(self.b, (num_units,),
                                name='b_{}'.format(self.name),
                                regularizable=False, **tags),
                self.nonlinearity)


class LSTMCell(CellLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    hid_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

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
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
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
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
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
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=Gate(name='ingate'),
                 forgetgate=Gate(name='forgetgate'),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh,
                           name='cell'),
                 outgate=Gate(name='outgate'),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 peepholes=True,
                 grad_clipping=0,
                 **kwargs):
        super(LSTMCell, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.inits = {'cell': cell_init, 'output': hid_init}
        self.peepholes = peepholes
        self.grad_clipping = grad_clipping
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        num_inputs = np.prod(get_cell_shape(incoming)[1:])

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = ingate.add_params_to(
            self, num_inputs, num_units, step=False)

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = forgetgate.add_params_to(
            self, num_inputs, num_units, step=False)

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = cell.add_params_to(
            self, num_inputs, num_units, step=False)

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = outgate.add_params_to(
            self, num_inputs, num_units, step=False)

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units,), name='W_cell_to_ingate')

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units,), name='W_cell_to_forgetgate')

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units,), name='W_cell_to_outgate')

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        self.W_in_stacked = self.add_param(T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1),
            (num_inputs, 4*num_units), step_only=True, precompute_input=False)

        # Same for hidden weight matrices
        self.W_hid_stacked = self.add_param(T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1),
            (num_units, 4*num_units), step_only=True)

        # Stack biases into a (4*num_units) vector
        self.b_stacked = self.add_param(T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0),
            (4*num_units,), step_only=True, precompute_input=False)

    def get_output_shape_for(self, input_shape):
        return {
            'cell': (input_shape[0], self.num_units),
            'output': (input_shape[0], self.num_units),
        }

    def precompute_for(self, input, **kwargs):
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because the input is given for all time steps, we can
        # precompute_input the inputs dot weight matrices before scanning.
        # W_in_stacked is (n_features, 4*num_units). input is then
        # (n_time_steps, n_batch, 4*num_units).
        return T.dot(input, self.W_in_stacked) + self.b_stacked

    def get_output_for(self, inputs, precompute_input=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        input, cell_previous, hid_previous = \
            inputs['input'], inputs['cell'], inputs['output']

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            s = x[:, n*self.num_units:(n+1)*self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        if not precompute_input:
            if input.ndim > 2:
                input = T.flatten(input, 2)
            input = T.dot(input, self.W_in_stacked) + self.b_stacked

        # Calculate gates pre-activations and slice
        gates = input + T.dot(hid_previous, self.W_hid_stacked)

        # Clip gradients
        if self.grad_clipping:
            gates = theano.gradient.grad_clip(
                gates, -self.grad_clipping, self.grad_clipping)

        # Extract the pre-activation gate values
        ingate = slice_w(gates, 0)
        forgetgate = slice_w(gates, 1)
        cell_input = slice_w(gates, 2)
        outgate = slice_w(gates, 3)

        if self.peepholes:
            # Compute peephole connections
            ingate += cell_previous*self.W_cell_to_ingate
            forgetgate += cell_previous*self.W_cell_to_forgetgate

        # Apply nonlinearities
        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)

        # Compute new cell value
        cell = forgetgate*cell_previous + ingate*cell_input

        if self.peepholes:
            outgate += cell*self.W_cell_to_outgate
        outgate = self.nonlinearity_outgate(outgate)

        # Compute new hidden unit activation
        hid = outgate*self.nonlinearity(cell)
        return {'cell': cell, 'output': hid}


class LSTMLayer(RecurrentContainerLayer):
    def __init__(self, incoming, num_units,
                 ingate=Gate(name='ingate'),
                 forgetgate=Gate(name='forgetgate'),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh,
                           name='cell'),
                 outgate=Gate(name='outgate'),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 peepholes=True,
                 grad_clipping=0,
                 **kwargs):
        cell_kwargs = {'name': kwargs['name']} if 'name' in kwargs else {}
        cell = LSTMCell(
            get_cell_shape(incoming, cell=False), num_units, ingate,
            forgetgate, cell, outgate, nonlinearity, cell_init, hid_init,
            peepholes, grad_clipping, **cell_kwargs)
        super(LSTMLayer, self).__init__(incoming, cell, **kwargs)


class GRUCell(CellLayer):
    r"""
    lasagne.layers.recurrent.GRULayer(incoming, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

    Gated Recurrent Unit (GRU) Layer

    Implements the recurrent step proposed in [1]_, which computes the output
    by

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
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
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
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

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
                 resetgate=Gate(W_cell=None, name='resetgate'),
                 updategate=Gate(W_cell=None, name='updategate'),
                 hidden_update=Gate(
                     W_cell=None, nonlinearity=nonlinearities.tanh,
                     name='hidden_update'),
                 hid_init=init.Constant(0.),
                 grad_clipping=0,
                 **kwargs):
        super(GRUCell, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.inits = {'output': hid_init}
        self.grad_clipping = grad_clipping

        num_inputs = np.prod(get_cell_shape(incoming)[1:])

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = updategate.add_params_to(
            self, num_inputs, num_units, step=False)
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = resetgate.add_params_to(
            self, num_inputs, num_units, step=False)
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = hidden_update\
            .add_params_to(self, num_inputs, num_units, step=False)

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        self.W_in_stacked = self.add_param(T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1),
            (num_inputs, 3*num_units), step_only=True, precompute_input=False)

        # Same for hidden weight matrices
        self.W_hid_stacked = self.add_param(T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1),
            (num_units, 3*num_units), step_only=True)

        # Stack gate biases into a (3*num_units) vector
        self.b_stacked = self.add_param(T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0),
            (3*num_units,), step_only=True, precompute_input=False)

    def get_output_shape_for(self, input_shape):
        return {'output': (input_shape[0], self.num_units)}

    def precompute_for(self, input, **kwargs):
        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # precompute_input inputs*W. W_in is (n_features, 3*num_units).
        # input is then (n_batch, n_time_steps, 3*num_units).
        return T.dot(input, self.W_in_stacked) + self.b_stacked

    def get_output_for(self, inputs, precompute_input=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        input, hid_previous = inputs['input'], inputs['output']

        # When theano.scan calls step, input_n will be (n_batch, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            s = x[:, n*self.num_units:(n+1)*self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        if not precompute_input:
            if input.ndim > 2:
                input = T.flatten(input, 2)

        # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
        hid_input = T.dot(hid_previous, self.W_hid_stacked)

        if self.grad_clipping:
            input = theano.gradient.grad_clip(
                input, -self.grad_clipping, self.grad_clipping)
            hid_input = theano.gradient.grad_clip(
                hid_input, -self.grad_clipping, self.grad_clipping)

        if not precompute_input:
            # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
            input = T.dot(input, self.W_in_stacked) + self.b_stacked

        # Reset and update gates
        resetgate = slice_w(hid_input, 0) + slice_w(input, 0)
        updategate = slice_w(hid_input, 1) + slice_w(input, 1)
        resetgate = self.nonlinearity_resetgate(resetgate)
        updategate = self.nonlinearity_updategate(updategate)

        # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
        hidden_update_in = slice_w(input, 2)
        hidden_update_hid = slice_w(hid_input, 2)
        hidden_update = hidden_update_in + resetgate*hidden_update_hid
        if self.grad_clipping:
            hidden_update = theano.gradient.grad_clip(
                hidden_update, -self.grad_clipping, self.grad_clipping)
        hidden_update = self.nonlinearity_hid(hidden_update)

        # Compute (1 - u_t)h_{t - 1} + u_t c_t
        hid = (1 - updategate)*hid_previous + updategate*hidden_update
        return {'output': hid}


class GRULayer(RecurrentContainerLayer):
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None, name='resetgate'),
                 updategate=Gate(W_cell=None, name='updategate'),
                 hidden_update=Gate(
                     W_cell=None, nonlinearity=nonlinearities.tanh,
                     name='hidden_update'),
                 hid_init=init.Constant(0.),
                 grad_clipping=0,
                 **kwargs):
        cell_kwargs = {'name': kwargs['name']} if 'name' in kwargs else {}
        cell = GRUCell(
            get_cell_shape(incoming, cell=False), num_units, resetgate,
            updategate, hidden_update, hid_init,
            grad_clipping, **cell_kwargs)
        super(GRULayer, self).__init__(incoming, cell, **kwargs)
