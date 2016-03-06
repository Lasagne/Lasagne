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
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from .. import nonlinearities
from .. import init
from .. import utils
from ..utils import unroll_scan

from .base import MergeLayer, Layer
from .input import InputLayer
from .dense import DenseLayer
from . import helper

__all__ = [
    "RecurrentContainerLayer",
    "CellLayer",
    "CustomRecurrentCell",
    "CustomRecurrentLayer",
    "DenseRecurrentCell",
    "RecurrentLayer",
    "Gate",
    "LSTMCell",
    "LSTMLayer",
    "GRUCell",
    "GRULayer"
]


class RecurrentContainerLayer(MergeLayer):
    def __init__(self, incomings, cell,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 n_steps=None,
                 **kwargs):
        """
        Parameters
        ----------
        precompute_input : Because the input is given for all time steps,
            we can precompute the inputs to hidden before scanning.
        """

        # This layer inherits from a MergeLayer, because it can have multiple
        # inputs - the layer inputs, the mask and the initial hidden state.  We
        # will just provide the layer inputs as incomings, unless a mask input
        # or initial hidden state was provided.

        # Obtain topological ordering of all layers the output layer(s)
        # depend on
        self.cells = helper.get_all_layers(cell)

        self.seq_incomings = incomings.copy()
        if mask_input is not None:
            incomings['mask'] = mask_input
        for i, cell_m in enumerate(self.cells):
            if isinstance(cell_m, CellLayer):
                for name, init in cell_m.inits.items():
                    if isinstance(init, Layer):
                        incomings[cell_m.input_layers[name]] = init

        super(RecurrentContainerLayer, self).__init__(incomings, **kwargs)

        self.cell = cell
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.n_steps = n_steps

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        for cell_m in self.seq_incomings:
            input_shape = self.input_shapes[cell_m]
            if unroll_scan and input_shape[1] is None:
                raise ValueError("Input sequence length cannot be specified "
                                 "as None when unroll_scan is True")

        # Check that params are valid as early as possible
        self._get_cell_params(
            step=True, precompute_input=self.precompute_input)

        # Initialize states
        self.inits = {}
        input_shapes = self._get_cell_input_shape_for(self.input_shapes)
        cell_input_shapes = self._precompute_cell_output_shape_for(
            input_shapes)
        for cell_m, shape_m in input_shapes.items():
            if cell_m in self.seq_incomings:
                input_shapes[cell_m] = self._get_cell_shape(shape_m)
        for cell_m, shapes_m in cell_input_shapes.items():
            cell_input_shapes[cell_m] = {}
            for name, shape in shapes_m.items():
                if cell_m.input_layers[name] in self.seq_incomings:
                    cell_input_shapes[cell_m][name] = self._get_cell_shape(
                        shape) if shape is not None else shape
        self.all_shapes = helper.get_output_shape(
            self.cells, input_shapes, cell_input_shapes)
        for cell_m, shape_m in zip(self.cells, self.all_shapes):
            if isinstance(cell_m, CellLayer):
                for name, init in cell_m.inits.items():
                    if isinstance(init, Layer):
                        self.inits[cell_m.input_layers[name]] = init
                    else:
                        self.inits[cell_m.input_layers[name]] = self.add_param(
                            init, (1,) + shape_m[name][1:],
                            name='hid_init', trainable=learn_init,
                            regularizable=False)

    @staticmethod
    def _get_cell_shape(incoming):
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        if isinstance(incoming, Layer):
            incoming = incoming.output_shape
        return (incoming[0],) + incoming[2:]

    def _get_cell_params(self, **tags):
        params = []
        tags_ = tags.copy()
        for tag in ('step', 'step_only', 'precompute_input'):
            tags_.pop(tag, None)
        for l in self.cells:
            params.extend(l.get_params(**(
                tags if isinstance(l, CellLayer) else tags_)))
        return utils.unique(params)

    def get_params(self, **tags):
        for tag in ('step', 'step_only', 'precompute_input'):
            tags.pop(tag, None)
        # Get all parameters from this layer, the master layer
        params = super(RecurrentContainerLayer, self).get_params(**tags)
        # Combine with all parameters from the cells
        params += self._get_cell_params(**tags)
        return params

    def _get_cell_input_shape_for(self, input_shapes):
        # initialize layer-to-shape mapping from all input layers
        all_shapes = dict((cell_m, cell_m.shape) for cell_m in self.cells
                          if isinstance(cell_m, InputLayer) and
                          cell_m not in input_shapes)
        # update layer-to-shape mapping from given input(s), if any
        all_shapes.update(input_shapes)
        # update inits
        for cell_m in self.cells:
            if isinstance(cell_m, CellLayer):
                for name, init in cell_m.inits.items():
                    if not isinstance(init, Layer):
                        all_shapes[cell_m.input_layers[name]] = None
        return all_shapes

    def _precompute_cell_output_shape_for(self, input_shapes):
        cell_input_shapes = {}
        for cell_m in self.cells:
            if isinstance(cell_m, CellLayer):
                if self.precompute_input and cell_m.is_precomputable():
                    cell_input_shapes[cell_m] = {}
                    for name, shape in cell_m.precompute_shape_for({
                        name: input_shapes[input_layer]
                        for name, input_layer in cell_m.input_layers.items()
                    }).items():
                        cell_input_shapes[cell_m][name] = shape
        return cell_input_shapes

    def _get_cell_input_for(self, inputs):
        # initialize layer-to-expression mapping from all input layers
        all_outputs = dict((cell_m, cell_m.input_var)
                           for cell_m in self.cells
                           if isinstance(cell_m, InputLayer) and
                           cell_m not in inputs)
        # update layer-to-expression mapping from given input(s), if any
        all_outputs.update((layer, utils.as_theano_expression(expr))
                           for layer, expr in inputs.items())
        # update inits
        n_batch = next(iter(inputs.values())).shape[0]
        ones = T.ones((n_batch, 1))
        for cell_m in self.cells:
            if isinstance(cell_m, CellLayer):
                for name, init in cell_m.inits.items():
                    if not isinstance(init, Layer):
                        init = self.inits[cell_m.input_layers[name]]
                        # The code below simply repeats self.hid_init
                        # num_batch times in its first dimension.  Turns
                        # out using a dot product and a dimshuffle is
                        # faster than T.repeat.
                        dot_dims = list(range(1, init.ndim - 1)) + [
                            0, init.ndim - 1]
                        init = T.dot(ones, init.dimshuffle(dot_dims))
                        all_outputs[cell_m.input_layers[name]] = init
        return all_outputs

    def _precompute_cell_output_for(self, inputs, **kwargs):
        cell_inputs = {}
        for cell_m in self.cells:
            if isinstance(cell_m, CellLayer):
                if self.precompute_input and cell_m.is_precomputable():
                    cell_inputs[cell_m] = {}
                    for name, input in cell_m.precompute_for({
                        name: inputs[input_layer]
                        for name, input_layer in cell_m.input_layers.items()
                    }, **kwargs).items():
                        cell_inputs[cell_m][name] = input
        return cell_inputs

    @staticmethod
    def _output_to_dict(output):
        return output if isinstance(output, dict) else {'output': output}

    def _output_from_dict(self, output):
        return output if isinstance(self.all_shapes[-1], dict) else \
            output['output']

    def get_output_shape_for(self, input_shapes):
        n_batch = next(iter(input_shapes.values()))[0]
        n_steps = self.n_steps if self.n_steps is not None else \
            self.input_shapes[next(iter(self.seq_incomings))][1]
        input_shapes = self._get_cell_input_shape_for(input_shapes)
        cell_input_shapes = self._precompute_cell_output_shape_for(
            input_shapes)
        for cell_m, shape_m in input_shapes.items():
            if cell_m in self.seq_incomings:
                input_shapes[cell_m] = self._get_cell_shape(shape_m)
        for cell_m, shapes_m in cell_input_shapes.items():
            for name, shape in shapes_m.items():
                if cell_m.input_layers[name] in self.seq_incomings:
                    cell_input_shapes[cell_m][name] = self._get_cell_shape(
                        shape) if shape is not None else shape
        output_shape = self._output_to_dict(helper.get_output_shape(
            self.cell, input_shapes, cell_input_shapes))
        for name, output_shape_m in output_shape.items():
            if self.only_return_final:
                # When only_return_final is true, the second (sequence step)
                # dimension will be flattened
                output_shape[name] = (n_batch,) + output_shape_m[1:]
            else:
                output_shape[name] = (n_batch, n_steps) + output_shape_m[1:]
        return self._output_from_dict(output_shape)

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
        inputs = self._get_cell_input_for(inputs)
        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        for seq_incoming in self.seq_incomings:
            inputs[seq_incoming] = inputs[seq_incoming].dimshuffle(
                1, 0, *range(2, inputs[seq_incoming].ndim))
        cell_inputs = self._precompute_cell_output_for(inputs, **kwargs)

        # Create seq states
        seqs, layer_seqs, all_seqs = {}, {}, []
        for cell_m in inputs:
            if cell_m in self.seq_incomings:
                seqs[cell_m] = inputs[cell_m]
        all_seqs.extend(seqs.values())
        for cell_m, inputs_m in cell_inputs.items():
            layer_seqs[cell_m] = {}
            for name, input in inputs_m.items():
                if cell_m.input_layers[name] in self.seq_incomings:
                    layer_seqs[cell_m][name] = cell_inputs[cell_m][name]
            all_seqs.extend(layer_seqs[cell_m].values())
        all_seqs_uniq, all_seqs_index = utils.unique(
            all_seqs, return_index=True)
        all_seqs_index_iter = iter(all_seqs_index)
        seqs_index = dict(zip(seqs, all_seqs_index_iter))
        layer_seqs_index = {}
        for cell_m, inputs_m in cell_inputs.items():
            layer_seqs_index[cell_m] = dict(zip(
                layer_seqs[cell_m], all_seqs_index_iter))

        # Create init states
        inits_uniq, inits_index = [], {}
        for cell_m in self.cells:
            if isinstance(cell_m, CellLayer):
                for name in cell_m.output_shape:
                    input_layer = cell_m.input_layers[name]
                    inits_uniq.append(inputs[input_layer])
                    inits_index[input_layer] = len(inits_uniq) - 1

        # Create non-sequence states
        non_seqs, non_seqs_index = self._get_cell_params(
            step=True, precompute_input=self.precompute_input), {}
        for cell_m in self.cells:
            if isinstance(cell_m, CellLayer):
                for name in set(cell_m.inits) - set(cell_m.output_shape):
                    input_layer = cell_m.input_layers[name]
                    non_seqs.append(inputs[input_layer])
                    non_seqs_index[input_layer] = len(non_seqs) - 1

        # Create output states. Only create when they don't already belong
        # to an intermediate cell's step output. This is necessary when the
        # mask is all 0 so we can use the cell's init, and beneficial when
        # the final layer is an index layer.
        cell_kwargs = {}
        for cell_m in self.cells:
            if isinstance(cell_m, CellLayer):
                cell_kwargs[cell_m] = {
                    'precompute_input':
                        self.precompute_input and cell_m.is_precomputable()}
        for cell_m in inputs:
            if cell_m in self.seq_incomings:
                inputs[cell_m] = inputs[cell_m][0]
        for cell_m, input_m in cell_inputs.items():
            for name, input in input_m.items():
                if cell_m.input_layers[name] in self.seq_incomings:
                    cell_inputs[cell_m][name] = cell_inputs[cell_m][name][0]
        outputs_n = helper.get_output(
            self.cells, inputs, layer_inputs=cell_inputs,
            layer_kwargs=cell_kwargs, **kwargs)
        output_uniq, output_index = [], {}
        output_n = self._output_to_dict(outputs_n[-1])
        output_shape_n = self._output_to_dict(self.output_shape)
        for name in output_n:
            for cell_m, outputs_m in zip(self.cells[:-1], outputs_n[:-1]):
                if isinstance(cell_m, CellLayer):
                    if output_n[name] in outputs_m.values():
                        cell_name = next(
                            cell_name for cell_name, value in outputs_m.items()
                            if value == output_n[name])
                        output_index[name] = -len(inits_uniq) + inits_index[
                            cell_m.input_layers[cell_name]]
                        break
            else:
                output_uniq.append(T.zeros(
                    self._get_cell_shape(output_shape_n[name])))
                output_index[name] = len(output_uniq) - 1

        # Create single recurrent computation step function
        def step(*args):
            inputs_n = {}
            inputs_n.update({seq: args[i] for seq, i in seqs_index.items()})
            inputs_n.update({init: args[len(all_seqs_uniq) + i]
                             for init, i in inits_index.items()})
            inputs_n.update({non_seq: non_seqs[i]
                             for non_seq, i in non_seqs_index.items()})
            cell_inputs_n = {}
            for cell_m, inputs_m in cell_inputs.items():
                cell_inputs_n[cell_m] = {seq: args[i] for seq, i in
                                         layer_seqs_index[cell_m].items()}
            outputs_n = helper.get_output(
                self.cells, inputs_n, layer_inputs=cell_inputs_n,
                layer_kwargs=cell_kwargs, **kwargs)
            step_outputs_n = []
            for cell_m, cell_output_n in zip(self.cells, outputs_n):
                if isinstance(cell_m, CellLayer):
                    for name in cell_m.output_shape:
                        step_outputs_n.append(cell_output_n[name])
            output_n = self._output_to_dict(outputs_n[-1])
            for name, index in output_index.items():
                if index >= 0:
                    step_outputs_n.append(output_n[name])
            return step_outputs_n

        def step_masked(*args):
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            mask, inputs = args[0], args[1:]
            outputs = step(*inputs)
            for i, output in enumerate(outputs):
                outputs[i] = T.switch(mask, output, inputs[
                    len(all_seqs_uniq) + i])
            return outputs

        if 'mask' in inputs:
            mask = inputs['mask'].dimshuffle(1, 0, 'x')
            sequences = [mask] + all_seqs_uniq
            step_fun = step_masked
        else:
            sequences = all_seqs_uniq
            step_fun = step

        outputs_info = inits_uniq + output_uniq

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            n_steps = self.n_steps if self.n_steps is not None else \
                self.input_shapes[next(iter(self.seq_incomings))][1]
            # Explicitly unroll the recurrence instead of using scan
            outputs = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=outputs_info,
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=n_steps)
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            outputs = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=outputs_info,
                non_sequences=non_seqs,
                n_steps=self.n_steps,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]
            if len(outputs_info) == 1:
                outputs = [outputs]

        # retrieve the output state
        output = {}
        for name, index in output_index.items():
            output_m = outputs[len(inits_uniq) + index]

            # When it is requested that we only return the final sequence step,
            # we need to slice it out immediately after scan is applied
            if self.only_return_final:
                output_m = output_m[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                output_m = output_m.dimshuffle(1, 0, *range(2, output_m.ndim))

                # if scan is backward reverse the output
                if self.backwards:
                    output_m = output_m[:, ::-1]

            output[name] = output_m

        return self._output_from_dict(output)


class StepInputLayer(Layer):
    def __init__(self, **kwargs):
        self.input_layer = None
        self.input_shape = None
        self.params = OrderedDict()


class CellLayer(MergeLayer):
    def __init__(self, incomings, inits, **kwargs):
        # states may change over time, so assign new input layers
        incomings.update({name: StepInputLayer() for name in inits})
        self.inits = inits
        super(CellLayer, self).__init__(incomings, **kwargs)

    def add_param(self, spec, shape, name=None, **tags):
        tags['step'] = tags.get('step', True)
        tags['step_only'] = tags.get('step_only', False)
        tags['precompute_input'] = tags.get('precompute_input', True)
        return super(CellLayer, self).add_param(spec, shape, name, **tags)

    def get_params(self, **tags):
        if not tags.get('step', False):
            tags['step_only'] = tags.get('step_only', False)
        if not (tags.get('precompute_input', True) and
                self.is_precomputable()):
            tags.pop('precompute_input')
        return super(CellLayer, self).get_params(**tags)

    def is_precomputable(self):
        return all(isinstance(self.input_layers[name], InputLayer)
                   for name in set(self.input_layers.keys()) -
                   set(self.inits.keys()))

    def precompute_shape_for(self, input_shapes):
        return input_shapes

    def precompute_for(self, inputs, **kwargs):
        return inputs


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
        super(CustomRecurrentCell, self).__init__(
            {'input': incoming} if input_to_hidden is not None else {},
            {'output': hid_init}, **kwargs)
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
        self.grad_clipping = grad_clipping
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

    def get_params(self, **tags):
        step = tags.pop('step', False)
        tags.pop('step_only', None)
        precompute_input = tags.pop('precompute_input', False)

        if self.input_to_hidden is not None:
            # Check that input_to_hidden and hidden_to_hidden output shapes
            #  match, but don't check a dimension if it's None for either shape
            if not all(s1 is None or s2 is None or s1 == s2 for s1, s2
                       in zip(self.input_to_hidden.output_shape[1:],
                              self.hidden_to_hidden.output_shape[1:])):
                raise ValueError("The output shape for input_to_hidden and "
                                 "hidden_to_hidden must be equal after the "
                                 "first dimension, but "
                                 "input_to_hidden.output_shape={} and "
                                 "hidden_to_hidden.output_shape={}".format(
                                     self.input_to_hidden.output_shape,
                                     self.hidden_to_hidden.output_shape))

            # Check that input_to_hidden's output shape is the same as
            # hidden_to_hidden's input shape but don't check a dimension if
            # it's None for either shape
            h_to_h_input_shape = self.hidden_to_hidden_in_layer.output_shape
            if not all(s1 is None or s2 is None or s1 == s2 for s1, s2
                       in zip(self.input_to_hidden.output_shape[1:],
                              h_to_h_input_shape[1:])):
                raise ValueError("The output shape for input_to_hidden "
                                 "must be equal to the input shape of "
                                 "hidden_to_hidden after the first "
                                 "dimension, but "
                                 "input_to_hidden.output_shape={} and "
                                 "hidden_to_hidden.input_shape={}".format(
                                     self.input_to_hidden.output_shape,
                                     h_to_h_input_shape))

            # Check that the first dimension of input_to_hidden and
            # hidden_to_hidden's outputs match when we won't precompute the
            # input dot product
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

    def precompute_shape_for(self, input_shapes):
        if self.input_to_hidden is not None:
            input_shape = input_shapes['input']
            # Check that the input_to_hidden connection can appropriately
            # handle a first dimension of input_shape[0]*input_shape[1] when
            # we will precompute the input dot product
            if (self.input_to_hidden.output_shape[0] is not None and
                    input_shape[0] is not None and
                    input_shape[1] is not None and
                    (self.input_to_hidden.output_shape[0] !=
                     input_shape[0]*input_shape[1])):
                raise ValueError(
                    'When precompute_input == True, '
                    'input_to_hidden.output_shape[0] must equal '
                    'incoming.output_shape[0]*incoming.output_shape[1] '
                    '(i.e. n_batch*sequence_length) or be None but '
                    'input_to_hidden.output_shape[0] = {} and '
                    'incoming.output_shape[0]*incoming.output_shape[1] = '
                    '{}'.format(self.input_to_hidden.output_shape[0],
                                input_shape[0]*input_shape[1]))
        return input_shapes

    def get_output_shape_for(self, input_shapes):
        return {'output': self.hidden_to_hidden.output_shape}

    def precompute_for(self, inputs, **kwargs):
        if self.input_to_hidden is not None:
            input = inputs['input']
            seq_len, n_batch = input.shape[0], input.shape[1]

            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, n_batch, trailing dimensions...) to
            # (seq_len*n_batch, trailing dimensions...)
            # This strange use of a generator in a tuple was because
            # input.shape[2:] was raising a Theano error
            trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
            input = T.reshape(input, (seq_len*n_batch,) + trailing_dims)
            input = helper.get_output(
                self.input_to_hidden, input, **kwargs)

            # Reshape back to (seq_len, n_batch, trailing dimensions...)
            trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
            input = T.reshape(input, (seq_len, n_batch) + trailing_dims)
            inputs['input'] = input
        return inputs

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
        hidden = inputs['output']

        # Compute the hidden-to-hidden activation
        hid_pre = helper.get_output(
            self.hidden_to_hidden, hidden, **kwargs)

        if self.input_to_hidden is not None:
            input = inputs['input']
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
        cell_in = InputLayer(self._get_cell_shape(incoming))
        cell_kwargs = {'name': kwargs['name']} if 'name' in kwargs else {}
        cell = CustomRecurrentCell(
            cell_in, input_to_hidden, hidden_to_hidden, nonlinearity, hid_init,
            grad_clipping, **cell_kwargs)['output']
        super(CustomRecurrentLayer, self).__init__(
            {cell_in: incoming}, cell, **kwargs)


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
        input_shape = incoming.output_shape[1:]

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

        in_to_hid = DenseLayer(InputLayer((None,) + input_shape),
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
        cell_in = InputLayer(self._get_cell_shape(incoming))
        cell_kwargs = {'name': kwargs['name']} if 'name' in kwargs else {}
        cell = DenseRecurrentCell(
            cell_in, num_units, W_in_to_hid, W_hid_to_hid, b, nonlinearity,
            hid_init, grad_clipping, **cell_kwargs)['output']

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = cell.input_layer.input_to_hidden.W
        self.W_hid_to_hid = cell.input_layer.hidden_to_hidden.W
        self.b = cell.input_layer.input_to_hidden.b

        super(RecurrentLayer, self).__init__(
            {cell_in: incoming}, cell, **kwargs)


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
        super(LSTMCell, self).__init__(
            {'input': incoming},
            {'cell': cell_init, 'output': hid_init}, **kwargs)
        self.num_units = num_units
        self.peepholes = peepholes
        self.grad_clipping = grad_clipping
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        num_inputs = np.prod(incoming.output_shape[1:])

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

    def get_output_shape_for(self, input_shapes):
        return {
            'cell': (input_shapes['input'][0], self.num_units),
            'output': (input_shapes['input'][0], self.num_units),
        }

    def precompute_for(self, inputs, **kwargs):
        input = inputs['input']

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because the input is given for all time steps, we can
        # precompute_input the inputs dot weight matrices before scanning.
        # W_in_stacked is (n_features, 4*num_units). input is then
        # (n_time_steps, n_batch, 4*num_units).
        input = T.dot(input, self.W_in_stacked) + self.b_stacked
        inputs['input'] = input
        return inputs

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
        cell_in = InputLayer(self._get_cell_shape(incoming))
        cell_kwargs = {'name': kwargs['name']} if 'name' in kwargs else {}
        cell = LSTMCell(
            cell_in, num_units, ingate, forgetgate, cell, outgate,
            nonlinearity, cell_init, hid_init, peepholes, grad_clipping,
            **cell_kwargs)['output']
        super(LSTMLayer, self).__init__(
            {cell_in: incoming}, cell, **kwargs)


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
        super(GRUCell, self).__init__(
            {'input': incoming}, {'output': hid_init}, **kwargs)
        self.num_units = num_units
        self.grad_clipping = grad_clipping

        num_inputs = np.prod(incoming.output_shape[1:])

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

    def get_output_shape_for(self, input_shapes):
        return {'output': (input_shapes['input'][0], self.num_units)}

    def precompute_for(self, inputs, **kwargs):
        input = inputs['input']

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # precompute_input inputs*W. W_in is (n_features, 3*num_units).
        # input is then (n_batch, n_time_steps, 3*num_units).
        input = T.dot(input, self.W_in_stacked) + self.b_stacked
        inputs['input'] = input
        return inputs

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
        cell_in = InputLayer(self._get_cell_shape(incoming))
        cell_kwargs = {'name': kwargs['name']} if 'name' in kwargs else {}
        cell = GRUCell(
            cell_in, num_units, resetgate, updategate, hidden_update, hid_init,
            grad_clipping, **cell_kwargs)['output']
        super(GRULayer, self).__init__(
            {cell_in: incoming}, cell, **kwargs)
