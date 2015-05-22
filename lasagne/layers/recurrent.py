"""
Layers to construct recurrent networks. Recurrent layers can be used similarly
to feed-forward layers except that the input shape is expected to be
(batch_size, sequence_length, num_inputs). The input is allowed to have more
than three dimensions in which case dimensions trailing the third dimension are
flattened.

The following recurrent layers are implemented:

* :func:`CustomRecurrentLayer()`
* :func:`RecurrentLayer()`
* :func:`LSTMLayer()`

Recurrent layers and feed-forward layers can be combined in the same network
by using a few reshape operations, please refer to the recurrent examples for
further explanations.

"""
import numpy as np
import theano
import theano.tensor as T
from .. import nonlinearities
from .. import init

from .base import Layer
from .input import InputLayer
from .dense import DenseLayer
from . import helper

__all__ = [
    "CustomRecurrentLayer",
    "RecurrentLayer",
    "LSTMLayer"
]


class CustomRecurrentLayer(Layer):
    """
    A layer which implements a recurrent connection.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    input_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects input to the hidden state
    hidden_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects the previous hidden state to the new state
    nonlinearity : function or theano.tensor.elemwise.Elemwise
        Nonlinearity to apply when computing new state
    hid_init : function, np.ndarray, theano.shared or TensorVariable
        :math:`h_0`. Passing in a TensorVariable allows the user to specify
        the value of hid_init. In this mode learn_init is ignored.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from x_1 to x_n.
    learn_init : bool
        If True, initial hidden values are learned. If hid_init or cell_init
        are TensorVariables learn_init is ignored.
    gradient_steps : int
        Number of timesteps to include in backpropagated gradient
        If -1, backpropagate through the entire sequence
    grad_clipping: False or float
        If float the gradient messages are clipped during the backward pass.
        See [1]_ (p. 6) for further explanation.

    References
    ----------
    .. [1] Alex Graves : Generating Sequences With Recurrent Neural
           Networks
    """
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=False):

        super(CustomRecurrentLayer, self).__init__(incoming)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping

        # check that output shapes match
        if input_to_hidden.output_shape != hidden_to_hidden.output_shape:
            raise ValueError("The output shape for input_to_hidden and "
                             "input_to_hidden must be equal was, ",
                             input_to_hidden.output_shape,
                             hidden_to_hidden.output_shape)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Get the batch size and number of units based on the expected output
        # of the input-to-hidden layer
        self.num_batch = self.input_shape[0]
        self.seq_len = self.input_shape[1]
        self.num_inputs = np.prod(self.input_shape[2:])
        self.num_units = input_to_hidden.output_shape[-1]

        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError("When a tensor hid_init should be a matrix")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        params = super(CustomRecurrentLayer, self).get_params(**tags)
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        return params

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
            then it assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape (n_batch, n_time_steps) where
            `mask[i, j] = 1` when `j <= (length of sequence i)` and
            `mask[i, j] = 0` when `j > (length of sequence i)`.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable
        """
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Because the input is given for all time steps, we can precompute
        # the inputs to hidden before scanning. First we need to reshape
        # from (seq_len, batch_size, num_inputs) to
        # (seq_len*batch_size, num_inputs)
        input = T.reshape(input,
                          (self.seq_len*self.num_batch, -1))
        input_dot_W = helper.get_output(
            self.input_to_hidden, input, **kwargs)

        # reshape to original (seq_len, batch_size, num_units)
        input_dot_W = T.reshape(input_dot_W,
                                (self.seq_len, self.num_batch, -1))

        # Create single recurrent computation step function
        def step(input_dot_W_n, hid_previous, *args):
            # For optimization reasons we need to replace the calculation
            # performed by hidden_to_hidden with weight values that scan
            # knows. The weights are given in args. We use theano.clone to
            # replace the relevant variables. This allows us to use
            # strict=True when calling theano.scan(...)
            original_hid_pre = helper.get_output(
                self.hidden_to_hidden, hid_previous, **kwargs)
            original_params = helper.get_all_params(self.hidden_to_hidden)
            new_params = args
            new_hid_pre = theano.clone(
                original_hid_pre,
                replace=dict(zip(original_params, new_params)))

            new_hid_pre += input_dot_W_n

            # clip gradients
            if self.grad_clipping is not False:
                new_hid_pre = theano.gradient.grad_clip(
                    new_hid_pre, -self.grad_clipping, self.grad_clipping)

            return self.nonlinearity(new_hid_pre)

        def step_masked(input_dot_W_n, mask_n, hid_previous, *args):
            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            hid = step(input_dot_W_n, hid_previous, *args)
            hid_out = hid*mask_n + hid_previous*(1 - mask_n)
            return [hid_out]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input_dot_W, mask]
            step_fun = step_masked
        else:
            sequences = input_dot_W
            step_fun = step

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # repeat num_batch times
            hid_init = T.dot(T.ones((self.num_batch, 1)), self.hid_init)

        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        hid_out = theano.scan(
            step_fun,
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
    A "vanilla" RNN layer, which has dense input-to-hidden and
    hidden-to-hidden connections.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        Number of hidden units in the layer
    W_in_to_hid : function or np.ndarray or theano.shared
        Initializer for input-to-hidden weight matrix
    W_hid_to_hid : function or np.ndarray or theano.shared
        Initializer for hidden-to-hidden weight matrix
    b : function or np.ndarray or theano.shared
        Initializer for bias vector
    nonlinearity : function or theano.tensor.elemwise.Elemwise
        Nonlinearity to apply when computing new state
    hid_init : function, np.ndarray, theano.shared or TensorVariable
        :math:`h_0`. Passing in a TensorVariable allows the user to specify
        the value of hid_init. In this mode learn_init is ignored.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from x_1 to x_n.
    learn_init : bool
        If True, initial hidden values are learned. If hid_init or cell_init
        are TensorVariables learn_init is ignored.
    gradient_steps : int
        Number of timesteps to include in backpropagated gradient
        If -1, backpropagate through the entire sequence
    grad_clipping: False or float
        If float the gradient messages are clipped during the backward pass.
        See [1]_ (p. 6) for further explanation.

    References
    ----------
    .. [1] Alex Graves : Generating Sequences With Recurrent Neural
           Networks
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
                 grad_clipping=False):
        input_shape = helper.get_output_shape(incoming)
        n_batch = input_shape[0]
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((n_batch,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((n_batch, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None)

        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping)


class LSTMLayer(Layer):
    """
    A long short-term memory (LSTM) layer.  Includes "peephole connections" and
    forget gate.  Based on the definition in [1]_, which is
    the current common definition.

    Parameters
    ----------
    incoming : a :class:`:class:`lasagne.layers.Layer`` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer
    W_in_to_ingate : function or np.ndarray or theano.shared
        :math:`W_{xi}`
    W_hid_to_ingate : function or np.ndarray or theano.shared
        :math:`W_{hi}`
    W_cell_to_ingate : function or np.ndarray or theano.shared
        :math:`W_{ci}`
    b_ingate : function or np.ndarray or theano.shared
        :math:`b_i`
    nonlinearity_ingate : function
        :math:`\sigma`
    W_in_to_forgetgate : function or np.ndarray or theano.shared
        :math:`W_{xf}`
    W_hid_to_forgetgate : function or np.ndarray or theano.shared
        :math:`W_{hf}`
    W_cell_to_forgetgate : function or np.ndarray or theano.shared
        :math:`W_{cf}`
    b_forgetgate : function or np.ndarray or theano.shared
        :math:`b_f`
    nonlinearity_forgetgate : function
        :math:`\sigma`
    W_in_to_cell : function or np.ndarray or theano.shared
        :math:`W_{ic}`
    W_hid_to_cell : function or np.ndarray or theano.shared
        :math:`W_{hc}`
    b_cell : function or np.ndarray or theano.shared
        :math:`b_c`
    nonlinearity_cell : function or np.ndarray or theano.shared
        :math:`\tanh`
    W_in_to_outgate : function or np.ndarray or theano.shared
        :math:`W_{io}`
    W_hid_to_outgate : function or np.ndarray or theano.shared
        :math:`W_{ho}`
    W_cell_to_outgate : function or np.ndarray or theano.shared
        :math:`W_{co}`
    b_outgate : function or np.ndarray or theano.shared
        :math:`b_o`
    nonlinearity_outgate : function
        :math:`\sigma`
    nonlinearity_out : function or np.ndarray or theano.shared
        :math:`\tanh`
    cell_init : function, np.ndarray, theano.shared or TensorVariable
        :math:`c_0`. Passing in a TensorVariable allows the user to specify
        the value of cell_init. In this mode learn_init is ignored.
    hid_init : function, np.ndarray, theano.shared or TensorVariable
        :math:`h_0`. Passing in a TensorVariable allows the user to specify
        the value of hid_init. In this mode learn_init is ignored.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from x_1 to x_n.
    learn_init : bool
        If True, initial hidden values are learned. If hid_init or cell_init
        are TensorVariables learn_init is ignored.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, W_cell_to_ingate, W_cell_to_forgetgate and
        W_cell_to_outgate are ignored.
    gradient_steps : int
        Number of timesteps to include in backpropagated gradient
        If -1, backpropagate through the entire sequence
    grad_clipping: False or float
        If float the gradient messages are clipped during the backward pass.
        See [1]_ (p. 6) for further explanation.

    References
    ----------
    .. [1] Alex Graves : Generating Sequences With Recurrent Neural
           Networks
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
                 grad_clipping=False):

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

        self.num_batch = self.input_shape[0]
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

        # Stack input to gate weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        self.W_in_to_gates = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden to gate weight matrices
        self.W_hid_to_gates = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack gate biases into a (4*num_units) vector
        self.b_gates = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        # Initialize peephole (cell to gate) connections.  These are
        # elementwise products with the cell state, so they are represented as
        # vectors.
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
                raise ValueError("When a tensor cell_init should be a matrix")
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError("When a tensor hid_init should be a matrix")
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
            then it assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape (n_batch, n_time_steps) where
            `mask[i, j] = 1` when `j <= (length of sequence i)` and
            `mask[i, j] = 0` when `j > (length of sequence i)`.
        """
        # Treat all layers after the first as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Because the input is given for all time steps, we can precompute
        # the inputs to the gates before scanning.
        # input is dimshuffled to (n_time_steps, n_batch, n_features)
        # W_in_to_gates is (n_features, 4*num_units). input_dot_W is then
        # (n_time_steps, n_batch, 4*num_units).
        input_dot_W = T.dot(input, self.W_in_to_gates) + self.b_gates

        # input_dot_W is (n_batch, n_time_steps, 4*num_units). We define a
        # slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_dot_W_n is the nth timestep of the input, dotted with W
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        def step(input_dot_W_n, cell_previous, hid_previous, W_hid_to_gates,
                 *args):

            if self.peepholes:
                [W_cell_to_ingate,
                 W_cell_to_forgetgate,
                 W_cell_to_outgate] = args

            # Calculate gates pre-activations and slice
            gates = input_dot_W_n + T.dot(hid_previous, W_hid_to_gates)

            # clip gradients
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

        def step_masked(input_dot_W_n, mask_n, cell_previous, hid_previous,
                        W_hid_to_gates, *args):

            cell, hid = step(input_dot_W_n, cell_previous, hid_previous,
                             W_hid_to_gates, *args)

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
            sequences = [input_dot_W, mask]
            step_fun = step_masked
        else:
            sequences = input_dot_W
            step_fun = step

        ones = T.ones((self.num_batch, 1))
        if isinstance(self.cell_init, T.TensorVariable):
            cell_init = self.cell_init
        else:
            cell_init = T.dot(ones, self.cell_init)  # repeat num_batch times

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            hid_init = T.dot(ones, self.hid_init)  # repeat num_batch times

        non_seqs = [self.W_hid_to_gates]

        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        cell_out, hid_out = theano.scan(
            step_fun,
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
