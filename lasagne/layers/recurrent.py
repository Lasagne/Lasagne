import theano
import theano.tensor as T
from .. import nonlinearities
from .. import init
from .. import utils

from .base import Layer
from .input import InputLayer
from .dense import DenseLayer
from . import helper


class CustomRecurrentLayer(Layer):
    '''
    A layer which implements a recurrent connection.

    Expects inputs of shape
        (n_batch, n_time_steps, n_features_1, n_features_2, ...)
    '''
    def __init__(self, input_layer, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,
                 learn_init=False, gradient_steps=-1):
        '''
        Create a recurrent layer.

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to the recurrent layer
            - input_to_hidden : nntools.layers.Layer
                Layer which connects input to the hidden state
            - hidden_to_hidden : nntools.layers.Layer
                Layer which connects the previous hidden state to the new state
            - nonlinearity : function or theano.tensor.elemwise.Elemwise
                Nonlinearity to apply when computing new state
            - hid_init : function or np.ndarray or theano.shared
                Initial hidden state
            - backwards : boolean
                If True, process the sequence backwards
            - learn_init : boolean
                If True, initial hidden values are learned
            - gradient_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''
        super(CustomRecurrentLayer, self).__init__(input_layer)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Get the batch size and number of units based on the expected output
        # of the input-to-hidden layer
        (n_batch, self.num_units) = self.input_to_hidden.get_output_shape()

        # Initialize hidden state
        self.hid_init = self.create_param(hid_init, (n_batch, self.num_units))

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = (helper.get_all_params(self.input_to_hidden) +
                  helper.get_all_params(self.hidden_to_hidden))

        if self.learn_init:
            return params + self.get_init_params()
        else:
            return params

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        return [self.hid_init]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return (helper.get_all_bias_params(self.input_to_hidden) +
                helper.get_all_bias_params(self.hidden_to_hidden))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, mask=None, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable
            - mask : theano.TensorType
                Theano variable denoting whether each time step in each
                sequence in the batch is part of the sequence or not.  If None,
                then it assumed that all sequences are of the same length.  If
                not all sequences are of the same length, then it must be
                supplied as a matrix of shape (n_batch, n_time_steps) where
                `mask[i, j] = 1` when `j <= (length of sequence i)` and
                `mask[i, j] = 0` when `j > (length of sequence i)`.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Create single recurrent computation step function
        def step(layer_input, hid_previous):
            return self.nonlinearity(
                self.input_to_hidden.get_output(layer_input) +
                self.hidden_to_hidden.get_output(hid_previous))

        def step_masked(layer_input, mask, hid_previous):
            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            hid = (step(layer_input, hid_previous)*mask
                   + hid_previous*(1 - mask))
            return [hid]

        if self.backwards and mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        output = theano.scan(step_fun, sequences=sequences,
                             go_backwards=self.backwards,
                             outputs_info=[self.hid_init],
                             truncate_gradient=self.gradient_steps)[0]

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        if self.backwards:
            output = output[:, ::-1, :]

        return output


class RecurrentLayer(CustomRecurrentLayer):
    '''
    A "vanilla" RNN layer, which has dense input-to-hidden and
    hidden-to-hidden connections.

    Expects inputs of shape
        (n_batch, n_time_steps, n_features_1, n_features_2, ...)
    '''
    def __init__(self, input_layer, num_units, W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.), backwards=False,
                 learn_init=False, gradient_steps=-1):
        '''
        Create a recurrent layer.

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to the recurrent layer
            - num_units : int
                Number of hidden units in the layer
            - W_in_to_hid : function or np.ndarray or theano.shared
                Initializer for input-to-hidden weight matrix
            - W_hid_to_hid : function or np.ndarray or theano.shared
                Initializer for hidden-to-hidden weight matrix
            - b : function or np.ndarray or theano.shared
                Initializer for bias vector
            - nonlinearity : function or theano.tensor.elemwise.Elemwise
                Nonlinearity to apply when computing new state
            - hid_init : function or np.ndarray or theano.shared
                Initial hidden state
            - backwards : boolean
                If True, process the sequence backwards
            - learn_init : boolean
                If True, initial hidden values are learned
            - gradient_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''

        input_shape = input_layer.get_output_shape()
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
            input_layer, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps)


class LSTMLayer(Layer):
    '''
    A long short-term memory (LSTM) layer.  Includes "peephole connections" and
    forget gate.  Based on the definition in [#graves2014generating]_, which is
    the current common definition. Gate names are taken from [#zaremba2014],
    figure 1.

    :references:
        .. [#graves2014generating] Alex Graves, "Generating Sequences With
            Recurrent Neural Networks"
        .. [#zaremba2014] Wojciech Zaremba et al.,  "Recurrent neural network
           regularization"
    '''
    def __init__(self, input_layer, num_units,
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
                 gradient_steps=-1):
        '''
        Initialize an LSTM layer.  For details on what the parameters mean, see
        (7-11) from [#graves2014generating]_.

        :parameters:
            - input_layer : layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_in_to_ingate : function or np.ndarray or theano.shared
                :math:`W_{xi}`
            - W_hid_to_ingate : function or np.ndarray or theano.shared
                :math:`W_{hi}`
            - W_cell_to_ingate : function or np.ndarray or theano.shared
                :math:`W_{ci}`
            - b_ingate : function or np.ndarray or theano.shared
                :math:`b_i`
            - nonlinearity_ingate : function
                :math:`\sigma`
            - W_in_to_forgetgate : function or np.ndarray or theano.shared
                :math:`W_{xf}`
            - W_hid_to_forgetgate : function or np.ndarray or theano.shared
                :math:`W_{hf}`
            - W_cell_to_forgetgate : function or np.ndarray or theano.shared
                :math:`W_{cf}`
            - b_forgetgate : function or np.ndarray or theano.shared
                :math:`b_f`
            - nonlinearity_forgetgate : function
                :math:`\sigma`
            - W_in_to_cell : function or np.ndarray or theano.shared
                :math:`W_{ic}`
            - W_hid_to_cell : function or np.ndarray or theano.shared
                :math:`W_{hc}`
            - b_cell : function or np.ndarray or theano.shared
                :math:`b_c`
            - nonlinearity_cell : function or np.ndarray or theano.shared
                :math:`\tanh`
            - W_in_to_outgate : function or np.ndarray or theano.shared
                :math:`W_{io}`
            - W_hid_to_outgate : function or np.ndarray or theano.shared
                :math:`W_{ho}`
            - W_cell_to_outgate : function or np.ndarray or theano.shared
                :math:`W_{co}`
            - b_outgate : function or np.ndarray or theano.shared
                :math:`b_o`
            - nonlinearity_outgate : function
                :math:`\sigma`
            - nonlinearity_out : function or np.ndarray or theano.shared
                :math:`\tanh`
            - cell_init : function or np.ndarray or theano.shared
                :math:`c_0`
            - hid_init : function or np.ndarray or theano.shared
                :math:`h_0`
            - backwards : boolean
                If True, process the sequence backwards and then reverse the
                output again such that the output from the layer is always
                from x_1 to x_n.
            - learn_init : boolean
                If True, initial hidden values are learned
            - peepholes : boolean
                If True, the LSTM uses peephole connections.
                When False, W_cell_to_ingate, W_cell_to_forgetgate and
                W_cell_to_outgate are ignored.
            - gradient_steps : int
                Number of timesteps to include in backpropagated gradient
                If -1, backpropagate through the entire sequence
        '''

        # Initialize parent layer
        super(LSTMLayer, self).__init__(input_layer)

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

        # Input dimensionality is the output dimensionality of the input layer
        (num_batch, _, num_inputs) = self.input_layer.get_output_shape()

        # Initialize parameters using the supplied args
        self.W_in_to_ingate = self.create_param(
            W_in_to_ingate, (num_inputs, num_units), name="W_in_to_ingate")

        self.W_hid_to_ingate = self.create_param(
            W_hid_to_ingate, (num_units, num_units), name="W_hid_to_ingate")

        self.b_ingate = self.create_param(
            b_ingate, (num_units), name="b_ingate")

        self.W_in_to_forgetgate = self.create_param(
            W_in_to_forgetgate, (num_inputs, num_units),
            name="W_in_to_forgetgate")

        self.W_hid_to_forgetgate = self.create_param(
            W_hid_to_forgetgate, (num_units, num_units),
            name="W_hid_to_forgetgate")

        self.b_forgetgate = self.create_param(
            b_forgetgate, (num_units,), name="b_forgetgate")

        self.W_in_to_cell = self.create_param(
            W_in_to_cell, (num_inputs, num_units), name="W_in_to_cell")

        self.W_hid_to_cell = self.create_param(
            W_hid_to_cell, (num_units, num_units), name="W_hid_to_cell")

        self.b_cell = self.create_param(
            b_cell, (num_units,), name="b_cell")

        self.W_in_to_outgate = self.create_param(
            W_in_to_outgate, (num_inputs, num_units), name="W_in_to_outgate")

        self.W_hid_to_outgate = self.create_param(
            W_hid_to_outgate, (num_units, num_units), name="W_hid_to_outgate")

        self.b_outgate = self.create_param(
            b_outgate, (num_units,), name="b_outgate")

        # Stack input to gate weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        self.W_in_to_gates = utils.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
            self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden to gate weight matrices
        self.W_hid_to_gates = utils.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
            self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack gate biases into a (4*num_units) vector
        self.b_gates = utils.concatenate(
            [self.b_ingate, self.b_forgetgate,
            self.b_cell, self.b_outgate], axis=0)

        # Initialize peephole (cell to gate) connections.  These are
        # elementwise products with the cell state, so they are represented as
        # vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.create_param(
                W_cell_to_ingate, (num_units), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.create_param(
                W_cell_to_forgetgate, (num_units), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.create_param(
                W_cell_to_outgate, (num_units), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        self.cell_init = self.create_param(
            cell_init, (num_batch, num_units), name="cell_init")
        self.hid_init = self.create_param(
            hid_init, (num_batch, num_units), name="hid_init")

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = self.get_weight_params() + self.get_bias_params()
        if self.peepholes:
            params.extend(self.get_peephole_params())

        if self.learn_init:
            params.extend(self.get_init_params())

        return params

    def get_weight_params(self):
        '''
        Get all weight matrix parameters of this layer

        :returns:
            - weight_params : list of theano.shared
                List of all weight matrix parameters
        '''
        return [self.W_in_to_ingate,
                self.W_hid_to_ingate,
                self.W_in_to_forgetgate,
                self.W_hid_to_forgetgate,
                self.W_in_to_cell,
                self.W_hid_to_cell,
                self.W_in_to_outgate,
                self.W_hid_to_outgate]

    def get_peephole_params(self):
        '''
        Get all peephole connection parameters of this layer.

        :returns:
            - peephole_params : list of theano.shared
                List of all peephole parameters.  If this LSTM layer doesn't
                use peephole connections (peepholes=False), then an empty list
                is returned.
        '''
        if self.peepholes:
            return [self.W_cell_to_ingate,
                    self.W_cell_to_forgetgate,
                    self.W_cell_to_outgate]
        else:
            return []

    def get_init_params(self):
        '''
        Get all initital state parameters of this layer.

        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        return [self.hid_init, self.cell_init]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return [self.b_ingate, self.b_forgetgate,
                self.b_cell, self.b_outgate]

    def get_output_shape_for(self, input_shape):
        '''
        Compute the expected output shape given the input.

        :parameters:
            - input_shape : tuple
                Dimensionality of expected input

        :returns:
            - output_shape : tuple
                Dimensionality of expected outputs given input_shape
        '''
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, mask=None, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable
            - mask : theano.TensorType
                Theano variable denoting whether each time step in each
                sequence in the batch is part of the sequence or not.  If None,
                then it assumed that all sequences are of the same length.  If
                not all sequences are of the same length, then it must be
                supplied as a matrix of shape (n_batch, n_time_steps) where
                `mask[i, j] = 1` when `j <= (length of sequence i)` and
                `mask[i, j] = 0` when `j > (length of sequence i)`.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
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

        # input_dot_w is (n_batch, n_time_steps, 4*num_units). We define a
        # slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_dot_W_n is the n'th timestep of the input, dotted with W
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        def step(input_dot_W_n, cell_previous, hid_previous):

            # Calculate gates pre-activations and slice
            gates = input_dot_W_n + T.dot(hid_previous, self.W_hid_to_gates)
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
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input
            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity_out(cell)
            return [cell, hid]

        def step_masked(input_dot_W_n, mask, cell_previous, hid_previous):

            cell, hid = step(input_dot_W_n, cell_previous, hid_previous)

            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            not_mask = 1 - mask
            cell = cell*mask + cell_previous*not_mask
            hid = hid*mask + hid_previous*not_mask

            return [cell, hid]

        if self.backwards and mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input_dot_W, mask]
            step_fun = step_masked
        else:
            sequences = input_dot_W
            step_fun = step

        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        output = theano.scan(step_fun, sequences=sequences,
                             outputs_info=[self.cell_init, self.hid_init],
                             go_backwards=self.backwards,
                             truncate_gradient=self.gradient_steps)[0][1]

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            output = output[:, ::-1, :]

        return output
