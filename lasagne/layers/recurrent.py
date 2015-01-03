import theano
import theano.tensor as T

from .. import nonlinearities
from .. import init

from .base import Layer
from . import helper


class RecurrentLayer(Layer):
    '''
    A layer which implements a recurrent connection.

    Expects inputs of shape
        (n_batch, n_time_steps, n_features_1, n_features_2, ...)
    '''
    def __init__(self, input_layer, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False):
        '''
        Create a recurrent layer.

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to the recurrent layer
            - input_to_hidden : nntools.layers.Layer
                Layer which connects input to thie hidden state
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
        '''
        super(RecurrentLayer, self).__init__(input_layer)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards

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
            - mask  : A theano shared variable of shape (BATCH_SIZE, SEG_LEN).
                      dtype is identical to input.
                      Mask must be given when backwards is true.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        if self.backwards:
            assert mask is not None, ("Mask must be given to get_output_for"
                                      " when backwards is true")

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        if self.backwards:
            mask = mask.dimshuffle(1, 0, 'x')

        # Create single recurrent computation step function
        def step(layer_input, hid_previous):
            return self.nonlinearity(
                self.input_to_hidden.get_output(layer_input) +
                self.hidden_to_hidden.get_output(hid_previous))

        def step_back(layer_input, mask, hid_previous):
            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            hid = (step(layer_input, hid_previous)*mask
                   + hid_previous*(1 - mask))
            return [hid]

        if self.backwards:
            sequences = [input, mask]
            step_fun = step_back
        else:
            sequences = input
            step_fun = step

        output = theano.scan(step_fun, sequences=sequences,
                             go_backwards=self.backwards,
                             outputs_info=[self.hid_init])[0]

        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        if self.backwards:
            output = output[:, ::-1, :]

        return output


class ReshapeLayer(Layer):
    def __init__(self, input_layer, shape):
        super(ReshapeLayer, self).__init__(input_layer)
        self.shape = shape

    def get_output_shape_for(self, input_shape):
        return self.shape

    def get_output_for(self, input, *args, **kwargs):
        return input.reshape(self.shape)


class BidirectionalLayer(Layer):
    '''
    Takes two layers and runs one forward and one backword on input sequences.
    '''
    def __init__(self, input_layer, forward_layer, backward_layer):
        '''
        Initialize the bidirectional layer with a forward and backward layer

        :parameters:
            - input_layer : nntools.layers.Layer
                Input to the bidirectional layer
            - forward_layer : nntools.layers.Layer
                Layer to run the sequence forward on
            - backward_layer : nntools.layers.Layer
                Layer to run the sequence backward on
        '''
        # In order to sum the outputs of the layers, they must have the same
        # number of units
        assert forward_layer.num_units == backward_layer.num_units
        self.input_layer = input_layer
        self.forward_layer = forward_layer
        self.backward_layer = backward_layer

    def get_params(self):
        '''
        Get all parameters of the forward and backward layers.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        return (self.forward_layer.get_params()
                + self.backward_layer.get_params())

    def get_bias_params(self):
        '''
        Get all bias parameters of the forward and backward layers

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return (self.forward_layer.get_bias_params()
                + self.backward_layer.get_bias_params())

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
        # In order to sum the outputs of the layers, they must have the same
        # number of units
        assert self.forward_layer.num_units == self.backward_layer.num_units
        return (input_shape[0], input_shape[1], self.forward_layer.num_units)

    def get_output_for(self, input, *args, **kwargs):
        '''
        Compute the output by running the sequence forwards through
        self.forward_layer and backwards through self.backward_layer

        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        forward_output = self.forward_layer.get_output_for(input)
        backward_output = self.backward_layer.get_output_for(
            input[:, ::-1, :])
        return forward_output + backward_output


class LSTMLayer(Layer):
    '''
    A long short-term memory (LSTM) layer.  Includes "peephole connections" and
    forget gate.  Based on the definition in [#graves2014generating]_, which is
    the current common definition.

    :references:
        .. [#graves2014generating] Alex Graves, "Generating Sequences With
            Recurrent Neural Networks".
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
                 learn_init=False):
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
                If True, process the sequence backwards
            - learn_init : boolean
                If True, initial hidden values are learned
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

        # Input dimensionality is the output dimensionality of the input layer
        (num_batch, _, num_inputs) = self.input_layer.get_output_shape()

        # Initialize parameters using the supplied args
        self.W_in_to_ingate = self.create_param(
            W_in_to_ingate, (num_inputs, num_units))

        self.W_hid_to_ingate = self.create_param(
            W_hid_to_ingate, (num_units, num_units))

        self.W_cell_to_ingate = self.create_param(
            W_cell_to_ingate, (num_units))

        self.b_ingate = self.create_param(b_ingate, (num_units))

        self.W_in_to_forgetgate = self.create_param(
            W_in_to_forgetgate, (num_inputs, num_units))

        self.W_hid_to_forgetgate = self.create_param(
            W_hid_to_forgetgate, (num_units, num_units))

        self.W_cell_to_forgetgate = self.create_param(
            W_cell_to_forgetgate, (num_units))

        self.b_forgetgate = self.create_param(b_forgetgate, (num_units,))

        self.W_in_to_cell = self.create_param(
            W_in_to_cell, (num_inputs, num_units))

        self.W_hid_to_cell = self.create_param(
            W_hid_to_cell, (num_units, num_units))

        self.b_cell = self.create_param(b_cell, (num_units,))

        self.W_in_to_outgate = self.create_param(
            W_in_to_outgate, (num_inputs, num_units))

        self.W_hid_to_outgate = self.create_param(
            W_hid_to_outgate, (num_units, num_units))

        self.W_cell_to_outgate = self.create_param(
            W_cell_to_outgate, (num_units))

        self.b_outgate = self.create_param(b_outgate, (num_units,))

        self.cell_init = self.create_param(cell_init, (num_batch, num_units))
        self.hid_init = self.create_param(hid_init, (num_batch, num_units))

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = self.get_weight_params() + self.get_bias_params()
        if self.learn_init:
            return params + self.get_init_params()
        else:
            return params

    def get_weight_params(self):
        '''
        Get all weights of this layer
        :returns:
            - weight_params : list of theano.shared
                List of all weight parameters
        '''
        return [self.W_in_to_ingate,
                self.W_hid_to_ingate,
                self.W_cell_to_ingate,
                self.W_in_to_forgetgate,
                self.W_hid_to_forgetgate,
                self.W_cell_to_forgetgate,
                self.W_in_to_cell,
                self.W_hid_to_cell,
                self.W_in_to_outgate,
                self.W_hid_to_outgate,
                self.W_cell_to_outgate]

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
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
            - mask  : A theano shared variable of shape (BATCH_SIZE, SEG_LEN).
                      dtype is identical to input.
                      Mask must be given when backwards is true.

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        if self.backwards:
            assert mask is not None, ("Mask must be given to get_output_for"
                                      " when backwards is true")
        # Treat all layers after the first as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        if self.backwards:
            mask = mask.dimshuffle(1, 0, 'x')

        # Create single recurrent computation step function
        def step(layer_input, cell_previous, hid_previous,
                 W_in_to_ingate, W_hid_to_ingate,
                 W_cell_to_ingate, b_ingate, W_in_to_forgetgate,
                 W_hid_to_forgetgate, W_cell_to_forgetgate, b_forgetgate,
                 W_in_to_cell, W_hid_to_cell, b_cell,
                 W_in_to_outgate, W_hid_to_outgate,
                 W_cell_to_outgate, b_outgate):
            # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
            ingate = self.nonlinearity_ingate(
                T.dot(layer_input, W_in_to_ingate) +
                T.dot(hid_previous, W_hid_to_ingate) +
                cell_previous*W_cell_to_ingate +
                b_ingate)
            # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
            forgetgate = self.nonlinearity_forgetgate(
                T.dot(layer_input, W_in_to_forgetgate) +
                T.dot(hid_previous, W_hid_to_forgetgate) +
                cell_previous*W_cell_to_forgetgate +
                b_forgetgate)
            # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
            cell = (forgetgate*cell_previous +
                    ingate*self.nonlinearity_cell(
                        T.dot(layer_input, W_in_to_cell) +
                        T.dot(hid_previous, W_hid_to_cell) +
                        b_cell))
            # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
            outgate = self.nonlinearity_outgate(
                T.dot(layer_input, W_in_to_outgate) +
                T.dot(hid_previous, W_hid_to_outgate) +
                cell*W_cell_to_outgate +
                b_outgate)
            # h_t = o_t \tanh(c_t)
            hid = outgate*self.nonlinearity_out(cell)

            return [cell, hid]

        def step_back(layer_input, mask, cell_previous, hid_previous,
                 W_in_to_ingate, W_hid_to_ingate,
                 W_cell_to_ingate, b_ingate, W_in_to_forgetgate,
                 W_hid_to_forgetgate, W_cell_to_forgetgate, b_forgetgate,
                 W_in_to_cell, W_hid_to_cell, b_cell,
                 W_in_to_outgate, W_hid_to_outgate,
                 W_cell_to_outgate, b_outgate):

            cell, hid = step(layer_input, cell_previous, hid_previous,
                    W_in_to_ingate, W_hid_to_ingate,
                    W_cell_to_ingate, b_ingate, W_in_to_forgetgate,
                    W_hid_to_forgetgate, W_cell_to_forgetgate, b_forgetgate,
                    W_in_to_cell, W_hid_to_cell, b_cell,
                    W_in_to_outgate, W_hid_to_outgate,
                    W_cell_to_outgate, b_outgate)

            # If mask is 0, use previous state until mask = 1 is found.
            # This propagates the layer initial state when moving backwards
            # until the end of the sequence is found.
            not_mask = 1 - mask
            cell = cell*mask + cell_previous*not_mask
            hid = hid*mask + hid_previous*not_mask

            return [cell, hid]

        if self.backwards:
            sequences = [input, mask]
            step_fun = step_back
        else:
            sequences = input
            step_fun = step

        # Scan op iterates over first dimension of input and repeatedly
        # applied the step function
        output = theano.scan(step_fun, sequences=sequences,
                             outputs_info=[self.cell_init, self.hid_init],
                             non_sequences=[self.W_in_to_ingate,
                                            self.W_hid_to_ingate,
                                            self.W_cell_to_ingate,
                                            self.b_ingate,
                                            self.W_in_to_forgetgate,
                                            self.W_hid_to_forgetgate,
                                            self.W_cell_to_forgetgate,
                                            self.b_forgetgate,
                                            self.W_in_to_cell,
                                            self.W_hid_to_cell, self.b_cell,
                                            self.W_in_to_outgate,
                                            self.W_hid_to_outgate,
                                            self.W_cell_to_outgate,
                                            self.b_outgate],
                             go_backwards=self.backwards)[0][1]
        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            output = output[:, ::-1, :]

        return output
