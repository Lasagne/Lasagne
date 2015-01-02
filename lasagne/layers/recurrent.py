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
                 h_init=init.Constant(0.)):
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
            - h_init : function or np.ndarray or theano.shared
                Initial hidden state
        '''
        super(RecurrentLayer, self).__init__(input_layer)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Get the batch size and number of units based on the expected output
        # of the input-to-hidden layer
        (n_batch, self.num_units) = self.input_to_hidden.get_output_shape()

        # Initialize hidden state
        self.h_init = self.create_param(h_init, (n_batch, self.num_units))

    def get_params(self):
        return (helper.get_all_params(self.input_to_hidden) +
                helper.get_all_params(self.hidden_to_hidden) +
                [self.h_init])

    def get_bias_params(self):
        return (helper.get_all_bias_params(self.input_to_hidden) +
                helper.get_all_bias_params(self.hidden_to_hidden))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, *args, **kwargs):
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Create single recurrent computation step function
        def step(layer_input, previous_output):
            return self.nonlinearity(
                self.input_to_hidden.get_output(layer_input) +
                self.hidden_to_hidden.get_output(previous_output))

        output = theano.scan(step, sequences=input,
                             outputs_info=[self.h_init])[0]
        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

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
                 W_input_to_input_gate=init.Normal(0.1),
                 W_hidden_to_input_gate=init.Normal(0.1),
                 W_cell_to_input_gate=init.Normal(0.1),
                 b_input_gate=init.Normal(0.1),
                 nonlinearity_input_gate=nonlinearities.sigmoid,
                 W_input_to_forget_gate=init.Normal(0.1),
                 W_hidden_to_forget_gate=init.Normal(0.1),
                 W_cell_to_forget_gate=init.Normal(0.1),
                 b_forget_gate=init.Normal(0.1),
                 nonlinearity_forget_gate=nonlinearities.sigmoid,
                 W_input_to_cell=init.Normal(0.1),
                 W_hidden_to_cell=init.Normal(0.1),
                 b_cell=init.Normal(0.1),
                 nonlinearity_cell=nonlinearities.tanh,
                 W_input_to_output_gate=init.Normal(0.1),
                 W_hidden_to_output_gate=init.Normal(0.1),
                 W_cell_to_output_gate=init.Normal(0.1),
                 b_output_gate=init.Normal(0.1),
                 nonlinearity_output_gate=nonlinearities.sigmoid,
                 nonlinearity_output=nonlinearities.tanh,
                 c_init=init.Constant(0.),
                 h_init=init.Constant(0.)):
        '''
        Initialize an LSTM layer.  For details on what the parameters mean, see
        (7-11) from [#graves2014generating]_.

        :parameters:
            - input_layer : layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_input_to_input_gate : function or np.ndarray or theano.shared
                :math:`W_{xi}`
            - W_hidden_to_input_gate : function or np.ndarray or theano.shared
                :math:`W_{hi}`
            - W_cell_to_input_gate : function or np.ndarray or theano.shared
                :math:`W_{ci}`
            - b_input_gate : function or np.ndarray or theano.shared
                :math:`b_i`
            - nonlinearity_input_gate : function
                :math:`\sigma`
            - W_input_to_forget_gate : function or np.ndarray or theano.shared
                :math:`W_{xf}`
            - W_hidden_to_forget_gate : function or np.ndarray or theano.shared
                :math:`W_{hf}`
            - W_cell_to_forget_gate : function or np.ndarray or theano.shared
                :math:`W_{cf}`
            - b_forget_gate : function or np.ndarray or theano.shared
                :math:`b_f`
            - nonlinearity_forget_gate : function
                :math:`\sigma`
            - W_input_to_cell : function or np.ndarray or theano.shared
                :math:`W_{ic}`
            - W_hidden_to_cell : function or np.ndarray or theano.shared
                :math:`W_{hc}`
            - b_cell : function or np.ndarray or theano.shared
                :math:`b_c`
            - nonlinearity_cell : function or np.ndarray or theano.shared
                :math:`\tanh`
            - W_input_to_output_gate : function or np.ndarray or theano.shared
                :math:`W_{io}`
            - W_hidden_to_output_gate : function or np.ndarray or theano.shared
                :math:`W_{ho}`
            - W_cell_to_output_gate : function or np.ndarray or theano.shared
                :math:`W_{co}`
            - b_output_gate : function or np.ndarray or theano.shared
                :math:`b_o`
            - nonlinearity_output_gate : function
                :math:`\sigma`
            - nonlinearity_output : function or np.ndarray or theano.shared
                :math:`\tanh`
            - c_init : function or np.ndarray or theano.shared
                :math:`c_0`
            - h_init : function or np.ndarray or theano.shared
                :math:`h_0`
        '''
        # Initialize parent layer
        super(LSTMLayer, self).__init__(input_layer)

        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_input_gate is None:
            self.nonlinearity_input_gate = nonlinearities.identity
        else:
            self.nonlinearity_input_gate = nonlinearity_input_gate

        if nonlinearity_forget_gate is None:
            self.nonlinearity_forget_gate = nonlinearities.identity
        else:
            self.nonlinearity_forget_gate = nonlinearity_forget_gate

        if nonlinearity_cell is None:
            self.nonlinearity_cell = nonlinearities.identity
        else:
            self.nonlinearity_cell = nonlinearity_cell

        if nonlinearity_output_gate is None:
            self.nonlinearity_output_gate = nonlinearities.identity
        else:
            self.nonlinearity_output_gate = nonlinearity_output_gate

        if nonlinearity_output is None:
            self.nonlinearity_output = nonlinearities.identity
        else:
            self.nonlinearity_output = nonlinearity_output

        self.num_units = num_units

        # Input dimensionality is the output dimensionality of the input layer
        (num_batch, _, num_inputs) = self.input_layer.get_output_shape()

        # Initialize parameters using the supplied args
        self.W_input_to_input_gate = self.create_param(
            W_input_to_input_gate, (num_inputs, num_units))

        self.W_hidden_to_input_gate = self.create_param(
            W_hidden_to_input_gate, (num_units, num_units))

        self.W_cell_to_input_gate = self.create_param(
            W_cell_to_input_gate, (num_units))

        self.b_input_gate = self.create_param(b_input_gate, (num_units))

        self.W_input_to_forget_gate = self.create_param(
            W_input_to_forget_gate, (num_inputs, num_units))

        self.W_hidden_to_forget_gate = self.create_param(
            W_hidden_to_forget_gate, (num_units, num_units))

        self.W_cell_to_forget_gate = self.create_param(
            W_cell_to_forget_gate, (num_units))

        self.b_forget_gate = self.create_param(b_forget_gate, (num_units,))

        self.W_input_to_cell = self.create_param(
            W_input_to_cell, (num_inputs, num_units))

        self.W_hidden_to_cell = self.create_param(
            W_hidden_to_cell, (num_units, num_units))

        self.b_cell = self.create_param(b_cell, (num_units,))

        self.W_input_to_output_gate = self.create_param(
            W_input_to_output_gate, (num_inputs, num_units))

        self.W_hidden_to_output_gate = self.create_param(
            W_hidden_to_output_gate, (num_units, num_units))

        self.W_cell_to_output_gate = self.create_param(
            W_cell_to_output_gate, (num_units))

        self.b_output_gate = self.create_param(b_output_gate, (num_units,))

        self.c_init = self.create_param(c_init, (num_batch, num_units))
        self.h_init = self.create_param(h_init, (num_batch, num_units))

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        return [self.W_input_to_input_gate,
                self.W_hidden_to_input_gate,
                self.W_cell_to_input_gate,
                self.b_input_gate,
                self.W_input_to_forget_gate,
                self.W_hidden_to_forget_gate,
                self.W_cell_to_forget_gate,
                self.b_forget_gate,
                self.W_input_to_cell,
                self.W_hidden_to_cell,
                self.b_cell,
                self.W_input_to_output_gate,
                self.W_hidden_to_output_gate,
                self.W_cell_to_output_gate,
                self.b_output_gate]

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        return [self.b_input_gate, self.b_forget_gate,
                self.b_cell, self.b_output_gate]

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

    def get_output_for(self, input, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        # Treat all layers after the first as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)

        # Create single recurrent computation step function
        def step(layer_input, previous_cell, previous_output,
                 W_input_to_input_gate, W_hidden_to_input_gate,
                 W_cell_to_input_gate, b_input_gate, W_input_to_forget_gate,
                 W_hidden_to_forget_gate, W_cell_to_forget_gate, b_forget_gate,
                 W_input_to_cell, W_hidden_to_cell, b_cell,
                 W_input_to_output_gate, W_hidden_to_output_gate,
                 W_cell_to_output_gate, b_output_gate):
            # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
            input_gate = self.nonlinearity_input_gate(
                T.dot(layer_input, W_input_to_input_gate) +
                T.dot(previous_output, W_hidden_to_input_gate) +
                previous_cell*W_cell_to_input_gate +
                b_input_gate)
            # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
            forget_gate = self.nonlinearity_forget_gate(
                T.dot(layer_input, W_input_to_forget_gate) +
                T.dot(previous_output, W_hidden_to_forget_gate) +
                previous_cell*W_cell_to_forget_gate +
                b_forget_gate)
            # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
            cell = (forget_gate*previous_cell +
                    input_gate*self.nonlinearity_cell(
                        T.dot(layer_input, W_input_to_cell) +
                        T.dot(previous_output, W_hidden_to_cell) +
                        b_cell))
            # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
            output_gate = self.nonlinearity_output_gate(
                T.dot(layer_input, W_input_to_output_gate) +
                T.dot(previous_output, W_hidden_to_output_gate) +
                cell*W_cell_to_output_gate +
                b_output_gate)
            # h_t = o_t \tanh(c_t)
            output = output_gate*self.nonlinearity_output(cell)
            return [cell, output]

        # Scan op iterates over first dimension of input and repeatedly
        # applied the step function
        output = theano.scan(step, sequences=input,
                             outputs_info=[self.c_init, self.h_init],
                             non_sequences=[self.W_input_to_input_gate,
                                            self.W_hidden_to_input_gate,
                                            self.W_cell_to_input_gate,
                                            self.b_input_gate,
                                            self.W_input_to_forget_gate,
                                            self.W_hidden_to_forget_gate,
                                            self.W_cell_to_forget_gate,
                                            self.b_forget_gate,
                                            self.W_input_to_cell,
                                            self.W_hidden_to_cell, self.b_cell,
                                            self.W_input_to_output_gate,
                                            self.W_hidden_to_output_gate,
                                            self.W_cell_to_output_gate,
                                            self.b_output_gate])[0][1]
        # Now, dimshuffle back to (n_batch, n_time_steps, n_features))
        output = output.dimshuffle(1, 0, 2)

        return output
