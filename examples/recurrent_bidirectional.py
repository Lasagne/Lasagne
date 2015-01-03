import numpy as np
import theano
import theano.tensor as T
import lasagne
# Sequence length
LENGTH = 10
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 4
# Number of training sequences in each batch
N_BATCH = 30
# Delay used to generate artificial training data
DELAY = 2
# SGD learning rate
LEARNING_RATE = 1e-5
# Number of iterations to train the net
N_ITERATIONS = 1000


def gen_data(length=LENGTH, n_batch=N_BATCH, delay=DELAY):
    '''
    Generate a simple lag sequence

    :parameters:
        - length : int
            Length of sequences to generate
        - n_batch : int
            Number of training sequences per batch
        - delay : int
            How much to delay one feature dimension in the target

    :returns:
        - X : np.ndarray, shape=(n_batch, length, 2)
            Input sequence
        - y : np.ndarray, shape=(n_batch, length, 1)
            Target sequence, where
            y[n] = X[:, n, 0] - X[:, n - delay, 1] + noise
    '''
    X = np.random.rand(n_batch, length, 2)
    y = X[:, :, 0].reshape((n_batch, length, 1))
    # Compute y[n] = X[:, n, 0] - X[:, n - delay, 1] + noise
    y[:, delay:, 0] -= (X[:, :-delay, 1]
                        + .01*np.random.randn(n_batch, length - delay))
    return X.astype(theano.config.floatX), y.astype(theano.config.floatX)


# Generate a "validation" sequence whose cost we will periodically compute
X_val, y_val = gen_data()
mask_val = np.ones(shape=(N_BATCH, LENGTH), dtype=theano.config.floatX)

# Construct vanilla RNN: One recurrent layer (with input weights) and one
# dense output layer
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, LENGTH, X_val.shape[-1]))

# This input layer is used to tell the input-to-hidden what shape to expect
# As we iterate over time steps, the input will be batch size x feature dim
l_recurrent_in_fwd = lasagne.layers.InputLayer(shape=(N_BATCH, X_val.shape[-1]))
l_input_to_hidden_fwd = lasagne.layers.DenseLayer(l_recurrent_in_fwd, N_HIDDEN,
                                              nonlinearity=None)

# As above, we need to tell the hidden-to-hidden layer what shape to expect
l_recurrent_hid_fwd = lasagne.layers.InputLayer(shape=(N_BATCH, N_HIDDEN))
l_hidden_to_hidden_1_fwd = lasagne.layers.DenseLayer(l_recurrent_hid_fwd,
                                                     N_HIDDEN,
                                                 nonlinearity=None,
                                                 b=lasagne.init.Constant(1.))
l_hidden_to_hidden_2_fwd = lasagne.layers.DenseLayer(l_hidden_to_hidden_1_fwd,
                                                 N_HIDDEN, nonlinearity=None,
                                                 b=lasagne.init.Constant(1.))

l_recurrent_fwd = lasagne.layers.RecurrentLayer(l_in,
                                            l_input_to_hidden_fwd,
                                            l_hidden_to_hidden_2_fwd,
                                            nonlinearity=None,
                                            backwards=False)


l_recurrent_in_bck = lasagne.layers.InputLayer(shape=(N_BATCH, X_val.shape[-1]))
l_input_to_hidden_bck = lasagne.layers.DenseLayer(l_recurrent_in_bck, N_HIDDEN,
                                              nonlinearity=None)

# As above, we need to tell the hidden-to-hidden layer what shape to expect
l_recurrent_hid_bck = lasagne.layers.InputLayer(shape=(N_BATCH, N_HIDDEN))
l_hidden_to_hidden_1_bck = lasagne.layers.DenseLayer(l_recurrent_hid_bck,
                                                     N_HIDDEN,
                                                 nonlinearity=None,
                                                 b=lasagne.init.Constant(1.))
l_hidden_to_hidden_2_bck = lasagne.layers.DenseLayer(l_hidden_to_hidden_1_bck,
                                                 N_HIDDEN, nonlinearity=None,
                                                 b=lasagne.init.Constant(1.))

l_recurrent_bck = lasagne.layers.RecurrentLayer(l_in,
                                            l_input_to_hidden_bck,
                                            l_hidden_to_hidden_2_bck,
                                            nonlinearity=None,
                                            backwards=True)


# concat layers
l_recurrent_fwd_rs = lasagne.layers.ReshapeLayer(l_recurrent_fwd,
                                                 (N_BATCH*LENGTH, N_HIDDEN))
l_recurrent_bck_rs = lasagne.layers.ReshapeLayer(l_recurrent_bck,
                                                 (N_BATCH*LENGTH, N_HIDDEN))



l_concat = lasagne.layers.ConcatLayer([l_recurrent_fwd_rs,
                                       l_recurrent_bck_rs], axis=1)


l_recurrent_out = lasagne.layers.DenseLayer(l_concat,
                                            num_units=y_val.shape[-1],
                                            nonlinearity=None)
l_out = lasagne.layers.ReshapeLayer(l_recurrent_out,
                                    (N_BATCH, LENGTH, y_val.shape[-1]))



print "Total parameters: {}".format(
    sum([p.get_value().size for p in lasagne.layers.get_all_params(l_out)]))

# Cost function is mean squared error
input = T.tensor3('input')
target_output = T.tensor3('target_output')
mask = T.matrix('mask')

# Cost = mean squared error, starting from delay point
cost = T.mean((l_out.get_output(input, mask=mask)[:, DELAY:, :]
               - target_output[:, DELAY:, :])**2)
# Use NAG for training
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output, mask], cost, updates=updates)
y_pred = theano.function([input, mask], l_out.get_output(input, mask=mask))
compute_cost = theano.function([input, target_output, mask], cost)

# Train the net
costs = np.zeros(N_ITERATIONS)
for n in range(N_ITERATIONS):
    X, y = gen_data()
    costs[n] = train(X, y, mask_val)
    if not n % 100:
        cost_val = compute_cost(X_val, y_val, mask_val)
        print "Iteration {} validation cost = {}".format(n, cost_val)
