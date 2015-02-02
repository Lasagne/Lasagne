import numpy as np
import theano
import theano.tensor as T
import lasagne
theano.config.compute_test_value = 'raise'
# Sequence length (number of time steps)
LENGTH = 10
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 4
# Number of training sequences in each batch
N_BATCH = 30
# Delay used to generate artificial training data
DELAY = 2
# SGD learning rate
LEARNING_RATE = 1e-1
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
    X = .5*np.random.rand(n_batch, length, 2)
    y = X[:, :, 0].reshape((n_batch, length, 1))
    # Compute y[n] = X[:, n, 0] - X[:, n - delay, 1] + noise
    y[:, delay:, 0] += (X[:, :-delay, 1]
                        + .01*np.random.randn(n_batch, length - delay))
    return X.astype(theano.config.floatX), y.astype(theano.config.floatX)


# Generate a "validation" sequence whose cost we will periodically compute
X_val, y_val = gen_data()

n_features = X_val.shape[-1]
n_output = y_val.shape[-1]
assert X_val.shape == (N_BATCH, LENGTH, n_features)
assert y_val.shape == (N_BATCH, LENGTH, n_output)

# Construct LSTM RNN: One LSTM layer and one dense output layer.
# Shape is (number of examples per batch,
#           maximum number of time steps per example,
#           number of features per example)
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, LENGTH, n_features))

# setup forward and backwards LSTM layers.
# Note that LSTMLayer takes a backwards flag. The backwards flag tells scan
# to go backwards before it returns the output from backwards layers.
# It is reversed again such that the output from the layer is always
# from x_1 to x_n.
l_fwd = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, backwards=False, learn_init=True, peepholes=True)
l_bck = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, backwards=True, learn_init=True, peepholes=True)

# concatenate forward and backward LSTM layers
l_fwd_reshape = lasagne.layers.ReshapeLayer(l_fwd, (N_BATCH*LENGTH, N_HIDDEN))
l_bck_reshape = lasagne.layers.ReshapeLayer(l_bck, (N_BATCH*LENGTH, N_HIDDEN))
l_concat = lasagne.layers.ConcatLayer([l_fwd_reshape, l_bck_reshape], axis=1)

# The ReshapeLayers are there because LSTMLayer expects a shape of
# (n_batch, n_time_steps, n_features) but the DenseLayer will flatten
# that shape to (n_batch, n_time_steps*n_features) by default which is wrong.
# So, you need to manually reshape before and after using a DenseLayer.
# Dimshuffling is done inside the LSTMLayer.  You need to dimshuffle
# because Theano's scan function iterates over the first dimension,
# and if the shape is (n_batch, n_time_steps, n_features) then you
# need to dimshuffle(1, 0, 2) in order to iterate over time steps.

l_recurrent_out = lasagne.layers.DenseLayer(
    l_concat, num_units=n_output, nonlinearity=None)
l_out = lasagne.layers.ReshapeLayer(
    l_recurrent_out, (N_BATCH, LENGTH, n_output))

input = T.tensor3('input')
target_output = T.tensor3('target_output')

# add test values
input.tag.test_value = np.random.rand(
    *X_val.shape).astype(theano.config.floatX)
target_output.tag.test_value = np.random.rand(
    *y_val.shape).astype(theano.config.floatX)

# Cost = mean squared error, starting from delay point
cost = T.mean((l_out.get_output(input)[:, DELAY:, :]
               - target_output[:, DELAY:, :])**2)
# Use NAG for training
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output],
                        cost, updates=updates, on_unused_input='warn',
                        allow_input_downcast=True)
y_pred = theano.function(
    [input], l_out.get_output(input), on_unused_input='warn',
    allow_input_downcast=True)

compute_cost = theano.function(
    [input, target_output], cost, on_unused_input='warn',
    allow_input_downcast=True)

# Train the net
costs = np.zeros(N_ITERATIONS)
for n in range(N_ITERATIONS):
    X, y = gen_data()

    # you should use your own training data mask instead of mask_val
    costs[n] = train(X, y)
    if not n % 100:
        cost_val = compute_cost(X_val, y_val)
        print "Iteration {} validation cost = {}".format(n, cost_val)

import matplotlib.pyplot as plt
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
