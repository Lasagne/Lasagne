import numpy as np
import theano
import theano.tensor as T
import lasagne
theano.config.compute_test_value = 'raise'
# Sequence length
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

N_FEATURES = X_val.shape[-1]
N_OUTPUT = y_val.shape[-1]
assert X_val.shape == (N_BATCH, LENGTH, N_FEATURES)
assert y_val.shape == (N_BATCH, LENGTH, N_OUTPUT)
# mask
mask_val = np.ones(shape=(N_BATCH, LENGTH), dtype=theano.config.floatX)

# Construct LSTM RNN: One LSTM layer and one dense output layer
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, LENGTH, N_FEATURES))


# setup fwd and bck LSTM layer.
l_fwd = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, backwards=False, learn_init=True, peepholes=True)
l_bck = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, backwards=True, learn_init=True, peepholes=True)

# concatenate forward and backward LSTM layers
l_fwd_reshape = lasagne.layers.ReshapeLayer(l_fwd, (N_BATCH*LENGTH, N_HIDDEN))
l_bck_reshape = lasagne.layers.ReshapeLayer(l_bck, (N_BATCH*LENGTH, N_HIDDEN))
l_concat = lasagne.layers.ConcatLayer([l_fwd_reshape, l_bck_reshape], axis=1)


l_recurrent_out = lasagne.layers.DenseLayer(
    l_concat, num_units=N_OUTPUT, nonlinearity=None)
l_out = lasagne.layers.ReshapeLayer(
    l_recurrent_out, (N_BATCH, LENGTH, N_OUTPUT))

input = T.tensor3('input')
target_output = T.tensor3('target_output')
mask = T.matrix('mask')

# add test values
input.tag.test_value = np.random.rand(
    *X_val.shape).astype(theano.config.floatX)
target_output.tag.test_value = np.random.rand(
    *y_val.shape).astype(theano.config.floatX)
mask.tag.test_value = np.random.rand(
    *mask_val.shape).astype(theano.config.floatX)

# Cost = mean squared error, starting from delay point
cost = T.mean((l_out.get_output(input, mask=mask)[:, DELAY:, :]
               - target_output[:, DELAY:, :])**2)
# Use NAG for training
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output, mask],
                        cost, updates=updates, on_unused_input='warn')
y_pred = theano.function(
    [input, mask], l_out.get_output(input, mask=mask), on_unused_input='warn')
compute_cost = theano.function(
    [input, target_output, mask], cost, on_unused_input='warn')

# Train the net
costs = np.zeros(N_ITERATIONS)
for n in range(N_ITERATIONS):
    X, y = gen_data()

    # you should use your own training data mask instead of mask_val
    costs[n] = train(X, y, mask_val)
    if not n % 100:
        cost_val = compute_cost(X_val, y_val, mask_val)
        print "Iteration {} validation cost = {}".format(n, cost_val)

import matplotlib.pyplot as plt
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
