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
    """
    Generate a simple lag sequence

    Parameters
    ----------
    length : int
        Length of sequences to generate
    n_batch : int
        Number of training sequences per batch
    delay : int
        How much to delay one feature dimension in the target

    Returns
    -------
    X : np.ndarray, shape=(n_batch, length, 2)
        Input sequence
    y : np.ndarray, shape=(n_batch, length, 1)
        Target sequence, where
        y[n] = X[:, n, 0] - X[:, n - delay, 1] + noise
    """
    X = np.random.rand(n_batch, length, 2)
    y = X[:, :, 0].reshape((n_batch, length, 1))
    # Compute y[n] = X[:, n, 0] - X[:, n - delay, 1] + noise
    y[:, delay:, 0] -= (X[:, :-delay, 1]
                        + .01*np.random.randn(n_batch, length - delay))
    return X, y


# Generate a "validation" sequence whose cost we will periodically compute
X_val, y_val = gen_data()

# Construct vanilla RNN
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, LENGTH, X_val.shape[-1]))

l_recurrent = lasagne.layers.RecurrentLayer(l_in, N_HIDDEN, nonlinearity=None)
# We need a reshape layer which combines the first (batch size) and second
# (number of timesteps) dimensions, otherwise the DenseLayer will treat the
# number of time steps as a feature dimension
l_reshape = lasagne.layers.ReshapeLayer(l_recurrent,
                                        (N_BATCH*LENGTH, N_HIDDEN))

l_recurrent_out = lasagne.layers.DenseLayer(l_reshape,
                                            num_units=y_val.shape[-1],
                                            nonlinearity=None)
l_out = lasagne.layers.ReshapeLayer(l_recurrent_out,
                                    (N_BATCH, LENGTH, y_val.shape[-1]))

print "Total parameters: {}".format(
    sum([p.get_value().size for p in lasagne.layers.get_all_params(l_out)]))

# Cost function is mean squared error
input = T.tensor3('input')
target_output = T.tensor3('target_output')

# Cost = mean squared error, starting from delay point
cost = T.mean((lasagne.layers.get_output(l_out, input)[:, DELAY:, :]
               - target_output[:, DELAY:, :])**2)
# Use NAG for training
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE)
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output], cost, updates=updates)
y_pred = theano.function([input], lasagne.layers.get_output(l_out, input))
compute_cost = theano.function([input, target_output], cost)

# Train the net
costs = np.zeros(N_ITERATIONS)
for n in range(N_ITERATIONS):
    X, y = gen_data()
    costs[n] = train(X, y)
    if not n % 100:
        cost_val = compute_cost(X_val, y_val)
        print "Iteration {} validation cost = {}".format(n, cost_val)
