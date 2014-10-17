import numpy as np
import theano
import theano.tensor as T
import nntools


def gen_data(length):
    '''
    Generate a simple lag sequence

    :parameters:
        - length : int
            Length of sequences to generate

    :returns:
        - X : np.ndarray, shape=(length, 2)
            Input sequence
        - y : np.ndarray, shape=(length, 1)
            Target sequence, where y[n] = X[n - 1] - *X[n - 2] + noise
    '''
    X = np.random.rand(length, 2)
    y = np.zeros((length, 1))
    # Compute y[n] = X[n - 1] - *X[n - 2] + noise
    delayed_X = X[1:-1, 0] - X[:-2, 1] + .01*np.random.randn(length - 2)
    # Store as target
    y[2:, :] = delayed_X.reshape(-1, 1)
    return X, y

# Sequence length
length = 10
# Number of units in the hidden (recurrent) layer
n_hidden = 4

# Generate a "validation" sequence whose cost we will periodically compute
X_val, y_val = gen_data(length)

# Construct vanilla RNN: One recurrent layer (with input weights) and one
# dense output layer
l_in = nntools.layers.InputLayer(shape=(1, X_val.shape[1]))
l_input_to_hidden = nntools.layers.DenseLayer(l_in, n_hidden,
                                              nonlinearity=None)
l_hidden_to_hidden = nntools.layers.DenseLayer(l_input_to_hidden, n_hidden,
                                               nonlinearity=None,
                                               b=nntools.init.Constant(1.))
l_recurrent = nntools.layers.RecurrentLayer(l_in, l_input_to_hidden,
                                            l_hidden_to_hidden,
                                            nonlinearity=None)
l_out = nntools.layers.DenseLayer(l_recurrent, num_units=y_val.shape[1],
                                  nonlinearity=None)

# Cost function is mean squared error
input = T.matrix('input')
target_output = T.matrix('target_output')
cost = T.mean((l_out.get_output(input)[2:] - target_output[2:])**2)
# Use SGD for training
learning_rate = .1
all_params = nntools.layers.get_all_params(l_out)
updates = nntools.updates.nesterov_momentum(cost, all_params, learning_rate)
# Theano functions for training, getting output, and computing cost
train = theano.function([input, target_output], cost, updates=updates)
y_pred = theano.function([input], l_out.get_output(input))
compute_cost = theano.function([input, target_output], cost)

# Train the net
n_iterations = 1000
costs = np.zeros(n_iterations)
for n in range(n_iterations):
    X, y = gen_data(length)
    costs[n] = train(X, y)
    if not n % 100:
        cost_val = compute_cost(X_val, y_val)
        print "Iteration {} validation cost = {}".format(n, cost_val)

import matplotlib.pyplot as plt
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
