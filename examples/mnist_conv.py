import gzip
import cPickle as pickle

import numpy as np 

import theano
import theano.tensor as T 

import nntools


NUM_EPOCHS = 500
BATCH_SIZE = 600
LEARNING_RATE = 0.01
MOMENTUM = 0.9


print "Loading data"

with gzip.open("mnist.pkl.gz", 'r') as f:
    data = pickle.load(f)

X_train, y_train = data[0]
X_valid, y_valid = data[1]
X_test, y_test = data[2]

# reshape for convolutions
X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
X_valid = X_valid.reshape((X_valid.shape[0], 1, 28, 28))
X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))

X_train_shared = theano.shared(nntools.utils.floatX(X_train))
y_train_shared = T.cast(theano.shared(y_train), 'int32')
X_valid_shared = theano.shared(nntools.utils.floatX(X_valid))
y_valid_shared = T.cast(theano.shared(y_valid), 'int32')
X_test_shared = theano.shared(nntools.utils.floatX(X_test))
y_test_shared = T.cast(theano.shared(y_test), 'int32')

num_examples_train = X_train.shape[0]
num_examples_valid = X_valid.shape[0]
num_examples_test = X_test.shape[0]

input_width = X_train.shape[2]
input_height = X_train.shape[3]
output_dim = 10

num_batches_train = num_examples_train // BATCH_SIZE
num_batches_valid = num_examples_valid // BATCH_SIZE
num_batches_test = num_examples_test // BATCH_SIZE


print "Building model"

l_in = nntools.layers.InputLayer(shape=(BATCH_SIZE, 1, input_width, input_height))

l_conv1 = nntools.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5, 5), nonlinearity=nntools.nonlinearities.rectify, W=nntools.init.Uniform())

l_pool1 = nntools.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

l_conv2 = nntools.layers.Conv2DLayer(l_pool1, num_filters=32, filter_size=(5, 5), nonlinearity=nntools.nonlinearities.rectify, W=nntools.init.Uniform())

l_pool2 = nntools.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

l_hidden1 = nntools.layers.DenseLayer(l_pool2, num_units=256, nonlinearity=nntools.nonlinearities.rectify, W=nntools.init.Uniform())

l_hidden1_dropout = nntools.layers.DropoutLayer(l_hidden1, p=0.5)

# l_hidden2 = nntools.layers.DenseLayer(l_hidden1_dropout, num_units=256, nonlinearity=nntools.nonlinearities.rectify)

# l_hidden2_dropout = nntools.layers.DropoutLayer(l_hidden2, p=0.5)

l_out = nntools.layers.DenseLayer(l_hidden1_dropout, num_units=output_dim, nonlinearity=nntools.nonlinearities.softmax, W=nntools.init.Uniform())


batch_index = T.iscalar("batch_index")
X_batch = T.tensor4('x')
y_batch = T.ivector('y')

def loss(output):
    return -T.mean(T.log(output)[T.arange(y_batch.shape[0]), y_batch])

loss_train = loss(l_out.get_output(X_batch))
loss_eval = loss(l_out.get_output(X_batch, deterministic=True))

pred = T.argmax(l_out.get_output(X_batch, deterministic=True), axis=1)
accuracy = T.mean(T.eq(pred, y_batch))

all_params = nntools.layers.get_all_params(l_out)
# updates = nntools.updates.sgd(loss_train, all_params, LEARNING_RATE)
updates = nntools.updates.nesterov_momentum(loss_train, all_params, LEARNING_RATE, MOMENTUM)

batch_slice = slice(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE)


print "Compiling functions"

iter_train = theano.function([batch_index], loss_train, updates=updates, givens={
        X_batch: X_train_shared[batch_slice],
        y_batch: y_train_shared[batch_slice],
    })

iter_valid = theano.function([batch_index], [loss_eval, accuracy], givens={
        X_batch: X_valid_shared[batch_slice],
        y_batch: y_valid_shared[batch_slice],
    })

iter_test = theano.function([batch_index], [loss_eval, accuracy], givens={
        X_batch: X_test_shared[batch_slice],
        y_batch: y_test_shared[batch_slice],
    })


print "Training model"

for e in range(NUM_EPOCHS):
    print "Epoch %d of %d" % (e + 1, NUM_EPOCHS)

    batch_train_losses = []
    for b in range(num_batches_train):
        batch_train_loss = iter_train(b)
        batch_train_losses.append(batch_train_loss)

    avg_train_loss = np.mean(batch_train_losses)

    batch_valid_losses = []
    batch_valid_accuracies = []
    for b in range(num_batches_valid):
        batch_valid_loss, batch_valid_accuracy = iter_valid(b)
        batch_valid_losses.append(batch_valid_loss)
        batch_valid_accuracies.append(batch_valid_accuracy)

    avg_valid_loss = np.mean(batch_valid_losses)
    avg_valid_accuracy = np.mean(batch_valid_accuracies)
    
    print "  training loss:\t\t%.6f" % avg_train_loss
    print "  validation loss:\t\t%.6f" % avg_valid_loss
    print "  validation accuracy:\t\t%.2f %%" % (avg_valid_accuracy * 100) 
