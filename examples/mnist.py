import cPickle as pickle
import gzip

import numpy as np
import nntools
import theano
import theano.tensor as T


NUM_EPOCHS = 500
BATCH_SIZE = 600
NUM_HIDDEN_UNITS = 512
LEARNING_RATE = 0.01
MOMENTUM = 0.9


class Bunch(object):
    def __init__(self, **kwargs):
        vars(self).update(kwargs)


def load_data(filename):
    with gzip.open(filename, 'r') as f:
        data = pickle.load(f)

    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    return Bunch(
        X_train=theano.shared(nntools.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(nntools.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(nntools.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=10,
        )


def build_model(input_dim, output_dim,
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):

    l_in = nntools.layers.InputLayer(
        shape=(batch_size, input_dim),
        )
    l_hidden1 = nntools.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=nntools.nonlinearities.rectify,
        )
    l_hidden1_dropout = nntools.layers.DropoutLayer(
        l_hidden1,
        p=0.5,
        )
    l_hidden2 = nntools.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=num_hidden_units,
        nonlinearity=nntools.nonlinearities.rectify,
        )
    l_hidden2_dropout = nntools.layers.DropoutLayer(
        l_hidden2,
        p=0.5,
        )
    l_out = nntools.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=nntools.nonlinearities.softmax,
        )
    return l_out


def create_iter_functions(dataset, l_out, batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    batch_index = T.iscalar('batch_index')
    X_batch = T.matrix('x')
    y_batch = T.ivector('y')
    batch_slice = slice(
        batch_index * batch_size, (batch_index + 1) * batch_size)

    def loss(output):
        return -T.mean(T.log(output)[T.arange(y_batch.shape[0]), y_batch])

    loss_train = loss(l_out.get_output(X_batch))
    loss_eval = loss(l_out.get_output(X_batch, deterministic=True))

    pred = T.argmax(l_out.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = nntools.layers.get_all_params(l_out)
    updates = nntools.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset.X_train[batch_slice],
            y_batch: dataset.y_train[batch_slice],
            },
        )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset.X_valid[batch_slice],
            y_batch: dataset.y_valid[batch_slice],
            },
        )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset.X_test[batch_slice],
            y_batch: dataset.y_test[batch_slice],
            },
        )

    return Bunch(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        )


def train(iter_funcs, dataset, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    num_batches_train = dataset.num_examples_train // batch_size
    num_batches_valid = dataset.num_examples_valid // batch_size
    num_batches_test = dataset.num_examples_test // batch_size

    for e in range(epochs):
        print "Epoch %d of %d" % (e + 1, epochs)

        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs.train(b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs.valid(b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        print "  training loss:\t\t%.6f" % avg_train_loss
        print "  validation loss:\t\t%.6f" % avg_valid_loss
        print "  validation accuracy:\t\t%.2f %%" % (avg_valid_accuracy * 100) 


def main():
    dataset = load_data('mnist.pkl.gz')
    output_layer = build_model(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        )
    iter_funcs = create_iter_functions(dataset, output_layer)
    train(iter_funcs, dataset)


main()
