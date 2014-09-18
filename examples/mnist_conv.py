from __future__ import print_function

import cPickle as pickle
import gzip
import nntools
import theano
import theano.tensor as T

from mnist import create_iter_functions
from mnist import train


NUM_EPOCHS = 500
BATCH_SIZE = 600
LEARNING_RATE = 0.01
MOMENTUM = 0.9


def load_data(filename):
    with gzip.open(filename, 'r') as f:
        data = pickle.load(f)

    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, 28, 28))
    X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))

    return dict(
        X_train=theano.shared(nntools.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(nntools.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(nntools.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_width=X_train.shape[2],
        input_height=X_train.shape[3],
        output_dim=10,
        )


def build_model(input_width, input_height, output_dim,
                batch_size=BATCH_SIZE):
    l_in = nntools.layers.InputLayer(
        shape=(BATCH_SIZE, 1, input_width, input_height),
        )

    l_conv1 = nntools.layers.Conv2DLayer(
        l_in,
        num_filters=32,
        filter_size=(5, 5),
        nonlinearity=nntools.nonlinearities.rectify,
        W=nntools.init.Uniform(),
        )
    l_pool1 = nntools.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

    l_conv2 = nntools.layers.Conv2DLayer(
        l_pool1,
        num_filters=32,
        filter_size=(5, 5),
        nonlinearity=nntools.nonlinearities.rectify,
        W=nntools.init.Uniform(),
        )
    l_pool2 = nntools.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

    l_hidden1 = nntools.layers.DenseLayer(
        l_pool2,
        num_units=256,
        nonlinearity=nntools.nonlinearities.rectify,
        W=nntools.init.Uniform(),
        )

    l_hidden1_dropout = nntools.layers.DropoutLayer(l_hidden1, p=0.5)

    # l_hidden2 = nntools.layers.DenseLayer(
    #     l_hidden1_dropout,
    #     num_units=256,
    #     nonlinearity=nntools.nonlinearities.rectify,
    #     )
    # l_hidden2_dropout = nntools.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = nntools.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=output_dim,
        nonlinearity=nntools.nonlinearities.softmax,
        W=nntools.init.Uniform(),
        )

    return l_out


def main(num_epochs=NUM_EPOCHS):
    dataset = load_data('mnist.pkl.gz')

    output_layer = build_model(
        input_width=dataset['input_width'],
        input_height=dataset['input_width'],
        output_dim=dataset['output_dim'],
        )

    iter_funcs = create_iter_functions(
        dataset,
        output_layer,
        X_tensor_type=T.tensor4,
        )

    print("Starting training...")

    for epoch in train(iter_funcs, dataset):
        print("Epoch %d of %d" % (epoch['number'], num_epochs))
        print("  training loss:\t\t%.6f" % epoch['train_loss'])
        print("  validation loss:\t\t%.6f" % epoch['valid_loss'])
        print("  validation accuracy:\t\t%.2f %%" %
              (epoch['valid_accuracy'] * 100))

        if epoch['number'] > num_epochs:
            break


if __name__ == '__main__':
    main()
