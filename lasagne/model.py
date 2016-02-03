from __future__ import print_function

import time

import numpy as np

import theano
import theano.tensor as T

from lasagne.layers import InputLayer
from lasagne.layers import get_output, get_output_shape
from lasagne.layers import get_all_params, get_all_layers

__all__ = ['Model']

class Model:

    def __init__(self, layer, objective, update, n_epochs=500, batch_size=64,
                 update_kwargs=None, output_kwargs=None, output_one_hot=False,
                 show_accuracy=False, shuffle=True, validation_split=0,
                 verbose=0):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.update_kwargs = update_kwargs
        self.output_kwargs = output_kwargs
        self.output_one_hot = output_one_hot
        self.show_accuracy = show_accuracy
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.verbose = verbose

        inputs = [layer.input_var for layer in get_all_layers(self.layer)
                  if isinstance(layer, InputLayer)]
        if len(inputs) != 1:
            raise ValueError('%s only supports networks with only a single'
                             'input layer' % self.__class__)
        input_var = inputs[0]

        if output_kwargs is None:
            output_kwargs = {}

        train_output = get_output(self.layer, input_var, **output_kwargs)
        test_output = get_output(self.layer, input_var, deterministic=True,
                                 **output_kwargs)

        output_ndim = len(get_output_shape(self.layer))
        if output_one_hot:
            output_var = T.TensorType('int32', [False] * (output_ndim - 1))()
        else:
            output_var = T.TensorType(train_output.dtype,
                                      [False] * output_ndim)()

        train_loss = T.mean(objective(train_output, output_var))
        test_loss = T.mean(objective(test_output, output_var))

        if update_kwargs is None:
            update_kwargs = {}
        params = get_all_params(self.layer, trainable=True)
        updates = update(train_loss, params, **update_kwargs)

        self.train_fn = theano.function([input_var, output_var], [train_loss],
                                        updates=updates)
        self.test_fn = theano.function([input_var, output_var], [test_loss])
        self.predict_fn = theano.function([input_var], test_output)


    def fit(self, X, y, X_val=None, y_val=None):
        assert (X_val is None) == (y_val is None)
        if X_val is None and self.validation_split:
            split = len(X) * (1 - self.validation_split)
            X_val, y_val = X[split:], y[split:]
            X, y = X[:split], y[:split]

        for epoch in range(self.n_epochs):
            start_time = time.time()
            err = 0
            batches = 0
            for batch in self.iter_batches(X, y):
                err += len(batch[0]) * self.train_fn(*batch)
                batches += len(batch[0])

            if X_val is not None:
                val_err = 0
                val_batches = 0
                for batch in self.iter_batches(X_val, y_val):
                    val_err += len(batch[0]) * self.test_fn(*batch)
                    val_batches += len(batch[0])

            if self.verbose:
                print('Epoch %s of %s took %.3fs'
                      % (epoch + 1, self.n_epochs, time.time() - start_time))
                print('\ttrain loss:\t\t%.6f' % (err / batches))
                if X_val is not None:
                    print('\tvalidation loss:\t%.6f' % (val_err / val_batches))


    def partial_fit(self, X, y):
        return self.train_fn(X, y)


    def score(self, X, y):
        err = 0
        batches = 0
        for batch in self.iter_batches(X, y):
            err += len(batch[0]) * self.test_fn(*batch)
            batches += len(batch[0])
        return err / batches


    def predict(self, X):
        prediction = []
        for batch in self.iter_batches(X, shuffle=False):
            prediction.append(self.predict_fn(*batch))
        return np.concatenate(prediction)


    def iter_batches(self, *args, shuffle=None):
        if not all(len(arg) == len(args[0]) for arg in args):
            raise ValueError('All arguments to iter_batches must have the '
                             'same length')
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            index = np.random.permutation(len(args[0]))
        for i in range(0, len(args[0]), self.batch_size):
            if shuffle:
                yield [arg[index[i:i+self.batch_size]] for arg in args]
            else:
                yield [arg[i:i+self.batch_size] for arg in args]
