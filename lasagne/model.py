from __future__ import print_function

from collections import OrderedDict
from time import time

import numpy as np

import theano
import theano.tensor as T

from lasagne.layers import InputLayer
from lasagne.layers import get_output, get_output_shape
from lasagne.layers import get_all_params, get_all_layers

__all__ = ['Model']


class Model:

    def __init__(self, layer, objective, update, target=None, evaluators=None,
                 output_kwargs=None, update_kwargs=None,
                 n_epochs=500, batch_size=64, shuffle=True, verbose=0):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.output_kwargs = output_kwargs
        self.update_kwargs = update_kwargs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

        if output_kwargs is None:
            output_kwargs = {}
        if update_kwargs is None:
            update_kwargs = {}

        if evaluators is None:
            evaluators = {}
        loss_dict = OrderedDict({'loss': objective})
        loss_dict.update(evaluators)
        evaluators = loss_dict
        self.evaluators = evaluators

        if target is None:
            output = get_output(layer, **output_kwargs)
            ndim = len(get_output_shape(layer))
            target = T.TensorType(output.dtype, [False] * ndim)()

        inputs = [layer.input_var for layer in get_all_layers(layer)
                      if isinstance(layer, InputLayer)]
        if len(inputs) != 1:
            raise ValueError('get_input expects a layer with only one input')
        input = inputs[0]

        output = get_output(layer, **output_kwargs)
        loss = T.mean(objective(output, target))
        params = get_all_params(layer, trainable=True)
        updates = update(loss, params, **update_kwargs)
        evaluations = [T.mean(evaluator(output, target))
                       for evaluator in evaluators.values()]
        self.fit_fn = theano.function([input, target], evaluations,
                                      updates=updates)

        output_kwargs = dict(output_kwargs)
        output_kwargs['deterministic'] = True
        output = get_output(layer, **output_kwargs)
        self.predict_fn = theano.function([input], output)

        evaluations = [T.mean(evaluator(output, target))
                       for evaluator in evaluators.values()]
        self.evaluate_fn = theano.function([input, target], evaluations)


    def fit(self, X, y, X_val=None, y_val=None):
        assert (X_val is None) == (y_val is None)

        for epoch in range(self.n_epochs):
            start_time = time()
            evaluations = 0
            batches = 0
            for batch in self.iter_batches(X, y):
                size = len(batch[0])
                evaluations += size * np.array(self.fit_fn(*batch))
                batches += size
            evaluations /= batches

            if X_val is not None:
                val_eval = 0
                val_batches = 0
                for batch in self.iter_batches(X_val, y_val):
                    size = len(batch[0])
                    val_eval += size *np.array(self.evaluate_fn(*batch))
                    val_batches += size
                val_eval /= val_batches

            if self.verbose:
                print('Epoch %s of %s took %.3fs'
                      % (epoch + 1, self.n_epochs, time() - start_time))

                for name, value in zip(self.evaluators, evaluations):
                    print('\ttrain %s:\t\t\t%.6f' % (name, value))
                if X_val is not None:
                    for name, value in zip(self.evaluators, val_eval):
                        print('\tvalidation %s:\t\t%.6f' % (name, value))

        return self


    def partial_fit(self, X, y):
        return self.train_fn(X, y)


    def loss(self, X, y):
        return self.evaluate(X, y)['loss']


    def evaluate(self, X, y):
        evaluations = 0
        batches = 0
        for batch in self.iter_batches(X, y):
            size = len(batch[0])
            evaluations += size * np.array(self.evaluate_fn(*batch))
            batches += size
        evaluations /= batches
        return OrderedDict(zip(self.evaluators, evaluations))


    def predict(self, X):
        prediction = []
        for batch in self.iter_batches(X, shuffle=False):
            prediction.append(self.predict_fn(*batch))
        return np.concatenate(prediction)


    def iter_batches(self, *args, batch_size=None, shuffle=None):
        if not all(len(arg) == len(args[0]) for arg in args):
            raise ValueError('All arguments to iter_batches must have the '
                             'same length')

        if batch_size is None:
            batch_size = self.batch_size

        if shuffle is None:
            shuffle = self.shuffle

        if shuffle:
            index = np.random.permutation(len(args[0]))

        for i in range(0, len(args[0]), self.batch_size):
            if shuffle:
                yield [arg[index[i:i+batch_size]] for arg in args]
            else:
                yield [arg[i:i+batch_size] for arg in args]

    def __repr__(self):
        return ''
