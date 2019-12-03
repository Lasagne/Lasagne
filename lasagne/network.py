"""
Classes that help with training and using networks.
"""

from __future__ import print_function

from collections import OrderedDict
from inspect import getargspec
from time import time

import numpy as np

import theano
import theano.tensor as T

from lasagne.layers import InputLayer
from lasagne.layers import get_output, get_output_shape
from lasagne.layers import get_all_params, get_all_layers

__all__ = ['Network']


class Network:
    """
    A Neural Network Model with a single input and a single output.

    This class provides a scikit-learn style interface for using Lasagne-based
    Neural Networks.

    Parameters
    ----------
    layer : lasagne.layers.Layer
        The network to be wrapped. The network must contain only a single input
        layer.
    objective : callable
        Accepts two theano expressions which represent the output of the network
        and target output. Returns a theano expression that represents the value
        to minimized.
    update : callable
        Accepts a theano expression which represent the loss to be minimized and
        a list of parameters. Returns a dictionary of updates.
    target : theano.TensorType or None
        Target output, passed to `objective`. If `None`, `target` defaults to
        the same dtype and shape as the output of the network.
    evaluators : dict of str: callable
        Additional evaluation metrics.
    output_kwargs : dict
        Passed as kwargs to `lasagne.layers.get_output`. If `None`, defaults to
        `{}`
    update_kwargs : dict
        Passed as kwargs to `objective`. If `None`, defaults to `{}`
    n_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Size of batches.
    shuffle : bool
        Whether to shuffle training data.
    verbose : bool
        Enable verbose output.

    Attributes
    ----------
    fit_batch : theano.function
        Fits the network to a single batch.
    evaluate_batch : theano.function
        Evaluates the network on a single batch.
    predict_batch : theano.function
        Predict the ouput of the network on a single batch.
    """

    def __init__(self, layer, objective, update, target=None, evaluators=None,
                 output_kwargs=None, update_kwargs=None,
                 n_epochs=500, batch_size=64, shuffle=True, verbose=False):
        self.layer = layer
        self.objective = objective
        self.update = update
        self.target = target
        self.evaluators = evaluators
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
            raise ValueError('Network expects a layer with only one input')
        input = inputs[0]

        output = get_output(layer, **output_kwargs)
        loss = T.mean(objective(output, target))
        params = get_all_params(layer, trainable=True)
        updates = update(loss, params, **update_kwargs)
        evaluations = [T.mean(evaluator(output, target))
                       for evaluator in evaluators.values()]
        self.fit_batch = theano.function([input, target], evaluations,
                                         updates=updates)

        output_kwargs = dict(output_kwargs)
        output_kwargs['deterministic'] = True
        output = get_output(layer, **output_kwargs)
        self.predict_batch = theano.function([input], output)

        evaluations = [T.mean(evaluator(output, target))
                       for evaluator in evaluators.values()]
        self.evaluate_batch = theano.function([input, target], evaluations)


    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the network.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values for the training data.
        X_val : array-like or None
            Validation data. If `None`, no validation is performed.
        y_val : array-like or None
            Target values for the validation data. If `None`, no validation is
            performed.

        Returns
        -------
        self : object
            Returns self.

        """
        assert (X_val is None) == (y_val is None)

        for epoch in range(self.n_epochs):

            if self.verbose:
                start_time = time()

            train_evaluation = self.partial_fit(X, y)

            if X_val is not None:
                val_evaluation = self.evaluate(X_val, y_val)

            if self.verbose:
                print('Epoch %s of %s took %.3fs'
                      % (epoch + 1, self.n_epochs, time() - start_time))

                for name, value in train_evaluation.items():
                    print('\ttrain %s:\t\t\t%.6f' % (name, value))

                if X_val is not None:
                    for name, value in val_evaluation.items():
                        print('\tvalidation %s:\t\t%.6f' % (name, value))

        return self


    def partial_fit(self, X, y):
        """
        Fit the network to a subset of the training data.

        Parameters
        ----------
        X : array-like
            Subset of training data.
        y : array-like
            Subset of target values.

        Returns
        -------
        self : object
            Returns self.

        """
        evaluations = 0
        batches = 0
        for batch in self._iter_batches(X, y, shuffle=self.shuffle):
            size = len(batch[0])
            evaluations += size * np.array(self.fit_batch(*batch))
            batches += size
        evaluations /= batches
        return OrderedDict(zip(self.evaluators, evaluations))


    def loss(self, X, y):
        """
        Returns the objective loss of the prediction.

        Parameters
        ----------
        X : array-like
            Test samples.
        y : array-like
            True values for X.

        Returns
        -------
        loss : float
            objective loss of self.predict(X) wrt. y.
        """
        return self.evaluate(X, y)['loss']


    def evaluate(self, X, y):
        """
        Returns the evaulations of the prediction.

        Parameters
        ----------
        X : array-like
            Test samples.
        y : array-like
            True values for X.

        Returns
        -------
        evaulations : OrderedDict of str: value
            evaluations of self.predict(X) wrt. y.
        """
        evaluations = 0
        batches = 0
        for batch in self._iter_batches(X, y):
            size = len(batch[0])
            evaluations += size * np.array(self.evaluate_batch(*batch))
            batches += size
        evaluations /= batches
        return OrderedDict(zip(self.evaluators, evaluations))


    def predict(self, X):
        """
        Predict target for X.

        Parameters
        ----------
        X : array-like
            The input samples.

        Returns
        -------
        y : array
            The predicted values.
        """
        prediction = []
        for batch in self._iter_batches(X, shuffle=False):
            prediction.append(self.predict_batch(*batch))
        return np.concatenate(prediction)


    def _iter_batches(self, *args, batch_size=None, shuffle=False):
        """
        Yields tuples of batches.

        Parameters
        ----------
        *args : list of array-like
            The data to be batched. All elements must have the same length
        batch_size : int or None
            The size of the batch. If None, defaults to self.batch_size
        shuffle : bool
            Whether to shuffle the data.

        Returns
        -------
        y : generator of tuples
            Each tuple has length `len(*args)`. Each element of the tuple has
            the same length, which is less than or equal to the batch_size.
        """
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
        param_names = getargspec(self.__init__).args[1:]
        params = [(name, getattr(self, name)) for name in param_names]
        strings = [self.__class__.__name__, '(']
        for name, value in params:
            strings.append('%s=%s' % (name, value))
            strings.append(', ')
        strings.pop()
        strings.append(')')
        return ''.join(strings)
