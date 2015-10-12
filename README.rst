.. image:: https://readthedocs.org/projects/lasagne/badge/
    :target: http://lasagne.readthedocs.org/en/latest/

.. image:: https://travis-ci.org/Lasagne/Lasagne.svg
    :target: https://travis-ci.org/Lasagne/Lasagne

.. image:: https://img.shields.io/coveralls/Lasagne/Lasagne.svg
    :target: https://coveralls.io/r/Lasagne/Lasagne

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

.. image:: https://zenodo.org/badge/16974/Lasagne/Lasagne.svg
   :target: https://zenodo.org/badge/latestdoi/16974/Lasagne/Lasagne

Lasagne
=======

Lasagne is a lightweight library to build and train neural networks in Theano.
Its main features are:

* Supports feed-forward networks such as Convolutional Neural Networks (CNNs),
  recurrent networks including Long Short-Term Memory (LSTM), and any
  combination thereof
* Allows architectures of multiple inputs and multiple outputs, including
  auxiliary classifiers
* Many optimization methods including Nesterov momentum, RMSprop and ADAM
* Freely definable cost function and no need to derive gradients due to
  Theano's symbolic differentiation
* Transparent support of CPUs and GPUs due to Theano's expression compiler

Its design is governed by `six principles
<http://lasagne.readthedocs.org/en/latest/user/development.html#philosophy>`_:

* Simplicity: Be easy to use, easy to understand and easy to extend, to
  facilitate use in research
* Transparency: Do not hide Theano behind abstractions, directly process and
  return Theano expressions or Python / numpy data types
* Modularity: Allow all parts (layers, regularizers, optimizers, ...) to be
  used independently of Lasagne
* Pragmatism: Make common use cases easy, do not overrate uncommon cases
* Restraint: Do not obstruct users with features they decide not to use
* Focus: "Do one thing and do it well"


Installation
------------

In short, you can install a known compatible version of Theano and the latest
Lasagne development version via:

.. code-block:: bash

  pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
  pip install https://github.com/Lasagne/Lasagne/archive/master.zip

For more details and alternatives, please see the `Installation instructions
<http://lasagne.readthedocs.org/en/latest/user/installation.html>`_.


Documentation
-------------

Documentation is available online: http://lasagne.readthedocs.org/

For support, please refer to the `lasagne-users mailing list
<https://groups.google.com/forum/#!forum/lasagne-users>`_.


Example
-------

.. code-block:: python

  import lasagne
  import theano
  import theano.tensor as T

  # create Theano variables for input and target minibatch
  input_var = T.tensor4('X')
  target_var = T.ivector('y')

  # create a small convolutional neural network
  from lasagne.nonlinearities import leaky_rectify, softmax
  network = lasagne.layers.InputLayer((None, 3, 32, 32), input_var)
  network = lasagne.layers.Conv2DLayer(network, 64, (3, 3),
                                       nonlinearity=leaky_rectify)
  network = lasagne.layers.Conv2DLayer(network, 32, (3, 3),
                                       nonlinearity=leaky_rectify)
  network = lasagne.layers.Pool2DLayer(network, (3, 3), stride=2, mode='max')
  network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                      128, nonlinearity=leaky_rectify,
                                      W=lasagne.init.Orthogonal())
  network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                      10, nonlinearity=softmax)

  # create loss function
  prediction = lasagne.layers.get_output(network)
  loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
  loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
          network, lasagne.regularization.l2)

  # create parameter update expressions
  params = lasagne.layers.get_all_params(network, trainable=True)
  updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
                                              momentum=0.9)

  # compile training function that updates parameters and returns training loss
  train_fn = theano.function([input_var, target_var], loss, updates=updates)

  # train network (assuming you've got some training data in numpy arrays)
  for epoch in range(100):
      loss = 0
      for input_batch, target_batch in training_data:
          loss += train_fn(input_batch, target_batch)
      print("Epoch %d: Loss %g" % (epoch + 1, loss / len(training_data)))

  # use trained network for predictions
  test_prediction = lasagne.layers.get_output(network, deterministic=True)
  predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
  print("Predicted class for first test input: %r" % predict_fn(test_data[0]))

For a fully-functional example, see `examples/mnist.py <examples/mnist.py>`_,
and check the `Tutorial
<http://lasagne.readthedocs.org/en/latest/user/tutorial.html>`_ for in-depth
explanations of the same. More examples, code snippets and reproductions of
recent research papers are maintained in the separate `Lasagne Recipes
<https://github.com/Lasagne/Recipes>`_ repository.


Development
-----------

Lasagne is a work in progress, input is welcome.

Please see the `Contribution instructions
<http://lasagne.readthedocs.org/en/latest/user/development.html>`_ for details
on how you can contribute!
