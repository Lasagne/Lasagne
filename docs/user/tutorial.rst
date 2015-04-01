.. _tutorial:

========
Tutorial
========

Run the MNIST example
=====================

In this first part of the tutorial, we will run the MNIST example that's
included in the source distribution of Lasagne.

It is assumed that you have already run through the :ref:`installation`.  If
you haven't done so already, get a copy of the source tree of Lasagne.  It
includes a number of examples in the ``examples`` folder.  Navigate to the
folder and list its contents:

.. code-block:: bash

  $ cd examples
  $ ls

Now run the ``mnist.py`` example:

.. code-block:: bash

  python mnist.py

You will get the following output:

.. code-block::text

  Loading data...
  Building model and compiling functions...
/usr/local/lib/python2.7/dist-packages/Lasagne-0.1dev-py2.7.egg/lasagne/layers/helper.py:52: UserWarning: get_all_layers() has been changed to return layers in topological order. The former implementation is still available as get_all_layers_old(), but will be removed before the first release of Lasagne. To ignore this warning, use `  warnings.filterwarnings('ignore', '.*topo.*')`.
    warnings.warn("get_all_layers() has been changed to return layers in "
  Starting training...
  Epoch 1 of 500 took 58.438s
    training loss:        1.360644
    validation loss:      0.467446
    validation accuracy:  87.55 %%
  Epoch 2 of 500 took 58.442s
    training loss:        0.597908
    validation loss:      0.330508
    validation accuracy:  90.62 %%
  Epoch 3 of 500 took 58.893s
    training loss:        0.467016
    validation loss:      0.278081
    validation accuracy:  91.92 %%
  Epoch 4 of 500 took 58.037s
    training loss:        0.406298
    validation loss:      0.248938
    validation accuracy:  92.76 %%





Understand the MNIST example
============================

The example starts with loading the MNIST dataset as train-, test- and
validation set in line 206. It continues with automatically building a model
with two hidden layers. It automatically detects the necessary number of input
neurons (= number of features, 28*28=784 pixels in case of MNIST) and the
necessary number of output neurons (= number of classes, 10 in case of MNIST).

Then it automatically trains and evaluates the feed forward neural network
with mini-batch gradient descent.

A detailed explanation of how the training of feed forward neural networks
works can be found in "Machine Learning" by  Tom Mitchell.