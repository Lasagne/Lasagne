.. _tutorial:

========
Tutorial
========

The following document will give a high-level overview of how to build a
handwritten digits classifier using the MNIST dataset with Lasagne.

If you want to get detailed information about neural networks and how to build
them using Theano (the library which Lasagne is built on top of), you should
read the `Deeplearning Tutorial`_, possibly along with an online course for
more theoretical background (`Neural Networks and Deep Learning`_ by Michael
Nielsen, `Convolutional Neural Networks for Visual Recognition`_ by
Andrej Karpathy et al.) or a standard text book such as "Machine Learning" by
Tom Mitchell.

Note that a basic understanding of how Theano works is required to be able to
use Lasagne.


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

If everything is set up correctly, you will get the following output:

.. code-block:: text

  Loading data...
  Building model and compiling functions...
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

TODO: flesh out this section with code fragments and descriptions of what they
are used for, once the examples are in their final format (see `GitHub issue
#215`_).

The example starts with loading the MNIST dataset as train-, test- and
validation set in line 206. It continues with automatically building a model
with two hidden layers. It automatically detects the necessary number of input
neurons (= number of features, 28*28=784 pixels in case of MNIST) and the
necessary number of output neurons (= number of classes, 10 in case of MNIST).

Then it automatically trains and evaluates the feed forward neural network
with mini-batch gradient descent.

.. _Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
.. _Deeplearning Tutorial: http://deeplearning.net/tutorial/
.. _Convolutional Neural Networks for Visual Recognition: http://cs231n.github.io/
.. _GitHub issue #215: https://github.com/Lasagne/Lasagne/issues/215
