:mod:`lasagne.layers`
=====================

.. automodule:: lasagne.layers


Helper functions
----------------

.. autofunction:: get_output
.. autofunction:: get_output_shape
.. autofunction:: get_all_layers
.. autofunction:: get_all_params
.. autofunction:: get_all_bias_params
.. autofunction:: get_all_non_bias_params
.. autofunction:: count_params
.. autofunction:: get_all_param_values
.. autofunction:: set_all_param_values


Layer base classes
------------------

.. autoclass:: Layer
   :members:

.. autoclass:: MergeLayer
    :members:

Layer classes: network input
----------------------------

.. autoclass:: InputLayer
   :members:

Layer classes: dense layers
---------------------------

.. autoclass:: DenseLayer
   :members:

.. autoclass:: NonlinearityLayer
   :members:

.. autoclass:: NINLayer
    :members:

Layer classes: convolutional layers
-----------------------------------

.. autoclass:: Conv1DLayer
    :members:

.. autoclass:: Conv2DLayer
    :members:

Layer classes: pooling layers
-----------------------------

.. autoclass:: MaxPool1DLayer
    :members:

.. autoclass:: MaxPool2DLayer
    :members:

.. autoclass:: GlobalPoolLayer
    :members:

.. autoclass:: FeaturePoolLayer
    :members:

.. autoclass:: FeatureWTALayer
    :members:

Layer classes: noise layers
---------------------------

.. autoclass:: DropoutLayer
    :members:

.. autofunction:: dropout

.. autoclass:: GaussianNoiseLayer
    :members:

Layer classes: shape layers
---------------------------

.. autoclass:: ReshapeLayer
    :members:

.. autofunction:: reshape

.. autoclass:: FlattenLayer
    :members:

.. autofunction:: flatten

.. autoclass:: PadLayer
    :members:

.. autofunction:: pad

Layer classes: merge layers
---------------------------

.. autoclass:: ConcatLayer
    :members:

.. autofunction:: concat

.. autoclass:: ElemwiseSumLayer
    :members:

Layer classes: embedding layers
---------------------------

.. autoclass:: EmbeddingLayer
    :members:

:mod:`lasagne.layers.corrmm`
============================

.. automodule:: lasagne.layers.corrmm
    :members:


:mod:`lasagne.layers.cuda_convnet`
==================================

.. automodule:: lasagne.layers.cuda_convnet
    :members:


:mod:`lasagne.layers.dnn`
=========================

.. automodule:: lasagne.layers.dnn
    :members:

