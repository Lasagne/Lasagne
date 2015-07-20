Convolutional layers
--------------------

.. automodule:: lasagne.layers.conv

.. currentmodule:: lasagne.layers

.. autoclass:: Conv1DLayer
    :members:

.. autoclass:: Conv2DLayer
    :members:

.. note::
    For experts: ``Conv2DLayer`` will create a convolutional layer using
    ``T.nnet.conv2d``, Theano's default convolution. On compilation for GPU,
    Theano replaces this with a `cuDNN`_-based implementation if available,
    otherwise falls back to a gemm-based implementation. For details on this,
    please see the `Theano convolution documentation`_.

    Lasagne also provides convolutional layers directly enforcing a specific
    implementation: :class:`lasagne.layers.dnn.Conv2DDNNLayer` to enforce
    cuDNN, :class:`lasagne.layers.corrmm.Conv2DMMLayer` to enforce the
    gemm-based one, :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` for
    Krizhevsky's `cuda-convnet`_.

.. _cuda-convnet: https://code.google.com/p/cuda-convnet/
.. _cuDNN: https://developer.nvidia.com/cudnn
.. _Theano convolution documentation: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html
