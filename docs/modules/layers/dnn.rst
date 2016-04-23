:mod:`lasagne.layers.dnn`
-------------------------

This module houses layers that require `cuDNN <https://developer.nvidia.com/cudnn>`_ to work. Its layers are not automatically imported into the :mod:`lasagne.layers` namespace: To use these layers, you need to ``import lasagne.layers.dnn`` explicitly.

Note that these layers are not required to use cuDNN: If cuDNN is available, Theano will use it for the default convolution and pooling layers anyway.
However, they allow you to enforce the usage of cuDNN or use features not available in :mod:`lasagne.layers`.

.. automodule:: lasagne.layers.dnn
    :members:

