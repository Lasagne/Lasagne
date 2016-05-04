:mod:`lasagne.layers`
=====================

.. automodule:: lasagne.layers

.. toctree::
    :hidden:

    layers/helper
    layers/base
    layers/input
    layers/dense
    layers/conv
    layers/pool
    layers/recurrent
    layers/noise
    layers/shape
    layers/merge
    layers/normalization
    layers/embedding
    layers/special
    layers/corrmm
    layers/cuda_convnet
    layers/dnn
   

.. rubric:: :doc:`layers/helper`

.. autosummary::
    :nosignatures:

    get_output
    get_output_shape
    get_all_layers
    get_all_params
    count_params
    get_all_param_values
    set_all_param_values


.. rubric:: :doc:`layers/base`

.. autosummary::
    :nosignatures:

    Layer
    MergeLayer


.. rubric:: :doc:`layers/input`

.. autosummary::
    :nosignatures:

    InputLayer


.. rubric:: :doc:`layers/dense`

.. autosummary::
    :nosignatures:

    DenseLayer
    NINLayer


.. rubric:: :doc:`layers/conv`

.. autosummary::
    :nosignatures:

    Conv1DLayer
    Conv2DLayer
    TransposedConv2DLayer
    Deconv2DLayer
    DilatedConv2DLayer


.. rubric:: :doc:`layers/pool`

.. autosummary::
    :nosignatures:

    MaxPool1DLayer
    MaxPool2DLayer
    Pool1DLayer
    Pool2DLayer
    Upscale1DLayer
    Upscale2DLayer
    GlobalPoolLayer
    FeaturePoolLayer
    FeatureWTALayer


.. rubric:: :doc:`layers/recurrent`

.. autosummary::
    :nosignatures:

    CustomRecurrentLayer
    RecurrentLayer
    LSTMLayer
    GRULayer
    Gate


.. rubric:: :doc:`layers/noise`

.. autosummary::
    :nosignatures:

    DropoutLayer
    dropout
    GaussianNoiseLayer


.. rubric:: :doc:`layers/shape`

.. autosummary::
    :nosignatures:

    ReshapeLayer
    reshape
    FlattenLayer
    flatten
    DimshuffleLayer
    dimshuffle
    PadLayer
    pad
    SliceLayer


.. rubric:: :doc:`layers/merge`

.. autosummary::
    :nosignatures:

    ConcatLayer
    concat
    ElemwiseMergeLayer
    ElemwiseSumLayer


.. rubric:: :doc:`layers/normalization`

.. autosummary::
    :nosignatures:

    LocalResponseNormalization2DLayer
    BatchNormLayer
    batch_norm


.. rubric:: :doc:`layers/embedding`

.. autosummary::
    :nosignatures:

    EmbeddingLayer


.. rubric:: :doc:`layers/special`

.. autosummary::
    :nosignatures:

    NonlinearityLayer
    BiasLayer
    ExpressionLayer
    InverseLayer
    TransformerLayer
    ParametricRectifierLayer
    prelu
    RandomizedRectifierLayer
    rrelu


.. rubric:: :doc:`layers/corrmm`

.. autosummary::
    :nosignatures:

    corrmm.Conv2DMMLayer


.. rubric:: :doc:`layers/cuda_convnet`

.. autosummary::
    :nosignatures:

    cuda_convnet.Conv2DCCLayer
    cuda_convnet.MaxPool2DCCLayer
    cuda_convnet.ShuffleBC01ToC01BLayer
    cuda_convnet.bc01_to_c01b
    cuda_convnet.ShuffleC01BToBC01Layer
    cuda_convnet.c01b_to_bc01
    cuda_convnet.NINLayer_c01b


.. rubric:: :doc:`layers/dnn`

.. autosummary::
    :nosignatures:

    dnn.Conv2DDNNLayer
    dnn.Conv3DDNNLayer
    dnn.MaxPool2DDNNLayer
    dnn.Pool2DDNNLayer
    dnn.MaxPool3DDNNLayer
    dnn.Pool3DDNNLayer
    dnn.SpatialPyramidPoolingDNNLayer

