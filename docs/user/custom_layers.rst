Creating custom layers
======================

A simple layer
--------------

To implement a custom layer in Lasagne, you will have to write a Python class
that subclasses :class:`Layer` and implement at least one method:
`get_output_for()`. This method computes the output of the layer given its
input. Note that both the output and the input are Theano expressions, so they
are symbolic.

The following is an example implementation of a layer that multiplies its input
by 2:

.. code:: python

    class DoubleLayer(lasagne.layers.Layer):
        def get_output_for(self, input, **kwargs):
            return 2 * input

This is all that's required to implement a functioning custom layer class in
Lasagne.

A layer that changes the shape
------------------------------

If the layer does not change the shape of the data (for example because it
applies an elementwise operation), then implementing only this one method is
sufficient. Lasagne will assume that the output of the layer has the same shape
as its input.

However, if the operation performed by the layer changes the shape of the data,
you also need to implement `get_output_shape_for()`. This method computes the
shape of the layer output given the shape of its input. Note that this shape
computation should result in a tuple of integers, so it is *not* symbolic.

This method exists because Lasagne needs a way to propagate shape information
when a network is defined, so it can determine what sizes the parameter tensors
should be, for example. This mechanism allows each layer to obtain the size of
its input from the previous layer, which means you don't have to specify the
input size manually. This also prevents errors stemming from inconsistencies
between the layers' expected and actual shapes.

We can implement a layer that computes the sum across the trailing axis of its
input as follows:

.. code:: python

    class SumLayer(lasagne.layers.Layer):
        def get_output_for(self, input, **kwargs):
            return input.sum(axis=-1)

        def get_output_shape_for(self, input_shape):
            return input_shape[:-1]


It is important that the shape computation is correct, as this shape
information may be used to initialize other layers in the network.

A layer with trainable parameters
---------------------------------

If the layer has trainable parameters, these should be initialized in the
constructor using the :meth:`lasagne.layers.Layer.create_param()` method. When
overriding the constructor, it is also important to call the base class
constructor as the first statement, passing ``kwargs`` as well.

A layer should declare its trainable parameters by implementing a
`get_params()` method, which returns a list of Theano shared variables
representing the trainable parameters.

TODO: flesh out this section, update it once the API is frozen.

TODO: replace create_param with add_param once #228 is merged, and remove 
any references to get_params

TODO: layer class creation tutorial?