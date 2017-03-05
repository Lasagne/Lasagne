# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import lasagne

__all__ = [
        'BatchNormLayer',
        'copy_batch_norm',
        'join_layer'
        ]


class BatchNormLayer(lasagne.layers.BatchNormLayer):
    def __init__(self, *args, **kwargs):
        self.init_kargs = kwargs
        self.init_args = args
        super(BatchNormLayer,  self).__init__(*args, **kwargs)

    def copy(self):
        return BatchNormLayer(*self.init_args, **self.init_kargs)


class CopyLayer(lasagne.layers.Layer):
    def __init__(self,  layer):
        self.layer = layer
        if hasattr(layer,  'input_layers'):
            self.input_layers = layer.input_layers
        if hasattr(layer,  'input_layer'):
            self.input_layer = layer.input_layer
        if hasattr(layer,  'input_shapes'):
            self.input_shapes = layer.input_shapes
        if hasattr(layer,  'input_shape'):
            self.input_shape = layer.input_shape
        self.name = layer.name
        self.params = layer.params
        self.get_output_kwargs = layer.get_output_kwargs

    def get_output_for(self,  *args,  **kwargs):
        return self.layer.get_output_for(*args, **kwargs)

    def get_output_shape_for(self,  *args,  **kwargs):
        return self.layer.get_output_shape_for(*args, **kwargs)

    @lasagne.layers.Layer.output_shape.getter
    def output_shape(self):
        return self.layer.output_shape


class CopyMergeLayer(CopyLayer, lasagne.layers.MergeLayer):
    pass


def copylayer(layer):
    assert layer is not CopyLayer
    if isinstance(layer, lasagne.layers.MergeLayer):
        return CopyMergeLayer(layer)
    else:
        return CopyLayer(layer)


def reallayer(p):
    while isinstance(p, CopyLayer):
        p = p.layer
    return p


def copy_batch_norm(layer):
    real = reallayer(layer)

    if isinstance(real, BatchNormLayer):
        real = real.copy()
    elif isinstance(real, lasagne.layers.BatchNormLayer):
        raise Exception('Can not handle lasagne.layers.BatchNormLayer, '
                        'you should use lasagne.layers.join.BatchNormLayer '
                        'instead.')
    if isinstance(real, lasagne.layers.InputLayer):
        return real
    else:
        res = layer
        if hasattr(layer,  'input_layer'):
            tmp = copy_batch_norm(layer.input_layer)
            if tmp != layer.input_layer:
                res = copylayer(real)
                res.input_layer = tmp
                assert res.input_layer == tmp
        if hasattr(layer,  'input_layers'):
            tmp = map(lambda x: copy_batch_norm(x), layer.input_layers)
            for t, r in zip(tmp, layer.input_layers):
                if t != r:
                    res = copylayer(real)
                    res.input_layers = tmp
                    assert res.input_layers == tmp
                    break
        return res


def join_layer(layer, m):
    """
    Substitute sub layers of `layer`.

    Parameters
    ----------

    layer : Layer object
        the original network

    m : dict from Layer to Layer
        key of m will be substituted by corresponding value

    Return
    ------

    New network.

    """

    real = reallayer(layer)

    if layer in m:
        return m[layer]
    elif isinstance(real, lasagne.layers.InputLayer):
        return real
    else:
        res = layer
        # res = copylayer(real)
        if hasattr(layer,  'input_layer'):
            tmp = join_layer(layer.input_layer, m)
            if tmp != layer.input_layer:
                res = copylayer(real)
                res.input_layer = tmp
                assert res.input_layer == tmp
        if hasattr(layer,  'input_layers'):
            tmp = map(lambda x: join_layer(x, m), layer.input_layers)
            for t, r in zip(tmp, layer.input_layers):
                if t != r:
                    res = copylayer(real)
                    res.input_layers = tmp
                    assert res.input_layers == tmp
                    break
        return res
