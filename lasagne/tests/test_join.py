import numpy as np
import os
# os.environ['THEANO_FLAGS'] = "device = cpu"
# import fcntl
# lck = open('theano.lck', 'w+')
# fcntl.flock(lck, fcntl.LOCK_EX)
# print 'start theano ...'
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.join import join_layer as JoinLayer
import pytest


def test_join_substitue_input():

    var1 = T.imatrix('var1')
    var2 = T.imatrix('var2')

    input = lasagne.layers.InputLayer(shape=(1, 1), input_var=var1)
    input2 = lasagne.layers.InputLayer(shape=(1, 1), input_var=var2)
    network = input
    network = lasagne.layers.NonlinearityLayer(network, T.invert)
    network = JoinLayer(network, {input: input2})

    out = lasagne.layers.get_output(network)
    fun = theano.function([var2], out)
    res = fun(np.zeros((1, 1), dtype='int8'))
    assert res[0, 0] == -1


def test_join_substitue_nest():

    var1 = T.imatrix('var1')
    var2 = T.imatrix('var2')
    var3 = T.imatrix('var3')

    input1 = lasagne.layers.InputLayer(shape=(1, 1), input_var=var1)
    input2 = lasagne.layers.InputLayer(shape=(1, 1), input_var=var2)
    input3 = lasagne.layers.InputLayer(shape=(1, 1), input_var=var3)
    network = input1
    network = lasagne.layers.NonlinearityLayer(network, T.invert)
    network = JoinLayer(network, {input1: input2})
    network = JoinLayer(network, {input2: input3})

    out = lasagne.layers.get_output(network)
    fun = theano.function([var3], out)
    res = fun(np.zeros((1, 1), dtype='int8'))
    assert res[0, 0] == -1


def test_join_substitue_both():

    var1 = T.imatrix('var1')
    var2 = T.imatrix('var2')
    var3 = T.imatrix('var3')

    input1 = lasagne.layers.InputLayer(shape=(1, 1), input_var=var1)
    input2 = lasagne.layers.InputLayer(shape=(1, 1), input_var=var2)
    input3 = lasagne.layers.InputLayer(shape=(1, 1), input_var=var3)
    network = lasagne.layers.ElemwiseMergeLayer((input1, input2), T.add)
    network = JoinLayer(network, {input1: input2})
    network = JoinLayer(network, {input2: input3})

    out = lasagne.layers.get_output(network)
    fun = theano.function([var3], out)
    res = fun(np.ones((1, 1), dtype='int8'))
    assert res[0, 0] == 2

# test_join_substitue_input()
