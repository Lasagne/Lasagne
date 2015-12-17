import pytest

from lasagne.layers import RecurrentLayer, LSTMLayer, CustomRecurrentLayer
from lasagne.layers import InputLayer, DenseLayer, GRULayer, Gate, Layer
from lasagne.layers import helper
import theano
import theano.tensor as T
import numpy as np
import lasagne
from mock import Mock


def test_recurrent_return_shape():
    num_batch, seq_len, n_features1, n_features2 = 5, 3, 10, 11
    num_units = 6
    x = T.tensor4()
    in_shp = (num_batch, seq_len, n_features1, n_features2)
    l_inp = InputLayer(in_shp)
    l_rec = RecurrentLayer(l_inp, num_units=num_units)

    x_in = np.random.random(in_shp).astype('float32')
    output = helper.get_output(l_rec, x)
    output_val = output.eval({x: x_in})

    assert helper.get_output_shape(l_rec, x_in.shape) == output_val.shape
    assert output_val.shape == (num_batch, seq_len, num_units)


def test_recurrent_grad():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_rec = RecurrentLayer(l_inp,
                           num_units=num_units)
    output = helper.get_output(l_rec)
    g = T.grad(T.mean(output), lasagne.layers.get_all_params(l_rec))
    assert isinstance(g, (list, tuple))


def test_recurrent_nparams():
    l_inp = InputLayer((2, 2, 3))
    l_rec = RecurrentLayer(l_inp, 5, learn_init=False, nonlinearity=None)

    # b, W_hid_to_hid and W_in_to_hid
    assert len(lasagne.layers.get_all_params(l_rec, trainable=True)) == 3

    # b + hid_init
    assert len(lasagne.layers.get_all_params(l_rec, regularizable=False)) == 2


def test_recurrent_nparams_learn_init():
    l_inp = InputLayer((2, 2, 3))
    l_rec = RecurrentLayer(l_inp, 5, learn_init=True)

    # b, W_hid_to_hid and W_in_to_hid + hid_init
    assert len(lasagne.layers.get_all_params(l_rec, trainable=True)) == 4

    # b + hid_init
    assert len(lasagne.layers.get_all_params(l_rec, regularizable=False)) == 2


def test_recurrent_tensor_init():
    # check if passing in a TensorVariable to hid_init works
    num_units = 5
    batch_size = 3
    seq_len = 2
    n_inputs = 4
    in_shp = (batch_size, seq_len, n_inputs)
    l_inp = InputLayer(in_shp)
    hid_init = T.matrix()
    x = T.tensor3()

    l_rec = RecurrentLayer(l_inp, num_units, learn_init=True,
                           hid_init=hid_init)
    # check that the tensor is used
    assert hid_init == l_rec.hid_init

    # b, W_hid_to_hid and W_in_to_hid, should not return any inits
    assert len(lasagne.layers.get_all_params(l_rec, trainable=True)) == 3

    # b, should not return any inits
    assert len(lasagne.layers.get_all_params(l_rec, regularizable=False)) == 1

    # check that it compiles and runs
    output = lasagne.layers.get_output(l_rec, x)
    x_test = np.ones(in_shp, dtype='float32')
    hid_init_test = np.ones((batch_size, num_units), dtype='float32')
    output_val = output.eval({x: x_test, hid_init: hid_init_test})
    assert isinstance(output_val, np.ndarray)


def test_recurrent_incoming_tuple():
    input_shape = (2, 3, 4)
    l_rec = lasagne.layers.RecurrentLayer(input_shape, 5)
    assert l_rec.input_shapes[0] == input_shape


def test_recurrent_init_val_error():
    # check if errors are raised when init is non matrix tensor
    hid_init = T.vector()
    with pytest.raises(ValueError):
        l_rec = RecurrentLayer(InputLayer((2, 2, 3)), 5, hid_init=hid_init)


def test_recurrent_name():
    l_in = lasagne.layers.InputLayer((2, 3, 4))
    layer_name = 'l_rec'
    l_rec = lasagne.layers.RecurrentLayer(l_in, 4, name=layer_name)
    assert l_rec.b.name == layer_name + '.input_to_hidden.b'
    assert l_rec.W_in_to_hid.name == layer_name + '.input_to_hidden.W'
    assert l_rec.W_hid_to_hid.name == layer_name + '.hidden_to_hidden.W'


def test_custom_recurrent_arbitrary_shape():
    # Check that the custom recurrent layer can handle more than 1 feature dim
    n_batch, n_steps, n_channels, width, height = (2, 3, 4, 5, 6)
    n_out_filters = 7
    filter_shape = (3, 3)
    l_in = lasagne.layers.InputLayer(
        (n_batch, n_steps, n_channels, width, height))
    l_in_to_hid = lasagne.layers.Conv2DLayer(
        lasagne.layers.InputLayer((None, n_channels, width, height)),
        n_out_filters, filter_shape, pad='same')
    l_hid_to_hid = lasagne.layers.Conv2DLayer(
        lasagne.layers.InputLayer((None, n_out_filters, width, height)),
        n_out_filters, filter_shape, pad='same')
    l_rec = lasagne.layers.CustomRecurrentLayer(
        l_in, l_in_to_hid, l_hid_to_hid)
    assert l_rec.output_shape == (n_batch, n_steps, n_out_filters, width,
                                  height)
    out = theano.function([l_in.input_var], lasagne.layers.get_output(l_rec))
    out_shape = out(np.zeros((n_batch, n_steps, n_channels, width, height),
                             dtype=theano.config.floatX)).shape
    assert out_shape == (n_batch, n_steps, n_out_filters, width, height)


def test_recurrent_init_shape_error():
    # Check that the custom recurrent layer throws errors for invalid shapes
    n_batch, n_steps, n_channels, width, height = (2, 3, 4, 5, 6)
    n_out_filters = 7
    filter_shape = (3, 3)
    l_in = lasagne.layers.InputLayer(
        (n_batch, n_steps, n_channels, width, height))
    l_hid_to_hid = lasagne.layers.Conv2DLayer(
        lasagne.layers.InputLayer((n_batch, n_out_filters, width, height)),
        n_out_filters, filter_shape, pad='same')

    # When precompute_input == True, input_to_hidden.shape[0] must be None
    # or n_batch*n_steps
    l_in_to_hid = lasagne.layers.Conv2DLayer(
        lasagne.layers.InputLayer((n_batch, n_channels, width, height)),
        n_out_filters, filter_shape, pad='same')
    with pytest.raises(ValueError):
        l_rec = lasagne.layers.CustomRecurrentLayer(
            l_in, l_in_to_hid, l_hid_to_hid, precompute_input=True)

    # When precompute_input = False, input_to_hidden.shape[1] must be None
    # or hidden_to_hidden.shape[1]
    l_in_to_hid = lasagne.layers.Conv2DLayer(
        lasagne.layers.InputLayer((n_batch + 1, n_channels, width, height)),
        n_out_filters, filter_shape, pad='same')
    with pytest.raises(ValueError):
        l_rec = lasagne.layers.CustomRecurrentLayer(
            l_in, l_in_to_hid, l_hid_to_hid, precompute_input=False)

    # In any case, input_to_hidden and hidden_to_hidden's output shapes after
    # the first dimension must match
    l_in_to_hid = lasagne.layers.Conv2DLayer(
        lasagne.layers.InputLayer((None, n_channels, width + 1, height)),
        n_out_filters, filter_shape, pad='same')
    with pytest.raises(ValueError):
        l_rec = lasagne.layers.CustomRecurrentLayer(
            l_in, l_in_to_hid, l_hid_to_hid)

    # And, the output shape of input_to_hidden must match the input shape
    # of hidden_to_hidden past the first dimension.  By not using padding,
    # the output of l_in_to_hid will be cropped, which will make the
    # shape inappropriate.
    l_in_to_hid = lasagne.layers.Conv2DLayer(
        lasagne.layers.InputLayer((None, n_channels, width, height)),
        n_out_filters, filter_shape)
    l_hid_to_hid = lasagne.layers.Conv2DLayer(
        lasagne.layers.InputLayer((n_batch, n_out_filters, width, height)),
        n_out_filters, filter_shape)
    with pytest.raises(ValueError):
        l_rec = lasagne.layers.CustomRecurrentLayer(
            l_in, l_in_to_hid, l_hid_to_hid)


def test_recurrent_grad_clipping():
    num_units = 5
    batch_size = 3
    seq_len = 2
    n_inputs = 4
    in_shp = (batch_size, seq_len, n_inputs)
    l_inp = InputLayer(in_shp)
    x = T.tensor3()
    l_rec = RecurrentLayer(l_inp, num_units, grad_clipping=1.0)
    output = lasagne.layers.get_output(l_rec, x)


def test_recurrent_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.ones(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_rec_fwd = RecurrentLayer(l_inp, num_units=num_units, backwards=False)
    lasagne.random.get_rng().seed(1234)
    l_rec_bck = RecurrentLayer(l_inp, num_units=num_units, backwards=True)
    l_out_fwd = helper.get_output(l_rec_fwd, x)
    l_out_bck = helper.get_output(l_rec_bck, x)

    output_fwd = l_out_fwd.eval({l_out_fwd: x_in})
    output_bck = l_out_bck.eval({l_out_bck: x_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_fwd, output_bck[:, ::-1])


def test_recurrent_variable_input_size():
    # check that seqlen and batchsize None works
    num_batch, n_features1 = 6, 5
    num_units = 13
    x = T.tensor3()

    in_shp = (None, None, n_features1)
    l_inp = InputLayer(in_shp)
    x_in1 = np.ones((num_batch+1, 10, n_features1)).astype('float32')
    x_in2 = np.ones((num_batch, 15, n_features1)).astype('float32')
    l_rec = RecurrentLayer(l_inp, num_units=num_units, backwards=False)
    output = helper.get_output(l_rec, x)
    output_val1 = output.eval({x: x_in1})
    output_val2 = output.eval({x: x_in2})


def test_recurrent_unroll_scan_fwd():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones(in_shp[:2]).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_rec_scan = RecurrentLayer(l_inp, num_units=num_units, backwards=False,
                                unroll_scan=False, mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_rec_unroll = RecurrentLayer(l_inp, num_units=num_units, backwards=False,
                                  unroll_scan=True, mask_input=l_mask_inp)
    output_scan = helper.get_output(l_rec_scan)
    output_unrolled = helper.get_output(l_rec_unroll)

    output_scan_val = output_scan.eval(
        {l_inp.input_var: x_in, l_mask_inp.input_var: mask_in})
    output_unrolled_val = output_unrolled.eval(
        {l_inp.input_var: x_in, l_mask_inp.input_var: mask_in})
    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_recurrent_unroll_scan_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    x_in = np.random.random(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_rec_scan = RecurrentLayer(l_inp, num_units=num_units, backwards=True,
                                unroll_scan=False)
    lasagne.random.get_rng().seed(1234)
    l_rec_unroll = RecurrentLayer(l_inp, num_units=num_units, backwards=True,
                                  unroll_scan=True)
    output_scan = helper.get_output(l_rec_scan, x)
    output_unrolled = helper.get_output(l_rec_unroll, x)
    output_scan_val = output_scan.eval({x: x_in})
    output_unrolled_val = output_unrolled.eval({x: x_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_recurrent_precompute():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones((num_batch, seq_len), dtype='float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_rec_precompute = RecurrentLayer(l_inp, num_units=num_units,
                                      precompute_input=True,
                                      mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_rec_no_precompute = RecurrentLayer(l_inp, num_units=num_units,
                                         precompute_input=False,
                                         mask_input=l_mask_inp)
    output_precompute = helper.get_output(
        l_rec_precompute).eval({l_inp.input_var: x_in,
                                l_mask_inp.input_var: mask_in})
    output_no_precompute = helper.get_output(
        l_rec_no_precompute).eval({l_inp.input_var: x_in,
                                   l_mask_inp.input_var: mask_in})

    np.testing.assert_almost_equal(output_precompute, output_no_precompute)


def test_recurrent_return_final():
    num_batch, seq_len, n_features = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features)
    x_in = np.random.random(in_shp).astype('float32')

    l_inp = InputLayer(in_shp)
    lasagne.random.get_rng().seed(1234)
    l_rec_final = RecurrentLayer(l_inp, num_units, only_return_final=True)
    lasagne.random.get_rng().seed(1234)
    l_rec_all = RecurrentLayer(l_inp, num_units, only_return_final=False)

    output_final = helper.get_output(l_rec_final).eval({l_inp.input_var: x_in})
    output_all = helper.get_output(l_rec_all).eval({l_inp.input_var: x_in})

    assert output_final.shape == (output_all.shape[0], output_all.shape[2])
    assert output_final.shape == lasagne.layers.get_output_shape(l_rec_final)
    assert np.allclose(output_final, output_all[:, -1])


def test_lstm_return_shape():
    num_batch, seq_len, n_features1, n_features2 = 5, 3, 10, 11
    num_units = 6
    x = T.tensor4()
    in_shp = (num_batch, seq_len, n_features1, n_features2)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')

    l_lstm = LSTMLayer(l_inp, num_units=num_units)
    output = helper.get_output(l_lstm, x)
    output_val = output.eval({x: x_in})
    assert helper.get_output_shape(l_lstm, x_in.shape) == output_val.shape
    assert output_val.shape == (num_batch, seq_len, num_units)


def test_lstm_grad():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_lstm = LSTMLayer(l_inp, num_units=num_units)
    output = helper.get_output(l_lstm)
    g = T.grad(T.mean(output), lasagne.layers.get_all_params(l_lstm))
    assert isinstance(g, (list, tuple))


def test_lstm_nparams_no_peepholes():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=False, learn_init=False)

    # 3*n_gates
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 12

    # bias params + init params
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_nparams_peepholes():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=True, learn_init=False)

    # 3*n_gates + peepholes(3).
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 15

    # bias params(4) + init params(2)
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_nparams_learn_init():
    l_inp = InputLayer((2, 2, 3))
    l_lstm = LSTMLayer(l_inp, 5, peepholes=False, learn_init=True)

    # 3*n_gates + inits(2).
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 14

    # bias params(4) + init params(2)
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 6


def test_lstm_tensor_init():
    # check if passing in TensorVariables to cell_init and hid_init works
    num_units = 5
    batch_size = 3
    seq_len = 2
    n_inputs = 4
    in_shp = (batch_size, seq_len, n_inputs)
    l_inp = InputLayer(in_shp)
    hid_init = T.matrix()
    cell_init = T.matrix()
    x = T.tensor3()

    l_lstm = LSTMLayer(l_inp, num_units, peepholes=False, learn_init=True,
                       hid_init=hid_init, cell_init=cell_init)

    # check that the tensors are used and not overwritten
    assert cell_init == l_lstm.cell_init
    assert hid_init == l_lstm.hid_init

    # 3*n_gates, should not return any inits
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 12

    # bias params(4), , should not return any inits
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 4

    # check that it compiles and runs
    output = lasagne.layers.get_output(l_lstm, x)

    x_test = np.ones(in_shp, dtype='float32')
    hid_init_test = np.ones((batch_size, num_units), dtype='float32')
    cell_init_test = np.ones_like(hid_init_test)
    output_val = output.eval(
        {x: x_test, cell_init: cell_init_test, hid_init: hid_init_test})

    assert isinstance(output_val, np.ndarray)


def test_lstm_init_val_error():
    # check if errors are raised when inits are non matrix tensor
    vector = T.vector()
    with pytest.raises(ValueError):
        l_rec = LSTMLayer(InputLayer((2, 2, 3)), 5, hid_init=vector)

    with pytest.raises(ValueError):
        l_rec = LSTMLayer(InputLayer((2, 2, 3)), 5, cell_init=vector)

    with pytest.raises(ValueError):
        l_rec = LSTMLayer(InputLayer((2, 2, 3)), 5, hid_init=0)


def test_lstm_grad_clipping():
    # test that you can set grad_clip variable
    x = T.tensor3()
    l_rec = LSTMLayer(InputLayer((2, 2, 3)), 5, grad_clipping=1)
    output = lasagne.layers.get_output(l_rec, x)


def test_lstm_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.ones(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_fwd = LSTMLayer(l_inp, num_units=num_units, backwards=False)
    lasagne.random.get_rng().seed(1234)
    l_lstm_bck = LSTMLayer(l_inp, num_units=num_units, backwards=True)
    output_fwd = helper.get_output(l_lstm_fwd, x)
    output_bck = helper.get_output(l_lstm_bck, x)

    output_fwd_val = output_fwd.eval({x: x_in})
    output_bck_val = output_bck.eval({x: x_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_fwd_val, output_bck_val[:, ::-1])


def test_lstm_precompute():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones((num_batch, seq_len), dtype='float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_precompute = LSTMLayer(
        l_inp, num_units=num_units, precompute_input=True,
        mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_lstm_no_precompute = LSTMLayer(
        l_inp, num_units=num_units, precompute_input=False,
        mask_input=l_mask_inp)
    output_precompute = helper.get_output(
        l_lstm_precompute).eval({l_inp.input_var: x_in,
                                 l_mask_inp.input_var: mask_in})
    output_no_precompute = helper.get_output(
        l_lstm_no_precompute).eval({l_inp.input_var: x_in,
                                    l_mask_inp.input_var: mask_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_precompute, output_no_precompute)


def test_lstm_variable_input_size():
    # that seqlen and batchsize None works
    num_batch, n_features1 = 6, 5
    num_units = 13
    x = T.tensor3()

    in_shp = (None, None, n_features1)
    l_inp = InputLayer(in_shp)
    x_in1 = np.ones((num_batch+1, 3+1, n_features1)).astype('float32')
    x_in2 = np.ones((num_batch, 3, n_features1)).astype('float32')
    l_rec = LSTMLayer(l_inp, num_units=num_units, backwards=False)
    output = helper.get_output(l_rec, x)
    output_val1 = output.eval({x: x_in1})
    output_val2 = output.eval({x: x_in2})


def test_lstm_unroll_scan_fwd():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones(in_shp[:2]).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_scan = LSTMLayer(l_inp, num_units=num_units, backwards=False,
                            unroll_scan=False, mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_lstm_unrolled = LSTMLayer(l_inp, num_units=num_units, backwards=False,
                                unroll_scan=True, mask_input=l_mask_inp)
    output_scan = helper.get_output(l_lstm_scan)
    output_unrolled = helper.get_output(l_lstm_unrolled)

    output_scan_val = output_scan.eval({l_inp.input_var: x_in,
                                        l_mask_inp.input_var: mask_in})
    output_unrolled_val = output_unrolled.eval({l_inp.input_var: x_in,
                                                l_mask_inp.input_var: mask_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_lstm_unroll_scan_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_lstm_scan = LSTMLayer(l_inp, num_units=num_units, backwards=True,
                            unroll_scan=False)
    lasagne.random.get_rng().seed(1234)
    l_lstm_unrolled = LSTMLayer(l_inp, num_units=num_units, backwards=True,
                                unroll_scan=True)
    output_scan = helper.get_output(l_lstm_scan, x)
    output_scan_unrolled = helper.get_output(l_lstm_unrolled, x)

    output_scan_val = output_scan.eval({x: x_in})
    output_unrolled_val = output_scan_unrolled.eval({x: x_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_lstm_passthrough():
    # Tests that the LSTM can simply pass through its input
    l_in = InputLayer((4, 5, 6))
    zero = lasagne.init.Constant(0.)
    one = lasagne.init.Constant(1.)
    pass_gate = Gate(zero, zero, zero, one, None)
    no_gate = Gate(zero, zero, zero, zero, None)
    in_pass_gate = Gate(
        np.eye(6).astype(theano.config.floatX), zero, zero, zero, None)
    l_rec = LSTMLayer(
        l_in, 6, pass_gate, no_gate, in_pass_gate, pass_gate, None)
    out = lasagne.layers.get_output(l_rec)
    inp = np.arange(4*5*6).reshape(4, 5, 6).astype(theano.config.floatX)
    np.testing.assert_almost_equal(out.eval({l_in.input_var: inp}), inp)


def test_lstm_return_final():
    num_batch, seq_len, n_features = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features)
    x_in = np.random.random(in_shp).astype('float32')

    l_inp = InputLayer(in_shp)
    lasagne.random.get_rng().seed(1234)
    l_rec_final = LSTMLayer(l_inp, num_units, only_return_final=True)
    lasagne.random.get_rng().seed(1234)
    l_rec_all = LSTMLayer(l_inp, num_units, only_return_final=False)

    output_final = helper.get_output(l_rec_final).eval({l_inp.input_var: x_in})
    output_all = helper.get_output(l_rec_all).eval({l_inp.input_var: x_in})

    assert output_final.shape == (output_all.shape[0], output_all.shape[2])
    assert output_final.shape == lasagne.layers.get_output_shape(l_rec_final)
    assert np.allclose(output_final, output_all[:, -1])


def test_gru_return_shape():
    num_batch, seq_len, n_features1, n_features2 = 5, 3, 10, 11
    num_units = 6
    x = T.tensor4()
    in_shp = (num_batch, seq_len, n_features1, n_features2)
    l_inp = InputLayer(in_shp)
    l_rec = GRULayer(l_inp, num_units=num_units)

    x_in = np.random.random(in_shp).astype('float32')
    output = helper.get_output(l_rec, x)
    output_val = output.eval({x: x_in})

    assert helper.get_output_shape(l_rec, x_in.shape) == output_val.shape
    assert output_val.shape == (num_batch, seq_len, num_units)


def test_gru_grad():
    num_batch, seq_len, n_features = 5, 3, 10
    num_units = 6
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_gru = GRULayer(l_inp,
                     num_units=num_units)
    output = helper.get_output(l_gru)
    g = T.grad(T.mean(output), lasagne.layers.get_all_params(l_gru))
    assert isinstance(g, (list, tuple))


def test_gru_nparams_learn_init_false():
    l_inp = InputLayer((2, 2, 3))
    l_gru = GRULayer(l_inp, 5, learn_init=False)

    # 3*n_gates
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_gru, trainable=True)) == 9

    # bias params(3) + hid_init
    assert len(lasagne.layers.get_all_params(l_gru, regularizable=False)) == 4


def test_gru_nparams_learn_init_true():
    l_inp = InputLayer((2, 2, 3))
    l_gru = GRULayer(l_inp, 5, learn_init=True)

    # 3*n_gates + hid_init
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_gru, trainable=True)) == 10

    # bias params(3) + init params(1)
    assert len(lasagne.layers.get_all_params(l_gru, regularizable=False)) == 4


def test_gru_tensor_init():
    # check if passing in a TensorVariable to hid_init works
    num_units = 5
    batch_size = 3
    seq_len = 2
    n_inputs = 4
    in_shp = (batch_size, seq_len, n_inputs)
    l_inp = InputLayer(in_shp)
    hid_init = T.matrix()
    x = T.tensor3()

    l_lstm = GRULayer(l_inp, num_units, learn_init=True, hid_init=hid_init)

    # check that the tensors are used and not overwritten
    assert hid_init == l_lstm.hid_init

    # 3*n_gates, should not return any inits
    # the 3 is because we have  hid_to_gate, in_to_gate and bias for each gate
    assert len(lasagne.layers.get_all_params(l_lstm, trainable=True)) == 9

    # bias params(3), , should not return any inits
    assert len(lasagne.layers.get_all_params(l_lstm, regularizable=False)) == 3

    # check that it compiles and runs
    output = lasagne.layers.get_output(l_lstm, x)
    x_test = np.ones(in_shp, dtype='float32')
    hid_init_test = np.ones((batch_size, num_units), dtype='float32')

    output_val = output.eval({x: x_test, hid_init: hid_init_test})
    assert isinstance(output_val, np.ndarray)


def test_gru_init_val_error():
    # check if errors are raised when init is non matrix tensorVariable
    vector = T.vector()
    with pytest.raises(ValueError):
        l_rec = GRULayer(InputLayer((2, 2, 3)), 5, hid_init=vector)


def test_gru_grad_clipping():
    # test that you can set grad_clip variable
    x = T.tensor3()
    l_rec = GRULayer(InputLayer((2, 2, 3)), 5, grad_clipping=1)
    output = lasagne.layers.get_output(l_rec, x)


def test_gru_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.ones(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_gru_fwd = GRULayer(l_inp, num_units=num_units, backwards=False)
    lasagne.random.get_rng().seed(1234)
    l_gru_bck = GRULayer(l_inp, num_units=num_units, backwards=True)
    output_fwd = helper.get_output(l_gru_fwd, x)
    output_bck = helper.get_output(l_gru_bck, x)

    output_fwd_val = output_fwd.eval({x: x_in})
    output_bck_val = output_bck.eval({x: x_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_fwd_val, output_bck_val[:, ::-1])


def test_gru_variable_input_size():
    # that seqlen and batchsize None works
    num_batch, n_features1 = 6, 5
    num_units = 13
    x = T.tensor3()

    in_shp = (None, None, n_features1)
    l_inp = InputLayer(in_shp)
    x_in1 = np.ones((num_batch+1, 10, n_features1)).astype('float32')
    x_in2 = np.ones((num_batch, 15, n_features1)).astype('float32')
    l_rec = GRULayer(l_inp, num_units=num_units, backwards=False)
    output = helper.get_output(l_rec, x)

    output.eval({x: x_in1})
    output.eval({x: x_in2})


def test_gru_unroll_scan_fwd():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones(in_shp[:2]).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_gru_scan = GRULayer(l_inp, num_units=num_units, backwards=False,
                          unroll_scan=False, mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_gru_unrolled = GRULayer(l_inp, num_units=num_units, backwards=False,
                              unroll_scan=True, mask_input=l_mask_inp)
    output_scan = helper.get_output(l_gru_scan)
    output_unrolled = helper.get_output(l_gru_unrolled)

    output_scan_val = output_scan.eval({l_inp.input_var: x_in,
                                        l_mask_inp.input_var: mask_in})
    output_unrolled_val = output_unrolled.eval({l_inp.input_var: x_in,
                                                l_mask_inp.input_var: mask_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_gru_unroll_scan_bck():
    num_batch, seq_len, n_features1 = 2, 5, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    x_in = np.random.random(in_shp).astype('float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_gru_scan = GRULayer(l_inp, num_units=num_units, backwards=True,
                          unroll_scan=False)
    lasagne.random.get_rng().seed(1234)
    l_gru_unrolled = GRULayer(l_inp, num_units=num_units, backwards=True,
                              unroll_scan=True)
    output_scan = helper.get_output(l_gru_scan, x)
    output_unrolled = helper.get_output(l_gru_unrolled, x)

    output_scan_val = output_scan.eval({x: x_in})
    output_unrolled_val = output_unrolled.eval({x: x_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_gru_precompute():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    l_mask_inp = InputLayer(in_shp[:2])

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones((num_batch, seq_len), dtype='float32')

    # need to set random seed.
    lasagne.random.get_rng().seed(1234)
    l_gru_precompute = GRULayer(l_inp, num_units=num_units,
                                precompute_input=True, mask_input=l_mask_inp)
    lasagne.random.get_rng().seed(1234)
    l_gru_no_precompute = GRULayer(l_inp, num_units=num_units,
                                   precompute_input=False,
                                   mask_input=l_mask_inp)
    output_precompute = helper.get_output(
        l_gru_precompute).eval({l_inp.input_var: x_in,
                                l_mask_inp.input_var: mask_in})
    output_no_precompute = helper.get_output(
        l_gru_no_precompute).eval({l_inp.input_var: x_in,
                                   l_mask_inp.input_var: mask_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_precompute, output_no_precompute)


def test_gru_passthrough():
    # Tests that the LSTM can simply pass through its input
    l_in = InputLayer((4, 5, 6))
    zero = lasagne.init.Constant(0.)
    one = lasagne.init.Constant(1.)
    pass_gate = Gate(zero, zero, None, one, None)
    no_gate = Gate(zero, zero, None, zero, None)
    in_pass_gate = Gate(
        np.eye(6).astype(theano.config.floatX), zero, None, zero, None)
    l_rec = GRULayer(l_in, 6, no_gate, pass_gate, in_pass_gate)
    out = lasagne.layers.get_output(l_rec)
    inp = np.arange(4*5*6).reshape(4, 5, 6).astype(theano.config.floatX)
    np.testing.assert_almost_equal(out.eval({l_in.input_var: inp}), inp)


def test_gru_return_final():
    num_batch, seq_len, n_features = 2, 3, 4
    num_units = 2
    in_shp = (num_batch, seq_len, n_features)
    x_in = np.random.random(in_shp).astype('float32')

    l_inp = InputLayer(in_shp)
    lasagne.random.get_rng().seed(1234)
    l_rec_final = GRULayer(l_inp, num_units, only_return_final=True)
    lasagne.random.get_rng().seed(1234)
    l_rec_all = GRULayer(l_inp, num_units, only_return_final=False)

    output_final = helper.get_output(l_rec_final).eval({l_inp.input_var: x_in})
    output_all = helper.get_output(l_rec_all).eval({l_inp.input_var: x_in})

    assert output_final.shape == (output_all.shape[0], output_all.shape[2])
    assert output_final.shape == lasagne.layers.get_output_shape(l_rec_final)
    assert np.allclose(output_final, output_all[:, -1])


def test_gradient_steps_error():
    # Check that error is raised if gradient_steps is not -1 and scan_unroll
    # is true
    l_in = InputLayer((2, 2, 3))
    with pytest.raises(ValueError):
        RecurrentLayer(l_in, 5, gradient_steps=3, unroll_scan=True)

    with pytest.raises(ValueError):
        LSTMLayer(l_in, 5, gradient_steps=3, unroll_scan=True)

    with pytest.raises(ValueError):
        GRULayer(l_in, 5, gradient_steps=3, unroll_scan=True)


def test_unroll_none_input_error():
    # Test that a ValueError is raised if unroll scan is True and the input
    # sequence length is specified as None.
    l_in = InputLayer((2, None, 3))
    with pytest.raises(ValueError):
        RecurrentLayer(l_in, 5, unroll_scan=True)

    with pytest.raises(ValueError):
        LSTMLayer(l_in, 5, unroll_scan=True)

    with pytest.raises(ValueError):
        GRULayer(l_in, 5, unroll_scan=True)


def test_CustomRecurrentLayer_child_kwargs():
    in_shape = (2, 3, 4)
    n_hid = 5
    # Construct mock for input-to-hidden layer
    in_to_hid = Mock(
        Layer,
        output_shape=(in_shape[0]*in_shape[1], n_hid),
        input_shape=(in_shape[0]*in_shape[1], in_shape[2]),
        input_layer=InputLayer((in_shape[0]*in_shape[1], in_shape[2])))
    # These two functions get called, need to return dummy values for them
    in_to_hid.get_output_for.return_value = T.matrix()
    in_to_hid.get_params.return_value = []
    # As above, for hidden-to-hidden layer
    hid_to_hid = Mock(
        Layer,
        output_shape=(in_shape[0], n_hid),
        input_shape=(in_shape[0], n_hid),
        input_layer=InputLayer((in_shape[0], n_hid)))
    hid_to_hid.get_output_for.return_value = T.matrix()
    hid_to_hid.get_params.return_value = []
    # Construct a CustomRecurrentLayer using these Mocks
    l_rec = lasagne.layers.CustomRecurrentLayer(
        InputLayer(in_shape), in_to_hid, hid_to_hid)
    # Call get_output with a kwarg, should be passd to in_to_hid and hid_to_hid
    helper.get_output(l_rec, foo='bar')
    # Retrieve the arguments used to call in_to_hid.get_output_for
    args, kwargs = in_to_hid.get_output_for.call_args
    # Should be one argument - the Theano expression
    assert len(args) == 1
    # One keywould argument - should be 'foo' -> 'bar'
    assert kwargs == {'foo': 'bar'}
    # Same as with in_to_hid
    args, kwargs = hid_to_hid.get_output_for.call_args
    assert len(args) == 1
    assert kwargs == {'foo': 'bar'}
