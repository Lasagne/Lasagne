import pytest
from lasagne.layers import RecurrentLayer, LSTMLayer, CustomRecurrentLayer
from lasagne.layers import InputLayer, DenseLayer, GRULayer, Gate
from lasagne.layers import helper
import theano
import theano.tensor as T
import numpy as np
import lasagne


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
    x = T.tensor3()
    mask = T.matrix()
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_rec = RecurrentLayer(l_inp,
                           num_units=num_units)
    output = helper.get_output(l_rec, x, mask=mask)
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


def test_recurrent_init_val_error():
    # check if errors are raised when init is non matrix tensor
    hid_init = T.vector()
    with pytest.raises(ValueError):
        l_rec = RecurrentLayer(InputLayer((2, 2, 3)), 5, hid_init=hid_init)


def test_recurrent_init_shape_error():
    # check if errors are raised if output shaped for subnetworks are not
    # correct
    num_hid = 5
    with pytest.raises(ValueError):
        l_rec = CustomRecurrentLayer(
            incoming=InputLayer((2, 2, 3)),
            input_to_hidden=DenseLayer((1, 5), num_hid),
            hidden_to_hidden=DenseLayer((1, num_hid+1), num_hid+1))


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
    np.random.seed(1234)
    l_rec_fwd = RecurrentLayer(l_inp, num_units=num_units, backwards=False)
    np.random.seed(1234)
    l_rec_bck = RecurrentLayer(l_inp, num_units=num_units, backwards=True)
    l_out_fwd = helper.get_output(l_rec_fwd, x)
    l_out_bck = helper.get_output(l_rec_bck, x)

    output_fwd = l_out_fwd.eval({l_out_fwd: x_in})
    output_bck = l_out_bck.eval({l_out_bck: x_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_fwd, output_bck[:, ::-1])


def test_recurrent_self_outvars():
    # check that outvars are correctly stored and returned
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.ones(in_shp).astype('float32')

    l_rec = RecurrentLayer(l_inp, num_units=num_units, backwards=True)
    l_out = helper.get_output(l_rec, x)

    output_hidout = l_rec.hid_out.eval({x: x_in})
    output = l_out.eval({x: x_in})

    np.testing.assert_almost_equal(output_hidout, output)


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
    x = T.tensor3()
    mask = T.matrix()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones(in_shp[:2]).astype('float32')

    # need to set random seed.
    np.random.seed(1234)
    l_rec_scan = RecurrentLayer(l_inp, num_units=num_units, backwards=False,
                                unroll_scan=False)
    np.random.seed(1234)
    l_rec_unroll = RecurrentLayer(l_inp, num_units=num_units, backwards=False,
                                  unroll_scan=True)
    output_scan = helper.get_output(l_rec_scan, x, mask=mask)
    output_unrolled = helper.get_output(l_rec_unroll, x, mask=mask)

    output_scan_val = output_scan.eval({x: x_in, mask: mask_in})
    output_unrolled_val = output_unrolled.eval({x: x_in, mask: mask_in})
    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_recurrent_unroll_scan_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    x_in = np.random.random(in_shp).astype('float32')

    # need to set random seed.
    np.random.seed(1234)
    l_rec_scan = RecurrentLayer(l_inp, num_units=num_units, backwards=True,
                                unroll_scan=False)
    np.random.seed(1234)
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
    x = T.tensor3()
    mask = T.matrix()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones((num_batch, seq_len), dtype='float32')

    # need to set random seed.
    np.random.seed(1234)
    l_rec_precompute = RecurrentLayer(l_inp, num_units=num_units,
                                      precompute_input=True)
    np.random.seed(1234)
    l_rec_no_precompute = RecurrentLayer(l_inp, num_units=num_units,
                                         precompute_input=False)
    output_precompute = helper.get_output(
        l_rec_precompute, x, mask=mask).eval({x: x_in, mask: mask_in})
    output_no_precompute = helper.get_output(
        l_rec_no_precompute, x, mask=mask).eval({x: x_in, mask: mask_in})

    np.testing.assert_almost_equal(output_precompute, output_no_precompute)


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
    x = T.tensor3()
    mask = T.matrix()
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_lstm = LSTMLayer(l_inp, num_units=num_units)
    output = helper.get_output(l_lstm, x, mask=mask)
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
    np.random.seed(1234)
    l_lstm_fwd = LSTMLayer(l_inp, num_units=num_units, backwards=False)
    np.random.seed(1234)
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
    x = T.tensor3()
    mask = T.matrix()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones((num_batch, seq_len), dtype='float32')

    # need to set random seed.
    np.random.seed(1234)
    l_lstm_precompute = LSTMLayer(
        l_inp, num_units=num_units, precompute_input=True)
    np.random.seed(1234)
    l_lstm_no_precompute = LSTMLayer(
        l_inp, num_units=num_units, precompute_input=False)
    output_precompute = helper.get_output(
        l_lstm_precompute, x, mask=mask).eval({x: x_in, mask: mask_in})
    output_no_precompute = helper.get_output(
        l_lstm_no_precompute, x, mask=mask).eval({x: x_in, mask: mask_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_precompute, output_no_precompute)


def test_lstm_self_outvars():
    # check that outvars are correctly stored and returned
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    x_in = np.ones(in_shp).astype('float32')

    l_lstm = LSTMLayer(l_inp, num_units=num_units, backwards=True,
                       peepholes=True)
    l_out = helper.get_output(l_lstm, x)

    output = l_out.eval({x: x_in})
    output_hidout_val = l_lstm.hid_out.eval({x: x_in})
    output_cellout_val = l_lstm.cell_out.eval({x: x_in})

    np.testing.assert_almost_equal(output, output_hidout_val)
    assert output_cellout_val.shape == output.shape


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
    x = T.tensor3()
    mask = T.matrix()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones(in_shp[:2]).astype('float32')

    # need to set random seed.
    np.random.seed(1234)
    l_lstm_scan = LSTMLayer(l_inp, num_units=num_units, backwards=False,
                            unroll_scan=False)
    np.random.seed(1234)
    l_lstm_unrolled = LSTMLayer(l_inp, num_units=num_units, backwards=False,
                                unroll_scan=True)
    output_scan = helper.get_output(l_lstm_scan, x, mask=mask)
    output_unrolled = helper.get_output(l_lstm_unrolled, x, mask=mask)

    output_scan_val = output_scan.eval({x: x_in, mask: mask_in})
    output_unrolled_val = output_unrolled.eval({x: x_in, mask: mask_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_lstm_unroll_scan_bck():
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')

    # need to set random seed.
    np.random.seed(1234)
    l_lstm_scan = LSTMLayer(l_inp, num_units=num_units, backwards=True,
                            unroll_scan=False)
    np.random.seed(1234)
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
    x = T.tensor3()
    mask = T.matrix()
    l_inp = InputLayer((num_batch, seq_len, n_features))
    l_gru = GRULayer(l_inp,
                     num_units=num_units)
    output = helper.get_output(l_gru, x, mask=mask)
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
    np.random.seed(1234)
    l_gru_fwd = GRULayer(l_inp, num_units=num_units, backwards=False)
    np.random.seed(1234)
    l_gru_bck = GRULayer(l_inp, num_units=num_units, backwards=True)
    output_fwd = helper.get_output(l_gru_fwd, x)
    output_bck = helper.get_output(l_gru_bck, x)

    output_fwd_val = output_fwd.eval({x: x_in})
    output_bck_val = output_bck.eval({x: x_in})

    # test that the backwards model reverses its final input
    np.testing.assert_almost_equal(output_fwd_val, output_bck_val[:, ::-1])


def test_gru_self_outvars():
    # check that outvars are correctly stored and returned
    num_batch, seq_len, n_features1 = 2, 3, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.ones(in_shp).astype('float32')

    l_gru = GRULayer(l_inp, num_units=num_units, backwards=True)
    output = helper.get_output(l_gru, x)

    output_val = output.eval({x: x_in})
    output_hidout_val = l_gru.hid_out.eval({x: x_in})

    np.testing.assert_almost_equal(output_val, output_hidout_val)


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
    x = T.tensor3()
    mask = T.matrix()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones(in_shp[:2]).astype('float32')

    # need to set random seed.
    np.random.seed(1234)
    l_gru_scan = GRULayer(l_inp, num_units=num_units, backwards=False,
                          unroll_scan=False)
    np.random.seed(1234)
    l_gru_unrolled = GRULayer(l_inp, num_units=num_units, backwards=False,
                              unroll_scan=True)
    output_scan = helper.get_output(l_gru_scan, x, mask=mask)
    output_unrolled = helper.get_output(l_gru_unrolled, x, mask=mask)

    output_scan_val = output_scan.eval({x: x_in, mask: mask_in})
    output_unrolled_val = output_unrolled.eval({x: x_in, mask: mask_in})

    np.testing.assert_almost_equal(output_scan_val, output_unrolled_val)


def test_gru_unroll_scan_bck():
    num_batch, seq_len, n_features1 = 2, 5, 4
    num_units = 2
    x = T.tensor3()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)
    x_in = np.random.random(in_shp).astype('float32')

    # need to set random seed.
    np.random.seed(1234)
    l_gru_scan = GRULayer(l_inp, num_units=num_units, backwards=True,
                          unroll_scan=False)
    np.random.seed(1234)
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
    x = T.tensor3()
    mask = T.matrix()
    in_shp = (num_batch, seq_len, n_features1)
    l_inp = InputLayer(in_shp)

    x_in = np.random.random(in_shp).astype('float32')
    mask_in = np.ones((num_batch, seq_len), dtype='float32')

    # need to set random seed.
    np.random.seed(1234)
    l_gru_precompute = GRULayer(
        l_inp, num_units=num_units, precompute_input=True)
    np.random.seed(1234)
    l_gru_no_precompute = GRULayer(
        l_inp, num_units=num_units, precompute_input=False)
    output_precompute = helper.get_output(
        l_gru_precompute, x, mask=mask).eval({x: x_in, mask: mask_in})
    output_no_precompute = helper.get_output(
        l_gru_no_precompute, x, mask=mask).eval({x: x_in, mask: mask_in})

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
