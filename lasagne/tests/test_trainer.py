import sys
import re
import itertools
import six
import numpy as np
import pytest


class TrainFunction(object):
    def __init__(self, results=None, states=None, on_invoke=None,
                 epoch_index_prepended=False,
                 return_array=False):
        """
        Training function to test `Trainer`.
        It is invoked like a regular function; see `__call__`.

        Parameters
        ----------
        results: [optional] list of lists
            The results that should be returned:
            `[[epoch0_resA, epoch0_resB, ...],
              [epoch1_resA, epoch1_resB, ...],
              ...]`

        states: [optional] list of states
            The states for each epoch; when this function is invoked, the
            `current_state` attribute will be set to the value of
            `self.states[current_epoch]`

        on_invoke: [optional] callback function
            Invoked whenever this function is invoked

        epoch_index_prepended: bool
            If True, this function will expect the current epoch index to
            be prepended to the arguments passed

        return_array: bool
            If True, the results will be converted to a NumPy array before
            being returned
        """
        self.count = 0
        self.results = results
        self.states = states
        self.current_state = None
        self.current_epoch = 0
        self.state_get_count = 0
        self.state_set_count = 0
        self.epoch_index_prepended = epoch_index_prepended
        self.return_array = return_array
        self.on_invoke = on_invoke

    def get_state(self):
        self.state_get_count += 1
        return self.current_state

    def set_state(self, state):
        self.state_set_count += 1
        self.current_state = state

    def pre_epoch(self, epoch_index):
        self.current_epoch = epoch_index

    @staticmethod
    def pre_epoch_for(*train_funcs):
        def pre_epoch(epoch_index):
            for f in train_funcs:
                f.pre_epoch(epoch_index)

        return pre_epoch

    def __call__(self, *args, **kwargs):
        if self.epoch_index_prepended:
            # Consume the first argument
            epoch_index = args[0]
            args = args[1:]
            assert epoch_index == self.current_epoch
        d0 = args[0]
        batch_size = d0.shape[0]
        if self.on_invoke is not None:
            self.on_invoke()
        self.count += 1
        if self.states is not None:
            self.current_state = self.states[self.current_epoch]
        if self.results is None:
            return None
        else:
            res = [r * batch_size for r in self.results[self.current_epoch]]
            if self.return_array:
                return np.array(res)
            else:
                return res


class EpochTracker (object):
    def __init__(self):
        self.epoch = None

    def __call__(self, epoch):
        self.epoch = epoch


def count_calls(fn):
    def f(*args, **kwargs):
        res = fn(*args, **kwargs)
        f.count += 1
        return res
    f.count = 0
    f.__name__ = fn.__name__
    return f


def test_no_train_fn():
    from lasagne.trainer import train

    with pytest.raises(ValueError):
        train([np.arange(10)], [np.arange(10)], None, batchsize=5)


def test_no_eval_fn():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    def train_batch(*batch):
        return [0.0]

    with pytest.raises(ValueError):
        train([np.arange(10)], [np.arange(10)], None, batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              log_stream=log, verbosity=VERBOSITY_NONE, log_final_result=False)
    with pytest.raises(ValueError):
        train([np.arange(10)], None, [np.arange(10)], batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_train():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    # No evaluation function or validation set so the get and set state
    # functions shouldn't be invoked
    def get_state():
        pytest.fail('get_state should not be invoked')

    def set_state():
        pytest.fail('set_state should not be invoked')

    res = train([np.arange(10)], None, None, batchsize=5,
                train_batch_func=train_batch, num_epochs=200,
                get_state_func=get_state,
                set_state_func=set_state,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_batch.count == 400
    # Logging is disable so it should be empty
    assert log.getvalue() == ''
    # Should have 200 training results; one per epoch
    assert len(res.train_results) == 200
    # No validation or test results
    assert res.validation_results is None
    assert res.test_results is None
    # No evaluation function or val set hence no best result
    assert res.best_val_epoch is None
    assert res.best_validation_results is None
    assert res.best_test_results is None
    # Last epoch should be the end
    assert res.last_epoch == 200


def test_train_prepend_epoch_number():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(epoch_index, *batch):
        assert epoch_index == epoch_track.epoch
        return [0.0]

    # No evaluation function or validation set so the get and set state
    # functions shouldn't be invoked
    def get_state():
        pytest.fail('get_state should not be invoked')

    def set_state():
        pytest.fail('set_state should not be invoked')

    res = train([np.arange(10)], None, None, batchsize=5,
                train_batch_func=train_batch, num_epochs=200,
                get_state_func=get_state,
                set_state_func=set_state,
                pre_epoch_callback=epoch_track,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False,
                train_pass_epoch_number=True)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_batch.count == 400
    # Logging is disable so it should be empty
    assert log.getvalue() == ''
    # Should have 200 training results; one per epoch
    assert len(res.train_results) == 200
    # No validation or test results
    assert res.validation_results is None
    assert res.test_results is None
    # No evaluation function or val set hence no best result
    assert res.best_val_epoch is None
    assert res.best_validation_results is None
    assert res.best_test_results is None
    # Last epoch should be the end
    assert res.last_epoch == 200


def test_train_return_array():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        # result should be sum across samples, hence multiplication by batch
        # size
        return np.array([float(epoch_track.epoch) * batch[0].shape[0]])

    res = train([np.arange(10)], None, None, batchsize=5,
                train_batch_func=train_batch, num_epochs=200,
                pre_epoch_callback=epoch_track,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_batch.count == 400
    assert res.train_results == [np.array([float(i)]) for i in range(200)]


def test_train_return_invalid():
    # Check that TypeError is raised when the training function returns
    # an invalid type
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    def train_batch(*args, **kwargs):
        return 'Strings wont work here'

    with pytest.raises(TypeError):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_training_failure():
    # Check that TypeError is raised when the training function returns
    # an invalid type
    from lasagne.trainer import train, TrainingFailedException, \
        VERBOSITY_MINIMAL
    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    # Fail at epoch 100
    def train_batch(*batch):
        if epoch_track.epoch == 100:
            return [float('nan')]
        else:
            return [float(epoch_track.epoch) * batch[0].shape[0]]

    # Check for failure
    def check_train_res(epoch, train_results):
        if np.isnan(train_results[0]).any():
            return 'NaN detected'
        else:
            return None

    with pytest.raises(TrainingFailedException):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              train_epoch_results_check_func=check_train_res,
              pre_epoch_callback=epoch_track,
              log_stream=log, verbosity=VERBOSITY_MINIMAL,
              log_final_result=False)

    lines = log.getvalue().split('\n')
    lines = [line for line in lines if line.strip() != '']
    assert lines[-1] == 'Training failed at epoch 100: NaN detected'


def test_training_failure_state_restore():
    # Check that TypeError is raised when the training function returns
    # an invalid type
    from lasagne.trainer import train, TrainingFailedException, \
        VERBOSITY_NONE
    log = six.moves.cStringIO()
    epoch_track = EpochTracker()

    # No stored state at the start
    current_state = [None]

    # Fail at epoch 100
    def train_batch(*batch):
        if epoch_track.epoch == 100:
            return [float('nan')]
        else:
            return [float(epoch_track.epoch) * batch[0].shape[0]]

    # Check for failure
    def check_train_res(epoch, train_results):
        if np.isnan(train_results[0]).any():
            return 'NaN detected'
        else:
            return None

    @count_calls
    def get_state():
        if epoch_track.epoch is None:
            # Initial state: 12345
            return 12345
        else:
            return range(1000, 2005, 5)[epoch_track.epoch]

    @count_calls
    def set_state(state):
        current_state[0] = state

    with pytest.raises(TrainingFailedException):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              train_epoch_results_check_func=check_train_res,
              get_state_func=get_state,
              set_state_func=set_state,
              pre_epoch_callback=epoch_track,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)

    assert current_state[0] == 12345


def test_validate():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        return [float(epoch_track.epoch)]

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_batch, num_epochs=200,
          eval_batch_func=eval_batch, pre_epoch_callback=epoch_track,
          log_stream=log, verbosity=VERBOSITY_NONE, log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_batch.count == 400
    # Called 200 times - 1x per epoch
    assert eval_batch.count == 200
    assert log.getvalue() == ''


def test_validate_with_store_state():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        if epoch_track.epoch <= 100:
            return [float(100 - epoch_track.epoch)]
        else:
            return [float(epoch_track.epoch - 99)]

    @count_calls
    def get_state():
        if epoch_track.epoch is None:
            return -12345
        else:
            return range(1000, 2005, 5)[epoch_track.epoch]

    current_state = [None]

    @count_calls
    def set_state(state):
        current_state[0] = state

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_batch, num_epochs=200,
          eval_batch_func=eval_batch,
          pre_epoch_callback=epoch_track,
          get_state_func=get_state,
          set_state_func=set_state,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_batch.count == 400
    assert eval_batch.count == 200
    assert log.getvalue() == ''
    # Called once at the start then 100 times (each epoch for 1st 100)
    assert get_state.count == 101
    # Called once at the end to restore the state
    assert set_state.count == 1
    # The state comes from the training function which should give a value
    # of 2005
    assert current_state[0] == 1500


def test_validate_with_store_layer_state():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne.layers import InputLayer, DenseLayer

    l_in = InputLayer(shape=(None, 5))
    l_out = DenseLayer(l_in, num_units=5)

    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        if epoch_track.epoch <= 100:
            return [float(100 - epoch_track.epoch)]
        else:
            return [float(epoch_track.epoch - 99)]

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_batch, num_epochs=200,
          eval_batch_func=eval_batch,
          pre_epoch_callback=epoch_track, layer_to_restore=l_out,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_batch.count == 400
    assert eval_batch.count == 200
    assert log.getvalue() == ''


def test_validate_with_store_layer_state_with_updates():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne import layers
    from lasagne.objectives import squared_error
    from lasagne.updates import adam

    l_in_x = layers.InputLayer(shape=(None, 5))
    l_in_y = layers.InputLayer(shape=(None, 5))
    l_out = layers.DenseLayer(l_in_x, num_units=5)
    loss = (squared_error(layers.get_output(l_out),
                          layers.get_output(l_in_y))).mean()
    updates = adam(loss, layers.get_all_params(l_out))

    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        if epoch_track.epoch <= 100:
            return [float(100 - epoch_track.epoch)]
        else:
            return [float(epoch_track.epoch - 99)]

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_batch, num_epochs=200,
          eval_batch_func=eval_batch,
          pre_epoch_callback=epoch_track, layer_to_restore=l_out,
          updates_to_restore=updates,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_batch.count == 400
    assert eval_batch.count == 200
    assert log.getvalue() == ''


def test_store_state_updates_without_layer():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne import layers
    from lasagne.objectives import squared_error
    from lasagne.updates import adam

    l_in_x = layers.InputLayer(shape=(None, 5))
    l_in_y = layers.InputLayer(shape=(None, 5))
    l_out = layers.DenseLayer(l_in_x, num_units=5)
    loss = (squared_error(layers.get_output(l_out),
                          layers.get_output(l_in_y))).mean()
    updates = adam(loss, layers.get_all_params(l_out))

    log = six.moves.cStringIO()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    with pytest.raises(ValueError):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              updates_to_restore=updates,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_restore_layer_incomplete_getstate_setstate():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne import layers

    l_in_x = layers.InputLayer(shape=(None, 5))
    l_out = layers.DenseLayer(l_in_x, num_units=5)

    log = six.moves.cStringIO()

    def train_batch(*batch):
        return [0.0]

    def get_state():
        # should never be invoked
        pytest.fail()

    def set_state():
        # should never be invoked
        pytest.fail()

    train([np.arange(10)], None, None, batchsize=5,
          train_batch_func=train_batch, num_epochs=200,
          layer_to_restore=l_out,
          get_state_func=get_state,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    train([np.arange(10)], None, None, batchsize=5,
          train_batch_func=train_batch, num_epochs=200,
          layer_to_restore=l_out,
          set_state_func=set_state,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)


def test_incomplete_getstate_setstate():
    from lasagne.trainer import train, VERBOSITY_NONE

    log = six.moves.cStringIO()

    def train_batch(*batch):
        return [0.0]

    def get_state():
        # should never be invoked
        pytest.fail()

    def set_state():
        # should never be invoked
        pytest.fail()

    with pytest.raises(ValueError):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              get_state_func=get_state,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)

    with pytest.raises(ValueError):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              set_state_func=set_state,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_validate_with_store_layer_state_with_updates_list():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne import layers
    from lasagne.objectives import squared_error
    from lasagne.updates import adam

    l_in_x = layers.InputLayer(shape=(None, 5))
    l_in_y = layers.InputLayer(shape=(None, 5))
    l_out = layers.DenseLayer(l_in_x, num_units=5)
    loss = (squared_error(layers.get_output(l_out),
                          layers.get_output(l_in_y))).mean()
    updates = adam(loss, layers.get_all_params(l_out))
    updates_list = [(p, v) for p, v in updates.items()]

    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        if epoch_track.epoch <= 100:
            return [float(100 - epoch_track.epoch)]
        else:
            return [float(epoch_track.epoch - 99)]

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_batch, num_epochs=200,
          eval_batch_func=eval_batch,
          pre_epoch_callback=epoch_track, layer_to_restore=l_out,
          updates_to_restore=updates_list,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_batch.count == 400
    assert eval_batch.count == 200
    assert log.getvalue() == ''


def test_validate_with_store_layer_state_with_updates_invalid():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne import layers

    l_in_x = layers.InputLayer(shape=(None, 5))
    l_out = layers.DenseLayer(l_in_x, num_units=5)
    updates = 'Strings are not valid update types'

    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        if epoch_track.epoch <= 100:
            return [float(100 - epoch_track.epoch)]
        else:
            return [float(epoch_track.epoch - 99)]

    with pytest.raises(TypeError):
        train([np.arange(10)], [np.arange(5)], None, batchsize=5,
              train_batch_func=train_batch, num_epochs=200,
              eval_batch_func=eval_batch,
              pre_epoch_callback=epoch_track, layer_to_restore=l_out,
              updates_to_restore=updates,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_validation_interval():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        assert epoch_track.epoch % 10 == 0
        if epoch_track.epoch <= 100:
            return [float(100 - epoch_track.epoch)]
        else:
            return [float(epoch_track.epoch - 99)]

    train([np.arange(5)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_batch, num_epochs=200,
          val_interval=10, eval_batch_func=eval_batch,
          pre_epoch_callback=epoch_track,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    assert train_batch.count == 200
    assert eval_batch.count == 20
    assert log.getvalue() == ''


def test_validation_score_fn():
    from lasagne.trainer import train, VERBOSITY_NONE

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        n = batch[0].shape[0]
        if epoch_track.epoch <= 100:
            return [float(epoch_track.epoch) * n,
                    float(100 - epoch_track.epoch) * n]
        else:
            return [float(epoch_track.epoch) * n,
                    float(epoch_track.epoch - 100) * n]

    log = six.moves.cStringIO()

    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_batch, num_epochs=200,
                eval_batch_func=eval_batch,
                val_improved_func=lambda a, b: a[1] < b[1],
                pre_epoch_callback=epoch_track,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    assert train_batch.count == 200
    assert eval_batch.count == 200
    assert log.getvalue() == ''

    assert res.validation_results[:101] == \
        [[float(i), float(100-i)] for i in range(101)]
    assert res.best_validation_results == [100.0, 0.0]


def test_pre_post_epoch_callbacks():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    epoch_track = EpochTracker()

    def train_batch(*batch):
        return [float(epoch_track.epoch) * batch[0].shape[0]]

    def post_epoch(epoch_index, train_results, val_results):
        assert epoch_index == epoch_track.epoch
        assert train_results[0] == float(epoch_index)

    train([np.arange(5)], None, None, batchsize=5,
          train_batch_func=train_batch, num_epochs=150,
          pre_epoch_callback=epoch_track,
          post_epoch_callback=post_epoch,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    def eval_batch(*batch):
        if epoch_track.epoch <= 75:
            return [float(75 - epoch_track.epoch) * batch[0].shape[0]]
        else:
            return [float(epoch_track.epoch - 75) * batch[0].shape[0]]

    def post_epoch_val(epoch_index, train_results, val_results):
        assert epoch_index == epoch_track.epoch
        assert train_results[0] == float(epoch_index)
        if epoch_track.epoch <= 75:
            assert val_results[0] == float(75 - epoch_track.epoch)
        else:
            assert val_results[0] == float(epoch_track.epoch - 75)

    train([np.arange(5)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_batch, eval_batch_func=eval_batch,
          num_epochs=150,
          pre_epoch_callback=epoch_track,
          post_epoch_callback=post_epoch_val,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)


def test_train_for_num_epochs():
    from lasagne.trainer import train, VERBOSITY_NONE

    epoch_track = EpochTracker()

    @count_calls
    def train_batch(*batch):
        return [0.0]

    @count_calls
    def eval_batch(*batch):
        if epoch_track.epoch <= 100:
            return [float(100 - epoch_track.epoch) * batch[0].shape[0]]
        else:
            return [float(epoch_track.epoch - 100) * batch[0].shape[0]]

    val_output = zip(range(200), itertools.chain(range(101, 1, -1),
                                                 range(1, 101, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_fn, num_epochs=150,
                eval_batch_func=eval_fn,
                val_improved_func=lambda a, b: a[1] < b[1],
                pre_epoch_callback=pre_epoch,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    assert train_fn.count == 150
    assert eval_fn.count == 150
    assert log.getvalue() == ''

    assert res.validation_results == val_output[:150]
    assert res.best_validation_results == [100, 1]


def test_train_for_min_epochs():
    from lasagne.trainer import train, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(101, 1, -1),
                                                 range(1, 101, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                min_epochs=95, eval_batch_func=eval_fn,
                val_improved_func=lambda a, b: a[1] < b[1],
                pre_epoch_callback=pre_epoch,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    assert train_fn.count == 102
    assert eval_fn.count == 102
    assert log.getvalue() == ''

    assert res.validation_results == val_output[:102]
    assert res.best_validation_results == [100, 1]


def test_train_for_val_improve_patience():
    from lasagne.trainer import train, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(101, 1, -1),
                                                 range(1, 101, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                min_epochs=95, val_improve_patience=10,
                eval_batch_func=eval_fn,
                val_improved_func=lambda a, b: a[1] < b[1],
                pre_epoch_callback=pre_epoch,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    assert train_fn.count == 111
    assert eval_fn.count == 111
    assert log.getvalue() == ''

    assert res.validation_results == val_output[:111]
    assert res.best_validation_results == [100, 1]


def test_train_for_val_improve_patience_factor():
    from lasagne.trainer import train, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(75, 0, -1),
                                                 range(0, 125, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                min_epochs=65, val_improve_patience_factor=2,
                eval_batch_func=eval_fn,
                val_improved_func=lambda a, b: a[1] < b[1],
                pre_epoch_callback=pre_epoch,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    assert train_fn.count == 152
    assert eval_fn.count == 152
    assert log.getvalue() == ''

    assert res.validation_results == val_output[:152]
    assert res.best_validation_results == [75, 0]


def test_report_verbosity_none():
    from lasagne.trainer import train, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(75, 0, -1),
                                                 range(0, 125, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)

    train([np.arange(5)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          min_epochs=65, val_improve_patience_factor=2,
          eval_batch_func=eval_fn,
          val_improved_func=lambda a, b: a[1] < b[1],
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    assert log.getvalue() == ''


def test_report_verbosity_minimal():
    from lasagne.trainer import train, VERBOSITY_MINIMAL
    val_output = zip(range(200), itertools.chain(range(75, 0, -1),
                                                 range(0, 125, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                min_epochs=65, val_improve_patience_factor=2,
                eval_batch_func=eval_fn,
                val_improved_func=lambda a, b: a[1] < b[1],
                pre_epoch_callback=pre_epoch,
                log_stream=log, verbosity=VERBOSITY_MINIMAL,
                log_final_result=False)

    assert train_fn.count == 152
    assert eval_fn.count == 152
    assert log.getvalue() == '*' * 76 + '-' * 76


def test_report_verbosity_minimal_log_final_result():
    from lasagne.trainer import train, VERBOSITY_MINIMAL
    val_output = zip(range(200), itertools.chain(range(75, 0, -1),
                                                 range(0, 125, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                min_epochs=65, val_improve_patience_factor=2,
                eval_batch_func=eval_fn,
                val_improved_func=lambda a, b: a[1] < b[1],
                pre_epoch_callback=pre_epoch,
                log_stream=log, verbosity=VERBOSITY_MINIMAL,
                log_final_result=True)

    log_lines = log.getvalue().split('\n')
    log_lines = [line for line in log_lines if line.strip() != '']
    assert log_lines[0] == '*' * 76 + '-' * 76
    assert log_lines[1] == 'Best result:'
    assert log_lines[2].startswith('Epoch 75')
    assert log_lines[3] == 'Final result:'
    assert log_lines[4].startswith('Epoch 151')

    # This time store the network state; this will cause this test to
    # exercise an alternate path for coverage purposes
    log = six.moves.cStringIO()
    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                min_epochs=65, val_improve_patience_factor=2,
                eval_batch_func=eval_fn,
                val_improved_func=lambda a, b: a[1] < b[1],
                get_state_func=train_fn.get_state,
                set_state_func=train_fn.set_state,
                pre_epoch_callback=pre_epoch,
                log_stream=log, verbosity=VERBOSITY_MINIMAL,
                log_final_result=True)

    log_lines = log.getvalue().split('\n')
    log_lines = [line for line in log_lines if line.strip() != '']
    assert log_lines[0] == '*' * 76 + '-' * 76
    assert log_lines[1] == 'Best result:'
    assert log_lines[2].startswith('Epoch 75')

    # Ensure that the last epoch has the best score
    val2_output = [[x] for x in range(200, -1, -1)]
    eval2_fn = TrainFunction(val2_output)
    pre_epoch2 = TrainFunction.pre_epoch_for(train_fn, eval2_fn)

    log = six.moves.cStringIO()
    res = train([np.arange(5)], [np.arange(5)], None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                min_epochs=65,
                eval_batch_func=eval2_fn,
                val_improved_func=lambda a, b: a[0] < b[0],
                pre_epoch_callback=pre_epoch2,
                log_stream=log, verbosity=VERBOSITY_MINIMAL,
                log_final_result=True)

    log_lines = log.getvalue().split('\n')
    log_lines = [line for line in log_lines if line.strip() != '']
    assert log_lines[0] == '*' * 200
    assert log_lines[1] == 'Best result:'
    assert log_lines[2].startswith('Epoch 199')


def test_report_verbosity_epoch_train():
    from lasagne.trainer import train, VERBOSITY_EPOCH
    val_output = zip(np.arange(200.0),
                     np.append(np.arange(75.0, 0.0, -1.0),
                               np.arange(0.0, 125.0, 1.0)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction(val_output)

    train([np.arange(5)], None, None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          pre_epoch_callback=train_fn.pre_epoch,
          log_stream=log, verbosity=VERBOSITY_EPOCH,
          log_final_result=False)

    assert train_fn.count == 200
    log_lines = log.getvalue().split('\n')
    for i, line in enumerate(log_lines):
        if line.strip() != '':
            pattern = re.escape('Epoch {0} ('.format(i)) + \
                      r'[0-9]+\.[0-9]+s' + \
                      re.escape('): train [{0}, {1}]'.format(val_output[i][0],
                                                             val_output[i][1]))
            match = re.match(pattern, line)
            if match is None or match.end(0) != len(line):
                pytest.fail(msg='No match "{}" with pattern '
                                '"{}"'.format(line, pattern))


def test_report_verbosity_epoch_train_val():
    from lasagne.trainer import train, VERBOSITY_EPOCH
    val_output = zip(np.arange(200.0),
                     np.append(np.arange(75.0, 0.0, -1.0),
                               np.arange(0.0, 125.0, 1.0)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction(val_output)
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    train([np.arange(5)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          min_epochs=65, val_improve_patience_factor=2,
          eval_batch_func=eval_fn,
          val_improved_func=lambda a, b: a[1] < b[1],
          pre_epoch_callback=pre_epoch,
          log_stream=log, verbosity=VERBOSITY_EPOCH,
          log_final_result=False)

    assert train_fn.count == 152
    assert eval_fn.count == 152
    log_lines = log.getvalue().split('\n')
    for i, line in enumerate(log_lines):
        if line.strip() != '':
            pattern = re.escape('Epoch {0} ('.format(i)) + \
                      r'[0-9]+\.[0-9]+s' + \
                      re.escape('): train [{0}, {1}], validation [{0}, '
                                '{1}]'.format(val_output[i][0],
                                              val_output[i][1]))
            match = re.match(pattern, line)
            if match is None or match.end(0) != len(line):
                pytest.fail(msg='No match "{}" with pattern '
                                '"{}"'.format(line, pattern))


def test_report_verbosity_epoch_train_test():
    from lasagne.trainer import train, VERBOSITY_EPOCH
    val_output = zip(np.arange(200.0),
                     np.append(np.arange(75.0, 0.0, -1.0),
                               np.arange(0.0, 125.0, 1.0)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction(val_output)
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    train([np.arange(5)], None, [np.arange(5)], batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          eval_batch_func=eval_fn,
          val_improved_func=lambda a, b: a[1] < b[1],
          pre_epoch_callback=pre_epoch,
          log_stream=log, verbosity=VERBOSITY_EPOCH,
          log_final_result=False)

    assert train_fn.count == 200
    assert eval_fn.count == 200
    log_lines = log.getvalue().split('\n')
    for i, line in enumerate(log_lines):
        if line.strip() != '':
            pattern = re.escape('Epoch {0} ('.format(i)) + \
                      r'[0-9]+\.[0-9]+s' + \
                      re.escape('): train [{0}, {1}], test [{0}, '
                                '{1}]'.format(val_output[i][0],
                                              val_output[i][1]))
            match = re.match(pattern, line)
            if match is None or match.end(0) != len(line):
                pytest.fail(msg='No match "{}" with pattern '
                                '"{}"'.format(line, pattern))


def test_report_verbosity_epoch_train_val_test():
    from lasagne.trainer import train, VERBOSITY_EPOCH
    output = zip(np.arange(200.0),
                 np.append(np.arange(75.0, 0.0, -1.0),
                           np.arange(0.0, 125.0, 1.0)))
    output = [list(xs) for xs in output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction(output)
    eval_fn = TrainFunction(output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    train([np.arange(5)], [np.arange(5)], [np.arange(5)], batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          min_epochs=65, val_improve_patience_factor=2,
          eval_batch_func=eval_fn,
          val_improved_func=lambda a, b: a[1] < b[1],
          pre_epoch_callback=pre_epoch,
          log_stream=log, verbosity=VERBOSITY_EPOCH,
          log_final_result=False)

    log_lines = log.getvalue().split('\n')
    for i, line in enumerate(log_lines):
        if line.strip() != '':
            if i <= 75:
                pattern = re.escape('Epoch {0} ('.format(i)) + \
                    r'[0-9]+\.[0-9]+s' + \
                    re.escape('): train [{0}, {1}], validation [{0}, '
                              '{1}], test [{0}, {1}]'.format(output[i][0],
                                                             output[i][1]))
            else:
                pattern = re.escape('Epoch {0} ('.format(i)) + \
                          r'[0-9]+\.[0-9]+s' + \
                          re.escape('): train [{0}, {1}], validation [{0}, '
                                    '{1}]'.format(output[i][0], output[i][1]))
            match = re.match(pattern, line)
            if match is None or match.end(0) != len(line):
                pytest.fail(msg='No match "{}" with pattern '
                                '"{}"'.format(line, pattern))


def test_progress_iter_func_batch_size_divide_exactly():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    epoch_track = EpochTracker()

    def train_batch(batch_X):
        return [0.0]

    totals = []
    descs = []

    def progress_iter_func(it, total, desc, leave):
        totals.append(total)
        descs.append(desc)
        assert not leave
        return it

    # Test situation where the batch size divides into the training set size
    # exactly
    train([np.arange(20)], None, None,
          batchsize=5, train_batch_func=train_batch, num_epochs=200,
          pre_epoch_callback=epoch_track,
          progress_iter_func=progress_iter_func,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # 20 samples, batch size of 5 = 4 batches, so we expect the `total`
    # parameter to have a value of 4 each time
    assert totals == [4] * 200
    assert descs == ['Epoch {} train'.format(i) for i in range(1, 201)]


def test_progress_iter_func_batch_size_remainder():
    # Test situation where the batch size doesn't divide into the training
    # set size exactly
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    epoch_track = EpochTracker()

    def train_batch(batch_X):
        return [0.0]

    totals = []
    descs = []

    def progress_iter_func(it, total, desc, leave):
        totals.append(total)
        descs.append(desc)
        assert not leave
        return it

    train([np.arange(20)], None, None,
          batchsize=7, train_batch_func=train_batch, num_epochs=200,
          pre_epoch_callback=epoch_track,
          progress_iter_func=progress_iter_func,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # 20 samples, batch size of 7 = 3 batches, so we expect the `total`
    # parameter to have a value of 3 each time
    assert totals == [3] * 200
    assert descs == ['Epoch {} train'.format(i) for i in range(1, 201)]


def test_progress_iter_func_unknown_size():
    # Test situation where the training set is provided as a callable
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    epoch_track = EpochTracker()

    def train_batch(batch_X):
        return [0.0]

    totals = []
    descs = []

    def progress_iter_func(it, total, desc, leave):
        totals.append(total)
        descs.append(desc)
        assert not leave
        return it

    def training_set(batchsize, shuffle_rng=None):
        for i in range(0, 20, batchsize):
            yield [np.arange(20)[i:i + batchsize]]

    train(training_set, None, None,
          batchsize=7, train_batch_func=train_batch, num_epochs=200,
          pre_epoch_callback=epoch_track,
          progress_iter_func=progress_iter_func,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Getting data from callable that returns an iterator, we we cannot
    # determine the number of batches ahead of time, so we expect the `total`
    # parameter to have a value of None each time
    assert totals == [None] * 200
    assert descs == ['Epoch {} train'.format(i) for i in range(1, 201)]


def test_progress_iter_func_val_test():
    # Test situation training, validation and test sets are provided
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    epoch_track = EpochTracker()

    val_results = list(range(200, 0, -1))

    def train_batch(batch_X):
        return [0.0]

    def eval_batch(batch_X):
        res = [float(val_results[epoch_track.epoch])]
        return res

    totals = []
    descs = []

    def progress_iter_func(it, total, desc, leave):
        totals.append(total)
        descs.append(desc)
        assert not leave
        return it

    train([np.arange(20)], [np.arange(20)], [np.arange(20)],
          batchsize=5, train_batch_func=train_batch, num_epochs=200,
          eval_batch_func=eval_batch,
          val_improved_func=lambda a, b: a[0] < b[0],
          pre_epoch_callback=epoch_track,
          progress_iter_func=progress_iter_func,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # 20 samples, batch size of 5 = 4 batches, so we expect the `total`
    # parameter to have a value of 4 each time
    assert totals == [4] * 600
    assert len(descs) == 600
    for i in range(200):
        assert descs[i * 3 + 0] == 'Epoch {} train'.format(i + 1)
        assert descs[i * 3 + 1] == 'Epoch {} val'.format(i + 1)
        assert descs[i * 3 + 2] == 'Epoch {} test'.format(i + 1)


def test_progress_iter_func_test():
    # Progress tracking with train and test set, no validation set
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    epoch_track = EpochTracker()

    val_results = list(range(200, 0, -1))

    def train_batch(batch_X):
        return [0.0]

    def eval_batch(batch_X):
        res = [float(val_results[epoch_track.epoch])]
        return res

    totals = []
    descs = []

    def progress_iter_func(it, total, desc, leave):
        totals.append(total)
        descs.append(desc)
        assert not leave
        return it

    train([np.arange(20)], None, [np.arange(20)],
          batchsize=5, train_batch_func=train_batch, num_epochs=200,
          eval_batch_func=eval_batch,
          val_improved_func=lambda a, b: a[0] < b[0],
          pre_epoch_callback=epoch_track,
          progress_iter_func=progress_iter_func,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # 20 samples, batch size of 5 = 4 batches, so we expect the `total`
    # parameter to have a value of 4 each time
    assert totals == [4] * 400
    assert len(descs) == 400
    for i in range(200):
        assert descs[i * 2 + 0] == 'Epoch {} train'.format(i + 1)
        assert descs[i * 2 + 1] == 'Epoch {} test'.format(i + 1)


def test_report_epoch_log_fn():
    from lasagne.trainer import train, VERBOSITY_EPOCH
    train_output = [[float(x)] for x in range(200)]
    val_output = zip(np.arange(200.0),
                     np.arange(200.0, -1.0, -1.0))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction(train_output)
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    def train_log(train_results):
        return '{}'.format(train_results[0])

    def eval_log(val_results):
        return '{} {}'.format(val_results[0], val_results[1])

    def epoch_log_fn(epoch_index, delta_time, train_str, val_str, test_str):
        return '{}: train: {}, val: {}, test: {}'.format(
            epoch_index, train_str, val_str, test_str)

    train([np.arange(5)], [np.arange(5)], [np.arange(5)], batchsize=5,
          train_batch_func=train_fn, train_log_func=train_log,
          num_epochs=200, min_epochs=65,
          val_improve_patience_factor=2,
          eval_batch_func=eval_fn, eval_log_func=eval_log,
          val_improved_func=lambda a, b: a[1] < b[1],
          pre_epoch_callback=pre_epoch,
          log_stream=log, verbosity=VERBOSITY_EPOCH,
          epoch_log_func=epoch_log_fn, log_final_result=False)

    assert train_fn.count == 200
    assert eval_fn.count == 400
    log_lines = log.getvalue().split('\n')
    for i, line in enumerate(log_lines):
        if line.strip() != '':
            assert '{0}: train: {1}, val: {2} {3}, test: {2} {3}'.format(
                i, train_output[i][0], val_output[i][0],
                val_output[i][1]) == line
