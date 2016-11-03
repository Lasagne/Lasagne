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


def test_no_train_fn():
    from lasagne.trainer import train

    with pytest.raises(ValueError):
        train([np.arange(10)], [np.arange(10)], None, batchsize=5)


def test_no_eval_fn():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction()

    with pytest.raises(ValueError):
        train([np.arange(10)], [np.arange(10)], None, batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              log_stream=log, verbosity=VERBOSITY_NONE, log_final_result=False)
    with pytest.raises(ValueError):
        train([np.arange(10)], None, [np.arange(10)], batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_train():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction()

    res = train([np.arange(10)], None, None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                get_state_func=train_fn.get_state,
                set_state_func=train_fn.set_state,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    # Logging is disable so it should be empty
    assert log.getvalue() == ''
    # No evaluation function or validation set so the get and set state
    # functions shouldn't be invoked
    assert train_fn.state_get_count == 0
    assert train_fn.state_set_count == 0
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
    # Tell the training function to expect the epoch index
    train_fn = TrainFunction(epoch_index_prepended=True)

    res = train([np.arange(10)], None, None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                get_state_func=train_fn.get_state,
                set_state_func=train_fn.set_state,
                pre_epoch_callback=train_fn.pre_epoch,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False,
                train_pass_epoch_number=True)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    # Logging is disable so it should be empty
    assert log.getvalue() == ''
    # No evaluation function or validation set so the get and set state
    # functions shouldn't be invoked
    assert train_fn.state_get_count == 0
    assert train_fn.state_set_count == 0
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
    results = [[i] for i in range(200)]
    # Tell the training function to expect the epoch index
    train_fn = TrainFunction(results, return_array=True)

    res = train([np.arange(10)], None, None, batchsize=5,
                train_batch_func=train_fn, num_epochs=200,
                get_state_func=train_fn.get_state,
                set_state_func=train_fn.set_state,
                pre_epoch_callback=train_fn.pre_epoch,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    assert res.train_results == results


def test_train_return_invalid():
    # Check that TypeError is raised when the training function returns
    # an invalid type
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()

    def train_fn(*args, **kwargs):
        return 'Strings wont work here'

    with pytest.raises(TypeError):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_training_failure():
    # Check that TypeError is raised when the training function returns
    # an invalid type
    from lasagne.trainer import train, TrainingFailedException, \
        VERBOSITY_MINIMAL
    log = six.moves.cStringIO()
    results = [[float('nan') if i == 100 else i] for i in range(200)]
    # Tell the training function to expect the epoch index
    train_fn = TrainFunction(results)

    def check_train_res(epoch, train_results):
        if np.isnan(train_results[0]).any():
            return 'NaN detected'
        else:
            return None

    with pytest.raises(TrainingFailedException):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              train_epoch_results_check_func=check_train_res,
              pre_epoch_callback=train_fn.pre_epoch,
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
    results = [[float('nan') if i == 100 else i] for i in range(200)]
    # Tell the training function to expect the epoch index
    train_fn = TrainFunction(results, states=np.arange(1000, 2005, 5))

    # Initialise state to 12345 so that this can be restored on training
    # failure
    train_fn.current_state = 12345

    def check_train_res(epoch, train_results):
        if np.isnan(train_results[0]).any():
            return 'NaN detected'
        else:
            return None

    with pytest.raises(TrainingFailedException):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              train_epoch_results_check_func=check_train_res,
              get_state_func=train_fn.get_state,
              set_state_func=train_fn.set_state,
              pre_epoch_callback=train_fn.pre_epoch,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)

    assert train_fn.current_state == 12345


def test_validate():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    val_out = [[i] for i in range(200)]
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_out)

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          eval_batch_func=eval_fn, log_stream=log,
          verbosity=VERBOSITY_NONE, log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    # Called 200 times - 1x per epoch
    assert eval_fn.count == 200
    assert log.getvalue() == ''
    # State is not stored
    assert train_fn.state_get_count == 0
    assert train_fn.state_set_count == 0


def test_validate_with_store_state():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction(states=np.arange(1000, 2005, 5))
    eval_fn = TrainFunction([[i] for i in range(200, -1, -2)] +
                            [[i] for i in range(0, 200, 2)])
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          eval_batch_func=eval_fn,
          pre_epoch_callback=pre_epoch,
          get_state_func=train_fn.get_state,
          set_state_func=train_fn.set_state,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    assert eval_fn.count == 200
    assert log.getvalue() == ''
    # Called once at the start then 100 times (each epoch for 1st 100)
    assert train_fn.state_get_count == 101
    # Called once at the end to restore the state
    assert train_fn.state_set_count == 1
    # The state comes from the training function which should give a value
    # of 2005
    assert train_fn.current_state == 1500


def test_validate_with_store_layer_state():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne.layers import InputLayer, DenseLayer

    l_in = InputLayer(shape=(None, 5))
    l_out = DenseLayer(l_in, num_units=5)

    log = six.moves.cStringIO()
    train_fn = TrainFunction(states=np.arange(1000, 3005, 5))
    eval_fn = TrainFunction([[i] for i in range(200, -1, -2)] +
                            [[i] for i in range(0, 200, 2)])

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          eval_batch_func=eval_fn,
          layer_to_restore=l_out,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    assert eval_fn.count == 200
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
    train_fn = TrainFunction(states=np.arange(1000, 3005, 5))
    eval_fn = TrainFunction([[i] for i in range(200, -1, -2)] +
                            [[i] for i in range(0, 200, 2)])

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          eval_batch_func=eval_fn,
          layer_to_restore=l_out,
          updates_to_restore=updates,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    assert eval_fn.count == 200
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
    train_fn = TrainFunction(states=np.arange(1000, 3005, 5))

    with pytest.raises(ValueError):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              updates_to_restore=updates,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_restore_layer_incomplete_getstate_setstate():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne import layers

    l_in_x = layers.InputLayer(shape=(None, 5))
    l_in_y = layers.InputLayer(shape=(None, 5))
    l_out = layers.DenseLayer(l_in_x, num_units=5)

    log = six.moves.cStringIO()
    train_fn = TrainFunction(states=np.arange(1000, 3005, 5))

    train([np.arange(10)], None, None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          layer_to_restore=l_out,
          get_state_func=train_fn.get_state,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    train([np.arange(10)], None, None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          layer_to_restore=l_out,
          set_state_func=train_fn.set_state,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)


def test_incomplete_getstate_setstate():
    from lasagne.trainer import train, VERBOSITY_NONE

    log = six.moves.cStringIO()
    train_fn = TrainFunction(states=np.arange(1000, 3005, 5))

    with pytest.raises(ValueError):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              get_state_func=train_fn.get_state,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)

    with pytest.raises(ValueError):
        train([np.arange(10)], None, None, batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              set_state_func=train_fn.set_state,
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
    train_fn = TrainFunction(states=np.arange(1000, 3005, 5))
    eval_fn = TrainFunction([[i] for i in range(200, -1, -2)] +
                            [[i] for i in range(0, 200, 2)])

    train([np.arange(10)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          eval_batch_func=eval_fn,
          layer_to_restore=l_out,
          updates_to_restore=updates_list,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    assert eval_fn.count == 200
    assert log.getvalue() == ''


def test_validate_with_store_layer_state_with_updates_invalid():
    from lasagne.trainer import train, VERBOSITY_NONE
    from lasagne import layers

    l_in_x = layers.InputLayer(shape=(None, 5))
    l_in_y = layers.InputLayer(shape=(None, 5))
    l_out = layers.DenseLayer(l_in_x, num_units=5)
    updates = 'Strings are not valid update types'

    log = six.moves.cStringIO()
    train_fn = TrainFunction(states=np.arange(1000, 3005, 5))
    eval_fn = TrainFunction([[i] for i in range(200, -1, -2)] +
                            [[i] for i in range(0, 200, 2)])

    with pytest.raises(TypeError):
        train([np.arange(10)], [np.arange(5)], None, batchsize=5,
              train_batch_func=train_fn, num_epochs=200,
              eval_batch_func=eval_fn,
              layer_to_restore=l_out,
              updates_to_restore=updates,
              log_stream=log, verbosity=VERBOSITY_NONE,
              log_final_result=False)


def test_validation_interval():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction()

    def on_eval():
        assert train_fn.count in range(1, 201, 10)

    eval_fn = TrainFunction([[i] for i in range(200)],
                            on_invoke=on_eval)

    train([np.arange(5)], [np.arange(5)], None, batchsize=5,
          train_batch_func=train_fn, num_epochs=200,
          val_interval=10, eval_batch_func=eval_fn,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)

    assert train_fn.count == 200
    assert eval_fn.count == 20
    assert log.getvalue() == ''


def test_validation_score_fn():
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
                eval_batch_func=eval_fn,
                val_improved_func=lambda a, b: a[1] < b[1],
                pre_epoch_callback=pre_epoch,
                log_stream=log, verbosity=VERBOSITY_NONE,
                log_final_result=False)

    assert train_fn.count == 200
    assert eval_fn.count == 200
    assert log.getvalue() == ''

    assert res.validation_results == val_output
    assert res.best_validation_results == [100, 1]


def test_pre_post_epoch_callbacks():
    from lasagne.trainer import train, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction()

    def post_epoch(epoch_index):
        assert epoch_index == train_fn.current_epoch

    train([np.arange(5)], None, None, batchsize=5,
          train_batch_func=train_fn, num_epochs=150,
          pre_epoch_callback=train_fn.pre_epoch,
          post_epoch_callback=post_epoch,
          log_stream=log, verbosity=VERBOSITY_NONE,
          log_final_result=False)


def test_train_for_num_epochs():
    from lasagne.trainer import train, VERBOSITY_NONE
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


def test_report_verbosity_batch():
    from lasagne.trainer import train, VERBOSITY_BATCH
    val_output = zip(np.arange(200.0),
                     np.arange(200.0, -1.0, -1.0))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)
    pre_epoch = TrainFunction.pre_epoch_for(train_fn, eval_fn)

    train([np.arange(20)], [np.arange(20)], [np.arange(20)],
          batchsize=5, train_batch_func=train_fn, num_epochs=200,
          eval_batch_func=eval_fn,
          val_improved_func=lambda a, b: a[1] < b[1],
          pre_epoch_callback=pre_epoch,
          log_stream=log, verbosity=VERBOSITY_BATCH,
          log_final_result=False)

    assert train_fn.count == 800
    assert eval_fn.count == 1600
    log_lines = log.getvalue().split('\n')
    for i, line in enumerate(log_lines):
        if line.strip() != '':
            tr_report = ''.join('\r[train {}]'.format(i) for i in range(4))
            val_report = ''.join('\r[val {}]'.format(i) for i in range(4))
            test_report = ''.join('\r[test {}]'.format(i) for i in range(4))
            assert line.startswith(tr_report)
            val_test_epoch = line[len(tr_report) + 1:]
            assert val_test_epoch.startswith(val_report)
            test_epoch = val_test_epoch[len(val_report) + 1:]
            assert test_epoch.startswith(test_report)
            epoch_report = test_epoch[len(test_report) + 1:]
            pattern_b = re.escape('Epoch {0} ('.format(i)) + \
                r'[0-9]+\.[0-9]+s' + \
                re.escape('): train None, validation [{0}, {1}], '
                          'test [{0}, {1}]'.format(val_output[i][0],
                                                   val_output[i][1]))
            match = re.match(pattern_b, epoch_report)
            if match is None or match.end(0) != len(epoch_report):
                pytest.fail(msg='No match "{}" with pattern '
                                '"{}"'.format(repr(epoch_report), pattern_b))


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
