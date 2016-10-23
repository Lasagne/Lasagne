import sys
import re
import itertools
import six
import numpy as np
import pytest


class TrainFunction (object):
    def __init__(self, results=None, states=None, on_invoke=None,
                 result_repeat=1):
        self.count = 0
        self.results = results
        self.states = states
        self.current_state = None
        self.state_get_count = 0
        self.state_set_count = 0
        self.on_invoke = on_invoke
        self.result_repeat = result_repeat

    def get_state(self):
        self.state_get_count += 1
        return self.current_state

    def set_state(self, state):
        self.state_set_count += 1
        self.current_state = state

    def __call__(self, *args, **kwargs):
        d0 = args[0]
        batch_N = d0.shape[0]
        i = self.count
        if self.on_invoke is not None:
            self.on_invoke()
        self.count += 1
        if self.states is not None:
            self.current_state = self.states[i]
        if self.results is None:
            return None
        else:
            return [r*batch_N for r in self.results[i//self.result_repeat]]


def test_no_train_fn():
    from lasagne.trainer import Trainer

    trainer = Trainer()

    with pytest.raises(ValueError):
        trainer.train([np.arange(10)], [np.arange(10)], None, batchsize=5)


def test_no_eval_fn():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction()

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    with pytest.raises(ValueError):
        trainer.train([np.arange(10)], [np.arange(10)], None, batchsize=5)
    with pytest.raises(ValueError):
        trainer.train([np.arange(10)], None, [np.arange(10)], batchsize=5)


def test_train():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction()

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      get_state_func=train_fn.get_state,
                      set_state_func=train_fn.set_state,
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    res = trainer.train([np.arange(10)], None, None, batchsize=5)

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
    assert res.final_test_results is None
    # Last epoch should be the end
    assert res.last_epoch == 200


def test_validate():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    log = six.moves.cStringIO()
    val_out = [[i] for i in range(200)]
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_out)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      eval_batch_func=eval_fn, log_stream=log,
                      verbosity=VERBOSITY_NONE, log_final_result=False)

    trainer.train([np.arange(10)], [np.arange(5)], None, batchsize=5)

    # Called 400 times - 2x per epoch for 200 epochs
    assert train_fn.count == 400
    # Called 200 times - 1x per epoch
    assert eval_fn.count == 200
    assert log.getvalue() == ''
    # State is not stored
    assert train_fn.state_get_count == 0
    assert train_fn.state_set_count == 0


def test_validate_with_store_state():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction(states=np.arange(1000, 3005, 5))
    eval_fn = TrainFunction([[i] for i in range(200, -1, -2)] +
                            [[i] for i in range(0, 200, 2)])

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      eval_batch_func=eval_fn,
                      get_state_func=train_fn.get_state,
                      set_state_func=train_fn.set_state,
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    trainer.train([np.arange(10)], [np.arange(5)], None, batchsize=5)

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
    assert train_fn.current_state == 2005


def test_validation_interval():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    log = six.moves.cStringIO()
    train_fn = TrainFunction()

    def on_eval():
        assert train_fn.count in range(1, 201, 10)

    eval_fn = TrainFunction([[i] for i in range(200)],
                            on_invoke=on_eval)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      val_interval=10, eval_batch_func=eval_fn,
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert train_fn.count == 200
    assert eval_fn.count == 20
    assert log.getvalue() == ''


def test_validation_score_fn():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(101, 1, -1),
                                                 range(1, 101, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    res = trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert train_fn.count == 200
    assert eval_fn.count == 200
    assert log.getvalue() == ''

    assert res.validation_results == val_output
    assert res.best_validation_results == [100, 1]


def test_train_for_num_epochs():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(101, 1, -1),
                                                 range(1, 101, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=150,
                      eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    res = trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert train_fn.count == 150
    assert eval_fn.count == 150
    assert log.getvalue() == ''

    assert res.validation_results == val_output[:150]
    assert res.best_validation_results == [100, 1]


def test_train_for_min_epochs():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(101, 1, -1),
                                                 range(1, 101, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      min_epochs=95, eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    res = trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert train_fn.count == 95
    assert eval_fn.count == 95
    assert log.getvalue() == ''

    assert res.validation_results == val_output[:95]
    assert res.best_validation_results == [94, 7]


def test_train_for_val_improve_patience():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(101, 1, -1),
                                                 range(1, 101, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      min_epochs=95, val_improve_patience=10,
                      eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    res = trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert train_fn.count == 111
    assert eval_fn.count == 111
    assert log.getvalue() == ''

    assert res.validation_results == val_output[:111]
    assert res.best_validation_results == [100, 1]


def test_train_for_val_improve_patience_factor():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(75, 0, -1),
                                                 range(0, 125, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      min_epochs=65, val_improve_patience_factor=2,
                      eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    res = trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert train_fn.count == 152
    assert eval_fn.count == 152
    assert log.getvalue() == ''

    assert res.validation_results == val_output[:152]
    assert res.best_validation_results == [75, 0]


def test_report_verbosity_none():
    from lasagne.trainer import Trainer, VERBOSITY_NONE
    val_output = zip(range(200), itertools.chain(range(75, 0, -1),
                                                 range(0, 125, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      min_epochs=65, val_improve_patience_factor=2,
                      eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_NONE,
                      log_final_result=False)

    trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert log.getvalue() == ''


def test_report_verbosity_minimal():
    from lasagne.trainer import Trainer, VERBOSITY_MINIMAL
    val_output = zip(range(200), itertools.chain(range(75, 0, -1),
                                                 range(0, 125, 1)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      min_epochs=65, val_improve_patience_factor=2,
                      eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_MINIMAL,
                      log_final_result=False)

    res = trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert train_fn.count == 152
    assert eval_fn.count == 152
    assert log.getvalue() == '*' * 76 + '-' * 76


def test_report_verbosity_epoch():
    from lasagne.trainer import Trainer, VERBOSITY_EPOCH
    val_output = zip(np.arange(200.0),
                     np.append(np.arange(75.0, 0.0, -1.0),
                               np.arange(0.0, 125.0, 1.0)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction(val_output)
    eval_fn = TrainFunction(val_output)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      min_epochs=65, val_improve_patience_factor=2,
                      eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_EPOCH,
                      log_final_result=False)

    trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

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


def test_report_verbosity_batch():
    from lasagne.trainer import Trainer, VERBOSITY_BATCH
    val_output = zip(np.arange(200.0),
                     np.append(np.arange(75.0, 0.0, -1.0),
                               np.arange(0.0, 125.0, 1.0)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction()
    eval_fn = TrainFunction(val_output, result_repeat=4)

    trainer = Trainer(train_batch_func=train_fn, num_epochs=200,
                      min_epochs=65, val_improve_patience_factor=2,
                      eval_batch_func=eval_fn,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_BATCH,
                      log_final_result=False)

    trainer.train([np.arange(20)], [np.arange(20)], None, batchsize=5)

    assert train_fn.count == 608
    assert eval_fn.count == 608
    log_lines = log.getvalue().split('\n')
    for i, line in enumerate(log_lines):
        if line.strip() != '':
            tr_report = ''.join('\r[train {}]'.format(i) for i in range(4))
            val_report = ''.join('\r[val {}]'.format(i) for i in range(4))
            assert line.startswith(tr_report)
            val_epoch = line[len(tr_report)+1:]
            assert val_epoch.startswith(val_report)
            epoch_report = val_epoch[len(val_report)+1:]
            pattern_b = re.escape('Epoch {0} ('.format(i)) + \
                r'[0-9]+\.[0-9]+s' + \
                re.escape('): train None, validation [{0}, '
                          '{1}]'.format(val_output[i][0],
                                        val_output[i][1]))
            match = re.match(pattern_b, epoch_report)
            if match is None or match.end(0) != len(epoch_report):
                pytest.fail(msg='No match "{}" with pattern '
                            '"{}"'.format(repr(epoch_report), pattern_b))


def test_report_epoch_log_fn():
    from lasagne.trainer import Trainer, VERBOSITY_EPOCH
    train_output = [[x] for x in range(200)]
    val_output = zip(np.arange(200.0),
                     np.append(np.arange(75.0, 0.0, -1.0),
                               np.arange(0.0, 125.0, 1.0)))
    val_output = [list(xs) for xs in val_output]
    log = six.moves.cStringIO()
    train_fn = TrainFunction(train_output)
    eval_fn = TrainFunction(val_output)

    def train_log(train_results):
        return '{}'.format(train_results[0])

    def eval_log(val_results):
        return '{} {}'.format(val_results[0], val_results[1])

    def epoch_log_fn(epoch_index, delta_time, train_str, val_str, test_str):
        return '{}: train: {}, val: {}'.format(epoch_index, train_str,
                                               val_str)

    trainer = Trainer(train_batch_func=train_fn, train_log_func=train_log,
                      num_epochs=200, min_epochs=65,
                      val_improve_patience_factor=2,
                      eval_batch_func=eval_fn, eval_log_func=eval_log,
                      val_improved_func=lambda a, b: a[1] < b[1],
                      log_stream=log, verbosity=VERBOSITY_EPOCH,
                      epoch_log_func=epoch_log_fn, log_final_result=False)

    trainer.train([np.arange(5)], [np.arange(5)], None, batchsize=5)

    assert train_fn.count == 152
    assert eval_fn.count == 152
    log_lines = log.getvalue().split('\n')
    for i, line in enumerate(log_lines):
        if line.strip() != '':
            if sys.version_info[0] == 2:
                assert '{}: train: {}, val: {} {}'.format(
                    i, train_output[i][0], val_output[i][0],
                    val_output[i][1]) == line
