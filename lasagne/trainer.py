import sys
import six
import time
import functools
import lasagne
from .batch import mean_batch_map


VERBOSITY_NONE = None
VERBOSITY_MINIMAL = 'minimal'
VERBOSITY_EPOCH = 'epoch'


# Helper functions
def _default_epoch_log_func(epoch_number, delta_time, train_str, val_str,
                            test_str):
    if val_str is None and test_str is None:
        return 'Epoch {} ({:.2f}s): train {}'.format(epoch_number,
                                                     delta_time, train_str)
    elif val_str is not None and test_str is None:
        return 'Epoch {} ({:.2f}s): train {}, validation {}'.format(
                epoch_number, delta_time, train_str, val_str)
    elif val_str is None and test_str is not None:
        return 'Epoch {} ({:.2f}s): train {}, test {}'.format(
                epoch_number, delta_time, train_str, test_str)
    else:
        return 'Epoch {} ({:.2f}s): train {}, validation {}, test {}'.format(
                epoch_number, delta_time, train_str, val_str, test_str)


def _default_val_improved_func(new_val_results, best_val_results):
    # Default validation improvement detetion function
    # Defined here so that pickle can find it if necessary
    return new_val_results[0] < best_val_results[0]


class TrainingFailedException (Exception):
    """
    This exception is raised to indicate that training failed for some reason;
    often indicating that the training

    Attributes
    ----------
    epoch: int
        The epoch at which training failed
    reason: str/unicode
        A string providing the reason that training failed
    parameters_reset: bool
        Indicates if the network parameters were successfully reset to the
        initial state
    """
    def __init__(self, epoch, reason, parameters_reset):
        super(TrainingFailedException, self).__init__(
                'Training failed at epoch {}: {}'.format(epoch, reason))
        self.epoch = epoch
        self.reason = reason
        self.parameters_reset = parameters_reset


class TrainingResults (object):
    """
    `TrainingResults` instance provide the results of training a neural
    network using a `Trainer` instance.

    Attributes
    ----------
    train_results: list
        Per epoch results of the training function
    validation_results: list
        Per epoch results of the evaluation function applied to the validation
        set
    test_results: list
        Per epoch results of the evaluation function applied to the test
        set; note test set is only evaluated *when* the validation score
        improves, so test_results will contain `None` for epochs in which
        no improvement was obtained
    best_val_epoch: int
        The index of the epoch at which the best validation score was obtained
    best_validation_results: list
        The best validation results obtained (equivalent to
        `self.validation_results[self.best_val_epoch]`)
    best_test_results: list
        The more recent test result obtained (equivalent to
        `self.test_results[self.best_val_epoch]` or
        `self.test_results[-1]` if no validation set was used)
    last_epoch: int
        The index of the last epoch that was executed; indicates when training
        stopped if early exit is enabled
    """
    def __init__(self, train_results, validation_results, best_val_epoch,
                 best_validation_results, test_results,
                 best_test_results, last_epoch):
        self.train_results = train_results
        self.validation_results = validation_results
        self.best_val_epoch = best_val_epoch
        self.best_validation_results = best_validation_results
        self.test_results = test_results
        self.best_test_results = best_test_results
        self.last_epoch = last_epoch


def train(train_set, val_set=None, test_set=None, train_batch_func=None,
          train_log_msg=None, train_epoch_results_check_func=None,
          train_pass_epoch_number=False, eval_batch_func=None,
          eval_log_msg=None, val_improved_func=None, val_interval=None,
          batchsize=128, num_epochs=100, min_epochs=None,
          val_improve_patience=1, val_improve_patience_factor=0.0,
          epoch_log_msg=None, pre_epoch_callback=None,
          post_epoch_callback=None, progress_iter_func=None,
          verbosity=VERBOSITY_EPOCH, log_stream=sys.stdout,
          log_final_result=True, get_state_func=None, set_state_func=None,
          layer_to_restore=None, updates_to_restore=None, shuffle_rng=None):
    """
    Neural network training loop, designed to be as generic as possible
    in order to simplify implementing a Theano/Lasagne training loop.

    Build with the constructor, providing arguments for required parameters.
    To train, invoke the `train` method, optionally overriding parameters
    that were passed to the constructor to customise.

    Parameters common to both the constructor and the `train` method will
    be documented here.

    The datasets (`train_set`, `val_set` and `test_set`) must either
    be a sequence of array-likes, an object with a `batch_iterator`
    method or a callable; see :func:`batch.batch_iterator`

    Parameters
    ----------
    train_set: dataset
        The training set
    val_set: [optional] dataset
        The validation set
    test_set: [optional] dataset
        The test set
    train_batch_func: [REQUIRED] callable
        `train_batch_func(*batch_data) -> batch_train_results`
        Mini-batch training function that updates the network parameters,
        where `batch_data` is a list of NumPy arrays that contain the training
        data for the batch and `batch_train_results` is a list of
        floats/arrays that represent loss/error rates/etc, or `None`. Note
        that the training function results should represent the *sum* of the
        loss/errors for the samples in the batch batch as the values will be
        accumulated and divided by the total number of training samples
        after all mini-batches has been processed.
    train_log_msg: [optional] str or callable
        If a string is provided the training log message is generated using
        the format method: `train_log_msg.format(*train_results)`, e.g.
        passing the string 'loss {0} err {1}' would be suitable to log loss
        and error values from the training results. If a callable is provided
        the training log message is generated using
        `train_log_msg(train_results)`. If a training log message is not
        provided, the default behaviour is to convert the train results to a
        string using `str(train_results)`.
    train_epoch_results_check_func: [optional] callable
        `train_epoch_results_check_func(epoch, train_epoch_results) -> str`
        Function that is invoked to check the results returned by
        `train_batch_func` accumulated during the epoch; if no training failure
        is detected, it should return `None`, whereas if a failure is
        detected - e.g. training loss is NaN - it should return
        a reason string that will be used to build a
        :class:`TrainingFailedException` that will be raised by the
        :method:`train` method. Note that should a training failure be
        detected, the trainer will attempt to restore the network's
        parameters to the same values it had before the `train` method
        was invoked (see :method:`<<ERROR>>`).
    train_pass_epoch_number: boolean
        If True, the epoch number will be passed to `train_batch_func` as the
        first argument, before the batch data.
    eval_batch_func: [optional] callable
        `eval_batch_func(*batch_data) -> eval_results`
        Mini-batch evaluation function (for validation/test) where
        `batch_data` is a list of numpy arrays that contain the data to
        evaluate for the batch and `eval_results` is a list of floats/arrays
        that represent loss/error rates/etc, or `None`. Note that as with
        `train_batch_func` the results should represent the *sum*
        of the loss/error rate over the batch. For the purpose of detecting
        improvements in validation score, the default behavior assumes
        that the first element of `eval_results` represents the score
        and that a *lower* value indicates a *better* result. This can be
        overridden by providing a callable for `val_improved_func`
        that can use a custom improvement detection strategy
    eval_log_msg: [optional] str or callable
        If a string is provided the evaluation log message is generated using
        the format method: `eval_log_msg.format(*eval_results)`, e.g.
        passing the string 'loss {0} err {1}' would be suitable to log loss
        and error values from the evaluation results. If a callable is provided
        the evaluation log message is generated using
        `eval_log_msg(eval_results)`. If a evaluation log message is not
        provided, the default behaviour is to convert the train results to a
        string using `str(eval_results)`.
    val_improved_func: [optional] callable
        `validation_improved_func(new_val_results, best_val_results) -> bool`
        Validation improvement detection function that determines if the
        most recent validation results `new_val_results` are an improvement
        on those in `best_val_results`.
    val_interval: int or `None`
        If not `None`, the validation set will be evaluated every
        `val_interval` epochs
    batchsize: int (default=128)
        The mini-batch size
    num_epochs: int (default=100)
        The maximum number of epochs to train for
    min_epochs: int or None (default=None)
        The minimum number of epochs to train for; training will not terminate
        early (due to lack of validation score improvements) until at least
        this number of epochs have been executed. If `None`, it will have
        the same value as `num_epochs`.
    val_improve_patience: int (default=0)
        If not `None`, training will terminate early if a run of
        `val_improve_patience` epochs executes with no improvement in
        validation score.
    val_improve_patience_factor: float (default=0.0)
        If not `None`, training will terminate early if a run of
        `(current_epoch + 1) * val_improve_patience_factor` epochs executes
        with no improvement in validation score.
    epoch_log_msg: [optional] str or callable
        If a string is provided the epoch log message is generated using
        the format method:
        `epoch_log_msg.format(epoch, d_time, train_str, val_str, test_str)`.
        If a callable is provided the epoch log message is generated using
        `epoch_log_msg(epoch, d_time, train_str, val_str, test_str)`.
        The arguments are as follows: `epoch` is the epoch index, `d_time` is
        the time elapsed in seconds. `train_str`, `val_str` and `test_str`
        are strings or `None`, that represent that training, validation and
        test results if available. They can be customised by providing values
        for `train_log_msg` for training results and `eval_log_msg` for
        validation and test results. The default behaviour produces a line
        reporting the epoch index, time elapsed, train, validation and test
        results, and should be suitable for most purposes.
    pre_epoch_callback: [optional] callable `pre_epoch_callback(epoch)`
        If provided this function will be invoked before the start of
        each epoch, with the epoch index provided as the (first)
        argument.
    post_epoch_callback: [optional] callable
        `post_epoch_callback(epoch, train_results, val_results)`
        If provided this function will be invoked after the end of
        each epoch, with the epoch index provided as the (first)
        argument, the mean training results as the second and the mean
        validation results as the third if validation was performed this
        epoch, `None` otherwise.
    progress_iter_func: [optional] callable
        `progress_iter_func(iterator, total=total, desc=desc, leave=leave)`
        A `tqdm` style function that will be passed the iterator that
        generates training batches along with the total number of batches,
        the current task description as a string and `False` for the
        `leave parameter. By passing either `tqdm.tqdm` or
        `tqdm.tqdm_notebook` as this argument you can have the training loop
        display a progress bar.
    verbosity: one of `VERBOSITY_NONE` (`None`), `VERBOSITY_MINIMAL`
        (`'minimal'`) or `VERBOSITY_EPOCH` (`'epoch'`)
        How much information is written to the log stream describing progress.
        If `VERBOSITY_NONE` then nothing is reported.
        If `VERBOSITY_MINIMAL` then after each epoch a '.' is written if no
        validation is performed (see  :param:`validation_interval`), a '-' if
        validation is performed but the score didn't improve or a '*' when
        the score does improve.
        If `VERBOSITY_EPOCH` then a single line report is produced for each
        epoch.
        Per-batch reporting can be achieved via the `progress_iter_func`
        argument.
    log_stream: a file like object (default=sys.stdout)
        The stream to which progress is logged. If `None`, no loggins is done.
    log_final_result: bool (default=True)
        If True the final result is reported in the log.
    get_state_func: [optional] callable `get_state_func() -> state`
        Gets the state of the parameters in the network being trained. Trainer
        invokes this function to save the state of the network after a
        validation score improvement. Restored using :param:`set_state_func`.
    set_state_func: [optional] callable `set_state_func(state)`
        Sets the state of the parameters in the network to that state passed
        as an argument. Used to restore the network to the state it was in
        at the start if a training error occurred (see
        :param:`train_epoch_results_check_func`) or the state it was in
        after the most recent improvement in validation score.
    layer_to_restore: [optional] Layer or list
        The :class:`Layer` instance or list of :class:`Layer` instances
        that are the final layer(s) in the network; the network state will
        be saved (see :param:`get_state_func`) by saving the values
        of the parameters in the layers of the network.
    updates_to_restore: [optional] None or sequence or dict
        An updates list or dictionary obtained by calling functions
        from :mod:`lasagne.udpates` that provide additional paremeters
        whose states should be saved and restored, in addition to
        those acquired from :param:`layer_to_restore`. Note that providing
        a value for :param:`update_to_restore` without a value for
        :param:`layer_to_restore` will result in the :method:`train`
        method raising `ValueError`.
    shuffle_rng: `None` or a `np.random.RandomState`
        A random number generator used to shuffle the order of samples
        during training. If one is not provided, `lasagne.rng.get_rng()`
        will be used.

    Returns
    -------
    `TrainingResults` instance:
    Provides the per-epoch history of results of training, validation and
    testing, along with the epoch that gave the best validation score
    and the best validation and final test results.

    Notes
    -----
    Early termination:
    If an improvement in validation score was last obtained at epoch
    `best_epoch`, training will terminate if no improvement is detected
    before epoch:
    `max(best_epoch + 1 + val_improve_patience,
         (best_epoch + 1) * val_improve_patience_factor,
         min_epoch)`

    Saving and restoring state:
    :param:`get_state_func` and :param:`set_state_func` should be provided
    together; the :method:`train` method will raise `ValueError` if
    one if provided but the other not.
    Providing a value for :param:`layer_to_restore` will override functions
    passed to :param:`get_state_func` and :param:`set_state_func`.
    If you provide a value for :param:`updates_to_restore` without a value for
    :param:`layer_to_restore`, thenthe :method:`train` method will raise a
    `ValueError`.

    """
    # Provide defaults
    if val_improved_func is None:
        val_improved_func = _default_val_improved_func
    if shuffle_rng is None:
        shuffle_rng = lasagne.random.get_rng()

    if min_epochs is None:
        # min_epochs not provided; default to num_epochs
        min_epochs = num_epochs
    else:
        # Ensure that min_epochs <= num_epochs
        min_epochs = min(min_epochs, num_epochs)

    # Check parameter sanity
    if train_batch_func is None:
        raise ValueError('no batch training function provided to '
                         'either the constructor or the `train` method')
    if val_set is not None and eval_batch_func is None:
        raise ValueError('validation set provided but no evaluation '
                         'function available providd to the constructor '
                         'or the `train` method')
    if test_set is not None and eval_batch_func is None:
        raise ValueError('test set provided but no evaluation '
                         'function available providd to the constructor '
                         'or the `train` method')
    if updates_to_restore is not None and layer_to_restore is None:
        raise ValueError('`updates_to_restore` provided without '
                         '`layer_to_restore`')
    # Handle log messages
    if train_log_msg is None:
        def train_log_func(train_res):
            return '{}'.format(train_res)
    elif isinstance(train_log_msg, six.string_types):
        def train_log_func(train_res):
            return train_log_msg.format(*train_res)
    elif callable(train_log_msg):
        train_log_func = train_log_msg
    else:
        raise TypeError('train_log_msg should be None, a string or a '
                        'callable, not a {}'.format(type(train_log_msg)))

    if eval_log_msg is None:
        def eval_log_func(eval_res):
            return '{}'.format(eval_res)
    elif isinstance(eval_log_msg, six.string_types):
        def eval_log_func(eval_res):
            return eval_log_msg.format(*eval_res)
    elif callable(eval_log_msg):
        eval_log_func = eval_log_msg
    else:
        raise TypeError('eval_log_msg should be None, a string or a '
                        'callable, not a {}'.format(type(eval_log_msg)))

    if epoch_log_msg is None:
        epoch_log_func = _default_epoch_log_func
    elif isinstance(epoch_log_msg, six.string_types):
        def epoch_log_func(epoch, d_time, train_str, val_str, test_str):
            return epoch_log_msg.format(epoch, d_time, train_str, val_str,
                                        test_str)
    elif callable(epoch_log_msg):
        epoch_log_func = epoch_log_msg
    else:
        raise TypeError('epoch_log_msg should be None, a string or a '
                        'callable, not a {}'.format(type(epoch_log_msg)))

    if get_state_func is not None and set_state_func is None:
        if layer_to_restore is not None:
            print('WARNING: `Trainer.train()`: `get_state_func` '
                  'provided without `set_state_func`; will ignore since '
                  'both are overridden by `layer_to_restore`')
        else:
            raise ValueError('`get_state_func` provided without '
                             '`set_state_func`')
    if get_state_func is None and set_state_func is not None:
        if layer_to_restore is not None:
            print('WARNING: `Trainer.train()`: `set_state_func` '
                  'provided without `get_state_func`; will ignore since '
                  'both are overridden by `layer_to_restore`')
        else:
            raise ValueError('`set_state_func` provided without '
                             '`get_state_func`')
    if layer_to_restore is not None:
        network_params = lasagne.layers.get_all_params(layer_to_restore)

        if updates_to_restore is not None:
            if isinstance(updates_to_restore, dict):
                params = list(updates_to_restore.keys())
            elif isinstance(updates_to_restore, (list, tuple)):
                params = [upd[0] for upd in updates_to_restore]
            else:
                raise TypeError(
                    'updates_to_restore should be a dict, list or tuple, '
                    'not a {}'.format(type(updates_to_restore)))

            for p in params:
                if p not in network_params:
                    network_params.append(p)

        # Override get_state_func and set_state_func
        def get_state_func():
            return [p.get_value() for p in network_params]

        def set_state_func(state):
            for p, v in zip(network_params, state):
                p.set_value(v)

    # Helper functions
    def _log(text):
        log_stream.write(text)
        log_stream.flush()

    def _should_validate(epoch_index):
        return val_interval is None or epoch_index % val_interval == 0

    def _log_epoch_results(epoch_index, delta_time, train_res,
                           val_res, test_res):
        train_str = val_str = test_str = None
        if train_res is not None:
            train_str = train_log_func(train_res)
        if val_res is not None:
            val_str = eval_log_func(val_res)
        if test_res is not None:
            test_str = eval_log_func(test_res)
        _log(epoch_log_func(epoch_index, delta_time, train_str, val_str,
                            test_str) + '\n')

    def _save_state():
        if get_state_func is not None:
            return get_state_func()
        else:
            return None

    def _restore_state(state):
        if set_state_func is not None:
            set_state_func(state)
            return True
        else:
            return False

    stop_at_epoch = min_epochs
    epoch = 0

    # If we have a training results check function, save the state
    if train_epoch_results_check_func is not None:
        state_at_start = _save_state()
    else:
        state_at_start = None

    validation_results = None
    best_train_results = None
    best_validation_results = None
    best_epoch = None
    best_state = None
    state_saved = False
    test_results = None

    all_train_results = []
    if val_set is not None and eval_batch_func is not None:
        all_val_results = []
    else:
        all_val_results = None
    if test_set is not None and eval_batch_func is not None:
        all_test_results = []
    else:
        all_test_results = None

    train_start_time = time.time()

    while epoch < min(stop_at_epoch, num_epochs):
        epoch_start_time = time.time()

        if pre_epoch_callback is not None:
            pre_epoch_callback(epoch)

        # TRAIN
        # Log start of training
        # Train
        train_epoch_args = (epoch,) if train_pass_epoch_number else None
        if progress_iter_func is not None:
            train_prog_iter = functools.partial(
                progress_iter_func, desc='Epoch {} train'.format(epoch + 1))
        else:
            train_prog_iter = None
        train_results = mean_batch_map(
            train_batch_func, train_set, batchsize, shuffle_rng=shuffle_rng,
            restartable=True, progress_iter_func=train_prog_iter,
            sum_axis=None, prepend_args=train_epoch_args)

        if train_epoch_results_check_func is not None:
            reason = train_epoch_results_check_func(epoch, train_results)
            if reason is not None:
                # Training failed: attempt to restore parameters to
                # initial state
                if state_at_start is not None:
                    params_restored = _restore_state(state_at_start)
                else:
                    params_restored = False

                if verbosity != VERBOSITY_NONE:
                    _log("\nTraining failed at epoch {}: {}\n".format(
                            epoch, reason))

                raise TrainingFailedException(epoch, reason,
                                              params_restored)

        validated = False
        tested = False
        validation_improved = False
        # VALIDATION
        if val_set is not None and _should_validate(epoch):
            validated = True

            if progress_iter_func is not None:
                val_prog_iter = functools.partial(
                    progress_iter_func, desc='Epoch {} val'.format(epoch + 1))
            else:
                val_prog_iter = None
            validation_results = mean_batch_map(
                eval_batch_func, val_set, batchsize, restartable=True,
                progress_iter_func=val_prog_iter, sum_axis=None)
            if best_validation_results is None or \
                    val_improved_func(validation_results,
                                      best_validation_results):
                validation_improved = True

                # Validation score improved
                best_train_results = train_results
                best_validation_results = validation_results
                best_epoch = epoch
                best_state = _save_state()
                state_saved = True

                stop_at_epoch = max(
                        epoch + 1 + val_improve_patience,
                        int((epoch + 1) * val_improve_patience_factor),
                        min_epochs)

                if test_set is not None:
                    tested = True
                    if progress_iter_func is not None:
                        test_prog_iter = functools.partial(
                            progress_iter_func,
                            desc='Epoch {} test'.format(epoch + 1))
                    else:
                        test_prog_iter = None
                    test_results = mean_batch_map(
                        eval_batch_func, test_set, batchsize,
                        restartable=True, progress_iter_func=test_prog_iter,
                        sum_axis=None)
        else:
            validation_results = None

        if not tested and test_set is not None and val_set is None:
            tested = True
            if progress_iter_func is not None:
                test_prog_iter = functools.partial(
                    progress_iter_func,
                    desc='Epoch {} test'.format(epoch + 1))
            else:
                test_prog_iter = None
            test_results = mean_batch_map(
                eval_batch_func, test_set, batchsize, restartable=True,
                progress_iter_func=test_prog_iter, sum_axis=None)

        if verbosity == VERBOSITY_EPOCH:
            _log_epoch_results(epoch, time.time() - epoch_start_time,
                               train_results,
                               validation_results if validated else None,
                               test_results if tested else None)
        elif verbosity == VERBOSITY_MINIMAL:
            if validation_improved:
                _log('*')
            elif validated:
                _log('-')
            else:
                _log('.')

        all_train_results.append(train_results)
        if all_val_results is not None:
            all_val_results.append(validation_results)
        if all_test_results is not None:
            if tested:
                all_test_results.append(test_results)
            else:
                all_test_results.append(None)

        if post_epoch_callback is not None:
            post_epoch_callback(epoch, train_results, validation_results)

        epoch += 1

    train_end_time = time.time()

    if state_saved:
        _restore_state(best_state)

    if log_final_result:
        if verbosity == VERBOSITY_MINIMAL:
            _log('\n')
        if state_saved and get_state_func is not None:
            _log("Best result:\n")
            _log_epoch_results(
                    best_epoch, train_end_time - train_start_time,
                    best_train_results, best_validation_results,
                    test_results)
        else:
            final_train_results = final_test_results = None
            if len(all_train_results) > 0:
                final_train_results = all_train_results[-1]
            if best_epoch == epoch - 1:
                final_test_results = test_results
            _log("Best result:\n")
            _log_epoch_results(
                    best_epoch, train_end_time - train_start_time,
                    best_train_results, best_validation_results,
                    test_results)
            _log("Final result:\n")
            _log_epoch_results(
                    epoch - 1, train_end_time - train_start_time,
                    final_train_results, validation_results,
                    final_test_results)

    return TrainingResults(
        train_results=all_train_results,
        validation_results=all_val_results,
        best_validation_results=best_validation_results,
        best_val_epoch=best_epoch,
        test_results=all_test_results,
        best_test_results=test_results,
        last_epoch=epoch,
    )
