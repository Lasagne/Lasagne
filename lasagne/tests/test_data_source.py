import pytest
import numpy as np


# Helper function to test the callable protocol
def make_batch_iterator_callable(X, Y):
    from lasagne import data_source

    def batch_iterator(batch_size, shuffle=None):
        # Make `data_source.ArrayDataSource.batch_iterator` do the work :)
        return data_source.ArrayDataSource([X, Y]).batch_iterator(
            batch_size, shuffle=shuffle)
    return batch_iterator


def test_length_of_batch():
    from lasagne import data_source

    X = np.arange(10)
    Y = np.arange(20)

    assert data_source._length_of_batch(X) == 10

    assert data_source._length_of_batch((X,)) == 10
    assert data_source._length_of_batch((X, Y)) == 10
    assert data_source._length_of_batch(((X,), Y)) == 10
    assert data_source._length_of_batch((Y, X)) == 20
    assert data_source._length_of_batch(((Y,), X)) == 20

    assert data_source._length_of_batch([X]) == 10
    assert data_source._length_of_batch([X, Y]) == 10
    assert data_source._length_of_batch([[X], Y]) == 10
    assert data_source._length_of_batch([Y, X]) == 20
    assert data_source._length_of_batch([[Y], X]) == 20


def test_num_batches():
    from lasagne import data_source

    assert data_source._num_batches(20, 5) == 4
    assert data_source._num_batches(21, 5) == 5


def test_trim_batch():
    from lasagne import data_source

    X = np.arange(10)
    Y = np.arange(20)

    assert (data_source._trim_batch(X, 5) == X[:5]).all()

    b = data_source._trim_batch([X, Y], 5)
    assert isinstance(b, list)
    assert (b[0] == X[:5]).all()
    assert (b[1] == Y[:5]).all()

    b = data_source._trim_batch((X, Y), 5)
    assert isinstance(b, list)
    assert (b[0] == X[:5]).all()
    assert (b[1] == Y[:5]).all()

    b = data_source._trim_batch([[X, Y], X, Y], 5)
    assert isinstance(b, list)
    assert (b[0][0] == X[:5]).all()
    assert (b[0][1] == Y[:5]).all()
    assert (b[1] == X[:5]).all()
    assert (b[2] == Y[:5]).all()


def test_AbstractDataSource():
    from lasagne import data_source

    ds = data_source.AbstractDataSource()

    with pytest.raises(NotImplementedError):
        _ = ds.num_samples()

    with pytest.raises(NotImplementedError):
        ds.batch_iterator(256)


def test_ArrayDataSource():
    from lasagne import data_source, random

    # Test `len(ds)`
    a3a = np.arange(3)
    a3b = np.arange(6).reshape((3, 2))
    a10 = np.arange(10)

    assert data_source.ArrayDataSource([a3a]).num_samples() == 3
    assert data_source.ArrayDataSource([a10]).num_samples() == 10
    assert data_source.ArrayDataSource([a3a, a3b]).num_samples() == 3

    # Test `batch_iterator`
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))
    ads = data_source.ArrayDataSource([X, Y])

    # Three in-order batches
    batches = list(ads.batch_iterator(batch_size=15))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[:15]).all()
    assert (batches[0][1] == Y[:15]).all()
    assert (batches[1][0] == X[15:30]).all()
    assert (batches[1][1] == Y[15:30]).all()
    assert (batches[2][0] == X[30:]).all()
    assert (batches[2][1] == Y[30:]).all()

    # Ensure that shuffle=False results in three in-order batches
    batches = list(ads.batch_iterator(batch_size=15, shuffle=False))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[:15]).all()
    assert (batches[0][1] == Y[:15]).all()
    assert (batches[1][0] == X[15:30]).all()
    assert (batches[1][1] == Y[15:30]).all()
    assert (batches[2][0] == X[30:]).all()
    assert (batches[2][1] == Y[30:]).all()

    # Three shuffled batches
    batches = list(ads.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[order[:15]]).all()
    assert (batches[0][1] == Y[order[:15]]).all()
    assert (batches[1][0] == X[order[15:30]]).all()
    assert (batches[1][1] == Y[order[15:30]]).all()
    assert (batches[2][0] == X[order[30:]]).all()
    assert (batches[2][1] == Y[order[30:]]).all()

    # Check that shuffle=True uses Lasagne's default RNG
    old_rng = random.get_rng()
    random.set_rng(np.random.RandomState(12345))
    batches = list(ads.batch_iterator(batch_size=15, shuffle=True))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[order[:15]]).all()
    assert (batches[0][1] == Y[order[:15]]).all()
    assert (batches[1][0] == X[order[15:30]]).all()
    assert (batches[1][1] == Y[order[15:30]]).all()
    assert (batches[2][0] == X[order[30:]]).all()
    assert (batches[2][1] == Y[order[30:]]).all()
    # Reset RNG
    random.set_rng(old_rng)

    # Check that constructing an array data source given input arrays
    # of differing lengths raises ValueError
    with pytest.raises(ValueError):
        _ = data_source.ArrayDataSource([
            np.arange(20), np.arange(50).reshape((25, 2))])

    # Check that `ArrayDataSource` raises TypeError if the list of arrays
    # is not a list
    with pytest.raises(TypeError):
        _ = data_source.ArrayDataSource(X)


def test_ArrayDataSource_indices():
    from lasagne import data_source

    # Test `len(ds)`
    a3a = np.arange(3)
    a3b = np.arange(6).reshape((3, 2))
    a10 = np.arange(10)

    assert data_source.ArrayDataSource([a3a]).num_samples() == 3
    assert data_source.ArrayDataSource([a10]).num_samples() == 10
    assert data_source.ArrayDataSource([a3a, a3b]).num_samples() == 3

    # Test `batch_iterator`
    X = np.arange(90)
    Y = np.arange(180).reshape((90, 2))
    indices = np.random.permutation(90)[:45]
    ads = data_source.ArrayDataSource([X, Y], indices=indices)

    # Three in-order batches
    batches = list(ads.batch_iterator(batch_size=15))
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[indices[:15]]).all()
    assert (batches[0][1] == Y[indices[:15]]).all()
    assert (batches[1][0] == X[indices[15:30]]).all()
    assert (batches[1][1] == Y[indices[15:30]]).all()
    assert (batches[2][0] == X[indices[30:]]).all()
    assert (batches[2][1] == Y[indices[30:]]).all()

    # Three shuffled batches
    batches = list(ads.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    # Get the expected order
    order = np.random.RandomState(12345).permutation(45)
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[indices[order[:15]]]).all()
    assert (batches[0][1] == Y[indices[order[:15]]]).all()
    assert (batches[1][0] == X[indices[order[15:30]]]).all()
    assert (batches[1][1] == Y[indices[order[15:30]]]).all()
    assert (batches[2][0] == X[indices[order[30:]]]).all()
    assert (batches[2][1] == Y[indices[order[30:]]]).all()


def test_ArrayDataSource_repeated():
    from lasagne import data_source

    X = np.arange(50)
    Y = np.arange(100).reshape((50, 2))
    ads = data_source.ArrayDataSource([X, Y])

    # Helper function for checking the resulting mini-batches
    def check_batches(batches, expected_n, order):
        # Eight batches
        assert len(batches) == expected_n
        # Verify contents
        for batch_i, batch in enumerate(batches):
            # Two items in each batch
            assert len(batch) == 2
            # Compute and wrap start and end indices
            start = batch_i * 20
            end = start + 20
            # Get the indices of the expected samples from the `order` array
            if end > start:
                batch_order = order[start:end]
            else:
                batch_order = np.append(order[start:], order[:end], axis=0)
            # Verify values
            assert batch[0].shape[0] == batch_order.shape[0]
            assert batch[1].shape[0] == batch_order.shape[0]
            assert (batch[0] == X[batch_order]).all()
            assert (batch[1] == Y[batch_order]).all()

    # Check size
    assert ads.num_samples(epochs=1) == 50
    assert ads.num_samples(epochs=2) == 100
    assert ads.num_samples(epochs=5) == 250
    assert ads.num_samples(epochs=-1) == np.inf

    # 3 repetitions; 150 samples, 8 in-order batches
    inorder_iter = ads.batch_iterator(batch_size=20, epochs=3)
    batches = list(inorder_iter)
    order = np.concatenate([np.arange(50)] * 3, axis=0)
    check_batches(batches, 8, order)

    # 3 repetitions; 150 samples, 8 shuffled batches
    shuffled_iter = ads.batch_iterator(batch_size=20, epochs=3,
                                       shuffle=np.random.RandomState(12345))
    batches = list(shuffled_iter)
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.concatenate(
        [order_shuffle_rng.permutation(50),
         order_shuffle_rng.permutation(50),
         order_shuffle_rng.permutation(50)], axis=0)
    check_batches(batches, 8, order)

    # Infinite repetitions; take 5 in-order batches
    inorder_iter = ads.batch_iterator(batch_size=20, epochs=-1)
    batches = [next(inorder_iter) for i in range(5)]
    order = np.concatenate([np.arange(50)] * 2, axis=0)
    check_batches(batches, 5, order)

    # Infinite repetitions; take 5 shuffled batches
    shuffled_iter = ads.batch_iterator(batch_size=20, epochs=-1,
                                       shuffle=np.random.RandomState(12345))
    batches = [next(shuffled_iter) for i in range(5)]
    # Get the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.append(order_shuffle_rng.permutation(50),
                      order_shuffle_rng.permutation(50), axis=0)
    check_batches(batches, 5, order)

    # Check invalid values for epochs
    with pytest.raises(ValueError):
        ads.num_samples(epochs=0)

    with pytest.raises(ValueError):
        ads.num_samples(epochs=-2)

    with pytest.raises(ValueError):
        for _ in ads.batch_iterator(batch_size=5, epochs=0):
            pass

    with pytest.raises(ValueError):
        for _ in ads.batch_iterator(batch_size=5, epochs=-2):
            pass


def test_ArrayDataSource_indices_repeated():
    from lasagne import data_source

    X = np.arange(100)
    Y = np.arange(200).reshape((100, 2))
    indices = np.random.permutation(100)[:50]

    # Helper function for checking the resulting mini-batches
    def check_batches(batches, expected_n, order):
        # Eight batches
        assert len(batches) == expected_n
        # Verify contents
        for batch_i, batch in enumerate(batches):
            # Two items in each batch
            assert len(batch) == 2
            # Compute and wrap start and end indices
            start = batch_i * 20
            end = start + 20
            # Get the indices of the expected samples from the `order` array
            batch_order = order[start:end]
            # Verify values
            assert batch[0].shape[0] == batch_order.shape[0]
            assert batch[1].shape[0] == batch_order.shape[0]
            assert (batch[0] == X[batch_order]).all()
            assert (batch[1] == Y[batch_order]).all()

    # 3 repetitions; 8 in-order batches
    ads = data_source.ArrayDataSource([X, Y], indices=indices)
    inorder_iter = ads.batch_iterator(batch_size=20, epochs=3)
    batches = list(inorder_iter)
    order = np.concatenate([indices, indices, indices], axis=0)
    check_batches(batches, 8, order)

    # 3 repetitions; 8 shuffled batches
    ads = data_source.ArrayDataSource([X, Y], indices=indices)
    inorder_iter = ads.batch_iterator(batch_size=20, epochs=3,
                                      shuffle=np.random.RandomState(12345))
    batches = list(inorder_iter)
    # Compute the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.concatenate(
        [order_shuffle_rng.permutation(indices),
         order_shuffle_rng.permutation(indices),
         order_shuffle_rng.permutation(indices)], axis=0)
    check_batches(batches, 8, order)

    # Infinite repetitions; take 5 in-order batches
    ads = data_source.ArrayDataSource([X, Y], indices=indices)
    inorder_iter = ads.batch_iterator(batch_size=20, epochs=-1)
    batches = [next(inorder_iter) for i in range(5)]
    order = np.concatenate([indices, indices], axis=0)
    check_batches(batches, 5, order)

    # Infinite repetitions; take 5 shuffled batches
    shuffled_iter = ads.batch_iterator(batch_size=20, epochs=-1,
                                       shuffle=np.random.RandomState(12345))
    batches = [next(shuffled_iter) for i in range(5)]
    # Compute the expected order
    order_shuffle_rng = np.random.RandomState(12345)
    order = np.append(order_shuffle_rng.permutation(indices),
                      order_shuffle_rng.permutation(indices), axis=0)
    check_batches(batches, 5, order)


def test_ApplyParamsDataSource():
    from lasagne import data_source

    X = np.arange(50)
    Y = np.arange(100).reshape((50, 2))

    ads = data_source.ArrayDataSource([X, Y])

    # No settings; normal batch iterator
    no_settings = ads.with_params()
    # Settings: `epochs=-1`
    inf_settings = ads.with_params(epochs=-1)

    assert isinstance(no_settings, data_source.ApplyParamsDataSource)
    assert no_settings.datasource is ads
    assert no_settings.params == {}

    assert isinstance(inf_settings, data_source.ApplyParamsDataSource)
    assert inf_settings.datasource is ads
    assert inf_settings.params == {'epochs': -1}

    # Test length
    assert no_settings.num_samples() == 50
    assert inf_settings.num_samples() == np.inf

    # Linear batch iterator via settings
    iter_linear = no_settings.batch_iterator(
        batch_size=20
    )
    batches = [next(iter_linear) for i in range(3)]
    # Three batches
    assert len(batches) == 3
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    # Verify values
    assert (batches[0][0] == X[:20]).all()
    assert (batches[0][1] == Y[:20]).all()
    assert (batches[1][0] == X[20:40]).all()
    assert (batches[1][1] == Y[20:40]).all()
    assert (batches[2][0] == X[40:]).all()
    assert (batches[2][1] == Y[40:]).all()

    # Circular batch iterator via settings
    iter_inf = inf_settings.batch_iterator(batch_size=20)
    batches = [next(iter_inf) for i in range(5)]
    # Five batches
    assert len(batches) == 5
    # Two items in each batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2
    assert len(batches[3]) == 2
    assert len(batches[4]) == 2
    # Verify values
    assert (batches[0][0] == X[:20]).all()
    assert (batches[0][1] == Y[:20]).all()
    assert (batches[1][0] == X[20:40]).all()
    assert (batches[1][1] == Y[20:40]).all()
    assert (batches[2][0] == np.append(X[40:50], X[0:10], axis=0)).all()
    assert (batches[2][1] == np.append(Y[40:50], Y[0:10], axis=0)).all()
    assert (batches[3][0] == X[10:30]).all()
    assert (batches[3][1] == Y[10:30]).all()
    assert (batches[4][0] == X[30:50]).all()
    assert (batches[4][1] == Y[30:50]).all()


def test_CallableDataSource():
    from lasagne import data_source

    # Data to extract batches from
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))

    # Helper functions to check
    def check_in_order_batches(batches):
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[:15]).all()
        assert (batches[0][1] == Y[:15]).all()
        assert (batches[1][0] == X[15:30]).all()
        assert (batches[1][1] == Y[15:30]).all()
        assert (batches[2][0] == X[30:]).all()
        assert (batches[2][1] == Y[30:]).all()

    def check_shuffled_batches(batches, order_seed=12345):
        # Get the expected order
        order = np.random.RandomState(order_seed).permutation(45)
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[order[:15]]).all()
        assert (batches[0][1] == Y[order[:15]]).all()
        assert (batches[1][0] == X[order[15:30]]).all()
        assert (batches[1][1] == Y[order[15:30]]).all()
        assert (batches[2][0] == X[order[30:]]).all()
        assert (batches[2][1] == Y[order[30:]]).all()

    # Function to make iterator factory (callable) that checks its
    # keyword arguments
    def make_iterator_callable(**expected):
        c = make_batch_iterator_callable(X, Y)

        def f(batch_size, **kwargs):
            assert kwargs == expected
            for k in expected.keys():
                del kwargs[k]
            return c(batch_size, **kwargs)
        return f

    cds = data_source.CallableDataSource(make_batch_iterator_callable(X, Y))

    # Length
    assert cds.num_samples() is None

    # Three in-order batches
    batches = list(cds.batch_iterator(batch_size=15))
    check_in_order_batches(batches)

    # Three shuffled batches
    batches = list(cds.batch_iterator(
        batch_size=15, shuffle=np.random.RandomState(12345)))
    check_shuffled_batches(batches)

    # Check that keyword args make it over
    cds_2 = data_source.CallableDataSource(
        make_iterator_callable(foo=42, bar=3.14))
    cds_2.batch_iterator(5, foo=42, bar=3.14)

    # Number of samples function
    def num_samples_fn(**kwargs):
        assert kwargs == {'foo': 42, 'bar': 3.14}
        return X.shape[0]

    cds_3 = data_source.CallableDataSource(
        make_iterator_callable(foo=42, bar=3.14),
        num_samples_fn
    )
    assert cds_3.num_samples(foo=42, bar=3.14) == X.shape[0]
    cds_3.batch_iterator(5, foo=42, bar=3.14)

    # Number of samples literal value
    cds_42 = data_source.CallableDataSource(
        make_iterator_callable(foo=42, bar=3.14), 42
    )
    assert cds_42.num_samples(foo=42, bar=3.14) == 42
    cds_42.batch_iterator(5, foo=42, bar=3.14)

    cds_inf = data_source.CallableDataSource(
        make_iterator_callable(foo=42, bar=3.14), np.inf
    )
    assert cds_inf.num_samples(foo=42, bar=3.14) == np.inf
    cds_inf.batch_iterator(5, foo=42, bar=3.14)

    with pytest.raises(TypeError):
        data_source.CallableDataSource(
            make_iterator_callable(foo=42, bar=3.14), 'invalid'
        )


def test_IteratorDataSource():
    from lasagne import data_source

    # Data to extract batches from
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))

    # Helper functions to check
    def check_in_order_batches(batches):
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[:15]).all()
        assert (batches[0][1] == Y[:15]).all()
        assert (batches[1][0] == X[15:30]).all()
        assert (batches[1][1] == Y[15:30]).all()
        assert (batches[2][0] == X[30:]).all()
        assert (batches[2][1] == Y[30:]).all()

    def check_shuffled_batches(batches, order_seed=12345):
        # Get the expected order
        order = np.random.RandomState(order_seed).permutation(45)
        # Three batches
        assert len(batches) == 3
        # Two items in each batch
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        # Verify values
        assert (batches[0][0] == X[order[:15]]).all()
        assert (batches[0][1] == Y[order[:15]]).all()
        assert (batches[1][0] == X[order[15:30]]).all()
        assert (batches[1][1] == Y[order[15:30]]).all()
        assert (batches[2][0] == X[order[30:]]).all()
        assert (batches[2][1] == Y[order[30:]]).all()

    # Re-use the function defined above to create the iterator
    in_order_batch_iter = make_batch_iterator_callable(X, Y)(15)
    ds = data_source.IteratorDataSource(in_order_batch_iter)
    assert ds.num_samples() is None
    batches = list(ds.batch_iterator(batch_size=15))
    check_in_order_batches(batches)

    # Three shuffled batches
    shuffled_batch_iter = make_batch_iterator_callable(X, Y)(
        15, shuffle=np.random.RandomState(12345))
    ds = data_source.IteratorDataSource(shuffled_batch_iter)
    assert ds.num_samples() is None
    batches = list(ds.batch_iterator(batch_size=15))
    check_shuffled_batches(batches)

    # With number of samples
    ds_42 = data_source.IteratorDataSource(in_order_batch_iter, 42)
    assert ds_42.num_samples() == 42

    ds_inf = data_source.IteratorDataSource(in_order_batch_iter, np.inf)
    assert ds_inf.num_samples() == np.inf

    with pytest.raises(TypeError):
        data_source.IteratorDataSource(in_order_batch_iter, 'invalid')


def test_CompositeDataSource():
    from lasagne import data_source

    # Test `CompositeDataSource` using an example layout; a generative
    # adversarial network (GAN) for semi-supervised learning
    # We have:
    # - 15 supervised samples with ground truths; `sup_X`, `sup_y`
    # - 33 unsupervised samples `unsup_X`
    sup_X = np.random.normal(size=(15, 10))
    sup_y = np.random.randint(0, 10, size=(15,))
    unsup_X = np.random.normal(size=(33, 10))

    # We need a dataset containing the supervised samples
    sup_ds = data_source.ArrayDataSource([sup_X, sup_y])
    # We need a dataset containing the unsupervised samples
    unsup_ds = data_source.ArrayDataSource([unsup_X])

    # We need to:
    # - repeatedly iterate over the supervised samples
    # - iterate over the unsupervised samples for the generator
    # - iterate over the unsupervised samples again in a different order
    #   for the discriminator
    gan_ds = data_source.CompositeDataSource([
        sup_ds.with_params(epochs=-1), unsup_ds, unsup_ds
    ])

    # Check number of samples
    assert gan_ds.num_samples() == 33

    def check_structured_batch_layout(batch):
        # Layout is:
        # [[sup_x, sup_y], [gen_x], [disc_x]]
        assert isinstance(batch, list)
        assert isinstance(batch[0], list)
        assert isinstance(batch[1], list)
        assert isinstance(batch[2], list)
        assert len(batch) == 3
        assert len(batch[0]) == 2
        assert len(batch[1]) == 1
        assert len(batch[2]) == 1

    batches = list(gan_ds.batch_iterator(
        batch_size=10, shuffle=np.random.RandomState(12345)))
    # Get the expected order for the supervised, generator and discriminator
    # sets
    # Note: draw in the same order that the data source will
    order_rng = np.random.RandomState(12345)
    order_sup = order_rng.permutation(15)
    order_gen = order_rng.permutation(33)
    order_dis = order_rng.permutation(33)
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_sup = np.append(order_sup, order_rng.permutation(15))
    order_gen = np.append(order_gen, order_rng.permutation(33))
    order_dis = np.append(order_dis, order_rng.permutation(33))
    # Four batches
    assert len(batches) == 4
    # Four items in each batch
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 4
    assert len(batches[3]) == 4
    # Verify values
    assert (batches[0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][2] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][3] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][2] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][3] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][2] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][3] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][2] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][3] == unsup_X[order_dis[30:33]]).all()

    # Now disable flattening, resulting in structured batches:
    batches = list(gan_ds.batch_iterator(
        batch_size=10, flatten=False,
        shuffle=np.random.RandomState(12345)))

    # Four batches
    assert len(batches) == 4
    # Two items in each batch
    check_structured_batch_layout(batches[0])
    check_structured_batch_layout(batches[1])
    check_structured_batch_layout(batches[2])
    check_structured_batch_layout(batches[3])
    # Verify values
    assert (batches[0][0][0] == sup_X[order_sup[:10]]).all()
    assert (batches[0][0][1] == sup_y[order_sup[:10]]).all()
    assert (batches[0][1][0] == unsup_X[order_gen[:10]]).all()
    assert (batches[0][2][0] == unsup_X[order_dis[:10]]).all()

    assert (batches[1][0][0] == sup_X[order_sup[10:20]]).all()
    assert (batches[1][0][1] == sup_y[order_sup[10:20]]).all()
    assert (batches[1][1][0] == unsup_X[order_gen[10:20]]).all()
    assert (batches[1][2][0] == unsup_X[order_dis[10:20]]).all()

    assert (batches[2][0][0] == sup_X[order_sup[20:30]]).all()
    assert (batches[2][0][1] == sup_y[order_sup[20:30]]).all()
    assert (batches[2][1][0] == unsup_X[order_gen[20:30]]).all()
    assert (batches[2][2][0] == unsup_X[order_dis[20:30]]).all()

    assert (batches[3][0][0] == sup_X[order_sup[30:33]]).all()
    assert (batches[3][0][1] == sup_y[order_sup[30:33]]).all()
    assert (batches[3][1][0] == unsup_X[order_gen[30:33]]).all()
    assert (batches[3][2][0] == unsup_X[order_dis[30:33]]).all()


def test_batch_map():
    from lasagne import data_source

    def sqr_sum(x):
        # Ensure that we receive batches of the expected size:
        assert x.shape[0] == 5
        return (x ** 2).sum(axis=1)

    # Construct data to process and create a data source:
    X = np.random.normal(size=(100, 10))
    ds = data_source.ArrayDataSource([X])

    # Apply the function defined above:
    batch_iter = ds.batch_iterator(batch_size=5)
    X_sqr_sum = data_source.batch_map(sqr_sum, batch_iter)
    assert (X_sqr_sum[0] == (X ** 2).sum(axis=1)).all()

    # Process 2 batches at a time:
    batch_iter = ds.batch_iterator(batch_size=5)
    for i in range(5):
        partial_result = data_source.batch_map(sqr_sum, batch_iter,
                                               batch_limit=2)
        # Should have 10 samples per partial result
        assert partial_result[0].shape[0] == 10
        j = i * 10
        assert (partial_result[0] == (X[j:j + 10]**2).sum(axis=1)).all()

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return [batch_X + 2, (batch_Y**2).sum(axis=1)]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 9
        # not leave
        assert not leave
        return iterator

    # Test `batch_iterator`
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))
    ads = data_source.ArrayDataSource([X, Y])

    [x, y] = data_source.batch_map(batch_func, ads.batch_iterator(5),
                                   progress_iter_func, batch_limit=9)

    assert (x == X + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    [x, y] = data_source.batch_map(batch_func, ads.batch_iterator(5),
                                   progress_iter_func, batch_limit=9)

    assert (x == X + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    # Test prepend_args
    ads_y = data_source.ArrayDataSource([Y])
    [x, y] = data_source.batch_map(batch_func, ads_y.batch_iterator(5),
                                   progress_iter_func, batch_limit=9,
                                   prepend_args=(np.array([5]),))

    assert (x == 5 + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()


def test_mean_batch_map():
    from lasagne import data_source

    # Define a function to compute the per-sample binary cross entropy
    # loss:
    def binary_crossentropy_loss(pred, target):
        e = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
        return e.mean(axis=1)

    # Now define a function that computes the *SUM* of the binary cross
    # entropy losses over the sample axis (axis 0), as the default
    # behaviour of `mean_batch_map` will sum them up and divide by the
    # number of samples at the end:
    def binary_crossentropy_loss_sum(pred, target):
        return binary_crossentropy_loss(pred, target).sum()

    # Construct prediction and target data
    pred = np.random.uniform(0.1, 0.9, size=(7, 10))
    tgt = np.random.uniform(0.1, 0.9, size=(7, 10))
    ds = data_source.ArrayDataSource([pred, tgt])

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 7 samples divided into batches of 5 means 2 batches
        assert total == 2
        # not leave
        assert not leave
        return iterator

    # Apply the loss sum function defined above:
    batch_iter = ds.batch_iterator(batch_size=5)
    loss = data_source.mean_batch_map(binary_crossentropy_loss_sum,
                                      batch_iter, batch_limit=2,
                                      progress_iter_func=progress_iter_func)
    assert np.allclose(
        loss, binary_crossentropy_loss(pred, tgt).mean())

    # Have `mean_batch_map` sum over axis 0:
    batch_iter = ds.batch_iterator(batch_size=5)
    loss = data_source.mean_batch_map(binary_crossentropy_loss, batch_iter,
                                      sum_axis=0)
    assert np.allclose(
        loss, binary_crossentropy_loss(pred, tgt).mean())

    # Construct a large data set and use `batch_limit` to limit the
    # number of batches processed in one go
    pred_large = np.random.uniform(0.1, 0.9, size=(100, 10))
    tgt_large = np.random.uniform(0.1, 0.9, size=(100, 10))
    ds_large = data_source.ArrayDataSource([pred_large, tgt_large])
    iter_large = ds_large.batch_iterator(batch_size=5)
    for i in range(10):
        partial_loss = data_source.mean_batch_map(
            binary_crossentropy_loss_sum, iter_large, batch_limit=2)
        j = i * 10
        assert np.allclose(
            partial_loss, binary_crossentropy_loss(
                pred_large[j:j + 10], tgt_large[j:j + 10]).mean())


def test_data_source_method_batch_map():
    from lasagne import data_source

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return [batch_X + 2, (batch_Y**2).sum(axis=1)]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 9
        # not leave
        assert not leave
        return iterator

    # Test `batch_iterator`
    X = np.arange(45)
    Y = np.arange(90).reshape((45, 2))
    ads = data_source.ArrayDataSource([X, Y])

    [x, y] = ads.batch_map(batch_func, 5, progress_iter_func, batch_limit=9)

    assert (x == X + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    [x, y] = ads.batch_map(batch_func, 5, progress_iter_func)

    assert (x == X + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    # Test prepend_args
    ads_y = data_source.ArrayDataSource([Y])
    [x, y] = ads_y.batch_map(batch_func, 5, progress_iter_func,
                             prepend_args=(np.array([5]),))

    assert (x == 5 + 2).all()
    assert (y == (Y**2).sum(axis=1)).all()

    # Test batch function with no return value
    def batch_func_no_ret(batch_X, batch_Y):
        pass

    r = ads.batch_map(batch_func_no_ret, 5, progress_iter_func)
    assert r is None

    # Test batch function with single array return value
    def batch_func_one_ret(batch_X, batch_Y):
        return (batch_Y**2).sum(axis=1)

    [y] = ads.batch_map(batch_func_one_ret, 5, progress_iter_func)

    assert (y == (Y**2).sum(axis=1)).all()

    # Test batch function that returns invalid type
    def batch_func_invalid_ret_type(batch_X, batch_Y):
        return 'invalid'

    with pytest.raises(TypeError):
        ads.batch_map(batch_func_invalid_ret_type, 5, progress_iter_func)

    # Check that using `epochs=-1` without specifying the number of
    # batches raises `ValueError`, as this results in a data source with
    # an infinite number of samples
    with pytest.raises(ValueError):
        ads.batch_map(batch_func, 5, progress_iter_func, epochs=-1)

    # Check that using `epochs=-1` while specifying the number of
    # batches is OK. Don't use progress_iter_func as it expects 9 batches,
    # not 15.
    [x, y] = ads.batch_map(batch_func, 5, n_batches=15, epochs=-1)

    assert (x == np.append(X, X[:30], axis=0) + 2).all()
    assert (y == (np.append(Y, Y[:30], axis=0)**2).sum(axis=1)).all()


def test_data_source_method_mean_batch_map_in_order():
    from lasagne import data_source

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))
    ads = data_source.ArrayDataSource([X, Y])

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return [batch_X.sum(), (batch_Y**2).sum(axis=1).sum()]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    [x, y] = data_source.mean_batch_map(
        batch_func, ads.batch_iterator(5),
        progress_iter_func=progress_iter_func, sum_axis=None,
        batch_limit=10)

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    [x, y] = ads.mean_batch_map(
        batch_func, 5, progress_iter_func=progress_iter_func, sum_axis=None)

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Single return value
    #
    def batch_func_single(batch_X, batch_Y):
        return batch_X.sum()

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    [x] = ads.mean_batch_map(
        batch_func_single, 5, progress_iter_func=progress_iter_func,
        sum_axis=None)

    assert np.allclose(x, X.mean())

    #
    # Batch function that returns no results
    #
    def batch_func_no_results(batch_X, batch_Y):
        return None

    res = ads.mean_batch_map(
        batch_func_no_results, 5, progress_iter_func=progress_iter_func,
        sum_axis=None)

    assert res is None

    #
    # Invalid return value
    #
    def batch_func_invalid(batch_X, batch_Y):
        return 'Should not return a string'

    with pytest.raises(TypeError):
        ads.mean_batch_map(batch_func_invalid, 5)

    #
    # Prepend arguments to batch function
    #
    def batch_func_prepend(a, b, batch_X, batch_Y):
        assert a == 42
        assert b == 3.14
        return [batch_X.sum(), (batch_Y**2).sum(axis=1).sum()]

    [x, y] = ads.mean_batch_map(
        batch_func_prepend, 5, progress_iter_func=progress_iter_func,
        sum_axis=None, prepend_args=(42, 3.14))

    assert np.allclose(x, X.mean())
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    # Check that using `epochs=-1` without specifying the number of
    # batches raises `ValueError`, as this results in a data source with
    # an infinite number of samples
    with pytest.raises(ValueError):
        ads.mean_batch_map(batch_func, 5, progress_iter_func, epochs=-1)

    # Check that using `epochs=-1` while specifying the number of
    # batches is OK. Don't use progress_iter_func as it expects 9 batches,
    # not 15.
    [x, y] = ads.mean_batch_map(batch_func, 5, n_batches=15, epochs=-1)

    assert np.allclose(x, np.append(X, X[:28], axis=0).mean())
    assert np.allclose(
        y, (np.append(Y, Y[:28], axis=0)**2).sum(axis=1).mean())

    # Test a typical loss scenario
    def binary_crossentropy(pred, target):
        e = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
        return e.mean(axis=1).sum(axis=0)

    pred = np.random.uniform(0.0, 1.0, size=(15, 10))
    tgt = np.random.uniform(0.0, 1.0, size=(15, 10))
    ds = data_source.ArrayDataSource([pred, tgt])

    loss = ds.mean_batch_map(binary_crossentropy, batch_size=5)
    assert np.allclose(loss, binary_crossentropy(pred, tgt) / pred.shape[0])

    loss = ds.mean_batch_map(binary_crossentropy, batch_size=5,
                             n_batches=1)
    assert np.allclose(loss, binary_crossentropy(pred[:5], tgt[:5]) / 5.0)


def test_data_source_method_mean_batch_map_in_order_per_sample_func():
    # Test `mean_batch_map` where the batch function returns per-sample
    # results
    from lasagne import data_source

    # Data to extract batches from
    rng = np.random.RandomState(12345)
    X = rng.normal(size=(47,))
    Y = rng.normal(size=(47, 2))
    ads = data_source.ArrayDataSource([X, Y])

    #
    # Multiple return values
    #
    def batch_func(batch_X, batch_Y):
        return [batch_X + 2, (batch_Y**2).sum(axis=1)]

    # Dummy progress function to check parameter values
    def progress_iter_func(iterator, total, leave):
        # 47 samples divided into batches of 5 means 10 batches
        assert total == 10
        # not leave
        assert not leave
        return iterator

    [x, y] = ads.mean_batch_map(batch_func, 5,
                                progress_iter_func=progress_iter_func,
                                sum_axis=0)

    assert np.allclose(x, X.mean() + 2.0)
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Single return value
    #
    def batch_func_single(batch_X, batch_Y):
        return batch_X + 2

    [x] = ads.mean_batch_map(batch_func_single, 5, sum_axis=0)

    assert np.allclose(x, X.mean() + 2.0)
    assert np.allclose(y, (Y**2).sum(axis=1).mean())

    #
    # Batch function that returns no results
    #
    def batch_func_no_results(batch_X, batch_Y):
        return None

    res = ads.mean_batch_map(batch_func_no_results, 5,
                             progress_iter_func=progress_iter_func,
                             sum_axis=0)

    assert res is None
