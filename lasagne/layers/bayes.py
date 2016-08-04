from functools import wraps

from lasagne import init
from lasagne.distributions import log_normal, log_normal3
from lasagne.random import get_rng
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__all__ = ['Accumulator', 'NormalApproximation',
           'NormalApproximationScMix', 'bbpwrap']


class Accumulator(object):
    """Simple class for collecting cost and providing RandomStreams
    """
    def __init__(self, seed=get_rng().randint(1, 2147462579)):
        self.srng = RandomStreams(seed)
        self.total = []

    def get_cost(self):
        """Computes and the total cost
            at first T.sum is called on all instances
            in total list to prepare for summing along list
        Returns
        -------
        Theano scalar
            all the cost collected before
        """
        return sum(map(T.sum, self.total))

    def add_cost(self, new):
        """Adds cost to accumulator

        Parameters
        ----------
        new : Theano tensor

        """
        self.total.append(new)


class NormalApproximation(object):
    """Helper class for providing logics of initializing
        random variable distributed like
            N(mean, (log(1+exp(rho))^2)
        with prior
            N(pm, pstd^2)
        where `mean`, `rho` are variational params fitted while training

        Parameters
        ----------
        pm1 : float - weight for first Gaussian
        pstd1 : float - prior mean for first Gaussian
    """
    def __init__(self, pm=0, pstd=T.exp(-3)):
        self.pm = pm
        self.pstd = pstd

    def log_prior(self, x):
        return log_normal(x, self.pm, self.pstd)

    def __call__(self, layer, spec, shape, **tags):
        """

        Parameters
        ----------
        layer : wrapped layer instance
        shape : tuple of int
                a tuple of integers representing the desired shape
                of the parameter tensor.
        tags : See :func: `lasagne.layers.base.Layer.add_param`
               for more information
        spec : Theano shared variable, expression, numpy array or callable
               Initial value, expression or initializer for the embedding
               matrix. This should be a matrix with shape
                ``(input_size, output_size)``.
               See :func:`lasagne.utils.create_param` for more information.
               .. Note
                    can also be a dict of same instances
                        {'mu': spec, 'rho':spec}
                    to avoid default rho initialization

        Returns
        -------
        Theano tensor
        """
        # case when user uses default init specs
        if not isinstance(spec, dict):
            spec = {'mu': spec}
        # important!
        # we declare that params we add next
        # are the ones we need to fit the distribution
        # they are variational
        tags['variational'] = True

        rho_spec = spec.get('rho', init.Normal(1))
        mu_spec = spec.get('mu', init.Normal(1))

        rho = layer.add_param(rho_spec, shape, **tags)
        mean = layer.add_param(mu_spec, shape, **tags)

        e = layer.acc.srng.normal(shape, std=1)
        W = mean + T.log1p(T.exp(rho)) * e

        q_p = self.log_posterior_approx(W, mean, rho) - self.log_prior(W)
        layer.acc.add_cost(q_p)
        return W

    @staticmethod
    def log_posterior_approx(W, mean, rho):
        return log_normal3(W, mean, rho)


class NormalApproximationScMix(NormalApproximation):
    """Helper class for providing logics of initializing
        random variable distributed like
            N(mean, (log(1+exp(rho))^2)

        with prior
           pi*N(pm1, pstd1^2) + (1-pi)*N(pm2, pstd2^2)

        where `mean`, `rho` are variational
        params fitted while training

    Parameters
    ----------
    pm1 : float - prior mean for first Gaussian
    pstd1 : float - prior std for first Gaussian
    pm2 : float - prior mean for second Gaussian
    pstd2 : float - prior std for second Gaussian
    pi : float in [0, 1] - first Gaussian weight
    """
    def __init__(self, pm1=.0, pstd1=.5, pm2=.0, pstd2=1e-3, pi=.5):
        assert .0 <= pi <= 1., 'Weight %d not in [0, 1]' % pi
        self.pi = pi
        self.pm1 = pm1
        self.pstd1 = pstd1
        self.pm2 = pm2
        self.pstd2 = pstd2

    def log_prior(self, x):
        return self.pi * log_normal(x, self.pm1, self.pstd1) + \
               (1 - self.pi) * log_normal(x, self.pm2, self.pstd2)


def bbpwrap(approximation=NormalApproximation()):
    """Wrapper function that allows to transform any
        reasonable layer to variational one.

    Parameters
    ----------
    approximation - callable
        supposed to take (layer, spec, shape, **tags) as params
        and return weight initialized weight
        See :class: `lasagne.layers.bayes.NormalApproximation`
            for more information and explanations

    Returns
    -------
        Wrapped layer

    Notes
    -----
        A layer should initialize weights in canonical way
        with ``add_param`` method, implementation heavily
        relies on it.

    References
    ----------
    .. [1] Charles Blundell, Julien Cornebise,
           Koray Kavukcuoglu, Daan Wierstra (2015):
           Weight Uncertainty in Neural Networks arXiv:1505.05424

    Usage
    -----
    >>> import lasagne
    >>> import theano.tensor as T
    >>> from lasagne.layers.bayes import (bbpwrap,
    ...                                   NormalApproximation,
    ...                                   Accumulator)
    >>> from lasagne.layers.dense import DenseLayer
    >>> from lasagne.layers.input import InputLayer
    >>> from lasagne.init import Normal
        # 1) Choose your favorite Layer
        # 2) Wrap your favorite Layer
    >>> @bbpwrap(NormalApproximation(pstd=1))
    ... class BayesDenseLayer(DenseLayer):
    ...     pass
        # Gracias! It's bayesian!
        # Constants for more clear code
    >>> N_HIDDEN = 5
    >>> N_BATCHES = 100
        # 3) Create the thing that will do some dirty work for us,
        #    it will collect all variational cost
    >>> acc = Accumulator()

    >>> l_in = InputLayer((10,100))
        # 4) Pass `acc` as first argument
    >>> l1_hidden = BayesDenseLayer(acc, l_in, num_units=N_HIDDEN,
    ...                             W=Normal(1), b=Normal(1),
    ...                             nonlinearity=lasagne.nonlinearities.tanh)
        # Also possible to specify both mu and rho
    >>> myW = {'mu':Normal(1.5), 'rho':Normal(.1)}
    >>> l_output = BayesDenseLayer(acc, l1_hidden, num_units=N_HIDDEN,
    ...                            W=myW, b=Normal(1),
    ...                            nonlinearity=lasagne.nonlinearities.sigmoid)
    >>> net_output = lasagne.layers.get_output(l_output).ravel()
    >>> true_output = T.ivector('true_output')
        # 5) That thing has collected the variational cost,
        #    all we need is to get it and add to our objective
        #    Do not forget to scale variational cost!
        #    .. Note
        #        Binary crossentropy is exactly the same as binomial
        #        likelihood
    >>> objective = lasagne.objectives.binary_crossentropy(net_output,
    ...                                                    true_output)
    >>> objective += (acc.get_cost() / N_BATCHES)
        # 6) Choose adam optimizer for training
    >>> all_params = lasagne.layers.get_all_params(l_output)
    >>> updates = lasagne.updates.adam(objective, all_params)
    """
    def decorator(cls):
        def add_param_wrap(add_param):
            @wraps(add_param)
            def wrapped(self, spec, shape, name=None, **tags):
                # we should take care about some user specification
                # to avoid bbp hook just set tags['variational'] = True
                if not tags.get('trainable', True) or \
                        tags.get('variational', False):
                    return add_param(self, spec, shape, name, **tags)
                else:
                    # they don't need to be regularized, strictly
                    tags['regularizable'] = False
                    param = self.approximation(self, spec, shape, **tags)
                    return param
            return wrapped

        def init_wrap(__init__):
            @wraps(__init__)
            def wrapped(self, acc, *args, **kwargs):
                self.acc = acc  # type: Accumulator
                __init__(self, *args, **kwargs)
            return wrapped

        cls.approximation = approximation
        cls.add_param = add_param_wrap(cls.add_param)
        cls.__init__ = init_wrap(cls.__init__)

        return cls
    return decorator
