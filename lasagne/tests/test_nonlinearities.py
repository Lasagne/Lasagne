import pytest
import numpy as np
import theano.tensor as T


class TestNonlinearities(object):
    def linear(self, x):
        return x

    def rectify(self, x):
        return x * (x > 0)

    def leaky_rectify(self, x):
        return x * (x > 0) + 0.01 * x * (x < 0)

    def leaky_rectify_0(self, x):
        return self.rectify(x)

    def elu(self, x, alpha=1):
        return np.where(x > 0, x, alpha * (np.expm1(x)))

    def selu(self, x, alpha=1, lmbda=1):
        return lmbda * np.where(x > 0, x, alpha * np.expm1(x))

    def selu_paper(self, x):
        return self.selu(x,
                         alpha=1.6732632423543772848170429916717,
                         lmbda=1.0507009873554804934193349852946)

    def selu_rect(self, x):
        return self.selu(x, alpha=0, lmbda=1)

    def selu_custom(self, x):
        return self.selu(x, alpha=0.12, lmbda=1.21)

    def softplus(self, x):
        return np.log1p(np.exp(x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def scaled_tanh(self, x):
        return np.tanh(x)

    def scaled_tanh_p(self, x):
        return 2.27 * np.tanh(0.5 * x)

    def softmax(self, x):
        return (np.exp(x).T / np.exp(x).sum(-1)).T

    @pytest.mark.parametrize('nonlinearity',
                             ['linear', 'rectify',
                              'leaky_rectify', 'elu',
                              'selu', 'selu_paper',
                              'selu_rect', 'selu_custom',
                              'sigmoid',
                              'tanh', 'scaled_tanh',
                              'softmax', 'leaky_rectify_0',
                              'scaled_tanh_p', 'softplus'])
    def test_nonlinearity(self, nonlinearity):
        import lasagne.nonlinearities

        if nonlinearity == 'leaky_rectify_0':
            from lasagne.nonlinearities import LeakyRectify
            theano_nonlinearity = LeakyRectify(leakiness=0)
        elif nonlinearity == 'scaled_tanh':
            from lasagne.nonlinearities import ScaledTanH
            theano_nonlinearity = ScaledTanH()
        elif nonlinearity == 'scaled_tanh_p':
            from lasagne.nonlinearities import ScaledTanH
            theano_nonlinearity = ScaledTanH(scale_in=0.5, scale_out=2.27)
        elif nonlinearity.startswith('selu'):
            from lasagne.nonlinearities import SELU, selu
            if nonlinearity == 'selu':
                theano_nonlinearity = SELU()
            elif nonlinearity == 'selu_paper':
                theano_nonlinearity = selu
            elif nonlinearity == 'selu_rect':
                theano_nonlinearity = SELU(scale=1, scale_neg=0)
            elif nonlinearity == 'selu_custom':
                theano_nonlinearity = SELU(scale=1.21, scale_neg=0.12)
        else:
            theano_nonlinearity = getattr(lasagne.nonlinearities,
                                          nonlinearity)
        np_nonlinearity = getattr(self, nonlinearity)

        X = T.matrix()
        X0 = lasagne.utils.floatX(np.random.uniform(-3, 3, (10, 10)))

        theano_result = theano_nonlinearity(X).eval({X: X0})
        np_result = np_nonlinearity(X0)

        assert np.allclose(theano_result, np_result)
