import theano.tensor as T


def mse(x, t):
    """Calculates the MSE mean across all dimensions, i.e. feature
     dimension AND minibatch dimension.
    """
    return T.mean((x - t) ** 2)


def crossentropy(x, t):
    """Calculates the binary crossentropy mean across all dimentions,
    i.e.  feature dimension AND minibatch dimension.
    """
    return T.mean(T.nnet.binary_crossentropy(x, t))


def multinomial_nll(x, t):
    """Calculates the multinomial negative-log-loss

    :param x: (predicted) class probabilities; a theano expression resulting
    in a 2D array; samples run along axis 0, class probabilities along axis 1
    :param t: (correct) class index values; a theano expression resulting in
    a 1D integer array that gives the class index of each sample

    :return: the mean negative log loss
    """
    return -T.mean(T.log(x)[T.arange(t.shape[0]), t])



class Objective(object):
    def __init__(self, input_layer, loss_function=mse):
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")

    def get_loss(self, input=None, target=None, *args, **kwargs):
        network_output = self.input_layer.get_output(input, *args, **kwargs)
        if target is None:
            target = self.target_var

        return self.loss_function(network_output, target)
