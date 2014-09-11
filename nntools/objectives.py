import theano
import theano.tensor as T


# MSE
def mse(x, t):
    return T.mean((x - t) ** 2) # note: mean across all dimensions, i.e. feature dimension AND minibatch dimension.

# cross-entropy
def crossentropy(x, t):
    T.mean(T.nnet.binary_crossentropy(x, t)) # note: mean across all dimensions, i.e. feature dimension AND minibatch dimension.


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
