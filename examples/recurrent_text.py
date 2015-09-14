'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne


data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#Lasagne Seed
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 25

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100

# Optimization learning rate
LEARNING_RATE = .01
# All gradients above this will be clipped
GRAD_CLIP = 5
# How often should we check the output?
PRINT_FREQ = 1000
# Number of epochs to train the net
NUM_EPOCHS = 10000

def gen_data(p):
    x = np.zeros((1,SEQ_LENGTH,vocab_size))
    y = np.zeros((SEQ_LENGTH,vocab_size))
    for i in range(SEQ_LENGTH):
        x[0,i,char_to_ix[data[p+i]]] = 1
        y[i,char_to_ix[data[p+i+1]]] = 1
    return x, np.array(y,dtype='int32')

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
    import ipdb
    
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(1, None, vocab_size))

    #l_forward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, grad_clipping=GRAD_CLIP)
 
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP, learn_init=False,
        W_in_to_hid=lasagne.init.Normal(),
        W_hid_to_hid=lasagne.init.Normal(),
        nonlinearity=lasagne.nonlinearities.tanh)



    l_forward_slice = lasagne.layers.SliceLayer(l_forward, 0, 0)

   # Our output layer is a simple dense connection, with 1 output unit
    l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.imatrix('target_output')
    
    


    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    #all_softmax, _ = theano.scan(fn = lambda x: T.nnet.softmax(x), outputs_info = None, sequences = network_output, n_steps = SEQ_LENGTH)
    
    # The value we care about is the final value produced for each sequence
    probs = theano.function([l_in.input_var],network_output[0])
    #probs = theano.function([l_in.input_var],T.exp(network_output[0]) / T.sum(T.exp(network_output[0])))

    # Our cost will be mean-squared error
    #ipdb.set_trace()
    cost = T.nnet.categorical_crossentropy(network_output,target_values).sum()

    # Retrieve all parameters from the network
    np.random.seed(0)
    all_params = lasagne.layers.get_all_params(l_out)
    #all_params[1].set_value(np.loadtxt('Wxh'))
    #all_params[3].set_value(np.loadtxt('Whh'))
    #all_params[4].set_value(np.loadtxt('Why'))

    def try_it_out(seed=8456):
        np.random.seed(0)
        sample_ix = []
        #print('\n\n \t\t\t\t TRYING THE NET')
        x = np.zeros((1,1,vocab_size))
        x[0,0,char_to_ix['N']] = 1.0
        for i in range(200):
            ix = np.random.choice(range(vocab_size), p=probs(x).ravel())
            sample_ix.append(ix)
            x = np.zeros((1,1,vocab_size))
            x[0,0,sample_ix[-1]] = 1.0
        #print(sample_ix)
        random_snippet = ''.join(ix_to_char[ix] for ix in sample_ix)    
        print("----\n %s \n----" % random_snippet)
        return sample_ix

    # def sample(h = np.zeros((N_HIDDEN,1)), seed_ix = char_to_ix['N'], n = 5):
    #   """ 
    #   sample a sequence of integers from the model 
    #   h is memory state, seed_ix is seed letter for first time step
    #   """
      
    #   ipdb.set_trace()
    #   Wxh = all_params[1].get_value().T
    #   Whh = all_params[3].get_value().T
    #   Why = all_params[4].get_value().T
    #   bh  = all_params[2].get_value()
    #   by  = all_params[5].get_value()

    #   Wxh = np.loadtxt('Wxh')
    #   Whh = np.loadtxt('Whh')
    #   Why = np.loadtxt('Why')
    #   x = np.zeros((vocab_size, 1))
    #   x[seed_ix] = 1
    #   ixes = []
    #   for t in xrange(n):
    #     h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh[0])
    #     y = np.dot(Why, h)[0] + by
    #     p = np.exp(y) / np.sum(np.exp(y))
    #     #print(p)
    #     np.random.seed(28)
    #     ix = np.random.choice(range(vocab_size), p=p.ravel())
    #     x = np.zeros((vocab_size, 1))
    #     x[ix] = 1
    #     ixes.append(ix)
    #   return ixes 
    

    try_it_out()
    
    # random_snippet = ''.join(ix_to_char[ix] for ix in sample())  
    # print("----\n %s \n----" % random_snippet) 
    # ipdb.set_trace()
    #import sys; sys.exit()
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_values], cost)
    

    # We'll use this "validation set" to periodically check progress
    X_val, y_val = gen_data(8456)
    #ipdb.set_trace()

    print("Training ...")
    p = 0
    try:
        for epoch in range(num_epochs):
            sample_ix = try_it_out()
            cost_val = compute_cost(X_val, y_val)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
            for _ in range(PRINT_FREQ):
                x,y = gen_data(p)
                p += SEQ_LENGTH
                train(x, y)

                if(p+1+SEQ_LENGTH>=data_size):
                    print('Carriage Returned to Start ...')
                    all_params[0].set_value(np.zeros((1,N_HIDDEN)))
                    p = 0
            
            
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
