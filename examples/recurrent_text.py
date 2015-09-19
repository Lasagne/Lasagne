'''
Recurrent network example.  Trains a 2 layered LSTM network to learn and generate
text from a user-provided input file. This example is partly based on Andrej
Karpathy's blog (http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
and a similar example in the Keras package (keras.io).
The inputs to the network are batches of sequences of characters and the corresponding
targets are the characters in the text shifted to the right by one. 
Assuming a sequence length of 5, a training point for a text file
"The quick brown fox jumps over the lazy dog" would be
INPUT : 'T','H','E',' ','Q'
OUTPUT: 'U'

The loss function compares (via categorical crossentropy) the prediction
with the output/target.

Also included is a function to generate text using the RNN given the first 
character.  

About 20 or so epochs are necessary to generate text that "makes sense".

Written by @keskarnitish
Pre-processing of text uses snippets of Karpathy's code (BSD License)
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys

#This snippet loads the text file and creates dictionaries to 
#encode characters into a vector-space representation and vice-versa. 
try:
    fname = 'input.txt' #The provided file is the work of Nietzsche downloaded from https://s3.amazonaws.com/text-datasets/nietzsche.txt
    data = open(fname, 'r').read() # should be simple plain text file
    data = data.decode("utf-8-sig").encode("utf-8")
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
    print("{0} does not exist. Please provide a valid txt file as input.".format(fname))
    print("A sample txt file can be downloaded from https://raw.githubusercontent.com/keskarnitish/Lasagne/master/examples/input.txt")
    sys.exit(1)

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 20

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 100

# Batch Size
BATCH_SIZE = 128


def gen_data(p, batch_size = BATCH_SIZE):
    '''
    This function produces a semi-redundant batch of training samples from the location 'p' in the text file.
    For instance, assuming SEQ_LENGTH = 25 and p=0, the function would create batches of 
    25 characters of the text file (starting from the 0th character and stepping by 1 for each semi-redundant batch)
    as the input and the next character as the target.
    '''
    x = np.zeros((batch_size,SEQ_LENGTH,vocab_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(SEQ_LENGTH):
            x[n,i,char_to_ix[data[p+ptr+i]]] = 1.
        y[n] = char_to_ix[data[p+ptr+SEQ_LENGTH]]
    return x, np.array(y,dtype='int32')

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    # The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
    # Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
    # The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, -1, 1)

    # The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    
    probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)

    # The next function generates text given a seed using the current state of the network. 
    # It takes as input a location in the provided text file. 
    # The function then uses the next SEQ_LENGTH characters as seed to start generating text. 
    # The input "N" is used to set the number of characters of text to predict. 

    def try_it_out(seed, N=200):
        '''
        This function uses the current state of the RNN to generate text. It takes as input the first character, 
        computes the probability distribution of the next (using the RNN), chooses one at random using the computed probabilities
        and continues the process for N characters. 
        Inputs
        seed (char) : The first letter of the generated text.
        N (int) : Number of characters of generated text
        '''
        sample_ix = []
        x,_ = gen_data(seed, 1)

        for i in range(N):
            ix = np.argmax(probs(x).ravel())
            sample_ix.append(ix)
            x[:,0:SEQ_LENGTH-1,:] = x[:,1:,:]
            x[:,SEQ_LENGTH-1,:] = 0
            x[0,SEQ_LENGTH-1,sample_ix[-1]] = 1. 

        random_snippet = data[seed] + ''.join(ix_to_char[ix] for ix in sample_ix)    
        print("----\n %s \n----" % random_snippet)

    
    print("Training ...")
    p = 0
    try:
        for it in xrange(data_size * num_epochs / BATCH_SIZE):
            sample_ix = try_it_out(p) # Generate text using the p^th character as the start. 
            
            avg_cost = 0;
            for _ in range(PRINT_FREQ):
                x,y = gen_data(p)
                
                #print(p)
                p += SEQ_LENGTH + BATCH_SIZE - 1 
                if(p+BATCH_SIZE+SEQ_LENGTH >= data_size):
                    print('Carriage Return')
                    p = 0;
                

                avg_cost += train(x, y)
            print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))
        import ipdb; ipdb.set_trace()
                    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
