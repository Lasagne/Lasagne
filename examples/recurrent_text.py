'''
Recurrent network example.  Trains a vanilla RNN to learn and generate
text from a user-provided input file. This example is partly based on Andrej
Karpathy's blog (http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
and more specifically, on his code min-char-rnn.py
(https://gist.github.com/karpathy/d4dee566867f8291f086).
The inputs to the RNN are a (semi-redundant) sequence of characters and the corresponding
targets are the characters in the text shifted to the right by one. 
Assuming a sequence length of 5, a training point for a text file
"The quick brown fox jumps over the lazy dog" would be
INPUT : 'T','H','E',' ','Q'
OUTPUT: 'U'

The loss function compares (via categorical crossentropy) the prediction
with the output/target.

Also included is a function to generate text using the RNN given the first 
character.  

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
    fname = 'input.txt'
    data = open(fname, 'r').read() # should be simple plain text file
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
    print("{0} does not exist. Please provide a valid txt file as input.".format(fname))
    print("A sample txt file can be downloaded from https://raw.githubusercontent.com/keskarnitish/Lasagne/master/examples/input.txt")
    sys.exit(1)


#As a sample file, you may use the complete works of Shakespeare
#available as a .txt file from http://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#Lasagne Seed
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 20

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 5

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 100

# Batch Size
BATCH_SIZE = 128


def gen_data(p, batch_size = BATCH_SIZE):
    '''
    This function produces a training sample from the location 'p' in the text file.
    For instance, assuming SEQ_LENGTH = 25 and p=0, the function would output the first 
    25 characters of the text file as the input and characters 1-26 (0 indexing) as the target.
    '''
    x = np.zeros((batch_size,SEQ_LENGTH,vocab_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(SEQ_LENGTH):
            try:
                x[n,i,char_to_ix[data[p+ptr+i]]] = 1.
            except:
                import ipdb; ipdb.set_trace()
        y[n] = char_to_ix[data[p+ptr+SEQ_LENGTH]]
    return x, np.array(y,dtype='int32')

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)
    # For simplicity, the batch size has been hardcoded as 1. However, this is easy to change. 

    l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))

    # We now build the recurrent layer which takes l_in as the input layer
    # The weights are initialized to be Normal with mean 0 and std 0.01
    # For simplicity, we assume that the initial value of the hidden weights is not learnt.
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    # The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
    # Since we hard-code the batch-size to be one, slice the output to obtain the matrix (SEQ_LENGTH, N_HIDDEN)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, -1, 1)

    # l_forward_slice now isolates the last value in the sequence (which corresponds to our prediction)
    #l_forward_slice = lasagne.layers.SliceLayer(l_forward_single_batch, -1, 0)

    # The sliced output is then passed through the softmax nonlinearity to create probability distribution
    # for each time step. In other words, the output of this step is (SEQ_LENGTH, N_HIDDEN)
    l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

       
    # In order to generate text from the network, we need the probability distribution of the next character given the current character
    # This is done using the compiled function probs. 
    # It takes the first character as input (in encoded form) and produces a probability distribution for the next. 
    probs = theano.function([l_in.input_var],network_output.mean(axis=0),allow_input_downcast=True)

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

        x,_ = gen_data(8456, 1)
        
        for i in range(200):
            ix = np.argmax(probs(x).ravel())
            sample_ix.append(ix)
            x = np.zeros((1,SEQ_LENGTH,vocab_size))
            x[0,0:SEQ_LENGTH-1,:] = x[0,1:,:]
            x[0,SEQ_LENGTH-1,sample_ix[-1]] = 1. 

        random_snippet = seed + ''.join(ix_to_char[ix] for ix in sample_ix)    
        print("----\n %s \n----" % random_snippet)
    
    # The loss function is calculated as the sum of the (categorical) cross-entropy between the prediction and target at each time step
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates)
    compute_cost = theano.function([l_in.input_var, target_values], cost)
    
    print("Training ...")
    p = 0
    try:
        for it in xrange(data_size * num_epochs / BATCH_SIZE):
            sample_ix = try_it_out(data[p]) # Generate text using RNN using the p^th character as the start. 
            
            avg_cost = 0;
            for _ in range(PRINT_FREQ):
                x,y = gen_data(p)
                #print(p)
                p += SEQ_LENGTH + BATCH_SIZE - 1 
                if(p+BATCH_SIZE+SEQ_LENGTH >= data_size):
                    print('Carriage Return')
                    p = 0;
                

                avg_cost += train(x, y)
            print("Epoch {} average loss = {}".format(it*1.0/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))
                    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()