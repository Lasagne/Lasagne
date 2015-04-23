About Lasagne
=============

Lasagne is a Python package for neural networks based on `Theano`_. It can
execute the training on Nvidia GPUs without having to adjust the code. Its
development was started by `Sander Dieleman`_ in September 2014. It is
developed by many people via GitHub Pull requests, e.g. by `Daniel Nouri`_,
`Colin Raffel`_, `Jan Schlüter`_ and some more.

Comparison with Caffe
---------------------

`Caffe`_ is developed by the Berkeley Vision and Learning Center and released
under the BSD 2-Clause license. Just like Lasagne, it can execute the neural
network training on Nvidia GPUs. The development of Caffe started in September
2013.

Caffe is designed to be easy to train existing models and limited in terms of
experimentation. Lasagne provides more flexibility at the cost of a being
more difficult to use.

Neural Networks are completely defined as Python code with Lasagne, whereas
Caffe makes use of `JSON-like text files`_ to describe the topology.

Lasagnes code base has about 3226 lines of Python code whereas Caffe has more
than 46,000 lines of C++ code and 3600 lines of Python code. This means
understanding the code base will cost a lot less time with Lasagne than with
Caffe.

.. _Caffe: http://caffe.berkeleyvision.org/
.. _Colin Raffel: http://colinraffel.com/
.. _Daniel Nouri: http://danielnouri.org/
.. _Jan Schlüter: http://ofai.at/~jan.schlueter/
.. _JSON-like text files: https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
.. _Sander Dieleman: http://benanne.github.io/about/
.. _Theano: http://deeplearning.net/software/theano/