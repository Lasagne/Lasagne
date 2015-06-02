.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: http://lasagne.readthedocs.org/en/latest/

.. image:: https://travis-ci.org/Lasagne/Lasagne.svg?branch=master
    :target: https://travis-ci.org/Lasagne/Lasagne

.. image:: https://img.shields.io/coveralls/Lasagne/Lasagne.svg
    :target: https://coveralls.io/r/Lasagne/Lasagne

Lasagne
=======

Lasagne is a lightweight library to build and train neural networks in Theano.

For support, please refer to the lasagne-users mailing list: https://groups.google.com/forum/#!forum/lasagne-users

Documentation: http://lasagne.readthedocs.org/ (work in progress)

Lasagne is a work in progress, input is welcome.

Design goals:

* Simplicity: it should be easy to use and extend the library. Whenever a feature is added, the effect on both of these should be considered. Every added abstraction should be carefully scrutinized, to determine whether the added complexity is justified.

* Small interfaces: as few classes and methods as possible. Try to rely on Theano's functionality and data types where possible, and follow Theano's conventions. Don't wrap things in classes if it is not strictly necessary. This should make it easier to both use the library and extend it (less cognitive overhead).

* Don't get in the way: unused features should be invisible, the user should not have to take into account a feature that they do not use. It should be possible to use each component of the library in isolation from the others.

* Transparency: don't try to hide Theano behind abstractions. Functions and methods should return Theano expressions and standard Python / numpy data types where possible.

* Focus: follow the Unix philosophy of "do one thing and do it well", with a strong focus on feed-forward neural networks.

* Pragmatism: making common use cases easy is more important than supporting every possible use case out of the box.
