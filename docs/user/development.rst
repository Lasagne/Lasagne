Development
===========

The Lasagne project was started by Sander Dieleman in September 2014. It is
developed by a `core team of seven people`_ and
`numerous additional contributors`_ on GitHub:
https://github.com/Lasagne/Lasagne

You can file feature requests and bug reports there. If you have questions
about how to use the library, please post on the lasagne-users mailing list
instead: https://groups.google.com/forum/#!forum/lasagne-users

Contributions
-------------

Everybody is welcome to contribute to Lasagne. You can do so by

* Testing it and giving us feedback / submitting bug reports on GitHub.

* Writing unit tests.

* Improving existing code.

* Writing new code and sending pull requests on GitHub. Note that Lasagne
  has a fairly narrow focus and we strictly follow a set of design principles
  (see below), so we cannot guarantee upfront that your pull request will
  be accepted. However, please don't hesitate to just propose your idea in a
  GitHub issue or on the mailing list first, so we can discuss it and/or guide
  you through the implementation.

* Improving the documentation.

* Suggestions for new features are also welcome.

We suggest reading the `issues page on GitHub`_ for more ideas how you can
contribute.


Tools
-----

* `pytest <http://pytest.org/>`_ for unit testing

* Install dependencies for testing and documentation with: ``pip
  install -r requirements-dev.txt``

* GitHub for hosting the source code

* http://lasagne.readthedocs.org/ for hosting the documentation


Code coverage can be tested with

.. code:: bash

    $ py.test

Testing will take over 5 minutes for the first run, but less than a minute for
subsequent runs when Theano can reuse compiled code.

You can add a so called pre-commit-hook to git to run all tests automatically
before you commit. The hook is installed by copying the following code to
`.git/hooks/pre-commit` and making this file executable:

.. code:: bash

    #!/usr/bin/env bash

    # Stash uncommited changes to make sure the commited ones will work
    git stash -q --keep-index
    # Run tests
    py.test
    RETURN_CODE=$?
    # Pop stashed changes
    git stash pop -q

    exit $RETURN_CODE


Documentation
-------------

The documentation is generated with `Sphinx <http://sphinx-doc.org/latest/index.html>`_.

Sphinx makes use of `reStructured Text <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html>`_

You should also `install pylearn2 <http://deeplearning.net/software/pylearn2/#download-and-installation>`_
to prevent warnings while generating the documentation.

The documentation can be built with ``make html``.

The documentation is written in numpydoc syntax. Information about numpydoc
can be found at the `numpydoc repository <https://github.com/numpy/numpydoc>`_,
especially `A Guide to NumPy/SciPy Documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.



Project structure
-----------------

The project structure is

::

    .
    ├── docs
    │   ├── modules
    │   └── user
    ├── examples
    └── lasagne
        ├── layers
        ├── tests
        │   └── layers
        └── theano_extensions



Philosophy
----------

Lasagne grew out of a need to combine the flexibility of Theano with the availability of the right building blocks for training neural networks. Its development is guided by a number of design goals:

* **Simplicity**: Be easy to use, easy to understand and easy to extend, to
  facilitate use in research. Interfaces should be kept small, with as few
  classes and methods as possible. Every added abstraction and feature should
  be carefully scrutinized, to determine whether the added complexity is
  justified.

* **Transparency**: Do not hide Theano behind abstractions, directly process
  and return Theano expressions or Python / numpy data types. Try to rely on
  Theano's functionality where possible, and follow Theano's conventions.

* **Modularity**: Allow all parts (layers, regularizers, optimizers, ...) to be
  used independently of Lasagne. Make it easy to use components in isolation or
  in conjunction with other frameworks.

* **Pragmatism**: Make common use cases easy, do not overrate uncommon cases.
  Ideally, everything should be possible, but common use cases shouldn't be
  made more difficult just to cater for exotic ones.

* **Restraint**: Do not obstruct users with features they decide not to use.
  Both in using and in extending components, it should be possible for users to
  be fully oblivious to features they do not need.

* **Focus**: "Do one thing and do it well". Do not try to provide a library for
  everything to do with deep learning.



.. _issues page on GitHub: https://github.com/Lasagne/Lasagne/issues
.. _core team of seven people: https://github.com/orgs/Lasagne/teams/core-team
.. _numerous additional contributors: https://github.com/Lasagne/Lasagne/graphs/contributors
