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

* Install testing dependencies with: ``pip install -r requirements-dev.txt``

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
On Debian derivates it can be installed with

.. code:: bash

    $ sudo apt-get install python-sphinx
    $ sudo -H pip install numpydoc

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

* **Simplicity**: it should be easy to use and extend the library. Whenever a feature is added, the effect on both of these should be considered. Every added abstraction should be carefully scrutinized, to determine whether the added complexity is justified.

* **Small interfaces**: as few classes and methods as possible. Try to rely on Theano's functionality and data types where possible, and follow Theano's conventions. Don't wrap things in classes if it is not strictly necessary. This should make it easier to both use the library and extend it (less cognitive overhead).

* **Don't get in the way**: unused features should be invisible, the user should not have to take into account a feature that they do not use. It should be possible to use each component of the library in isolation from the others.

* **Transparency**: don't try to hide Theano behind abstractions. Functions and methods should return Theano expressions and standard Python / numpy data types where possible.

* **Focus**: follow the Unix philosophy of "do one thing and do it well", with a strong focus on feed-forward neural networks.

* **Pragmatism**: making common use cases easy is more important than supporting every possible use case out of the box.



.. _issues page on GitHub: https://github.com/Lasagne/Lasagne/issues
.. _core team of seven people: https://github.com/orgs/Lasagne/teams/core-team
.. _numerous additional contributors: https://github.com/Lasagne/Lasagne/graphs/contributors
