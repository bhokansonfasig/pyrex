PyREx - (\ **Py**\ thon package for **R**\ adio **Ex**\ periments)
***************************************************************************

PyREx (\ **Py**\ thon package for **R**\ adio **Ex**\ periments) is a Python package designed to simulate the measurement of Askaryan pulses via a in-ice radio antenna arrays.
The code was written for the ARA collaboration with considerations for future radio arrays.
As such, the package is designed to be highly modular so that it can easily be used for other radio projects (e.g. ARIANNA, RNO, and IceCube Gen2).


Useful Links
============

* Source (GitHub): https://github.com/bhokansonfasig/pyrex/
* Documentation: https://bhokansonfasig.github.io/pyrex/
* Release notes: https://bhokansonfasig.github.io/pyrex/build/html/versions.html


Getting Started
===============

Requirements
------------

PyREx requires python version 3.6+ as well as numpy version 1.13+, scipy version 0.19+, and h5py version 2.7+.
After installing python from https://www.python.org/downloads/, the required packages can be installed with ``pip`` as follows, or they will be installed automatically by simply installing pyrex as specified in the next section.

.. code-block:: shell

    pip install numpy>=1.13
    pip install scipy>=0.19
    pip install h5py>=2.7

Installing
----------

The easiest way to get the PyREx package is using ``pip`` as follows

.. code-block:: shell

    pip install git+https://github.com/bhokansonfasig/pyrex#egg=pyrex

Note that since PyREx is not currently available on PyPI, a simple ``pip install pyrex`` will not have the intended effect.

Optional Dependencies
---------------------

The following packages are not required for running PyREx by default, but may be useful or required for running some specific parts of the code:

`matplotlib <https://matplotlib.org>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recommended version: 2.1+

Used for creating plots in example code and auxiliary scripts.

`PySpice <https://pyspice.fabrice-salvaire.fr>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recommended version: 1.1

Used by IREX sub-package for some complex front-end circuits. Not needed for default front-ends.


Examples
========

For examples of how to use PyREx, see the `usage page <https://bhokansonfasig.github.io/pyrex/build/html/usage.html>`_ and the `examples page <https://bhokansonfasig.github.io/pyrex/build/html/examples.html>`_ in the documentation, or the python notebooks in the `examples <https://github.com/bhokansonfasig/pyrex/tree/master/examples>`_ directory.


Contributing
============

Contributions to the code base are mostly handled through pull requests. Before contributing, for more information please read the `contribution page <https://bhokansonfasig.github.io/pyrex/build/html/contributing.html>`_ in the documentation.


Authors
=======

* Ben Hokanson-Fasig


License
=======

`MIT License <https://github.com/bhokansonfasig/pyrex/blob/master/LICENSE>`_

Copyright (c) 2018 Ben Hokanson-Fasig
