PyREx - (\ **Py**\ thon package for an IceCube **R**\ adio **Ex**\ tension)
***************************************************************************

PyREx (\ **Py**\ thon package for an IceCube **R**\ adio **Ex**\ tension) is, as its name suggests, a Python package designed to simulate the measurement of Askaryan pulses via a radio antenna array around the IceCube South Pole Neutrino Observatory.
The code is designed to be modular so that it can also be applied to other askaryan radio antennas (e.g. the ARA and ARIANA collaborations).


Useful Links
============

* Source (GitHub): https://github.com/bhokansonfasig/pyrex
* Documentation: https://bhokansonfasig.github.io/pyrex/
* Release notes: https://bhokansonfasig.github.io/pyrex/build/html/versions.html


Getting Started
===============

Requirements
------------

PyREx requires python version 3.6+ as well as numpy version 1.13+ and scipy version 0.19+.
After installing python from https://www.python.org/downloads/, numpy and scipy can be installed with ``pip`` as follows, or by simply installing pyrex as specified in the next section.

.. code-block:: shell
    pip install numpy>=1.13
    pip install scipy>=0.19

Installing
----------

The easiest way to get the PyREx package is using ``pip`` as follows

..code-block:: shell
    pip install git+https://github.com/bhokansonfasig/pyrex#egg=pyrex

Note that since PyREx is not currently available on PyPI, a simple ``pip install pyrex`` will not have the intended effect.


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
