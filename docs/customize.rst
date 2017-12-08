.. _custom-package:

Custom Sub-Package
==================

While the PyREx package provides a basis for simulation, the real benefits come in customizing the analysis for different purposes. To this end the custom sub-package allows for plug-in style modules to be distributed for different collaborations.

By default PyREx comes with a custom module for IREX (IceCube Radio Extension) accessible at ``pyrex.custom.irex``. This module includes a more thorough ``IREXAntenna`` class inheriting from the ``Antenna`` class which adds a front-end for amplifying the signal, processing signal envelopes, and downsampling the result. It also includes an ``IREXDetector`` class designed to easily produce different geometries of ``IREXAntenna`` objects.

Other institutions and research groups are encouraged to create their own custom modules to integrate with PyREx. These modules have full access to PyREx as if they were a native part of the package. When PyREx is loaded it automatically scans for these custom modules in certain parts of the filesystem and includes any modules that it can find.
The first place searched is the ``custom`` directory in the PyREx package itself. Next, if a ``.pyrex-custom`` directory exists in the user's home directory (note the leading ``.``), its subdirectories are searched for ``custom`` directories and any modules in these directories are included. Finally, if a ``pyrex-custom`` directory exists in the current working directory (this time without the leading ``.``), its subdirectories are similarly scanned for modules inside ``custom`` directories. Note that if any name-clashing occurs, the first result found takes precedence (without warning). Additionally, none of these ``custom`` directories should contain an ``__init__.py`` file, or else the plug-in system may not work (For more information on the implementation, see PEP 420 and/or David Beazley's 2015 PyCon talk on Modules and Packages at https://youtu.be/0oTh1CXRaQ0?t=1h25m45s).

As an example, in the following filesystem layout available custom modules are ``pyrex.custom.pyspice``, ``pyrex.custom.irex``, ``pyrex.custom.ara``, ``pyrex.custom.ariana``, and ``pyrex.custom.my_analysis``:

.. raw:: latex

    \newpage

.. code-block:: none

    /path/to/site-packages/pyrex/
    |-- __init__.py
    |-- signals.py
    |-- antenna.py
    |-- ...
    |-- custom/
    |   |-- pyspice.py
    |   |-- irex/
    |   |   |-- __init__.py
    |   |   |-- antenna.py
    |   |   |-- ...

    /path/to/home_dir/.pyrex-custom/
    |-- ara/
    |   |-- custom/
    |   |   |-- ara/
    |   |   |   |-- __init__.py
    |   |   |   |-- antenna.py
    |   |   |   |-- ...
    |-- ariana/
    |   |-- custom/
    |   |   |-- ariana.py

    /path/to/cwd/pyrex-custom/
    |-- my_analysis_module/
    |   |-- custom/
    |   |   |-- my_analysis.py
