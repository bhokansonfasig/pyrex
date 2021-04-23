About PyREx
***********

PyREx (\ **Py**\ thon package for **R**\ adio **Ex**\ periments) is a Python package designed to simulate the measurement of Askaryan pulses via in-ice radio antenna arrays.
The code was written for the ARA collaboration with considerations for future radio arrays.
As such, the package is designed to be highly modular so that it can easily be used for other radio projects (e.g. ARIANNA, RNO, and IceCube Gen2).


Installation
============

The easiest way to get the PyREx package is using ``pip`` as follows:

.. code-block:: shell

    pip install git+https://github.com/bhokansonfasig/pyrex#egg=pyrex

PyREx requires python version 3.6+ as well as numpy version 1.17+, scipy version 1.4+, and h5py version 3.0+, which should be automatically installed when installing via ``pip``.

Alternatively, you can download the code from https://github.com/bhokansonfasig/pyrex/ and then either include the ``pyrex`` directory (the one containing the python modules) in your ``PYTHON_PATH``, or just copy the ``pyrex`` directory into your working directory.
PyREx is not currently available on PyPI, so a simple ``pip install pyrex`` will not have the intended effect.


.. currentmodule:: pyrex

Quick Code Example
==================

The most basic simulation can be produced as follows:

First, import the package::

    import pyrex

Then, create a particle generator object that will produce random neutrino interactions in a cylinder with radius and depth of 1 km and with a fixed energy of 100 PeV::

    particle_generator = pyrex.CylindricalGenerator(dr=1000, dz=1000, energy=1e8)

An array of antennas that represent the detector is also needed. The base :class:`Antenna` class provides a basic antenna with a flat frequency response and no trigger condition. Here we make a single vertical "string" of four antennas with no noise::

    antenna_array = []
    for z in [-100, -150, -200, -250]:
        antenna_array.append(
            pyrex.Antenna(position=(0,0,z), noisy=False)
        )

Finally, we want to pass these into the :class:`EventKernel` and produce an event::

    kernel = pyrex.EventKernel(generator=particle_generator,
                               antennas=antenna_array)
    kernel.event()

Now the signals which triggered each antenna can be accessed by each antenna's :attr:`waveforms` parameter::

    import matplotlib.pyplot as plt
    for ant in kernel.antennas:
        for wave in ant.waveforms:
            plt.plot(wave.times, wave.values)
            plt.show()

Note that it may take a few attempts before an event is generated which produces a visible signal on the antennas!


Units
=====

For ease of use, PyREx tries to use consistent units in all classes and functions. The units used are mostly SI with a few exceptions listed in bold below:

======================= ========================================
Metric                  Unit
======================= ========================================
time                    seconds (s)
frequency               hertz (Hz)
distance                meters (m)
**density**             **grams per cubic centimeter (g/cm^3)**
**material thickness**  **grams per square centimeter (g/cm^2)**
temperature             kelvin (K)
**energy**              **gigaelectronvolts (GeV)**
resistance              ohms
voltage                 volts (V)
electric field          volts per meter (V/m)
======================= ========================================
