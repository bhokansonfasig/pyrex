About PyREx
***********

PyREx (\ **Py**\ thon package for an IceCube **R**\ adio **Ex**\ tension) is, as its name suggests, a Python package designed to simulate the measurement of Askaryan pulses via a radio antenna array around the IceCube South Pole Neutrino Observatory.
The code is designed to be modular so that it can also be applied to other askaryan radio antennas (e.g. the ARA and ARIANA collaborations).


Installation
============

The easiest way to get the PyREx package is using ``pip`` as follows::

    pip install git+https://github.com/bhokansonfasig/pyrex#egg=pyrex

PyREx requires python version 3.6+ as well as numpy version 1.13+ and scipy version 0.19+, which should be automatically installed when installing via ``pip``.

Alternatively, you can download the code from https://github.com/bhokansonfasig/pyrex and then either include the ``pyrex`` directory (the one containing the python modules) in your ``PYTHON_PATH``, or just copy the ``pyrex`` directory into your working directory.
PyREx is not currently available on PyPI, so a simple ``pip install pyrex`` will not have the intended effect.


.. currentmodule:: pyrex

Quick Code Example
==================

The most basic simulation can be produced as follows:

First, import the package::

    import pryex

Then, create a particle generator object that will produce random particles in  a cube of 1 km on each side with a fixed energy of 100 PeV::

    particle_generator = pyrex.ShadowGenerator(dx=1000, dy=1000, dz=1000,
                                               energy=1e8)

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

Now the signals received by each antenna can be accessed by their :attr:`waveforms` parameter::

    import matplotlib.pyplot as plt
    for ant in kernel.ant_array:
        for wave in ant.waveforms:
            plt.figure()
            plt.plot(wave.times, wave.values)
            plt.show()


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
resistance              ohms (Ω)
voltage                 volts (V)
electric field          volts per meter (V/m)
======================= ========================================
