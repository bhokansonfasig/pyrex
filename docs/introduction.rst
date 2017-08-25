Introduction
============

PyREx (\ **Py**\ thon package for an IceCube **R**\ adio **Ex**\ tension) is, as its name suggests, a python package designed to simulate the measurement of Askaryan pulses via a radio antenna array around the IceCube South Pole Neutrino Observatory.
The code is designed to be modular so that it can also be applied to other askaryan radio antennas, as in the ARA and ARIANA collaborations.


The most basic simulation can be produced as follows:

First, import the package::

    import pryex

Then, create a particle generator object that will produce random particles in  a cube of 1 km on each side with a fixed energy of 100 PeV::

    particle_generator = pyrex.ShadowGenerator(dx=1000, dy=1000, dz=1000, egen=lambda: 1e8)

An array of antennas that represent the detector is also needed. The base Antenna class provides a basic antenna with a flat frequency response and no trigger condition. Here we make a single vertical "string" of four antennas with no additional noise::

    antenna_array = []
    for z in [-100, -150, -200, -250]:
        antenna_array.append(
            pyrex.Antenna(position=(0,0,depth), temperature=300, freq_range=(1e6,1e9), noisy=False)
        )

Finally, we want to pass these into the EventKernel and produce an event::

    kernel = pyrex.EventKernel(generator=particle_generator, ice_model=pyrex.IceModel, antennas=antenna_array)
    kernel.event()

Now the signals received by each antenna can be accessed by their waveforms parameter::

    import matplotlib.pyplot as plt
    for ant in kernel.antennas:
        for wave in ant.waveforms:
            plt.figure()
            plt.plot(wave.times, wave.values)
            plt.show()
