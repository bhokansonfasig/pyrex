Code Examples
=============

The following code examples assume these imports::

    import numpy as np
    import matplotlib.pyplot as plt
    import pyrex

All of the following examples can also be found (and quickly run) in the Code Examples python notebook.

Working with Signal objects
---------------------------

The base ``Signal`` class is simply an array of times and an array of signal values, and is instantiated with these two arrays. The ``times`` array is assumed to be in units of seconds, but there are no general units for the ``values`` array (though it is commonly assumed to be in volts or volts per meter). It is worth noting that the Singal object stores shallow copies of the passed arrays, so changing the original arrays will not affect the ``Signal`` object. ::

    time_array = np.linspace(0, 10)
    value_array = np.sin(time_array)
    my_signal = pyrex.Signal(times=time_array, values=value_array)

Plotting the ``Signal`` object is as simple as plotting the times vs the values::

    plt.plot(my_signal.times, my_signal.values)

``Signal`` objects can be added as long as they have the same time array, and also support the python ``sum`` function::

    time_array = np.linspace(0, 10)
    values1 = np.sin(time_array)
    values2 = np.cos(time_array)
    signal1 = pyrex.Signal(time_array, values1)
    plt.plot(signal1.times, signal1.values, label="signal1")
    signal2 = pyrex.Signal(time_array, values2)
    plt.plot(signal2.times, signal2.values, label="signal2")
    signal3 = signal1 + signal2
    plt.plot(signal3.times, signal3.values, label="signal3")
    all_signals = [signal1, signal2, signal3]
    signal4 = sum(all_signals)
    plt.plot(signal4.times, signal4.values, label="signal4")
    plt.legend()

The ``Signal`` class provides many convenience attributes for dealing with signals::

    my_signal.dt == my_signal.times[1] - my_signal.times[0]
    my_signal.spectrum == scipy.fftpack.fft(my_signal.values)
    my_signal.frequencies == scipy.fftpack.fftfreq(n=len(my_signal.values),
                                                   d=my_signal.dt)
    my_signal.envelope == np.abs(scipy.signal.hilbert(my_signal.values))

The ``Signal`` class also provides functions for manipulating the signal. The ``resample`` function will resample the times and values arrays to the given number of points (with the same endpoints)::

    my_signal.resample(1001)
    len(my_signal.times) == len(my_signal.values) == 1001
    my_signal.times[0] == 0
    my_signal.times[-1] == 10

The ``filter_frequencies`` function will apply a frequency-domain filter to the values array based on the passed frequency response function::

    def lowpass_filter(frequency):
        if frequency < 1:
            return 1
        else:
            return 0
        
    time_array = np.linspace(0, 10, 1001)
    value_array = np.sin(0.1*2*np.pi*time_array) + np.sin(2*2*np.pi*time_array)
    my_signal = pyrex.Signal(times=time_array, values=value_array)

    plt.plot(my_signal.times, my_signal.values)
    my_signal.filter_frequencies(lowpass_filter)
    plt.plot(my_signal.times, my_signal.values)


A number of classes which inherit from the Signal class are included in PyREx: ``EmptySignal``, ``FunctionSignal``, ``AskaryanSignal``, and ``ThermalNoise``. ``EmptySignal`` is simply a signal whose values are all zero::

    time_array = np.linspace(0,10)
    empty = pyrex.EmptySignal(times=time_array)
    plt.plot(empty.times, empty.values)

``FunctionSignal`` takes a function of time and creates a signal based on that function::

    time_array = np.linspace(0,10)
    def square_wave(time):
        if int(time)%2==0:
            return 1
        else:
            return -1
    square_signal = pyrex.FunctionSignal(times=time_array, function=square_wave)
    plt.plot(square_signal.times, square_signal.values)

``AskaryanSignal`` produces an Askaryan pulse on a time array due to a neutrino of given energy observed at a given angle from the shower axis::

    time_array = np.linspace(-10e-9, 40e-9, 1001)
    neutrino_energy = 1e5 # TeV
    observation_angle = 45 * np.pi/180 # radians
    askaryan = pyrex.AskaryanSignal(times=time_array, energy=neutrino_energy,
                                    theta=observation_angle)
    plt.plot(askaryan.times, askaryan.values)

``ThermalNoise`` produces Rayleigh noise at a given temperature and resistance which has been passed through a bandpass filter of the given frequency range::

    time_array = np.linspace(-10e-9, 40e-9, 1001)
    noise_temp = 300 # K
    system_resistance = 1000 # ohm
    frequency_range = (550e6, 750e6) # Hz
    noise = pyrex.ThermalNoise(times=time_array, temperature=noise_temp,
                               resistance=system_resistance,
                               f_band=frequency_range)
    plt.plot(noise.times, noise.values)



More Examples
-------------

For more code examples, see the PyREx Demo python notebook.
