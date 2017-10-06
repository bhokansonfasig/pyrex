Code Examples
=============

The following code examples assume these imports::

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.fftpack
    import pyrex

All of the following examples can also be found (and quickly run) in the Code Examples python notebook.



Working with Signal Objects
---------------------------

The base ``Signal`` class is simply an array of times and an array of signal values, and is instantiated with these two arrays. The ``times`` array is assumed to be in units of seconds, but there are no general units for the ``values`` array (though it is commonly assumed to be in volts or volts per meter). It is worth noting that the Singal object stores shallow copies of the passed arrays, so changing the original arrays will not affect the ``Signal`` object. ::

    time_array = np.linspace(0, 10)
    value_array = np.sin(time_array)
    my_signal = pyrex.Signal(times=time_array, values=value_array)

Plotting the ``Signal`` object is as simple as plotting the times vs the values::

    plt.plot(my_signal.times, my_signal.values)
    plt.show()

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
    plt.show()

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
    plt.show()


A number of classes which inherit from the Signal class are included in PyREx: ``EmptySignal``, ``FunctionSignal``, ``AskaryanSignal``, and ``ThermalNoise``. ``EmptySignal`` is simply a signal whose values are all zero::

    time_array = np.linspace(0,10)
    empty = pyrex.EmptySignal(times=time_array)
    plt.plot(empty.times, empty.values)
    plt.show()

``FunctionSignal`` takes a function of time and creates a signal based on that function::

    time_array = np.linspace(0,10)
    def square_wave(time):
        if int(time)%2==0:
            return 1
        else:
            return -1
    square_signal = pyrex.FunctionSignal(times=time_array, function=square_wave)
    plt.plot(square_signal.times, square_signal.values)
    plt.show()

``AskaryanSignal`` produces an Askaryan pulse on a time array due to a neutrino of given energy observed at a given angle from the shower axis::

    time_array = np.linspace(-10e-9, 40e-9, 1001)
    neutrino_energy = 1e8 # GeV
    observation_angle = 45 * np.pi/180 # radians
    askaryan = pyrex.AskaryanSignal(times=time_array, energy=neutrino_energy,
                                    theta=observation_angle)
    plt.plot(askaryan.times, askaryan.values)
    plt.show()

``ThermalNoise`` produces Rayleigh noise at a given temperature and resistance which has been passed through a bandpass filter of the given frequency range::

    time_array = np.linspace(-10e-9, 40e-9, 1001)
    noise_temp = 300 # K
    system_resistance = 1000 # ohm
    frequency_range = (550e6, 750e6) # Hz
    noise = pyrex.ThermalNoise(times=time_array, temperature=noise_temp,
                               resistance=system_resistance,
                               f_band=frequency_range)
    plt.plot(noise.times, noise.values)
    plt.show()



Antenna Class and Subclasses
----------------------------

The base ``Antenna`` class provided by PyREx is designed to be inherited from to match the needs of each project. At its core, an ``Antenna`` object is initialized with a position, a temperature, and a frequency range, as well as optionally a resistance for noise calculations and a boolean dictating whether or not noise should be added to the antenna's signals (note that if noise is to be added, a resistance must be specified). ::

    position = (0, 0, -100) # m
    temperature = 300 # K
    resistance = 1 # ohm
    frequency_range = (0, 1e3) # Hz
    basic_antenna = pyrex.Antenna(position=position, temperature=temperature,
                                  resistance=resistance,
                                  freq_range=frequency_range)
    noiseless_antenna = pyrex.Antenna(position=position, noisy=False)

The basic properties of an ``Antenna`` object are ``is_hit`` and ``waveforms``. ``is_hit`` specifies whether or not the antenna has been triggered by an event. ``waveforms`` is a list of all the waveforms which have triggered the antenna. The antenna also defines ``signals``, which is a list of all signals the antenna has received, and ``all_waveforms`` which is a list of all waveforms (signal plus noise) the antenna has received including those which didn't trigger. ::

    basic_antenna.is_hit == False
    basic_antenna.waveforms == []

The ``Antenna`` class defines three methods which are expected to be overwritten: ``trigger``, ``response``, and ``receive``. ``trigger`` takes a ``Signal`` object as an argument and returns a boolean of whether or not the antenna would trigger on that signal (default always returns ``True``). ``response`` takes a frequency or list of frequencies (in Hz) and returns the frequency response of the antenna at each frequency given (default always returns ``1``). ::

    basic_antenna.trigger(pyrex.Signal([0],[0])) == True
    freqs = [1, 2, 3, 4, 5]
    basic_antenna.response(freqs) == [1, 1, 1, 1, 1]

The ``receive`` method is a bit different in that it contains some default functionality::

    def receive(self, signal):
        copy = Signal(signal.times, signal.values)
        copy.filter_frequencies(self.response)
        self.signals.append(copy)

In this sense, the ``receive`` function is intended to be extended instead of overwritten. In derived classes, it is recommended that a newly defined ``receive`` function call ``super().receive(signal)``. For example, if a polarization is to be applied, the following ``receive`` function could be implemented::

    def receive(self, signal, signal_polarization):
        polarization_factor = np.vdot(self.polariztion, signal_polarization)
        polarized_signal = Signal(signal.times,
                                  signal.values * polarization_factor)
        super().receive(polarized_signal)

To use the ``receive`` function, simply pass it the ``Signal`` object the antenna sees, and the ``Antenna`` class will handle the rest::

    incoming_singal = pyrex.FunctionSignal(np.linspace(0,10), np.sin)
    basic_antenna.receive(incoming_singal)
    basic_antenna.is_hit == True
    for wave in basic_antenna.waveforms:
        plt.figure()
        plt.plot(wave.times, wave.values)
        plt.show()
    for pure_signal in basic_antenna.signals:
        plt.figure()
        plt.plot(pure_signal.times, pure_signal.values)
        plt.show()

The ``Antenna`` class also defines a ``clear`` method which will reset the antenna to a state of having received no signals::

    basic_antenna.clear()
    basic_antenna.is_hit == False
    len(basic_antenna.waveforms) == 0


To create a custom antenna, simply inherit from the ``Antenna`` class::

    class NoiselessThresholdAntenna(pyrex.Antenna):
        def __init__(self, position, threshold):
            super().__init__(position=position, noisy=False)
            self.threshold = threshold

        def trigger(self, signal):
            if max(np.abs(signal.values)) > self.threshold:
                return True
            else:
                return False

Our custom ``NoiselessThresholdAntenna`` should only trigger when the amplitude of a signal exceeds its threshold value::

    my_antenna = NoiselessThresholdAntenna(position=(0, 0, 0), threshold=2)

    incoming_singal = pyrex.FunctionSignal(np.linspace(0,10), np.sin)
    my_antenna.receive(incoming_singal)
    my_antenna.is_hit == False
    len(my_antenna.waveforms) == 0
    len(my_antenna.all_waveforms) == 1

    incoming_singal = pyrex.Signal(incoming_singal.times,
                                   5*incoming_singal.values)
    my_antenna.receive(incoming_singal)
    my_antenna.is_hit == True
    len(my_antenna.waveforms) == 1
    len(my_antenna.all_waveforms) == 2

    for wave in my_antenna.waveforms:
        plt.figure()
        plt.plot(wave.times, wave.values)
        plt.show()


PyREx defines ``DipoleAntenna`` which as a subclass of ``Antenna``, which provides a basic threshold trigger, a basic bandpass filter, and a polarization effect on the reception of a signal. A ``DipoleAntenna`` object is created as follows::

    antenna_identifier = "antenna 1"
    position = (0, 0, -100)
    center_frequency = 250e6 # Hz
    bandwidth = 300e6 # Hz
    resistance = 100 # ohm
    antenna_length = 3e8/center_frequency/2 # m
    polarization_direction = (0, 0, 1)
    trigger_threshold = 1e-5 # V
    dipole = pyrex.DipoleAntenna(name=antenna_identifier,position=position,
                                 center_frequency=center_frequency,
                                 bandwidth=bandwidth, resistance=resistance,
                                 effective_height=antenna_length,
                                 polarization=polarization_direction,
                                 trigger_threshold=trigger_threshold)



Ice and Earth Models
--------------------

PyREx provides a class ``IceModel``, which is an alias for whichever south pole ice model class is the preferred (currently just the basic ``AntarcticIce``). The ``IceModel`` class provides class methods for calculating characteristics of the ice at different depths and frequencies outlined below::

    depth = -1000 # m
    pyrex.IceModel.temperature(depth)
    pyrex.IceModel.index(depth)
    pyrex.IceModel.gradient(depth)
    frequency = 1e8 # Hz
    pyrex.IceModel.attenuation_length(depth, frequency)

PyREx also provides two functions realted to its earth model: ``prem_density`` and ``slant_depth``. ``prem_density`` calculates the density in grams per cubic centimeter of the earth at a given radius::

    radius = 6360000 # m
    pyrex.prem_density(radius)

``slant_depth`` calculates the material thickness in grams per square centimeter of a chord cutting through the earth at a given nadir angle, starting from a given depth::

    nadir_angle = 60 * np.pi/180 # radians
    depth = 1000 # m
    pyrex.slant_depth(nadir_angle, depth)



Particle Generation
-------------------

PyREx includes ``Particle`` as a container for information about neutrinos which are generated to produce Askaryan pulses. ``Particle`` contains three attributes: ``vertex``, ``direction``, and ``energy``::

    initial_position = (0,0,0) # m
    direction_vector = (0,0,-1)
    particle_energy = 1e8 # GeV
    pyrex.Particle(vertex=initial_position, direction=direction_vector,
                   energy=particle_energy)

PyREx also includes a ``ShadowGenerator`` class for generating random neutrinos, taking into account some Earth shadowing. The neutrinos are generated in a box of given size, and with an energy given by an energy generation function::

    box_width = 1000 # m
    box_depth = 500 # m
    const_energy_generator = lambda: 1e8 # GeV
    my_generator = pyrex.ShadowGenerator(dx=box_width, dy=box_width,
                                         dz=box_depth,
                                         energy_generator=const_energy_generator)
    my_generator.create_particle()



Ray Tracing
-----------

While PyREx does not currently support full ray tracing, it does provide a ``PathFinder`` class which implements some basic ray analysis by Snell's law. ``PathFinder`` takes an ice model and two points as arguments and provides a number of properties and methods regarding the path between the points. ::

    start = (0, 0, -100) # m
    finish = (0, 0, -250) # m
    my_path = pyrex.PathFinder(ice_model=pyrex.IceModel,
                               from_point=start, to_point=finish)

``PathFinder.exists`` is a boolean value of whether or not the path between the points is traversable according to the indices of refraction. ``PathFinder.emitted_ray`` is a unit vector giving the direction from ``from_point`` to ``to_point``. ``PathFinder.path_length`` is the length in meters of the straight line path between the two points. ::

    my_path.exists
    my_path.emitted_ray
    my_path.path_length

``PathFinder.time_of_flight()`` calculates the time it takes for light to traverse the path, with an optional parameter ``n_steps`` defining the precision used. ``PathFinder.tof`` is a convenience property set to the time of flight using the default value of ``n_steps``. ::

    my_path.time_of_flight(n_steps=100)
    my_path.time_of_flight() == my_path.tof

``PathFinder.attenuation()`` calculates the attenuation factor along the path for a signal of given frequency. Here again there is an optional parameter ``n_steps`` defining the precision used. ::

    frequency = 1e9 # Hz
    my_path.attenuation(f=frequency, n_steps=100)

Finally, ``PathFinder.propagate()`` propagates a ``Signal`` object from ``from_point`` to ``to_point`` by applying a ``1/PathFinder.path_length`` factor, applying the frequency attenuation of ``PathFinder.attenuation()``, and shifting the signal times by ``PathFinder.tof``::

    time_array = np.linspace(0, 5e-9, 1001)
    my_signal = (pyrex.FunctionSignal(time_array, lambda t: np.sin(1e9*2*np.pi*t))
                + pyrex.FunctionSignal(time_array, lambda t: np.sin(1e10*2*np.pi*t)))
    plt.plot(my_signal.times, my_signal.values)
    plt.show()

    my_path.propagate(my_signal)
    plt.plot(my_signal.times, my_signal.values)
    plt.show()



Full Simulation
---------------

PyREx provides the ``EventKernel`` class to control a basic simulation including the creation of neutrinos, the propagation of their pulses to the antennas, and the triggering of the antennas::

    particle_generator = pyrex.ShadowGenerator(dx=1000, dy=1000, dz=500,
                                               energy_generator=lambda: 1e8)
    detector = []
    for i, z in enumerate([-100, -150, -200, -250]):
        detector.append(
            pyrex.DipoleAntenna(name="antenna_"+str(i), position=(0, 0, z),
                                center_frequency=250e6, bandwidth=300e6,
                                resistance=0, effective_height=0.6,
                                trigger_threshold=0, noisy=False)
        )
    kernel = pyrex.EventKernel(generator=particle_generator,
                               ice_model=pyrex.IceModel,
                               antennas=detector)

    triggered = False
    while not triggered:
        kernel.event()
        for antenna in detector:
            if antenna.is_hit:
                triggered = True
                break
    
    for antenna in detector:
        for i, wave in enumerate(antenna.waveforms):
            plt.plot(wave.times * 1e9, wave.values)
            plt.xlabel("Time (ns)")
            plt.ylabel("Voltage (V)")
            plt.title(antenna.name + " - waveform "+str(i))




More Examples
-------------

For more code examples, see the PyREx Demo python notebook.
