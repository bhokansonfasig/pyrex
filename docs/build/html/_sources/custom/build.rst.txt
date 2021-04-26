
In the course of using PyREx you may wish to change some behavior of parts of the code. Due to the modularity of the code, many behaviors should be customizable by substituting in your own classes inheriting from those already in PyREx. By adding these classes to your own custom module, your code can behave as though it was a native part of the PyREx package. Below the classes which can be easily substituted with your own version are listed, and descriptions of the behavior expected of the classes is outlined.


.. currentmodule:: pyrex

Askaryan Signal
---------------

The :class:`AskaryanSignal` class is responsible for storing the time-domain signal of the Askaryan signal produced by a particle shower. The :meth:`__init__` method of an :class:`AskaryanSignal`-like class must accept the arguments listed below:

==================== ================================================================
Attribute            Description
==================== ================================================================
``times``            A list-type (usually a numpy array) of time values at which to calculate the amplitude of the Askaryan pulse.
``particle``         A ``Particle`` object representing the neutrino that causes the event. Should have an ``energy``, ``vertex``, ``id``, and an ``interaction`` with an ``em_frac`` and ``had_frac``.
``viewing_angle``    The viewing angle in radians measured from the shower axis.
``viewing_distance`` The distance of the observation point from the shower vertex.
``ice``              The ice model to be used for describing the medium's index of refraction.
``t0``               The starting time of the Askaryan pulse / showers (default 0).
==================== ================================================================

The :meth:`__init__` method should result in a :class:`Signal` object with :attr:`values` being a numpy array of amplitudes corresponding to the given :attr:`times` and should have a proper :attr:`value_type`. Additionally, all methods of the :class:`Signal` class should be implemented (typically by just inheriting from :class:`Signal`).


Antenna / Antenna System
------------------------

The :class:`Antenna` class is primarily responsible for receiving and triggering on :class:`Signal` objects. The :meth:`__init__` method of an :class:`Antenna`-like class must accept a ``position`` argument, and any other arguments may be specified as desired. The :meth:`__init__` method should set the :attr:`position` attribute to the given argument. If not inheriting from :class:`Antenna`, the following methods and attributes must be implemented and may require the :meth:`__init__` method to set some other attributes. :class:`AntennaSystem`-like classes must expose the same required methods and attributes as :class:`Antenna`-like classes, typically by passing calls down to an underlying :class:`Antenna`-like object and applying some extra processing.

The :attr:`signals` attribute should contain a list of all pure :class:`Signal` objects that the antenna has seen. This is different from the :attr:`all_waveforms` attribute, which should contain a list of all waveform (pure signal + noise) :class:`Signal` objects the antenna has seen. Yet again different from the :attr:`waveforms` attribute, which should contain only those waveforms which have triggered the antenna.

If using the default :attr:`all_waveforms` and :attr:`waveforms`, a :attr:`_noises` attribute and :attr:`_triggers` attribute must be initialized to empty lists in :meth:`__init__`. Additionally a :meth:`make_noise` method must be defined which takes a ``times`` array and returns a :class:`Signal` object with noise amplitudes in the :attr:`values` attribute. If using the default :meth:`make_noise` method, a :attr:`_noise_master` attribute must be set in :meth:`__init__` to either ``None`` or a :class:`Signal` object that can generate noise waveforms (setting :attr:`_noise_master` to ``None`` and handling noise generation with the attributes :attr:`freq_range` and :attr:`noise_rms`, or :attr:`temperature` and :attr:`resistance`, is recommended).

A :meth:`full_waveform` method is required which will take a ``times`` array and return a :class:`Signal` object of the waveform the antenna sees at those times. If using the default :meth:`full_waveform`, a :attr:`noisy` attribute is required which contains a boolean value of whether or not the antenna includes noise in its waveforms. If :attr:`noisy` is ``True`` then a :meth:`make_noise` method is also required, as described in the previous paragraph.

An :attr:`is_hit` attribute is required which will be a boolean of whether or not the antenna has been triggered by any waveforms. Similarly an :meth:`is_hit_during` method is required which will take a ``times`` array and return a boolean of whether the antenna is triggered during those times.

The :meth:`trigger` method of the antenna should take a :class:`Signal` object and return a boolean of whether or not that signal would trigger the antenna.

The :meth:`clear` method should reset the antenna to a state of having received no signals (i.e. the state just after initialization), and should accept a boolean for ``reset_noise`` which will force the noise waveforms to be recalculated. If using the default :meth:`clear` method, the :attr:`_noises` and :attr:`_triggers` attributes must be lists.

A :meth:`receive` method is required which will take a :class:`Signal` object as ``signal``, a 3-vector (list) as ``direction``, and a 3-vector (list) as ``polarization``. This function doesn't return anything, but instead processes the input signal and stores it to the :attr:`signals` list (and anything else needed for the antenna to have officially received the signal). This is the final required method, but if using the default :meth:`receive` method, an :attr:`antenna_factor` attribute is needed to define the conversion from electric field to voltage and an :attr:`efficiency` attribute is required, along with four more methods which must be defined:

The :meth:`_convert_to_antenna_coordinates` method should take a point in cartesian coordinates and return the ``r``, ``theta``, and ``phi`` values of that point relative to the antenna. The :meth:`directional_gain` method should take ``theta`` and ``phi`` in radians and return a (complex) gain based on the directional response of the antenna. Similarly the :meth:`polarization_gain` method should take a ``polarization`` 3-vector (list) of an incoming signal and return a (complex) gain based on the polarization response of the antenna. Finally, the :meth:`response` method should take a list of frequencies and return the (complex) gains of the frequency response of the antenna. This assumes that the directional and frequency responses are separable. If this is not the case then the gains may be better handled with a custom :meth:`receive` method.


Detector
--------

The preferred method of creating your own detector class is to inherit from the :class:`Detector` class and then implement the :meth:`set_positions` method, the :meth:`triggered` method, and potentially the :meth:`build_antennas` method. However the only requirement of a :class:`Detector`-like object is that iterating over it will visit each antenna exactly once. This means that a simple list of antennas is an acceptable rudimentary detector. The advantages of using the :class:`Detector` class are easy breaking into subsets (a detector could be made up of stations, which in turn are made up of strings) and the simpler :meth:`triggered` method for trigger checks.


Ice Model
---------

Ice model classes are responsible for describing the properties of the ice as functions of depth and frequency. While not explicitly required, all ice model classes in PyREx are defined only with static and class methods, so no :meth:`__init__` method is actually necessary. The necessary methods, however, are as follows:

The :meth:`index` method should take a depth (or numpy array of depths) and return the corresponding index of refraction. Conversely, the :meth:`depth_with_index` method should take an index of refraction (or numpy array of indices) and return the corresponding depths. In the case of degeneracy here (for example with uniform ice), the recommended behavior is to return the shallowest depth with the given index, though PyREx's behavior in cases of non-monotonic index functions is not well defined.

The :meth:`temperature` method should take a depth (or numpy array of depths) and return the corresponding ice temperature in Kelvin.

Finally, the :meth:`attenuation_length` function should take a depth (or numpy array of depths) and a frequency (or numpy array of frequencies) and return the corresponding attenuation length. In the case of one scalar and one array argument, a simple 1D array should be returned. In the case of both arguments being arrays, the return value should be a 2D array where each row represents different frequencies at a single depth and each column represents different depths at a single frequency.


Ray Tracer / Ray Trace Path
---------------------------

The :class:`RayTracer` and :class:`RayTracePath` classes are responsible for handling ray tracing through the ice between shower vertices and antenna positions. The :class:`RayTracer` class finds the paths between the two points and the :class:`RayTracePath` calculates values along the path. Due to the potential for high calculation costs, the PyREx :class:`RayTracer` and :class:`RayTracePath` classes inherit from a :class:`LazyMutableClass` which allows the use of a :func:`lazy_property` decorator to cache results of attribute calculations. It is recommended that any other ray tracing classes consider doing this as well.

The :meth:`__init__` method of a :class:`RayTracer`-like class should take as arguments a 3-vector (list) ``from_point``, a 3-vector (list) ``to_point``, and an :class:`IceModel`-like ``ice_model``. The only required features of the class are a boolean attribute :attr:`exists` recording whether or not paths exist between the given points, and an iterable attribute :attr:`solutions` which iterates over :class:`RayTracePath`-like objects between the points.

A :class:`RayTracePath`-like class will be initialized by a corresponding :class:`RayTracer`-like object, so there are no requirements on its :meth:`__init__` method. The path must have :attr:`emitted_direction` and :attr:`received_direction` attributes which are numpy arrays of the cartesian direction the ray is pointing at the :attr:`from_point` and :attr:`to_point` of the ray tracer, respectively. The path must also have attributes for the :attr:`path_length` and :attr:`tof` (time of flight) along the path.

The path class must have a :meth:`propagate` method which takes a :class:`Signal` object as its argument and propagates that signal by applying any attenuation and time of flight. This method does not have a return value. Additionally, note that any 1/R factor that the signal could have is not applied in this method, but externally by dividing the signal values by the :attr:`path_length`. If using the default :meth:`propagate` method, an :meth:`attenuation` method is required which takes an array of frequencies ``f`` and returns the attenuation factors for a signal along the path at those frequencies.

Finally, though not required it is recommended that the path have a :attr:`coordinates` attribute which is a list of lists of the x, y, and z coordinates along the path (with some reasonable step size). This method is used for plotting purposes and does not need to have the accuracy necessary for calculations.


Interaction Model
-----------------

The interaction model used for :class:`Particle` interactions in ice handles the cross sections and interaction lengths of neutrinos, as well as the ratios of their interaction types and the resulting shower fractions. An interaction class should inherit from :class:`Interaction` (preferably keeping its :meth:`__init__` method) and should implement the following methods:

The :attr:`cross_section` property method should return the neutrino cross section for the :attr:`Interaction.particle` parent, specific to the :attr:`Interaction.kind`. Similarly the :attr:`total_cross_section` property method should return the neutrino cross section for the :attr:`Interaction.particle` parent, but this should be the total cross section for both charged-current and neutral-current interactions. The :attr:`interaction_length` and :attr:`total_interaction_length` properties will convert these cross sections to interaction lengths automatically.

The :meth:`choose_interaction` method should return a value from :class:`Interaction.Type` representing the interaction type based on a random choice. Similarly the :meth:`choose_inelasticity` method should return an inelasticity value based on a random choice, and the :meth:`choose_shower_fractions` method return calculate electromagnetic and hadronic fractions based on the :attr:`inelasticity` attribute storing the inelasticity value from :meth:`choose_inelasticity`. The :meth:`choose_shower_fractions` can be either chosen based on random processes like secondary generation or deterministic.


Particle Generator
------------------

The particle generator classes are quite flexible. The only requirement is that they possess an :meth:`create_event` method which returns a :class:`Event` object consisting of at least one :class:`Particle`. The :class:`Generator` base class provides a solid foundation for basic uniform generators in a volume, requiring only implementation of the :meth:`get_vertex` and :meth:`get_exit_points` methods for the specific volume at a minimum.
