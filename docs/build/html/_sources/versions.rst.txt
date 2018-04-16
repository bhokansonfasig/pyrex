Version History
***************

Version 1.4.1
=============

.. rubric:: Changes

* Improved ray tracing and defaulted to the almost completely analytical ``SpecializedRayTracer`` and ``SpecializedRayTracePath`` classes as ``RayTracer`` and ``RayTracePath``

* Added ray tracer into ``EventKernel`` to replace ``PathFinder`` completely



Version 1.4.0
=============

.. rubric:: New Features

* Implemented full ray tracing in the ``RayTracer`` and ``RayTracePath`` classes.



Version 1.3.1
=============

.. rubric:: New Features

* Added diode bridge rectifier envelope circuit analytic model to ``irex.frontends`` and made it the default analytic envelope model in ``IREXAntennaSystem``.

* Added ``allow_reflection`` attribute to ``EventKernel`` class to determine whether ``ReflectedPathFinder`` solutions should be allowed.


.. rubric:: Changes

* Changed neutrino interaction model to include all neutrino and anti-neutrino interactions rather than only charged-current neutrino (relevant for ``ShadowGenerator`` class).



Version 1.3.0
=============

.. rubric:: New Features

* Added and implemented ``ReflectedPathFinder`` class for rays which undergo total internal reflection and subsequently reach an antenna.


.. rubric:: Changes

* Change ``AksaryanSignal`` angle to always be positive and remove < 90 degree restriction (Alvarez-Muniz, Romero-Wolf, & Zas paper suggests the algorithm should work for all angles).


.. rubric:: Performance Improvements

* Improve performance of ice index calculated at many depths.



Version 1.2.1
=============

.. rubric:: New Features

* Added ``set_orientation`` function to ``Antenna`` class for setting the ``z_axis`` and ``x_axis`` attributes appropriately.


.. rubric:: Bug Fixes

* Fixed bug where ``Antenna._convert_to_antenna_coordinates`` function was returning coordinates relative to (0,0,0) rather than the antenna's position.



Version 1.2.0
=============

.. rubric:: Changes

* Changed ``custom`` module to a package containing ``irex`` module.

* ``custom`` package leverages "Implicit Namespace Package" structure to allow plug-in style additions to the package in either the user's ``~/.pyrex-custom/`` directory or the ``./pyrex-custom`` directory.



Version 1.1.2
=============

.. rubric:: New Features

* Added ``with_times`` method to ``Signal`` class for interpolation/extrapolation of signals to different times.

* Added ``full_waveform`` and ``is_hit_during`` methods to ``Antenna`` class for calculation of waveform over arbitrary time array and whether said waveform triggers the antenna, respectively.

* Added ``front_end_processing`` method to ``IREXAntenna`` for processing envelope, amplifying signal, and downsampling result (downsampling currently inactive).


.. rubric:: Changes

* Change ``Antenna.make_noise`` to use a single master noise object and use ``with_times`` to calculate noise at different times.

    * To ensure noise is not obviously periodic (for <100 signals), uses 100 times the recommended number of frequencies, which results in longer computation time for noise waveforms.



Version 1.1.1
=============

.. rubric:: Changes

* Moved ``ValueTypes`` inside ``Signal`` class. Now access as ``Signal.ValueTypes.voltage``, etc.

* Changed signal envelope calculation in custom ``IREXAntenna`` from hilbert transform to a basic model. Spice model also available, but slower.



Version 1.1.0
=============

.. rubric:: New Features

* Added ``directional_gain`` and ``polarization_gain`` methods to base ``Antenna``.

    * ``receive`` method should no longer be overwritten in most cases.

    * ``Antenna`` now has orientation defined by ``z_axis`` and ``x_axis``.

    * ``antenna_factor`` and ``efficiency`` attributes added to ``Antenna`` for more flexibility.

* Added ``value_type`` attribute to ``Signal`` class and derived classes.

    * Current value types are ``ValueTypes.undefined``, ``ValueTypes.voltage``, ``ValueTypes.field``, and ``ValueTypes.power``.

    * ``Signal`` objects now must have the same ``value_type`` to be added (though those with ``ValueTypes.undefined`` can be coerced).


.. rubric:: Changes

* Made units consistent across PyREx.

* Added ability to define ``Antenna`` noise by RMS voltage rather than temperature and resistance if desired.

* Allow ``DipoleAntenna`` to guess at ``effective_height`` if not specified.


.. rubric:: Performance Improvements

* Increase speed of ``IceModel.__atten_coeffs`` method, resulting in increased speed of attenuation length calculations.



Version 1.0.3
=============

.. rubric:: New Features

* Added ``custom`` module to contain classes and functions specific to the IREX project.



Version 1.0.2
=============

.. rubric:: New Features

* Added ``Antenna.make_noise()`` method so custom antennas can use their own noise functions.


.. rubric:: Changes

* Allow passing of numpy arrays of depths and frequencies into most ``IceModel`` methods.

    * ``IceModel.gradient()`` must still be calculated at individual depths.

* Added ability to specify RMS voltage of ``ThermalNoise`` without providing temperature and resistance.

* Removed (deprecated) ``Antenna.isHit()``.


.. rubric:: Performance Improvements

* Allowing for ``IceModel`` to calculate many attenuation lengths at once improves speed of ``PathFinder.propagate()``.

* Improved speed of ``PathFinder.time_of_flight()`` and ``PathFinder.attenuation()`` (and improved accuracy to boot).



Version 1.0.1
=============

.. rubric:: Changes

* Changed ``Antenna`` not require a temperature and frequency range if no noise is produced.


.. rubric:: Bug Fixes

* Fixed bugs in ``AskaryanSignal`` that caused the convolution to fail.

* Fixed bugs resulting from converting ``IceModel.temperature()`` from Celsius to Kelvin.



Version 1.0.0
=============

* Created PyREx package based on original notebook.

* Added all signal classes to produce full-waveform Askaryan pulses and thermal noise.

* Changed ``Antenna`` class to ``DipoleAntenna`` to allow ``Antenna`` to be a base class.

* Changed ``Antenna.isHit()`` method to ``Antenna.is_hit`` property.

* Introduced ``IceModel`` alias for ``AntarcticIce`` (or any future preferred ice model).

* Moved ``AntarcticIce.attenuationLengthMN`` to its own ``NewcombIce`` class inheriting from ``AntarcticIce``.

* Added ``PathFinder.propagate()`` to propagate a ``Signal`` object in a customizable way.

* Changed naming conventions to be more consistent, verbose, and "pythonic":

    * ``AntarcticIce.attenuationLength()`` becomes ``AntarcticIce.attenuation_length()``.

    * In ``pyrex.earth_model``, ``RE`` becomes ``EARTH_RADIUS``.

    * In ``pyrex.particle``, ``neutrino_interaction`` becomes ``NeutrinoInteraction``.

    * In ``pyrex.particle``, ``NA`` becomes ``AVOGADRO_NUMBER``.

    * ``particle`` class becomes ``Particle`` namedtuple.

        * ``Particle.vtx`` becomes ``Particle.vertex``.

        * ``Particle.dir`` becomes ``Particle.direction``.

        * ``Particle.E`` becomes ``Particle.energy``.

    * In ``pyrex.particle``, ``next_direction()`` becomes ``random_direction()``.

    * ``shadow_generator`` becomes ``ShadowGenerator``.

    * ``PathFinder`` methods become properties where reasonable:

        * ``PathFinder.exists()`` becomes ``PathFinder.exists``.

        * ``PathFinder.getEmittedRay()`` becomes ``PathFinder.emitted_ray``.

        * ``PathFinder.getPathLength()`` becomes ``PathFinder.path_length``.

    * ``PathFinder.propagateRay()`` split into ``PathFinder.time_of_flight()`` (with corresponding ``PathFinder.tof`` property) and ``PathFinder.attenuation()``.



Version 0.0.0
=============

Original PyREx python notebook written by Kael Hanson:

https://gist.github.com/physkael/898a64e6fbf5f0917584c6d31edf7940
