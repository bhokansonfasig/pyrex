Version History
===============

Version 1.0.2
-------------

* Allow passing of numpy arrays of depths and frequencies into most ``IceModel`` methods.

    * ``IceModel.gradient()`` must still be calculated at individual depths.

* Added ability to specify RMS voltage of ``ThermalNoise`` without providing temperature and resistance.

* Removed (deprecated) ``Antenna.isHit()``.

* Added ``Antenna.make_noise()`` method so custom antennas can use their own noise functions.

* Performance improvements:

    * Allowing for ``IceModel`` to calculate many attenuation lengths at once improves speed of ``PathFinder.propagate()``.

    * Improved speed of ``PathFinder.time_of_flight()`` and ``PathFinder.attenuation()`` (and improved accuracy to boot).

Version 1.0.1
-------------

* Fixed bugs in ``AskaryanSignal`` that caused the convolution to fail.

* Changed ``Antenna`` not require a temperature and frequency range if no noise is produced.

* Fixed bugs resulting from converting ``IceModel.temperature()`` from Celsius to Kelvin.



Version 1.0.0
-------------

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
-------------

Original PyREx python notebook written by Kael Hanson:

https://gist.github.com/physkael/898a64e6fbf5f0917584c6d31edf7940
