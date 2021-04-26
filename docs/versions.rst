Version History
***************

.. currentmodule:: pyrex

Version 1.10.0
==============

.. rubric:: New Features

* Added a cut in the :class:`EventKernel` to skip the simulation of events with observation angles larger than a given value away from the Cherenkov angle.

* Added a cut in the :class:`EventKernel` to skip the simulation of events with weight(s) below a given value.

* Added an option to interpolate attenuation calculations to speed up ray tracing steps.

* Added a toggle in event generators to opt in/out of using the secondary interaction approximations.

* Added the ability to toggle generators between the expected neutrino ratios from astrophysical or cosmogenic origins.

* Added :attr:`volume` and :attr:`solid_angle` properties to generator objects.

* Earth model moved into :data:`earth`, an instance of the new :class:`PREM` class.

* Added the :class:`CoreMantleCrustModel` class to replicate the Earth model used in AraSim.

* Added :mod:`pyrex.custom.ara.stations` which implements the layouts of the deployed ARA stations in :class:`Detector` subclasses.

* New HDF5 file version 1.1 which changes how string data is handled due to changes in the :mod:`h5py` package.

* Added the :meth:`File.get_data` method to read arbitrary datasets from files.

.. rubric:: Changes

* Minimum versions of dependencies increased (:mod:`numpy`>=1.17, :mod:`scipy`>=1.4, :mod:`h5py`>=3.0).

* Added support for Askaryan signals observed directly at the Cherenkov angle.

* Implemented alternative form factor for hadronic interactions in Askaryan model.

* Askaryan models now inherit from :class:`FunctionSignal`, improving the accuracy of their values after the application of multiple filters.

* Added optional buffers to the start and end of :class:`FunctionSignal` times.

* Changed the default times array of the :class:`EventKernel` to be symmetric about zero.

* Implemented extrapolation of attenuation data in :class:`ArasimIce` to match the implementation in AraSim.

* All physical constants now pulled from :mod:`scipy.constants` rather than hard-coded.

* Removed the previously deprecated classes: :class:`IceModel`, :class:`ShadowGenerator`, and :class:`NumpyFileGenerator`.

* Renamed :class:`ARVZAskaryanSignal` to :class:`ARZAskaryanSignal` to properly match the paper authors.

.. rubric:: Bug Fixes

* Fixed the relative timing of Askaryan pulses inside vs outside of the Cherenkov cone.

* Fixed a numerical error in the attenuation calculation in deep ice with slowly changing index of refraction.

* Fixed an error in the indirect ray launch angle minimization in deep ice.

* Improved the slant depth calculation in the Earth models to use the proper endpoint in the ice rather than (0, 0).

* Fixed the calculation of the :class:`ThermalNoise` rms value based on a proper antenna system connected to ground.

.. rubric:: Performance Improvements

* :class:`FunctionSignal` values are now lazily evaluated, including delayed application of filters.

* Improved performance of Askaryan signal calculation by shifting the relevant portion of the convolution rather than calculating and truncating a larger convolution.

* Default :class:`ThermalNoise` class now uses a faster FFT-based calculation. The old behavior was moved to the :class:`FullThermalNoise` class.

* HDF5 file reader improved to read data in chunks rather than line-by-line.



Version 1.9.0
=============

.. rubric:: New Features

* Added a :class:`GreenlandIce` ice model based on measurements at Summit Station.

* Added the :class:`AVZAskaryanSignal` Askaryan model from J. Alvarez-Muniz et al (2000) for comparison across simulation packages.

* Added :class:`UniformIce` ice model with a constant index of refraction, along with the corresponding ray tracing with :class:`UniformRayTracer` and :class:`UniformRayTracePath`.

* Added :mod:`pyrex.custom.layered_ice` for ice models with indices of refraction defined in stratified layers in the z-direction and the corresponding ray tracing.

* Added :func:`complex_interp` and :func:`complex_bilinear_interp` functions for interpolating arrays of complex values using Cartesian or Euler methods.

.. rubric:: Changes

* :class:`IceModel` has been deprecated in favor of the :data:`ice` object, an object of the preferred ice model class (currently :class:`AntarcticIce`).

* Removed the :class:`NewcombIce` ice model.

* Handling of s-polarized and p-polarized signals by the :class:`RayTracer` have now been split. This resulted in different behavior for the :meth:`RayTracePath.propagate` and :meth:`Antenna.receive` methods, as well as a new method :meth:`Antenna.apply_response`.

* Renamed :meth:`Antenna.response` method to :meth:`Antenna.frequency_response`.

* ARA antennas have been updated to use the latest XFDTD simulations from the Chiba group.

* The behavior of the :class:`CylindricalShadowGenerator` and :class:`RectangularShadowGenerator` classes have been moved into the :class:`CylindricalGenerator` and :class:`RectangularGenerator` classes, respectively, when the ``shadow`` argument is set to ``True``.

.. rubric:: Bug Fixes

* The handling of phase shifts between s-polarized and p-polarized signals during total internal reflection is now properly implemented.

* Fixed a bug in the antenna response of ARA antennas which resulted in an extra factor of sin(theta) for :class:`VpolAntenna` objects and had an unexplored effect on :class:`HpolAntenna` objects.

* Fixed an off-by-one error in the final event count given by :class:`ListGenerator` objects.

* Fixed a bug in the Earth shadowing effect produced by :class:`CylindricalGenerator` and :class:`RectangularGenerator` classes.



Version 1.8.2
=============

.. rubric:: New Features

* Added :class:`CylindricalGenerator`, :class:`RectangularGenerator`, :class:`CylindricalShadowGenerator`, and :class:`RectangularShadowGenerator` classes to provide options for generation volumes and how to account for shadowing by the Earth. :class:`RectangularShadowGenerator` has the same behavior as the existing :class:`ShadowGenerator`, so :class:`ShadowGenerator` is being deprecated.

* Added ability to add :class:`Detector` (and :class:`Antenna` or :class:`AntennaSystem`) objects into a :class:`CombinedDetector` for ease of use.

* Added ability to multiply (and divide) :class:`Signal` objects by numeric types.

* Added support for total events thrown in simulation, accessed by the :attr:`File.total_events_thrown` attribute of file readers.

.. rubric:: Changes

* Separated :attr:`Particle.weight` into :attr:`Particle.survival_weight` and :attr:`Particle.interaction_weight`. Now :attr:`Particle.weight` serves as a convenience attribute which gives the product of the two weights.

* Changed :class:`FileGenerator` to read from simulation output files rather than numpy files. For the time being numpy files can be read with :class:`NumpyFileGenerator`.

* All generator classes now have a :attr:`count` attribute for keeping track of the total number of events thrown.

.. rubric:: Bug Fixes

* Fixed error in :class:`ARAAntenna` signal amplitudes introduced in version 1.8.1.

* Fixed minor bugs in :class:`File` interfaces.

.. rubric:: Performance Improvements

* Changed :class:`AskaryanSignal` charge profile and RAC calculations to accept numpy arrays. Should result in marginal improvements in signal calculation speed.



Version 1.8.1
=============

.. rubric:: New Features

* Added ability to write (and subsequently read) simulation data files using :class:`File` objects.

* File I/O supports HDF5 files, but should be considered to be in a public-beta state until the release of version 1.9.0.

* :class:`EventKernel` now accepts ``event_writer`` and ``triggers`` arguments for writing simulation data to output files.

.. rubric:: Changes

* :class:`ThermalNoise` now uses Rayleigh-distributed amplitudes in frequency space by default.

* Handling of signal polarizations has been more closely integrated with the ray tracer; :meth:`RayTracer.propagate` now propagating the polarization vector as well as the signal.

* 3 dB splitter effect moved from :meth:`ARAAntenna.response` to :meth:`ARAAntennaSystem.front_end` for a more logical separation of antenna and front-end.

* Adjusted default noise rms of :class:`ARIANNAAntennaSystem` to the expected value in ARIANNA.

.. rubric:: Bug Fixes

* Corrected signal polarization calculation.

* Fixed calculation of fresnel factors in surface reflection.

* Fixed bug in antenna gains of asymmetric antennas for theta angles near 180 degrees.

* Corrected effective height of antennas modeled by WIPL-D (i.e. :class:`LPDA`).



Version 1.8.0
=============

.. rubric:: New Features

* Added model of the ARIANNA LPDA based primarily on the implementation in `NuRadioReco <https://github.com/nu-radio/NuRadioReco>`_.

* Added :attr:`Antenna.is_hit_mc` and :attr:`AntennaSystem.is_hit_mc` which test noise-only triggers to determine whether a triggered antenna as truly triggered by signal or not.

* Added ``require_mc_truth`` argument to :meth:`Detector.triggered` to toggle whether a true Monte Carlo signal trigger (described above with :attr:`Antenna.is_hit_mc`) is required for a detector trigger.

* Added :attr:`AntennaSystem.lead_in_time` which allows front-end systems time to equilibrate before waveforms are recorded.

.. rubric:: Changes

* :attr:`Antenna.waveforms` and :attr:`Antenna.all_waveforms` now include all relevant signals in the waveform during that time, similar to :meth:`Antenna.full_waveform`.

* :meth:`ARAAntenna.interpolate_filter` moved to :meth:`ARAAntennaSystem.interpolate_filter`, since this better matches the logical location of the front-end electronics.

.. rubric:: Bug Fixes

* Fixed error in calculation of ARA Hpol polarization gain.

* Corrected amplification of :class:`ARAAntennaSystem` (previously was silently ignored).

* Corrected tunnel diode and other triggers to use standard deviation from mean rather than rms.

* Fixed accidental duplication of antennas when :meth:`Detector.build_antennas` is called more than once.

* Fixed numerical issue when checking that antenna axes are perpendicular.



Version 1.7.0
=============

.. rubric:: New Features

* Moved :mod:`pyrex.custom.ara` module into main PyREx package instead of being a plug-in.

* All docstrings now follow numpy docstring style.

* Added particle types and interaction information to :class:`Particle` class.

* Added :class:`Interaction` classes :class:`GQRSInteraction` and :class:`CTWInteraction` for defining different neutrino interaction models. Preferred model (:class:`CTWInteraction`) aliased to :class:`NeutrinoInteraction`.

* Added :meth:`ShadowGenerator.get_vertex`, :meth:`ShadowGenerator.get_direction`, :meth:`ShadowGenerator.get_particle_type`, :meth:`ShadowGenerator.get_exit_points`, and :meth:`ShadowGenerator.get_weight` methods for generating neutrinos more modularly.

* Added :class:`Event` class for holding a tree of :class:`Particle` objects. :class:`Event` objects are now returned by generators and the :class:`EventKernel`.

* Added :class:`ZHSAskaryanSignal` class for the Zas, Halzen, Stanev parameterization of Askaryan pulses. Mostly for comparison purposes.

.. rubric:: Changes

* :meth:`ShadowGenerator.create_particle` changed to :meth:`ShadowGenerator.create_event` and now returns an `Event` object.

* Generator classes moved to :mod:`pyrex.generation` module.

* :class:`Signal.ValueTypes` changed to :class:`Signal.Type` to match :class:`Particle.Type` and :class:`Interaction.Type`.

* :class:`FastAskaryanSignal` changed to :class:`ARVZAskaryanSignal`. This class is still the preferred parameterization aliased to :class:`AskaryanSignal`.

* Arguments of :class:`AskaryanSignal` changed to take a :class:`Particle` object rather than taking its parameters individually.

* Removed unused :class:`SlowAskaryanSignal`.

* Now that :class:`AskaryanSignal` can handle different particle and shower types, secondary particle generation was added to determine shower fractions: :attr:`NeutrinoInteraction.em_frac` and :attr:`NeutrinoInteraction.had_frac`.

* Changed IREX envelope antennas to be an envelope front-end on top of an ARA antenna. Results in :class:`IREXAntennaSystem` becoming :class:`EnvelopeHpol` and :class:`EnvelopeVpol`.



Version 1.6.0
=============

.. rubric:: New Features

* :class:`EventKernel` can now take arguments to specify the ray tracer to be used and the times array to be used in signal generation.

* Added shell scripts to more easily work with git branching model.

.. rubric:: Changes

* :class:`ShadowGenerator` ``energy_generator`` argument changed to ``energy`` and can now take a function or a scalar value, in which case all particles will have that scalar value for their energy.

* :class:`EventKernel` now uses :class:`pyrex.IceModel` as its ice model by default.

* :meth:`Antenna.receive` method (and :meth:`receive` method of all inheriting antennas) now uses ``direction`` argument instead of ``origin`` argument to calculate directional gain.

* :meth:`Antenna.clear` and :meth:`Detector.clear` functions can now optionally reset the noise calculation by using the ``reset_noise`` argument.

* :class:`Antenna` classes can now set the ``unique_noise_waveforms`` argument to specify the expected number of unique noise waveforms needed.

* :meth:`ArasimIce.attenuation_length` changed to more closely match AraSim.

* :class:`IceModel` reverted to :class:`AntarcticIce` with new index of refraction coefficients matching those of :class:`ArasimIce`.

* :func:`prem_density` can now be calculated for an array of radii.

.. rubric:: Performance Improvements

* Improved performance of :func:`slant_depth` calculation.

* Improved performance of :meth:`IceModel.attenuation_length` calculation.

* Using the :class:`Antenna` ``unique_noise_waveforms`` argument can improve noise waveform calculation speed (previously assumed 100 unique waveforms were necessary).

.. rubric:: Bug Fixes

* Fixed received direction bug in :class:`EventKernel`, which had still been assuming a straight-ray path.

* Lists in function keyword arguments were changed to tuples to prevent unexpected mutability issues.

* Fixed potential errors in :class:`BasicRayTracer` and :class:`BasicRayTracePath`.



Version 1.5.0
=============

.. rubric:: Changes

* Changed structure of :class:`Detector` class so a detector can be built up from strings to stations to the full detector.

* :attr:`Detector.antennas` attribute changed to :attr:`Detector.subsets`, which contains the pieces which make up the detector (e.g. antennas on a string, strings in a station).

* Iterating the :class:`Detector` class directly retains its effect of iterating each antenna in the detector directly.

.. rubric:: New Features

* Added :meth`Detector.triggered` and :meth:`Detector.clear` methods.

* Added two new neutrino generators :class:`ListGenerator` and :class:`FileGenerator` designed to pull pre-generated :class:`Particle` objects.

.. rubric:: Bug Fixes

* Preserve :attr:`value_type` of :class:`Signal` objects passed to :meth:`IREXAntennaSystem.front_end`.



Version 1.4.2
=============

.. rubric:: Performance Improvements

* Improved performance of :class:`FastAskaryanSignal` by reducing the size of the convolution.

.. rubric:: Changes

* Adjusted time step of signals generated by kernel slightly (2000 steps instead of 2048).



Version 1.4.1
=============

.. rubric:: Changes

* Improved ray tracing and defaulted to the almost completely analytical :class:`SpecializedRayTracer` and :class:`SpecializedRayTracePath` classes as :class:`RayTracer` and :class:`RayTracePath`.

* Added ray tracer into :class:`EventKernel` to replace :class:`PathFinder` completely.



Version 1.4.0
=============

.. rubric:: New Features

* Implemented full ray tracing in the :class:`RayTracer` and :class:`RayTracePath` classes.



Version 1.3.1
=============

.. rubric:: New Features

* Added diode bridge rectifier envelope circuit analytic model to :mod:`irex.frontends` and made it the default analytic envelope model in :class:`IREXAntennaSystem`.

* Added :attr:`allow_reflection` attribute to :class:`EventKernel` class to determine whether :class:`ReflectedPathFinder` solutions should be allowed.


.. rubric:: Changes

* Changed neutrino interaction model to include all neutrino and anti-neutrino interactions rather than only charged-current neutrino (relevant for :class:`ShadowGenerator` class).



Version 1.3.0
=============

.. rubric:: New Features

* Added and implemented :class:`ReflectedPathFinder` class for rays which undergo total internal reflection and subsequently reach an antenna.


.. rubric:: Changes

* Change :class:`AskaryanSignal` angle to always be positive and remove < 90 degree restriction (Alvarez-Muniz, Romero-Wolf, & Zas paper suggests the algorithm should work for all angles).


.. rubric:: Performance Improvements

* Improve performance of ice index calculated at many depths.



Version 1.2.1
=============

.. rubric:: New Features

* Added :meth:`Antenna.set_orientation` method for setting the :attr:`z_axis` and :attr:`x_axis` attributes appropriately.


.. rubric:: Bug Fixes

* Fixed bug where :meth:`Antenna._convert_to_antenna_coordinates` function was returning coordinates relative to (0,0,0) rather than the antenna's position.



Version 1.2.0
=============

.. rubric:: Changes

* Changed :mod:`custom` module to a package containing :mod:`irex` module.

* :mod:`custom` package leverages "Implicit Namespace Package" structure to allow plug-in style additions to the package in either the user's ``~/.pyrex-custom/`` directory or the ``./pyrex-custom`` directory.



Version 1.1.2
=============

.. rubric:: New Features

* Added :meth:`Signal.with_times` method for interpolation/extrapolation of signals to different times.

* Added :meth:`Antenna.full_waveform` and :meth:`Antenna.is_hit_during` methods for calculation of waveform over arbitrary time array and whether said waveform triggers the antenna, respectively.

* Added :meth:`IREXAntenna.front_end_processing` method for processing envelope, amplifying signal, and downsampling result (downsampling currently inactive).


.. rubric:: Changes

* Change :meth:`Antenna.make_noise` to use a single master noise object and use :meth:`ThermalNoise.with_times` to calculate noise at different times.

  * To ensure noise is not obviously periodic (for <100 signals), uses 100 times the recommended number of frequencies, which results in longer computation time for noise waveforms.



Version 1.1.1
=============

.. rubric:: Changes

* Moved :obj:`ValueTypes` inside :class:`Signal` class. Now access as :attr:`Signal.ValueTypes.voltage`, etc.

* Changed signal envelope calculation in custom :class:`IREXAntenna` from hilbert transform to a basic model. Spice model also available, but slower.



Version 1.1.0
=============

.. rubric:: New Features

* Added :meth:`Antenna.directional_gain` and :meth:`Antenna.polarization_gain` methods to base :class:`Antenna`.

  * :meth:`Antenna.receive` method should no longer be overwritten in most cases.

  * :class:`Antenna` now has orientation defined by :attr:`z_axis` and :class:`x_axis`.

  * :attr:`antenna_factor` and :attr:`efficiency` attributes added to :class:`Antenna` for more flexibility.

* Added :attr:`value_type` attribute to :class:`Signal` class and derived classes.

  * Current value types are :attr:`ValueTypes.undefined`, :attr:`ValueTypes.voltage`, :attr:`ValueTypes.field`, and :attr:`ValueTypes.power`.

  * :class:`Signal` objects now must have the same :attr:`value_type` to be added (though those with :attr:`ValueTypes.undefined` can be coerced).


.. rubric:: Changes

* Made units consistent across PyREx.

* Added ability to define :class:`Antenna` noise by RMS voltage rather than temperature and resistance if desired.

* Allow :class:`DipoleAntenna` to guess at :attr:`effective_height` if not specified.


.. rubric:: Performance Improvements

* Increase speed of :meth:`IceModel.__atten_coeffs` method, resulting in increased speed of attenuation length calculations.



Version 1.0.3
=============

.. rubric:: New Features

* Added :mod:`custom` module to contain classes and functions specific to the IREX project.



Version 1.0.2
=============

.. rubric:: New Features

* Added :meth:`Antenna.make_noise` method so custom antennas can use their own noise functions.


.. rubric:: Changes

* Allow passing of numpy arrays of depths and frequencies into most :class:`IceModel` methods.

  * :meth:`IceModel.gradient` must still be calculated at individual depths.

* Added ability to specify RMS voltage of :class:`ThermalNoise` without providing temperature and resistance.

* Removed (deprecated) :meth:`Antenna.isHit`.


.. rubric:: Performance Improvements

* Allowing for :class:`IceModel` to calculate many attenuation lengths at once improves speed of :meth:`PathFinder.propagate`.

* Improved speed of :meth:`PathFinder.time_of_flight` and :meth:`PathFinder.attenuation` (and improved accuracy to boot).



Version 1.0.1
=============

.. rubric:: Changes

* Changed :class:`Antenna` to not require a temperature and frequency range if no noise is produced.


.. rubric:: Bug Fixes

* Fixed bugs in :class:`AskaryanSignal` that caused the convolution to fail.

* Fixed bugs resulting from converting :meth:`IceModel.temperature` from Celsius to Kelvin.



Version 1.0.0
=============

* Created PyREx package based on original notebook.

* Added all signal classes to produce full-waveform Askaryan pulses and thermal noise.

* Changed :class:`Antenna` class to :class:`DipoleAntenna` to allow :class:`Antenna` to be a base class.

* Changed :meth:`Antenna.isHit` method to :attr:`Antenna.is_hit` property.

* Introduced :class:`IceModel` alias for :class:`AntarcticIce` (or any future preferred ice model).

* Moved :meth:`AntarcticIce.attenuationLengthMN` to its own :class:`NewcombIce` class inheriting from :class:`AntarcticIce`.

* Added :meth:`PathFinder.propagate` to propagate a :class:`Signal` object in a customizable way.

* Changed naming conventions to be more consistent, verbose, and "pythonic":

  * :meth:`AntarcticIce.attenuationLength` becomes :meth:`AntarcticIce.attenuation_length`.

  * In :mod:`pyrex.earth_model`, :const:`RE` becomes :const:`EARTH_RADIUS`.

  * In :mod:`pyrex.particle`, :class:`neutrino_interaction` becomes :class:`NeutrinoInteraction`.

  * In :mod:`pyrex.particle`, :const:`NA` becomes :const:`AVOGADRO_NUMBER`.

  * :class:`particle` class becomes :class:`Particle` namedtuple.

  * :attr:`Particle.vtx` becomes :attr:`Particle.vertex`.

  * :attr:`Particle.dir` becomes :attr:`Particle.direction`.

  * :attr:`Particle.E` becomes :attr:`Particle.energy`.

  * In :mod:`pyrex.particle`, :func:`next_direction()` becomes :func:`random_direction()`.

  * :class:`shadow_generator` becomes :class:`ShadowGenerator`.

  * :meth:`PathFinder.exists()` method becomes :attr:`PathFinder.exists` property.

  * :meth:`PathFinder.getEmittedRay()` method becomes :attr:`PathFinder.emitted_ray` property.

  * :meth:`PathFinder.getPathLength()` method becomes :attr:`PathFinder.path_length` property.

  * :meth:`PathFinder.propagateRay()` split into :meth:`PathFinder.time_of_flight()` (with corresponding :attr:`PathFinder.tof` property) and :meth:`PathFinder.attenuation()`.



Version 0.0.0
=============

Original PyREx python notebook written by Kael Hanson:

https://gist.github.com/physkael/898a64e6fbf5f0917584c6d31edf7940
