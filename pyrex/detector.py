"""
Module containing higher-level detector-related classes.

The classes in this module are responsible for higher-level operations of
the antennas and detectors than in the antenna module. This includes
functions like front-end electronics chains and trigger systems.

"""

from collections.abc import Iterable
import inspect
import logging
import numpy as np
from pyrex.internal_functions import flatten, mirror_func

logger = logging.getLogger(__name__)


class AntennaSystem:
    """
    Base class for antenna system with front-end processing.

    Behaves similarly to an antenna by passing some functionality downward to
    an antenna class, but additionally applies some front-end processing (e.g.
    an electronics chain) to the signals received.

    Parameters
    ----------
    antenna : Antenna
        ``Antenna`` class or subclass to be extended with a front end. Can also
        accept an ``Antenna`` object directly.

    Attributes
    ----------
    antenna : Antenna
        ``Antenna`` object extended by the front end.
    lead_in_time : float
        Lead-in time (s) required for the front end to equilibrate.
        Automatically added in before calculation of signals and waveforms.
    is_hit
    is_hit_mc_truth
    signals
    waveforms
    all_waveforms

    See Also
    --------
    pyrex.Antenna : Base class for antennas.

    """
    lead_in_time = 0

    def __init__(self, antenna):
        if inspect.isclass(antenna):
            self._antenna_class = antenna
        else:
            self.antenna = antenna
            self._antenna_class = antenna.__class__

        self._signals = []
        self._all_waves = []
        self._triggers = []

    def __str__(self):
        return (self.__class__.__name__+"(position="+
                repr(self.antenna.position)+")")

    @property
    def _metadata(self):
        """Metadata dictionary for writing `AntennaSystem` information."""
        return self.antenna._metadata

    def setup_antenna(self, *args, **kwargs):
        """
        Setup the antenna by passing along its init arguments.

        Any arguments passed to this method are directly passed to the
        ``__init__`` methods of the ``antenna``'s class. This function can be
        overridden if desired, just make sure to assign the ``self.antenna``
        attribute in the function.

        """
        self.antenna = self._antenna_class(*args, **kwargs)

    def set_orientation(self, z_axis=(0,0,1), x_axis=(1,0,0)):
        """
        Sets the orientation of the antenna system.

        Sets up the z-axis and the x-axis of the antenna according to the given
        parameters. Fails if the z-axis and x-axis aren't perpendicular.

        Parameters
        ----------
        z_axis : array_like, optional
            Vector direction of the z-axis of the antenna.
        x_axis : array_like, optional
            Vector direction of the x-axis of the antenna.

        Raises
        ------
        ValueError
            If the z-axis and x-axis aren't perpendicular.

        """
        self.antenna.set_orientation(z_axis=z_axis, x_axis=x_axis)

    def front_end(self, signal):
        """
        Apply front-end processes to a signal and return the output.

        This function defines the front end of the antenna (e.g. electronics
        chain). It is expected to be overridden in subclasses, as for the base
        class it simply passes along the given `signal`.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to apply the front-end processes.

        Returns
        -------
        Signal
            Signal processed by the antenna front end.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        logger.debug("Using default front_end from "+
                     "pyrex.detector.AntennaSystem")
        return signal

    def _calculate_lead_in_times(self, times):
        t0 = times[0]
        t_min = t0-self.lead_in_time
        t_max = times[-1]
        dt = times[1]-t0
        # Number of points in the lead-in array
        n_pts = int((t_max-t_min)/dt)+2 - len(times)
        # Proper starting point of lead-in array to preserve dt
        t_min = t0-n_pts*dt
        return np.concatenate(
            (np.linspace(t_min, t0, n_pts, endpoint=False),
             times)
        )

    @property
    def is_hit(self):
        """Boolean of whether the antenna system has been triggered."""
        return len(self.waveforms)>0

    @property
    def is_hit_mc_truth(self):
        """
        Boolean of whether the antenna has been triggered by signal.

        The decision is based on the Monte Carlo truth of whether noise would
        have triggered without the signal. If a signal triggered, but the noise
        alone in the same timeframe would have triggered as well, the trigger
        is not counted.
        """
        for wave in self.waveforms:
            if not self.trigger(self.make_noise(wave.times)):
                return True
        return False

    def is_hit_during(self, times):
        """
        Check if the antenna system is triggered in a time range.

        Generate the full waveform of the antenna system over the given `times`
        array and check whether it triggers the antenna system.

        Parameters
        ----------
        times : array_like
            1D array of times during which to check for a trigger.

        Returns
        -------
        boolean
            Whether or not the antenna system triggered during the given
            `times`.

        See Also
        --------
        pyrex.Antenna.is_hit_during : Check if an antenna is triggered in a
                                      time range.

        """
        return self.trigger(self.full_waveform(times))

    def clear(self, reset_noise=False):
        """
        Reset the antenna system to an empty state.

        Clears all signals, noises, and triggers from the antenna state. Can
        also optionally recalibrate the noise so that a new signal arriving
        at the same times as a previous signal will not have the same noise.

        Parameters
        ----------
        reset_noise : boolean, optional
            Whether or not to recalibrate the noise.

        See Also
        --------
        pyrex.Antenna.clear : Reset an antenna to an empty state.

        """
        self._signals.clear()
        self._all_waves.clear()
        self._triggers.clear()
        self.antenna.clear(reset_noise=reset_noise)

    @property
    def signals(self):
        """The signals received by the antenna with front-end processing."""
        # Process any unprocessed antenna signals, including the appropriate
        # amount of front-end lead-in time
        while len(self._signals)<len(self.antenna.signals):
            signal = self.antenna.signals[len(self._signals)]
            long_times = self._calculate_lead_in_times(signal.times)
            preprocessed = signal.with_times(long_times)
            processed = self.front_end(preprocessed)
            self._signals.append(processed.with_times(signal.times))
        # Return processed antenna signals
        return self._signals

    @property
    def waveforms(self):
        """The antenna system signal + noise for each triggered hit."""
        # Process any unprocessed triggers
        all_waves = self.all_waveforms
        while len(self._triggers)<len(all_waves):
            waveform = all_waves[len(self._triggers)]
            self._triggers.append(self.trigger(waveform))

        return [wave for wave, triggered in zip(all_waves, self._triggers)
                if triggered]

    @property
    def all_waveforms(self):
        """The antenna system signal + noise for all hits."""
        # Process any unprocessed antenna signals, including the appropriate
        # amount of front-end lead-in time
        while len(self._all_waves)<len(self.antenna.signals):
            self._all_waves.append(
                self.full_waveform(
                    self.antenna.signals[len(self._all_waves)].times
                )
            )
        return self._all_waves

    def full_waveform(self, times):
        """
        Signal + noise for the antenna system for the given times.

        Creates the complete waveform of the antenna system including noise and
        all received signals for the given `times` array. Includes front-end
        processing with the required lead-in time.

        Parameters
        ----------
        times : array_like
            1D array of times during which to produce the full waveform.

        Returns
        -------
        Signal
            Complete waveform with noise and all signals.

        See Also
        --------
        pyrex.Antenna.full_waveform : Signal + noise for an antenna for the
                                      given times.

        """
        # Process full antenna waveform
        long_times = self._calculate_lead_in_times(times)
        preprocessed = self.antenna.full_waveform(long_times)
        processed = self.front_end(preprocessed)
        return processed.with_times(times)

    def make_noise(self, times):
        """
        Creates a noise signal over the given times.

        Passes a noise signal of the antenna through the front-end.

        Parameters
        ----------
        times : array_like
            1D array of times during which to produce noise values.

        Returns
        -------
        Signal
            Antenna system noise values during the `times` array.

        Raises
        ------
        ValueError
            If not enough noise-related attributes are defined for the antenna.

        """
        long_times = self._calculate_lead_in_times(times)
        preprocessed = self.antenna.make_noise(long_times)
        processed = self.front_end(preprocessed)
        return processed.with_times(times)

    def trigger(self, signal):
        """
        Check if the antenna system triggers on a given signal.

        By default just matches the antenna's trigger. It may be overridden in
        subclasses.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to test the trigger condition.

        Returns
        -------
        boolean
            Whether or not the antenna triggers on `signal`.

        See Also
        --------
        pyrex.Antenna.trigger : Check if an antenna triggers on a given signal.
        pyrex.Signal : Base class for time-domain signals.

        """
        return self.antenna.trigger(signal)

    def apply_response(self, signal, direction=None, polarization=None,
                       force_real=False):
        """
        Process the complete antenna response for an incoming signal.

        Processes the incoming signal by passing the call along to the
        ``antenna`` object.

        Parameters
        ----------
        signal : Signal
            Incoming ``Signal`` object to process.
        direction : array_like, optional
            Vector denoting the direction of travel of the signal as it reaches
            the antenna (in the global coordinate frame). If ``None`` no
            directional response will be applied.
        polarization : array_like, optional
            Vector denoting the signal's polarization direction (in the global
            coordinate frame). If ``None`` no polarization gain will be applied.
        force_real : boolean, optional
            Whether or not the frequency response should be redefined in the
            negative-frequency domain to keep the values of the filtered signal
            real.

        Returns
        -------
        Signal
            Processed ``Signal`` object after the complete antenna response has
            been applied. Should have a ``value_type`` of ``voltage``.

        Raises
        ------
        ValueError
            If the given `signal` does not have a ``value_type`` of ``voltage``
            or ``field``.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        return self.antenna.apply_response(signal, direction=direction,
                                           polarization=polarization,
                                           force_real=force_real)

    def receive(self, signal, direction=None, polarization=None,
                force_real=False):
        """
        Process and store one or more incoming (polarized) signals.

        Processes the incoming signal(s) by passing the call along to the
        ``antenna`` object.

        Parameters
        ----------
        signal : Signal or array_like
            Incoming ``Signal`` object(s) to process and store. May be separate
            polarization representations, but therefore should have the same
            times.
        direction : array_like, optional
            Vector denoting the direction of travel of the signal(s) as they
            reach the antenna (in the global coordinate frame). If ``None`` no
            directional gain will be applied.
        polarization : array_like, optional
            Vector(s) denoting the signal's polarization direction (in the
            global coordinate frame). Number of vectors should match the number
            of elements in `signal` argument. If ``None`` no polarization gain
            will be applied.
        force_real : boolean, optional
            Whether or not the frequency response should be redefined in the
            negative-frequency domain to keep the values of the filtered signal
            real.

        Raises
        ------
        ValueError
            If the number of polarizations does not match the number of signals.
            Or if the signals do not have the same `times` array.

        See Also
        --------
        pyrex.Antenna.receive : Process and store one or more incoming
                                (polarized) signals.
        pyrex.Signal : Base class for time-domain signals.

        """
        return self.antenna.receive(signal, direction=direction,
                                    polarization=polarization,
                                    force_real=force_real)



class Detector:
    """
    Base class for detectors for easily building up sets of antennas.

    Designed for automatically generating antenna positions based on geometry
    parameters, then building all the antennas with some properties. Any
    parameters to the ``__init__`` method are automatically passed on to the
    `set_positions` method. Once the antennas have been built, the object can
    be directly iterated over to iterate over the antennas (as if the object
    were just a list of the antennas).

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    See Also
    --------
    pyrex.Antenna : Base class for antennas.
    AntennaSystem : Base class for antenna system with front-end processing.

    Notes
    -----
    When this class is subclassed, the ``__init__`` method will mirror the
    signature of the `set_positions` method so that parameters can be easily
    discovered.

    The class is designed to be flexible in what defines a "detector". This
    should allow for easier modularization by defining detectors whose subsets
    are detectors themselves, and so on. For example, a string of antennas
    could be set up as a subclass of `Detector` which sets up some antennas
    in a vertical line. Then a station could be set up as a subclass of
    `Detector` which sets up multiple instances of the string class at
    different positions. Then a final overarching detector class can subclass
    `Detector` and set up multiple instances of the station class at
    different positions. In this example the ``subsets`` of the overarching
    detector class would be the station objects, the ``subsets`` of the station
    objects would be the string objects, and the ``subsets`` of the string
    objects would finally be the antenna objects. But the way the iteration of
    the `Detector` class is built, iterating over that overarching detector
    class would iterate directly over each antenna in each string in each
    station as a simple 1D list.

    """
    test_antenna_positions = True
    def __init__(self, *args, **kwargs):
        self.antenna_positions = []
        self.subsets = []
        self.set_positions(*args, **kwargs)

        # Pull antenna positions from any subsets
        if not self._is_base_subset:
            self.antenna_positions = [sub.antenna_positions
                                      for sub in self.subsets]

        self._test_positions()
        self._mirror_build_function()

    def __init_subclass__(cls, mirror_set_positions=True, **kwargs):
        """
        Automates function mirroring when subclasses are created.

        When this class is subclassed, set the subclass's ``__init__`` method
        to mirror its ``set_positions`` method (since all ``__init__``
        arguments are passed to ``set_positions`` anyway).

        Parameters
        ----------
        mirror_set_positions : boolean, optional
            Whether or not to mirror the set_positions method.

        Warnings
        --------
        If `mirror_set_positions` is ``True``, the ``__init__`` method of the
        class is replaced by ``super().__init__``.

        """
        super().__init_subclass__(**kwargs)
        if mirror_set_positions:
            cls.__init__ = mirror_func(cls.set_positions, Detector.__init__)

    def __add__(self, other):
        """
        Adds two detectors into a ``CombinedDetector``.

        Supports addition of ``Detector`` objects, ``Antenna``-like objects,
        or lists of ``Antenna``-like objects.

        """
        return CombinedDetector(self, other)

    def __radd__(self, other):
        """
        Allows for adding ``Detector`` object to 0.

        Since the python ``sum`` function starts by adding the first element
        to 0, to use ``sum`` with ``Detector`` objects we need to be able to
        add a ``Detector`` object to 0. If adding to anything else, perform the
        usual addition.

        """
        if other==0:
            return self
        else:
            return CombinedDetector(other, self)

    @property
    def _metadata(self):
        """List of metadata dictionaries of the `Antenna` objects."""
        return [antenna._metadata for antenna in self]

    def set_positions(self, *args, **kwargs):
        """
        Sets the positions of antennas in the detector.

        For the base `Detector` class, this method is not implemented.
        Subclasses should override this method with their own procedure for
        setting antenna positions based on some parameters.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides the function.

        """
        logger.debug("Using default set_positions from "+
                     "pyrex.detector.Detector")
        raise NotImplementedError("set_positions method must be implemented "
                                  +"by inheriting class")

    def build_antennas(self, *args, **kwargs):
        """
        Creates antenna objects at the set antenna positions.

        This method takes an antenna class and, for each position in the
        ``antenna_positions`` attribute, creates an antenna object of that
        class with the position as the `position` argument, followed by any
        other arguments that were passed to the method. These antenna objects
        are stored to the ``subsets`` attribute with the same list structure.
        Subclasses may extend this method, but generally shouldn't need to as
        subset `build_antennas` methods are automatically called as needed.

        """
        if self._is_base_subset:
            logger.debug("Using default build_antennas from "+
                         "pyrex.detector.Detector")
            if "antenna_class" in kwargs:
                antenna_class = kwargs["antenna_class"]
                kwargs.pop("antenna_class")
            else:
                if len(args)<1:
                    raise TypeError("build_antennas() missing 1 required "+
                                    "positional argument: 'antenna_class'")
                antenna_class = args[0]
                args = args[1:]
            self.subsets = []
            for p in self.antenna_positions:
                self.subsets.append(antenna_class(position=p, *args, **kwargs))
        else:
            if self._subset_builds_match:
                # If the signatures match, passing down args is fine
                for sub in self.subsets:
                    if hasattr(sub, 'build_antennas'):
                        sub.build_antennas(*args, **kwargs)
            else:
                # If the signatures don't match, only pass down kwargs as needed
                if len(args)>0:
                    raise TypeError("Detector build_antennas cannot handle "+
                                    "positional arguments when its subsets "+
                                    "aren't identical")
                for sub in self.subsets:
                    if hasattr(sub, 'build_antennas'):
                        sig = inspect.signature(sub.build_antennas)
                        keys = sig.parameters.keys()
                        sub_kwargs = {key: val for key, val in kwargs.items()
                                      if key in keys}
                        sub.build_antennas(**sub_kwargs)

    def triggered(self, *args, require_mc_truth=False, **kwargs):
        """
        Check if the detector is triggered based on its current state.

        By default just checks whether any antenna in the detector is hit.
        This method may be overridden in subclasses.

        Parameters
        ----------
        require_mc_truth : boolean, optional
            Whether or not the trigger should be based on the Monte-Carlo
            truth. If ``True``, noise-only triggers are removed.

        Returns
        -------
        boolean
            Whether or not the detector is triggered in its current state.

        See Also
        --------
        pyrex.Antenna.trigger : Check if an antenna triggers on a given signal.
        pyrex.AntennaSystem.trigger : Check if an antenna system triggers on a
                                      given signal.

        """
        for ant in self:
            if ((require_mc_truth and ant.is_hit_mc_truth) or
                    (not require_mc_truth and ant.is_hit)):
                return True
        return False

    def clear(self, reset_noise=False):
        """
        Reset the detector to an empty state.

        Convenience method to clear all signals, noises, and triggers from all
        antennas in the detector. Can also optionally recalibrate the noise for
        all antennas as well.

        Parameters
        ----------
        reset_noise : boolean, optional
            Whether or not to recalibrate the noise.

        See Also
        --------
        pyrex.Antenna.clear : Reset an antenna to an empty state.
        pyrex.AntennaSystem.clear : Reset an antenna system to an empty state.

        """
        for ant in self:
            ant.clear(reset_noise=reset_noise)

    @property
    def _is_base_subset(self):
        """Whether the detector is a base subset."""
        return (len(self.subsets)==0 or
                True not in [isinstance(sub, Iterable)
                             for sub in self.subsets])

    @property
    def _subset_builds_match(self):
        """
        Whether the subsets of the detector have the same ``build_antennas``.

        """
        return (self._is_base_subset or
                len(set([inspect.signature(sub.build_antennas)
                         for sub in self.subsets
                         if hasattr(sub, 'build_antennas')])) == 1)

    def _test_positions(self):
        """
        Check that all antenna positions are below the ice surface.

        Tests each antenna position to ensure no antennas are unexpectedly
        placed above the ice. Also pulls all antenna positions up from subsets.

        Raises
        ------
        ValueError
            If an antenna position is found above the ice surface.

        """
        if not self.test_antenna_positions:
            return
        if self._is_base_subset:
            for pos in self.antenna_positions:
                if pos[2]>0:
                    raise ValueError("Antenna placed outside of ice may cause "
                                     +"unexpected issues")
        else:
            for sub in self.subsets:
                if hasattr(sub, '_test_positions'):
                    sub._test_positions()
                elif isinstance(sub, Iterable):
                    for ant in sub:
                        if ant.position[2]>0:
                            raise ValueError("Antenna placed outside of ice "+
                                             "may cause unexpected issues")
                else:
                    if sub.position[2]>0:
                        raise ValueError("Antenna placed outside of ice "+
                                         "may cause unexpected issues")

    def _mirror_build_function(self):
        """
        Mirror the build function of the subsets.

        For a detector comprised of subsets which hasn't overwritten the
        default ``build_antennas`` method, mirrors the function signature of
        ``build_antennas`` from the base subset (as long as the signature is
        the same for all subsets).

        """
        if hasattr(self, '_actual_build'):
            self.build_antennas = self._actual_build
        if (not self._is_base_subset and
                self.build_antennas.__func__==Detector.build_antennas and
                self._subset_builds_match):
            self._actual_build = self.build_antennas
            for sub in self.subsets:
                if hasattr(sub, 'build_antennas'):
                    break
            self.build_antennas = mirror_func(sub.build_antennas,
                                              Detector.build_antennas,
                                              self=self)
        else:
            self._actual_build = self.build_antennas

    # Allow direct iteration of the detector to be treated as iteration over
    # the flat list of all its antennas
    def __iter__(self):
        yield from flatten(self.subsets)

    def __len__(self):
        return len(list(flatten(self.subsets)))

    def __getitem__(self, key):
        return list(flatten(self.subsets))[key]



class CombinedDetector(Detector, mirror_set_positions=False):
    """
    Class for detectors which have been added together.

    Designed to allow addition of ``Detector`` and ``Antenna``-like objects
    which can still build all antennas and trigger by smartly passing down
    keyword arguments to the subsets. Maintains all other properties of the
    ``Detector`` objects.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    See Also
    --------
    pyrex.Antenna : Base class for antennas.
    AntennaSystem : Base class for antenna system with front-end processing.
    Detector : Base class for detectors for easily building up sets of antennas.

    """
    def __init__(self, *detectors):
        self.subsets = list(detectors)
        self._test_positions()
        self._mirror_build_function()

    @property
    def antenna_positions(self):
        """List of the positions of the antennas in the detector."""
        positions = []
        for sub in self.subsets:
            if hasattr(sub, 'antenna_positions'):
                positions.append(sub.antenna_positions)
            elif isinstance(sub, Iterable):
                positions.append([ant.position for ant in sub])
            else:
                positions.append(sub.position)
        return positions

    def __add__(self, other):
        """
        Adds two detectors into a ``CombinedDetector``.

        Supports addition of ``Detector`` objects, ``Antenna``-like objects,
        or lists of ``Antenna``-like objects. Adds additional support for
        adding ``CombinedDetector`` objects so the subsets of the two are
        treated on equal footing.

        """
        if isinstance(other, CombinedDetector):
            return CombinedDetector(*self.subsets, *other.subsets)
        else:
            return CombinedDetector(*self.subsets, other)

    def __radd__(self, other):
        """
        Allows for adding ``CombinedDetector`` object to 0.

        Since the python ``sum`` function starts by adding the first element
        to 0, to use ``sum`` with ``CombinedDetector`` objects we need to be
        able to add a ``CombinedDetector`` object to 0. If adding to anything
        else, perform the usual addition.

        """
        if other==0:
            return self
        elif isinstance(other, CombinedDetector):
            return CombinedDetector(*other.subsets, *self.subsets)
        else:
            return CombinedDetector(other, *self.subsets)

    def __iadd__(self, other):
        """
        Adds a detector in-place.

        Supports addition of ``Detector`` objects, ``Antenna``-like objects,
        or lists of ``Antenna``-like objects. Adds additional support for
        adding ``CombinedDetector`` objects so the subsets of the two are
        treated on equal footing.

        """
        if isinstance(other, CombinedDetector):
            self.subsets.extend(other.subsets)
        else:
            self.subsets.append(other)
        self._test_positions()
        self._mirror_build_function()
        return self

    @property
    def _subset_triggers_match(self):
        """Whether the subsets of the detector have the same triggered."""
        return len(set([inspect.signature(sub.triggered)
                        for sub in self.subsets
                        if hasattr(sub, 'triggered')])) == 1

    def triggered(self, *args, require_mc_truth=False, **kwargs):
        """
        Check if the detector is triggered based on its current state.

        By default triggers if any subset of the detector is triggered. Passes
        arguments down to the appropriate subsets if the subsets are
        ``Detector`` objects, or checks for any triggering waveforms in
        ``Antenna``-like objects.

        Parameters
        ----------
        require_mc_truth : boolean, optional
            Whether or not the trigger should be based on the Monte-Carlo
            truth. If ``True``, noise-only triggers are removed.
        *args, **kwargs
            Positional and keyword arguments to be passed down to `triggered`
            methods of the detector subsets.

        Returns
        -------
        boolean
            Whether or not the detector is triggered in its current state.

        See Also
        --------
        pyrex.Antenna.trigger : Check if an antenna triggers on a given signal.
        pyrex.AntennaSystem.trigger : Check if an antenna system triggers on a
                                      given signal.
        pyrex.Detector.triggered : Check if the detector is triggered based on
                                   its current state.

        """
        if not self._subset_triggers_match and len(args)>0:
            raise TypeError("Combined detector trigger cannot handle "+
                            "positional arguments when its subsets "+
                            "aren't identical")

        # Add require_mc_truth to kwargs for ease of use
        kwargs['require_mc_truth'] = require_mc_truth

        for sub in self.subsets:
            if hasattr(sub, 'triggered'):
                if self._subset_triggers_match:
                    # If the signatures match, passing down args is fine
                    logger.debug("Called %s with args %s and kwargs %s",
                                 sub.triggered, args, kwargs)
                    if sub.triggered(*args, **kwargs):
                        return True
                else:
                    # Keep trying the subset trigger, removing keyword
                    # arguments which cause errors one by one
                    sub_kwargs = kwargs
                    while True:
                        prev_kwargs = sub_kwargs
                        try:
                            triggered = sub.triggered(**sub_kwargs)
                        except TypeError as e:
                            # Remove the problematic argument from sub_kwargs
                            msg = e.args[0]
                            if 'got an unexpected keyword argument' in msg:
                                parts = msg.split("'")
                                bad_kw = parts[1]
                                sub_kwargs = {key: val
                                              for key, val in sub_kwargs.items()
                                              if key!=bad_kw}
                            else:
                                raise e
                        else:
                            logger.debug("Called %s with kwargs %s",
                                         sub.triggered, sub_kwargs)
                            if triggered:
                                return True
                            else:
                                break
                        if sub_kwargs==prev_kwargs:
                            raise TypeError("Unable to pass keyword arguments"+
                                            " down to subset triggers")
            elif isinstance(sub, Iterable):
                # Check for any antenna trigger in the subset
                for ant in sub:
                    if ((require_mc_truth and ant.is_hit_mc_truth) or
                            (not require_mc_truth and ant.is_hit)):
                        return True
            else:
                # Check for single antenna trigger
                if ((require_mc_truth and sub.is_hit_mc_truth) or
                        (not require_mc_truth and sub.is_hit)):
                    return True

        return False
