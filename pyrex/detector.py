"""
Module containing higher-level detector-related classes.

The classes in this module are responsible for higher-level operations of
the antennas and detectors than in the antenna module. This includes
functions like front-end electronics chains and trigger systems.

"""

import collections
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

    def receive(self, signal, direction=None, polarization=None,
                force_real=False):
        """
        Process and store an incoming signal.

        Processes the incoming signal by passing the call along to the
        ``antenna`` object.

        Parameters
        ----------
        signal : Signal
            Incoming ``Signal`` object to process and store.
        direction : array_like, optional
            Vector denoting the direction of travel of the signal as it reaches
            the antenna. If ``None`` no directional response will be applied.
        polarization : array_like, optional
            Vector denoting the signal's polarization direction. If ``None``
            no polarization gain will be applied.
        force_real : boolean, optional
            Whether or not the frequency response should be redefined in the
            negative-frequency domain to keep the values of the filtered signal
            real.

        Raises
        ------
        ValueError
            If the given `signal` does not have a ``value_type`` of ``voltage``
            or ``field``.

        See Also
        --------
        pyrex.Antenna.receive : Process and store an incoming signal.
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
    `Detector` and set up mutliple instances of the station class at
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
        if self.test_antenna_positions:
            self._test_positions()

        # For a detector comprised of subsets which hasn't overwritten
        # build_antennas, mirror the function signature of build_antennas from
        # the base subset
        if (not self._is_base_subset and
                self.build_antennas.__func__==Detector.build_antennas):
            self.build_antennas = mirror_func(self.subsets[0].build_antennas,
                                              Detector.build_antennas,
                                              self=self)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # When this class is sublassed, set the subclass's __init__ to mirror
        # its set_positions function (since all __init__ arguments are passed
        # to set_positions anyway)
        cls.__init__ = mirror_func(cls.set_positions, Detector.__init__)

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
                antenna_class = args[0]
                args = args[1:]
            self.subsets = []
            for p in self.antenna_positions:
                self.subsets.append(antenna_class(position=p, *args, **kwargs))
        else:
            for sub in self.subsets:
                sub.build_antennas(*args, **kwargs)

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
        return (len(self.subsets)==0 or
                not isinstance(self.subsets[0], collections.Iterable))

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
        if self._is_base_subset:
            for pos in self.antenna_positions:
                if pos[2]>0:
                    raise ValueError("Antenna placed outside of ice may cause "
                                     +"unexpected issues")
        else:
            for sub in self.subsets:
                sub._test_positions()

    # Allow direct iteration of the detector to be treated as iteration over
    # the flat list of all its antennas
    def __iter__(self):
        yield from flatten(self.subsets)

    def __len__(self):
        return len(list(flatten(self.subsets)))

    def __getitem__(self, key):
        return list(flatten(self.subsets))[key]
