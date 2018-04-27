"""Module containing higher-level AntennaSystem and Detector classes"""

import collections
import inspect
import logging
from pyrex.internal_functions import flatten, mirror_func

logger = logging.getLogger(__name__)


class AntennaSystem:
    """Base class for an antenna system consisting of an antenna and some
    front-end processes."""
    def __init__(self, antenna):
        if inspect.isclass(antenna):
            self._antenna_class = antenna
        else:
            self.antenna = antenna
            self._antenna_class = antenna.__class__

        self._signals = []
        self._all_waveforms = []
        self._triggers = []

    def __str__(self):
        return (self.__class__.__name__+"(position="+
                repr(self.antenna.position)+")")

    def setup_antenna(self, *args, **kwargs):
        """Setup the antenna by passing along its init arguments.
        This function can be overwritten if desired, just make sure to assign
        the self.antenna attribute in the function."""
        self.antenna = self._antenna_class(*args, **kwargs)

    def front_end(self, signal):
        """This function should take the signal passed (from the antenna) and
        return the resulting signal after all processing by the antenna system's
        front-end. By default it just returns the given signal."""
        logger.debug("Using default front_end from "+
                     "pyrex.detector.AntennaSystem")
        return signal

    @property
    def is_hit(self):
        return len(self.waveforms)>0

    def is_hit_during(self, times):
        return self.trigger(self.full_waveform(times))

    @property
    def signals(self):
        # Process any unprocessed antenna signals
        while len(self._signals)<len(self.antenna.signals):
            signal = self.antenna.signals[len(self._signals)]
            self._signals.append(self.front_end(signal))
        # Return processed antenna signals
        return self._signals

    @property
    def waveforms(self):
        # Process any unprocessed triggers
        all_waves = self.all_waveforms
        while len(self._triggers)<len(all_waves):
            waveform = all_waves[len(self._triggers)]
            self._triggers.append(self.trigger(waveform))

        return [wave for wave, triggered in zip(all_waves, self._triggers)
                if triggered]

    @property
    def all_waveforms(self):
        # Process any unprocessed antenna waveforms
        while len(self._all_waveforms)<len(self.antenna.signals):
            wave = self.antenna.all_waveforms[len(self._all_waveforms)]
            self._all_waveforms.append(self.front_end(wave))
        # Return processed antenna waveforms
        return self._all_waveforms

    def full_waveform(self, times):
        # Process full antenna waveform
        preprocessed = self.antenna.full_waveform(times)
        return self.front_end(preprocessed)

    def receive(self, signal, origin=None, polarization=None):
        return self.antenna.receive(signal, origin=origin,
                                    polarization=polarization)

    def clear(self):
        """Reset the antenna system to a state of having received no signals."""
        self._signals.clear()
        self._all_waveforms.clear()
        self._triggers.clear()
        self.antenna.clear()

    def trigger(self, signal):
        """Antenna system trigger. Should return True or False for whether the
        passed signal triggers the antenna system. By default just matches
        the antenna's trigger."""
        return self.antenna.trigger(signal)



class Detector:
    """Class for automatically generating antenna positions based on geometry
    criteria. The set_positions method creates a list of antenna positions and
    the build_antennas method is responsible for actually placing antennas
    at the generated positions. Once antennas are placed, the class can be
    directly iterated over to iterate over the antennas (as if it were just
    a list of antennas itself)."""
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
        """Not implemented. Should generate positions for the antennas based
        on the given arguments and assign those positions to the
        antenna_positions attribute."""
        logger.debug("Using default set_positions from "+
                     "pyrex.detector.Detector")
        raise NotImplementedError("set_positions method must be implemented "
                                  +"by inheriting class")

    def build_antennas(self, *args, **kwargs):
        """Sets up antenna objects at the positions stored in the class.
        By default takes an antenna class and passes a position to the
        'position' argument, followed by any other arguments to be passed to
        this class."""
        if self._is_base_subset:
            logger.debug("Using default build_antennas from "+
                         "pyrex.detector.Detector")
            if "antenna_class" in kwargs:
                antenna_class = kwargs["antenna_class"]
                kwargs.pop("antenna_class")
            else:
                antenna_class = args[0]
                args = args[1:]
            for p in self.antenna_positions:
                self.subsets.append(antenna_class(position=p, *args, **kwargs))
        else:
            for sub in self.subsets:
                sub.build_antennas(*args, **kwargs)

    def triggered(self, *args, **kwargs):
        """Test for whether the detector is triggered based on the current
        state of the antennas."""
        for ant in self:
            if ant.is_hit:
                return True
        return False

    def clear(self):
        """Convenience method for clearing all antennas in the detector."""
        for ant in self:
            ant.clear()

    @property
    def _is_base_subset(self):
        return (len(self.subsets)==0 or
                not isinstance(self.subsets[0], collections.Iterable))

    def _test_positions(self):
        """Tests to ensure no antennas are placed above the ice.
        Also pulls all antenna positions up from subsets."""
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
