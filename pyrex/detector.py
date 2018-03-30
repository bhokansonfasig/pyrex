"""Module containing higher-level AntennaSystem and Detector classes"""

import inspect
import logging

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

    def setup_antenna(self, *args, **kwargs):
        """Setup the antenna by passing along its init arguments.
        This function can be overwritten if desired, just make sure to assign
        the self.antenna attribute in the function."""
        self.antenna = self._antenna_class(*args, **kwargs)

    def front_end(self, signal):
        """This function should take the signal passed (from the antenna) and
        return the resulting signal after all processing by the antenna system's
        front-end. By default it just returns the given signal."""
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
        # Return envelopes of antenna signals
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
        # Return envelopes of antenna waveforms
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
    def __init__(self, *args, **kwargs):
        self.antenna_positions = []

        self.set_positions(*args, **kwargs)

        for pos in self.antenna_positions:
            if pos[2]>0:
                raise ValueError("Antenna placed outside of ice may cause "
                                 +"unexpected issues")

        self.antennas = []

    def set_positions(self, *args, **kwargs):
        """Not implemented. Should generates positions for the antennas based
        on the given arguments and assign those positions to the
        antenna_positions attribute."""
        raise NotImplementedError("set_positions method must be implemented "
                                  +"by inheriting class")

    def build_antennas(self, antenna_class, **kwargs):
        """Sets up antenna objects at the positions stored in the class.
        By default takes an antenna class and passes a position to the
        'position' argument, followed by the keyword arguments passed to this
        function."""
        self.antennas = []
        for pos in self.antenna_positions:
            self.antennas.append(antenna_class(position=pos, **kwargs))

    def __iter__(self):
        self._iter_counter = 0
        self._iter_max = len(self.antennas)
        return self

    def __next__(self):
        self._iter_counter += 1
        if self._iter_counter > self._iter_max:
            raise StopIteration
        else:
            return self.antennas[self._iter_counter-1]

    def __len__(self):
        return len(self.antennas)

    def __getitem__(self, key):
        return self.antennas[key]

    def __setitem__(self, key, value):
        self.antennas[key] = value
