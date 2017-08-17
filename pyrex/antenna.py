"""Module containing antenna class capable of receiving signals"""

import warnings
import numpy as np
import scipy.fftpack
from pyrex.signals import ThermalNoise
from pyrex.ice_model import IceModel


class Antenna:
    """Base class for an antenna with a given position (m), temperature (K),
    allowable frequency range, total resistance (ohm) used for Johnson noise,
    and whether or not to include noise in the antenna's waveforms.
    Defines default trigger, frequency response, and signal reception functions
    that can be overwritten in base classes to customize the antenna."""
    def __init__(self, position, temperature, freq_range,
                 resistance=None, noisy=True):
        self.position = position
        self.temperature = temperature
        self.freq_range = freq_range
        if noisy and resistance is None:
            raise ValueError("A resistance is required to generate antenna noise")
        self.resistance = resistance
        self.noisy = noisy

        self.signals = []
        self._waveforms_generated = 0
        self._noises = []

    @property
    def is_hit(self):
        """Test for whether the antenna has received a signal."""
        return len(self.signals)>0

    def isHit(self):
        """Deprecated. Replaced by is_hit property."""
        warnings.warn("Antenna.isHit has been replaced by Antenna.is_hit",
                      DeprecationWarning, stacklevel=2)
        return self.is_hit()

    def clear(self):
        """Reset the antenna to a state of having received no signals."""
        self.signals.clear()
        self._waveforms_generated = 0
        self._noises.clear()

    @property
    def waveforms(self):
        """Signal + (optional) noise at each antenna hit."""
        if not(self.noisy):
            return self.signals

        if self._waveforms_generated!=len(self.signals):
            new_signal = self.signals[-1]
            new_noise = ThermalNoise(new_signal.times,
                                     temperature=self.temperature,
                                     resistance=self.resistance,
                                     f_band=self.freq_range)
            self._noises.append(new_noise)

        return [s+n for s,n in zip(self.signals, self._noises)]

    def trigger(self, signal):
        """Function to determine whether or not the antenna is triggered by
        the given Signal object."""
        return True

    def response(self, frequency):
        """Function to return the frequency response of the antenna at the
        given frequency (Hz). This function should return the amplitude
        response as well as the phase response."""
        # TODO: Figure out how to deal with phase response as well
        return 1

    def receive(self, signal):
        """Process incoming signal according to the filter function and
        store it to the signals list."""
        signal.filter_frequencies(self.response)
        # TODO: Figure out where to apply trigger, since noise should be
        # included in the trigger. Maybe noise should be generated here
        self.signals.append(signal)



class DipoleAntenna(Antenna):
    """Antenna with a given name, position (m), center frequency (MHz),
    bandwidth (MHz), resistance (ohm), effective height (m)
    and trigger threshold (V)."""
    def __init__(self, name, position, center_frequency, bandwidth, resistance,
                 effective_height, threshold, noisy=True):
        # Get the critical frequencies in Hz
        f_low = (center_frequency - bandwidth/2) * 1e6
        f_high = (center_frequency + bandwidth/2) * 1e6
        super().__init__(position, IceModel.temperature(position[2]),
                         (f_low, f_high), resistance, noisy)
        # gets the time of the hit [ns]
        # induced signal strength [V]
        self.name = name
        self.effective_height = effective_height
        self.threshold = threshold


    def trigger(self, signal):
        return max(signal.values) > self.threshold

    def response(self, frequency):
        if self.freq_range[0] < frequency < self.freq_range[1]:
            return 1
        else:
            return 0

    def receive(self, signal, polarization):
        # Apply antenna polarization effect
        signal *= self.effective_height * np.abs(polarization[2])

        # Pass polarized signal to base class's receive function
        super().receive(signal)
