"""Module containing antenna class capable of receiving signals"""

import warnings
import numpy as np
import scipy.fftpack
import scipy.signal
from pyrex.signals import ThermalNoise
from pyrex.ice_model import IceModel


class Antenna:
    """Base class for an antenna with a given position (m), temperature (K),
    allowable frequency range (Hz), total resistance (ohm) used for Johnson
    noise, and whether or not to include noise in the antenna's waveforms.
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
        self._noises = []
        self.triggers = []

    @property
    def is_hit(self):
        """Test for whether the antenna has received a signal."""
        return len(self.waveforms)>0

    def isHit(self):
        """Deprecated. Replaced by is_hit property."""
        warnings.warn("Antenna.isHit has been replaced by Antenna.is_hit",
                      DeprecationWarning, stacklevel=2)
        return self.is_hit()

    def clear(self):
        """Reset the antenna to a state of having received no signals."""
        self.signals.clear()
        self._noises.clear()

    @property
    def waveforms(self):
        """Signal + noise (if noisy) at each triggered antenna hit."""
        # Process any unprocessed triggers
        all_waves = self.all_waveforms
        while len(self.triggers)<len(all_waves):
            waveform = all_waves[len(self.triggers)]
            self.triggers.append(self.trigger(waveform))

        return [wave for wave, triggered in zip(all_waves, self.triggers)
                if triggered]

    @property
    def all_waveforms(self):
        """Signal + noise (if noisy) at all antenna hits, even those that
        didn't trigger."""
        if not(self.noisy):
            return self.signals

        # Generate noise as necessary
        while len(self._noises)<len(self.signals):
            self._noises.append(
                ThermalNoise(self.signals[len(self._noises)].times,
                             temperature=self.temperature,
                             resistance=self.resistance,
                             f_band=self.freq_range)
            )

        return [s + n for s, n in zip(self.signals, self._noises)]


    def trigger(self, signal):
        """Function to determine whether or not the antenna is triggered by
        the given Signal object."""
        return True

    def response(self, frequencies):
        """Function to return the frequency response of the antenna at the
        given frequencies (Hz). This function should return the response as
        imaginary numbers, where the real part is the amplitude response and
        the imaginary part is the phase response."""
        return np.ones(len(frequencies))

    def receive(self, signal, polarization=[0,0,1]):
        """Process incoming signal according to the filter function and
        store it to the signals list. Subclasses may extend this fuction,
        but should end with super().receive(signal)."""
        signal.filter_frequencies(self.response)
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

        # Build scipy butterworth filter to speed up response function
        b, a  = scipy.signal.butter(1, 2*np.pi*np.array(self.freq_range),
                                    btype='bandpass', analog=True)
        self.filter_coeffs = (b, a)


    def trigger(self, signal):
        """Trigger on the signal if the maximum signal value is above the
        given threshold."""
        return max(signal.values) > self.threshold

    def response(self, frequencies):
        """Butterworth filter response for the antenna's frequency range."""
        angular_freqs = np.array(frequencies) * 2*np.pi
        w, h = scipy.signal.freqs(self.filter_coeffs[0], self.filter_coeffs[1],
                                  angular_freqs)
        return h

    def receive(self, signal, polarization=[0,0,1]):
        """Apply polarization effect to signal, then proceed with usual
        antenna reception."""
        signal *= self.effective_height * np.abs(polarization[2])
        super().receive(signal)
