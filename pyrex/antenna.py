"""Module containing antenna class capable of receiving signals"""

import warnings
import numpy as np
from pyrex.digsig import GaussianNoise

class Antenna:
    """Antenna with a given name, position (m), center frequency (MHz),
    bandwidth (MHz), effective height (m) and trigger threshold (V)."""
    def __init__(self, name, position, center_frequency, bandwidth,
                 effective_height, threshold, noisy=True):
        # gets the time of the hit [ns]
        # induced signal strength [V]
        self.name = name
        self.pos = position
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.effective_height = effective_height
        self.threshold = threshold
        self.noisy = noisy

        self.signals = []
        self._waveforms_generated = 0
        self._noises = []

        # Get critical frequencies in rad/s
        self.f_low = 2*np.pi * (self.center_frequency - self.bandwidth/2)*1e6
        self.f_high = 2*np.pi * (self.center_frequency + self.bandwidth/2)*1e6
        
        self.clear()

    @property
    def waveforms(self):
        """Signal + (optional) noise at each antenna hit."""
        if not(self.noisy):
            return self.signals

        if self._waveforms_generated!=len(self.signals):
            new_signal = self.signals[-1]
            new_noise = GaussianNoise(new_signal.times, max(new_signal.values)/5)
            # new_noise = ThermalNoise(signal.times, [self.f_low, self.f_high])
            self._noises.append(new_noise)

        return [s+n for s,n in zip(self.signals, self._noises)]

    def is_hit(self):
        """Test for whether the antenna has received a signal."""
        return len(self.signals)>0

    def isHit(self):
        """Deprecated. Replaced by is_hit."""
        warnings.warn("Antenna.isHit has been replaced by Antenna.is_hit",
                      DeprecationWarning, stacklevel=2)
        return self.is_hit()

    def clear(self):
        """Reset the antenna to having received no signals."""
        self.signals.clear()
        self._waveforms_generated = 0
        self._noises.clear()

    def receive(self):
        """Process incoming signal and store it in antenna's signals"""
        pass
