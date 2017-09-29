"""Module containing customized classes for IREX"""

import numpy as np
import scipy.signal
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.ice_model import IceModel

class IREXBaseAntenna(Antenna):
    """Antenna to be used in IREXAntenna class. Has a position (m),
    center frequency (MHz), bandwidth (MHz), resistance (ohm),
    effective height (m), and polarization direction."""
    def __init__(self, position, center_frequency, bandwidth, resistance,
                 effective_height, polarization=(0,0,1), noisy=True):
        # Get the critical frequencies in Hz
        f_low = (center_frequency - bandwidth/2) * 1e6
        f_high = (center_frequency + bandwidth/2) * 1e6
        super().__init__(position=position,
                         temperature=IceModel.temperature(position[2]),
                         freq_range=(f_low, f_high), resistance=resistance,
                         noisy=noisy)
        self.effective_height = effective_height
        self.polarization = (np.array(polarization)
                             / np.linalg.norm(polarization))

        # Build scipy butterworth filter to speed up response function
        b, a  = scipy.signal.butter(1, 2*np.pi*np.array(self.freq_range),
                                    btype='bandpass', analog=True)
        self.filter_coeffs = (b, a)

    def response(self, frequencies):
        """Butterworth filter response for the antenna's frequency range."""
        angular_freqs = np.array(frequencies) * 2*np.pi
        w, h = scipy.signal.freqs(self.filter_coeffs[0], self.filter_coeffs[1],
                                  angular_freqs)
        return h

    def receive(self, signal, polarization=[0,0,1]):
        """Apply polarization effect to signal, then proceed with usual
        antenna reception."""
        scaled_values = (signal.values * self.effective_height
                         * np.abs(np.vdot(self.polarization, polarization)))
        super().receive(Signal(signal.times, scaled_values))


class IREXAntenna:
    """IREX antenna system consisting of dipole antenna, low-noise amplifier,
    optional bandpass filter, and envelope circuit."""
    def __init__(self, name, position, trigger_threshold, time_over_threshold=0,
                 polarization=(0,0,1), noisy=True):
        self.name = str(name)
        self.position = position
        self.change_antenna(polarization=polarization, noisy=noisy)

        self.trigger_threshold = trigger_threshold
        self.time_over_threshold = time_over_threshold

        self._triggers = []

    def change_antenna(self, center_frequency=250, bandwidth=300,
                       resistance=100, polarization=(0,0,1), noisy=True):
        """Changes attributes of the antenna including center frequency (MHz),
        bandwidth (MHz), and resistance (ohms)."""
        h = 3e8 / (center_frequency*1e6) / 2
        self.antenna = IREXBaseAntenna(position=self.position,
                                       center_frequency=center_frequency,
                                       bandwidth=bandwidth,
                                       resistance=resistance,
                                       effective_height=h,
                                       polarization=polarization, noisy=noisy)

    @property
    def is_hit(self):
        return len(self.waveforms)>0

    @property
    def signals(self):
        # Return envelopes of antenna signals
        return [Signal(s.times, s.envelope) for s in self.antenna.signals]

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
        # Return envelopes of antenna waveforms
        return [Signal(w.times, w.envelope) for w in self.antenna.all_waveforms]

    def receive(self, signal, polarization=[0,0,1]):
        return self.antenna.receive(signal, polarization=polarization)

    def clear(self):
        return self.antenna.clear()

    def trigger(self, signal):
        imax = len(signal.times)
        i = 0
        while i<imax:
            j = i
            while i<imax-1 and signal.values[i]>self.trigger_threshold:
                i += 1
            if i!=j:
                time = signal.times[i]-signal.times[j]
                if time>self.time_over_threshold:
                    return True
            i += 1
        return False
