"""Module containing customized antenna classes for ARA"""

import os.path
import numpy as np
from pyrex.internal_functions import normalize
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.detector import AntennaSystem
from pyrex.ice_model import IceModel


def read_response_data(filename):
    """Gather antenna response data from a data file. Returns the data as a
    dictionary with keys (freq, theta, phi) and values (gain, phase).
    Also returns a set of the frequencies appearing in the keys."""
    data = {}
    freqs = set()
    freq = 0
    with open(filename) as f:
        for line in f:
            words = line.split()
            if line.startswith('freq'):
                freq = 1
                if words[-1]=="Hz":
                    pass
                elif words[-1]=="kHz":
                    freq *= 1e3
                elif words[-1]=="MHz":
                    freq *= 1e6
                elif words[-1]=="GHz":
                    freq *= 1e9
                else:
                    raise ValueError("Cannot parse line: '"+line+"'")
                freq *= float(words[-2])
                freqs.add(freq)
            elif line.startswith('SWR'):
                swr = float(words[-1])
            elif len(words)==5 and words[0]!="Theta":
                theta = int(words[0])
                phi = int(words[1])
                db_gain = float(words[2])
                gain = float(words[3])
                phase = float(words[4])
                data[(freq, theta, phi)] = (gain, phase)

    return data, freqs


ARA_DATA_DIR = os.path.dirname(__file__)
VPOL_DATA_FILE = os.path.join(ARA_DATA_DIR, "ARA_bicone6in_output_MY.txt")
HPOL_DATA_FILE = os.path.join(ARA_DATA_DIR, "ARA_dipoletest1_output_MY.txt")
VPOL_RESPONSE, VPOL_FREQS = read_response_data(VPOL_DATA_FILE)
HPOL_RESPONSE, HPOL_FREQS = read_response_data(HPOL_DATA_FILE)


class ARAAntenna(Antenna):
    """Antenna to be used in ARA antenna systems. Has a position (m),
    center frequency (Hz), bandwidth (Hz), resistance (ohm),
    effective height (m), and polarization direction."""
    def __init__(self, position, center_frequency, bandwidth, resistance,
                 orientation=(0,0,1), effective_height=None,
                 response_data=None, response_freqs=None, noisy=True):
        if effective_height is None:
            # Calculate length of half-wave dipole
            self.effective_height = 3e8 / center_frequency / 2
        else:
            self.effective_height = effective_height

        # Get the critical frequencies in Hz
        f_low = center_frequency - bandwidth/2
        f_high = center_frequency + bandwidth/2

        # Get arbitrary x-axis orthogonal to orientation
        tmp_vector = np.zeros(3)
        while np.array_equal(np.cross(orientation, tmp_vector), (0,0,0)):
            tmp_vector = np.random.rand(3)
        ortho = np.cross(orientation, tmp_vector)
        # Note: ortho is not normalized, but will be normalized by Antenna's init

        super().__init__(position=position, z_axis=orientation, x_axis=ortho,
                         antenna_factor=1/self.effective_height,
                         temperature=IceModel.temperature(position[2]),
                         freq_range=(f_low, f_high), resistance=resistance,
                         noisy=noisy)

        self._response_data = response_data
        self._response_freqs = response_freqs
        if self._response_freqs is None and self._response_data is not None:
            self._response_freqs = set()
            for key in self._response_data:
                self._response_freqs.add(key[0])


    def polarization_gain(self, polarization):
        """Polarization gain is simply the dot product of the polarization
        with the antenna's z-axis."""
        return np.vdot(self.z_axis, polarization)


    def generate_freq_gains(self, theta, phi):
        """Generate arrays of frequencies and gains for given angles."""
        if self._response_data is None:
            return np.array([1]), np.array([1])

        theta = np.degrees(theta) % 180
        phi = np.degrees(phi) % 360
        theta_under = 5*int(theta/5)
        theta_over = 5*(int(theta/5)+1)
        phi_under = 5*int(phi/5)
        phi_over = 5*(int(phi/5)+1)
        t = (theta - theta_under) / (theta_over - theta_under)
        u = (phi - phi_under) / (phi_over - phi_under)

        theta_over %= 180
        phi_over %= 360

        nfreqs = len(self._response_freqs)
        gain_ij = np.zeros(nfreqs)
        gain_i1j = np.zeros(nfreqs)
        gain_ij1 = np.zeros(nfreqs)
        gain_i1j1 = np.zeros(nfreqs)
        for f, freq in enumerate(self._response_freqs):
            # TODO: Implement phase shift as imaginary part of gain
            gain_ij[f] = self._response_data[(freq, theta_under, phi_under)][0]
            gain_i1j[f] = self._response_data[(freq, theta_over, phi_under)][0]
            gain_ij1[f] = self._response_data[(freq, theta_under, phi_over)][0]
            gain_i1j1[f] = self._response_data[(freq, theta_over, phi_over)][0]

        freqs = np.array(list(self._response_freqs))
        gains = ((1-t)*(1-u)*gain_ij + t*(1-u)*gain_i1j +
                 (1-t)*u*gain_ij1 + t*u*gain_i1j1)

        return freqs, gains


    def receive(self, signal, origin=None, polarization=None):
        """Process incoming signal according to the filter function and
        store it to the signals list. Subclasses may extend this fuction,
        but should end with super().receive(signal)."""
        copy = Signal(signal.times, signal.values, value_type=Signal.ValueTypes.voltage)
        copy.filter_frequencies(self.response)

        if origin is not None:
            # Calculate theta and phi relative to the orientation
            r, theta, phi = self._convert_to_antenna_coordinates(origin)
            freq_data, gain_data = self.generate_freq_gains(theta, phi)
            def interpolated_response(frequencies):
                return np.interp(frequencies, freq_data, gain_data)
            copy.filter_frequencies(interpolated_response)

        if polarization is None:
            p_gain = 1
        else:
            p_gain = self.polarization_gain(normalize(polarization))

        signal_factor = p_gain * self.efficiency

        if signal.value_type==Signal.ValueTypes.voltage:
            pass
        elif signal.value_type==Signal.ValueTypes.field:
            signal_factor /= self.antenna_factor
        else:
            raise ValueError("Signal's value type must be either "
                             +"voltage or field. Given "+str(signal.value_type))

        copy.values *= signal_factor
        self.signals.append(copy)



class ARAAntennaSystem(AntennaSystem):
    """ARA antenna system consisting of antenna, amplification,
    and tunnel diode response."""
    def __init__(self, name, position, power_threshold, response_data=None,
                 response_freqs=None, orientation=(0,0,1), amplification=1,
                 amplifier_clipping=3, noisy=True):
        super().__init__(ARAAntenna)

        self.name = str(name)
        self.position = position
        self.setup_antenna(response_data=response_data,
                           response_freqs=response_freqs,
                           orientation=orientation, noisy=noisy)

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.power_threshold = power_threshold

    def setup_antenna(self, center_frequency=500e6, bandwidth=700e6,
                      resistance=100, orientation=(0,0,1),
                      response_data=None, response_freqs=None,
                      effective_height=None, noisy=True):
        """Sets attributes of the antenna including center frequency (Hz),
        bandwidth (Hz), resistance (ohms), orientation, and effective
        height (m)."""
        super().setup_antenna(position=self.position,
                              center_frequency=center_frequency,
                              bandwidth=bandwidth,
                              resistance=resistance,
                              orientation=orientation,
                              effective_height=effective_height,
                              response_data=response_data,
                              response_freqs=response_freqs,
                              noisy=noisy)

    def tunnel_diode(self, signal):
        """Return the signal response from the tunnel diode."""
        return signal

    def front_end(self, signal):
        """Apply the front-end processing of the antenna signal, including
        amplification, clipping, and envelope processing."""
        amplified_values = np.clip(signal.values*self.amplification,
                                   a_min=-self.amplifier_clipping,
                                   a_max=self.amplifier_clipping)
        copy = Signal(signal.times, amplified_values)
        return self.tunnel_diode(copy)

    def trigger(self, signal):
        if self.antenna._noise_master is None:
            self.antenna.make_noise([0,1])
        rms = self.antenna._noise_master.rms * self.amplification
        return np.max(signal.values**2) > -1 * self.power_threshold * rms**2



class HpolAntenna(ARAAntennaSystem):
    """ARA Hpol ("quad-slot") antenna system consisting of antenna,
    amplification, and tunnel diode response."""
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=3, noisy=True):
        super().__init__(name=name, position=position,
                         power_threshold=power_threshold,
                         response_data=HPOL_RESPONSE,
                         response_freqs=HPOL_FREQS,
                         orientation=(0,0,1),
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy)


class VpolAntenna(ARAAntennaSystem):
    """ARA Vpol ("bicone" or "birdcage") antenna system consisting of antenna,
    amplification, and tunnel diode response."""
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=3, noisy=True):
        super().__init__(name=name, position=position,
                         power_threshold=power_threshold,
                         response_data=VPOL_RESPONSE,
                         response_freqs=VPOL_FREQS,
                         orientation=(0,0,1),
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy)
