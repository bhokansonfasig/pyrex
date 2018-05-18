"""Module containing customized antenna classes for ARA"""

import os.path
import numpy as np
import scipy.signal
from pyrex.internal_functions import normalize
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.detector import AntennaSystem
from pyrex.ice_model import IceModel


def _read_response_data(filename):
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


def _read_filter_data(filename):
    """Gather electronics chain filtering data from a data file. Returns the
    data as a dictionary with frequency keys and values (gain, phase)."""
    data = {}
    freq_scale = 0
    with open(filename) as f:
        for line in f:
            words = line.split()
            if line.startswith('Freq'):
                _, scale = words[0].split("(")
                scale = scale.rstrip(")")
                if scale=="Hz":
                    freq_scale = 1
                elif scale=="kHz":
                    freq_scale = 1e3
                elif scale=="MHz":
                    freq_scale = 1e6
                elif scale=="GHz":
                    freq_scale = 1e9
                else:
                    raise ValueError("Cannot parse line: '"+line+"'")
            elif len(words)==3 and words[0]!="Total":
                f, g, p = line.split(",")
                freq = float(f) * freq_scale
                gain = float(g)
                phase = float(p)
                data[freq] = (gain, phase)

    return data


ARA_DATA_DIR = os.path.dirname(__file__)
VPOL_DATA_FILE = os.path.join(ARA_DATA_DIR, "ARA_bicone6in_output_MY.txt")
HPOL_DATA_FILE = os.path.join(ARA_DATA_DIR, "ARA_dipoletest1_output_MY.txt")
FILTER_DATA_FILE = os.path.join(ARA_DATA_DIR,
                                "ARA_Electronics_TotalGain_TwoFilters.txt")
VPOL_RESPONSE, VPOL_FREQS = _read_response_data(VPOL_DATA_FILE)
HPOL_RESPONSE, HPOL_FREQS = _read_response_data(HPOL_DATA_FILE)
ALL_FILTERS = _read_filter_data(FILTER_DATA_FILE)


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
        # Just in case the frequencies don't get set, set them now
        if self._response_freqs is None and self._response_data is not None:
            self._response_freqs = set()
            for key in self._response_data:
                self._response_freqs.add(key[0])

        self._filter_data = ALL_FILTERS


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
        for f, freq in enumerate(sorted(self._response_freqs)):
            # TODO: Implement phase shift as imaginary part of gain
            gain_ij[f] = self._response_data[(freq, theta_under, phi_under)][0]
            gain_i1j[f] = self._response_data[(freq, theta_over, phi_under)][0]
            gain_ij1[f] = self._response_data[(freq, theta_under, phi_over)][0]
            gain_i1j1[f] = self._response_data[(freq, theta_over, phi_over)][0]

        freqs = np.array(sorted(self._response_freqs))
        gains = ((1-t)*(1-u)*gain_ij + t*(1-u)*gain_i1j +
                 (1-t)*u*gain_ij1 + t*u*gain_i1j1)

        return freqs, gains

    def interpolate_filter(self, frequencies):
        """Generate interpolated filter values for given frequencies."""
        freqs = sorted(self._filter_data.keys())
        gains = []
        for f in freqs:
            # TODO: Implement phase shift as imaginary part of gain
            gains.append(self._filter_data[f][0])
        return np.interp(frequencies, freqs, gains, left=0, right=0)


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
            def interpolate_response(frequencies):
                return np.interp(frequencies, freq_data, gain_data)
            copy.filter_frequencies(interpolate_response, force_causality=True)

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
                 amplifier_clipping=1, noisy=True):
        super().__init__(ARAAntenna)

        self.name = str(name)
        self.position = position

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.setup_antenna(response_data=response_data,
                           response_freqs=response_freqs,
                           orientation=orientation, noisy=noisy)

        self.power_threshold = power_threshold
        self._power_mean = None
        self._power_rms = None

    def setup_antenna(self, center_frequency=500e6, bandwidth=800e6,
                      resistance=5.45, orientation=(0,0,1),
                      response_data=None, response_freqs=None,
                      effective_height=None, noisy=True):
        """Sets attributes of the antenna including center frequency (Hz),
        bandwidth (Hz), resistance (ohms), orientation, and effective
        height (m)."""
        # Noise rms should be about 40 mV (after filtering with gain of ~5000).
        # This is satisfied for most ice temperatures by using an effective
        # resistance of ~5.45 Ohm
        # Additionally, the bandwidth of the antenna is set slightly larger
        # than the nominal bandwidth of the true ARA antenna system (700 MHz),
        # but the extra frequencies should be killed by the front-end filter
        super().setup_antenna(position=self.position,
                              center_frequency=center_frequency,
                              bandwidth=bandwidth,
                              resistance=resistance,
                              orientation=orientation,
                              effective_height=effective_height,
                              response_data=response_data,
                              response_freqs=response_freqs,
                              noisy=noisy)

    # Tunnel diode response functions pulled from arasim
    _td_args = {
        'down1': (-0.8, 15e-9, 2.3e-9, 0),
        'down2': (-0.2, 15e-9, 4e-9, 0),
        'up': (1, 18e-9, 7e-9, 1e9)
    }
    # Set td_args['up'][0] based on the other args, like in arasim
    _td_args['up'] = (-np.sqrt(2*np.pi) *
                      (_td_args['down1'][0]*_td_args['down1'][2] +
                       _td_args['down2'][0]*_td_args['down2'][2]) /
                      (2e18*_td_args['up'][2]**3),) + _td_args['up'][1:]

    # Set "down" and "up" functions as in arasim
    @classmethod
    def _td_fdown1(cls, x):
        return (cls._td_args['down1'][3] + cls._td_args['down1'][0] *
                np.exp(-(x-cls._td_args['down1'][1])**2 /
                       (2*cls._td_args['down1'][2]**2)))

    @classmethod
    def _td_fdown2(cls, x):
        return (cls._td_args['down2'][3] + cls._td_args['down2'][0] *
                np.exp(-(x-cls._td_args['down2'][1])**2 /
                       (2*cls._td_args['down2'][2]**2)))

    @classmethod
    def _td_fup(cls, x):
        return (cls._td_args['up'][0] *
                (cls._td_args['up'][3] * (x-cls._td_args['up'][1]))**2 *
                np.exp(-(x-cls._td_args['up'][1])/cls._td_args['up'][2]))

    def tunnel_diode(self, signal):
        """Return the signal response from the tunnel diode."""
        if signal.value_type!=Signal.ValueTypes.voltage:
            raise ValueError("Tunnel diode only accepts voltage signals")
        t_max = 1e-7
        n_pts = int(t_max/signal.dt)
        times = np.linspace(0, t_max, n_pts+1)
        diode_resp = self._td_fdown1(times) + self._td_fdown2(times)
        t_slice = times>self._td_args['up'][1]
        diode_resp[t_slice] += self._td_fup(times[t_slice])
        conv = scipy.signal.convolve(signal.values**2 / self.antenna.resistance,
                                     diode_resp, mode='full')
        # Signal class will automatically only take the first part of conv,
        # which is what we want.
        # conv multiplied by dt so that the amplitude stays constant for
        # varying dts (determined emperically, see FastAskaryanSignal comments)
        output = Signal(signal.times, conv*signal.dt,
                        value_type=Signal.ValueTypes.power)
        return output

    def front_end(self, signal):
        """Apply the front-end processing of the antenna signal, including
        electronics chain filters/amplification and clipping."""
        copy = Signal(signal.times, signal.values)
        copy.filter_frequencies(self.antenna.interpolate_filter,
                                force_causality=True)
        clipped_values = np.clip(copy.values,
                                 a_min=-self.amplifier_clipping,
                                 a_max=self.amplifier_clipping)
        return Signal(signal.times, clipped_values,
                      value_type=signal.value_type)

    def trigger(self, signal):
        if self._power_mean is None or self._power_rms is None:
            # Prepare for antenna trigger by finding rms of noise waveform
            # (1 microsecond) convolved with tunnel diode response
            long_noise = self.antenna.make_noise(np.linspace(0, 1e-6, 10001))
            power_noise = self.tunnel_diode(self.front_end(long_noise))
            self._power_mean = np.mean(power_noise.values)
            self._power_rms = np.sqrt(np.mean(power_noise.values**2))

        power_signal = self.tunnel_diode(signal)
        low_trigger = (self._power_mean -
                       self._power_rms*np.abs(self.power_threshold))
        high_trigger = (self._power_mean +
                        self._power_rms*np.abs(self.power_threshold))
        return (np.min(power_signal.values)<low_trigger or
                np.max(power_signal.values)>high_trigger)



class HpolAntenna(ARAAntennaSystem):
    """ARA Hpol ("quad-slot") antenna system consisting of antenna,
    amplification, and tunnel diode response."""
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=1, noisy=True):
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
                 amplification=1, amplifier_clipping=1, noisy=True):
        super().__init__(name=name, position=position,
                         power_threshold=power_threshold,
                         response_data=VPOL_RESPONSE,
                         response_freqs=VPOL_FREQS,
                         orientation=(0,0,1),
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy)
