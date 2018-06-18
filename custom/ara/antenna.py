"""Module containing customized antenna classes for ARA"""

import os.path
import numpy as np
import scipy.signal
from pyrex.internal_functions import normalize
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.detector import AntennaSystem
from pyrex.ice_model import IceModel


def _read_directionality_data(filename):
    """Gather antenna directionality data from a data file. Returns the data as
    a dictionary with keys (freq, theta, phi) and values (gain, phase).
    Also returns a set of the frequencies appearing in the keys."""
    data = {}
    freqs = set()
    thetas = set()
    phis = set()
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
                thetas.add(theta)
                phi = int(words[1])
                phis.add(phi)
                db_gain = float(words[2])
                # AraSim actually only seems to use the sqrt of the gain
                # (must be gain in power, not voltage)
                # gain = np.sqrt(float(words[3]))
                gain = np.sqrt(10**(db_gain/10))
                phase = np.radians(float(words[4]))
                data[(freq, theta, phi)] = (gain, phase)

    for theta in thetas:
        for phi in phis:
            phase_offset = 0
            prev_phase = 0
            for freq in sorted(freqs):
                # In order to smoothly interpolate phases, don't allow the phase
                # to wrap from -pi to +pi, but instead apply an offset
                gain, phase = data[(freq, theta, phi)]
                if phase-prev_phase>np.pi:
                    phase_offset -= 2*np.pi
                elif prev_phase-phase>np.pi:
                    phase_offset += 2*np.pi
                prev_phase = phase
                data[(freq, theta, phi)] = (gain, phase+phase_offset)

    return data, freqs


def _read_filter_data(filename):
    """Gather electronics chain filtering data from a data file. Returns the
    data as a dictionary with frequency keys and values (gain, phase)."""
    data = {}
    freq_scale = 0
    phase_offset = 0
    prev_phase = 0
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
                # In order to smoothly interpolate phases, don't allow the phase
                # to wrap from -pi to +pi, but instead apply an offset
                if phase-prev_phase>np.pi:
                    phase_offset -= 2*np.pi
                elif prev_phase-phase>np.pi:
                    phase_offset += 2*np.pi
                prev_phase = phase
                data[freq] = (gain, phase+phase_offset)

    return data


ARA_DATA_DIR = os.path.dirname(__file__)
VPOL_DATA_FILE = os.path.join(ARA_DATA_DIR, "ARA_bicone6in_output.txt")
HPOL_DATA_FILE = os.path.join(ARA_DATA_DIR, "ARA_dipoletest1_output.txt")
FILTER_DATA_FILE = os.path.join(ARA_DATA_DIR,
                                "ARA_Electronics_TotalGain_TwoFilters.txt")
VPOL_DIRECTIONALITY, VPOL_FREQS = _read_directionality_data(VPOL_DATA_FILE)
HPOL_DIRECTIONALITY, HPOL_FREQS = _read_directionality_data(HPOL_DATA_FILE)
ALL_FILTERS = _read_filter_data(FILTER_DATA_FILE)


class ARAAntenna(Antenna):
    """Antenna to be used in ARA antenna systems. Has a position (m),
    center frequency (Hz), bandwidth (Hz), resistance (ohm),
    effective height (m), and polarization direction."""
    def __init__(self, position, center_frequency, bandwidth, resistance,
                 orientation=(0,0,1), efficiency=1, noisy=True,
                 unique_noise_waveforms=10,
                 directionality_data=None, directionality_freqs=None):
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
                         efficiency=efficiency, freq_range=(f_low, f_high),
                         temperature=IceModel.temperature(position[2]),
                         resistance=resistance, noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)

        self._dir_data = directionality_data
        self._dir_freqs = directionality_freqs
        # Just in case the frequencies don't get set, set them now
        if self._dir_freqs is None and self._dir_data is not None:
            self._dir_freqs = set()
            for key in self._dir_data:
                self._dir_freqs.add(key[0])

        self._filter_data = ALL_FILTERS


    def polarization_gain(self, polarization):
        """Polarization gain is simply the dot product of the polarization
        with the antenna's z-axis."""
        return np.vdot(self.z_axis, polarization)


    def generate_directionality_gains(self, theta, phi):
        """Generate arrays of frequencies and gains for given angles."""
        if self._dir_data is None:
            return np.array([1]), np.array([1]), np.array([0])

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

        nfreqs = len(self._dir_freqs)
        gain_ij = np.zeros(nfreqs)
        phase_ij = np.zeros(nfreqs)
        gain_i1j = np.zeros(nfreqs)
        phase_i1j = np.zeros(nfreqs)
        gain_ij1 = np.zeros(nfreqs)
        phase_ij1 = np.zeros(nfreqs)
        gain_i1j1 = np.zeros(nfreqs)
        phase_i1j1 = np.zeros(nfreqs)
        for f, freq in enumerate(sorted(self._dir_freqs)):
            gain_ij[f] = self._dir_data[(freq, theta_under, phi_under)][0]
            phase_ij[f] = self._dir_data[(freq, theta_under, phi_under)][1]
            gain_i1j[f] = self._dir_data[(freq, theta_over, phi_under)][0]
            phase_i1j[f] = self._dir_data[(freq, theta_over, phi_under)][1]
            gain_ij1[f] = self._dir_data[(freq, theta_under, phi_over)][0]
            phase_ij1[f] = self._dir_data[(freq, theta_under, phi_over)][1]
            gain_i1j1[f] = self._dir_data[(freq, theta_over, phi_over)][0]
            phase_i1j1[f] = self._dir_data[(freq, theta_over, phi_over)][1]

        freqs = np.array(sorted(self._dir_freqs))
        gains = ((1-t)*(1-u)*gain_ij + t*(1-u)*gain_i1j +
                 (1-t)*u*gain_ij1 + t*u*gain_i1j1)
        phases = ((1-t)*(1-u)*phase_ij + t*(1-u)*phase_i1j +
                  (1-t)*u*phase_ij1 + t*u*phase_i1j1)

        return freqs, gains, phases

    def interpolate_filter(self, frequencies):
        """Generate interpolated filter values for given frequencies (Hz)."""
        freqs = sorted(self._filter_data.keys())
        gains = [self._filter_data[f][0] for f in freqs]
        phases = [self._filter_data[f][1] for f in freqs]
        interp_gains = np.interp(frequencies, freqs, gains, left=0, right=0)
        interp_phases = np.interp(frequencies, freqs, phases, left=0, right=0)
        return interp_gains * np.exp(1j * interp_phases)

    def response(self, frequencies):
        """Frequency response of the antenna for given frequencies (Hz)
        incorporating effective height and some electronics gains."""
        # From AraSim GaintoHeight function, removing gain to receive function.
        # gain=4*pi*A_eff/lambda^2 and h_eff=2*sqrt(A_eff*Z_rx/Z_air)
        heff = np.zeros(len(frequencies))
        n = IceModel.index(self.position[2])
        heff[frequencies!=0] = 2*np.sqrt((3e8/frequencies[frequencies!=0]/n)**2
                                         * n*50/377 /(4*np.pi))
        # From AraSim ApplyAntFactors function, removing polarization.
        # sqrt(2) for 3dB splitter for TURF, SURF,
        # 0.5 to calculate power with heff
        return heff * 0.5 / np.sqrt(2)


    def receive(self, signal, direction=None, polarization=None,
                force_real=True):
        """Process incoming signal according to the filter function and
        store it to the signals list. Subclasses may extend this fuction,
        but should end with super().receive(signal)."""
        copy = Signal(signal.times, signal.values, value_type=Signal.ValueTypes.voltage)
        copy.filter_frequencies(self.response, force_real=force_real)

        if direction is not None:
            # Calculate theta and phi relative to the orientation
            origin = self.position - normalize(direction)
            r, theta, phi = self._convert_to_antenna_coordinates(origin)
            freq_data, gain_data, phase_data = self.generate_directionality_gains(theta, phi)
            def interpolate_directionality(frequencies):
                interp_gains = np.interp(frequencies, freq_data, gain_data,
                                         left=0, right=0)
                interp_phases = np.interp(frequencies, freq_data, phase_data,
                                          left=0, right=0)
                return interp_gains * np.exp(1j * interp_phases)
            copy.filter_frequencies(interpolate_directionality,
                                    force_real=force_real)

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
    def __init__(self, name, position, power_threshold,
                 directionality_data=None, directionality_freqs=None,
                 orientation=(0,0,1), amplification=1, amplifier_clipping=1,
                 noisy=True, unique_noise_waveforms=10):
        super().__init__(ARAAntenna)

        self.name = str(name)
        self.position = position

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.setup_antenna(directionality_data=directionality_data,
                           directionality_freqs=directionality_freqs,
                           orientation=orientation, noisy=noisy,
                           unique_noise_waveforms=unique_noise_waveforms)

        self.power_threshold = power_threshold
        self._power_mean = None
        self._power_rms = None

    def setup_antenna(self, center_frequency=500e6, bandwidth=800e6,
                      resistance=8.5, orientation=(0,0,1),
                      directionality_data=None, directionality_freqs=None,
                      efficiency=1, noisy=True, unique_noise_waveforms=10):
        """Sets attributes of the antenna including center frequency (Hz),
        bandwidth (Hz), resistance (ohms), orientation, and effective
        height (m)."""
        # Noise rms should be about 40 mV (after filtering with gain of ~5000).
        # This is satisfied for most ice temperatures by using an effective
        # resistance of ~8.5 Ohm
        # Additionally, the bandwidth of the antenna is set slightly larger
        # than the nominal bandwidth of the true ARA antenna system (700 MHz),
        # but the extra frequencies should be killed by the front-end filter
        super().setup_antenna(position=self.position,
                              center_frequency=center_frequency,
                              bandwidth=bandwidth,
                              resistance=resistance,
                              orientation=orientation,
                              efficiency=efficiency,
                              directionality_data=directionality_data,
                              directionality_freqs=directionality_freqs,
                              noisy=noisy,
                              unique_noise_waveforms=unique_noise_waveforms)

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
                                force_real=True)
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

    def receive(self, signal, direction=None, polarization=None,
                force_real=True):
        """Process incoming signal according to the filter function and
        store it to the signals list. Forces real filtered signals by
        default."""
        super().receive(signal=signal, direction=direction,
                        polarization=polarization,
                        force_real=force_real)



class HpolAntenna(ARAAntennaSystem):
    """ARA Hpol ("quad-slot") antenna system consisting of antenna,
    amplification, and tunnel diode response."""
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=1, noisy=True,
                 unique_noise_waveforms=10):
        super().__init__(name=name, position=position,
                         power_threshold=power_threshold,
                         directionality_data=HPOL_DIRECTIONALITY,
                         directionality_freqs=HPOL_FREQS,
                         orientation=(0,0,1),
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)


class VpolAntenna(ARAAntennaSystem):
    """ARA Vpol ("bicone" or "birdcage") antenna system consisting of antenna,
    amplification, and tunnel diode response."""
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=1, noisy=True,
                 unique_noise_waveforms=10):
        super().__init__(name=name, position=position,
                         power_threshold=power_threshold,
                         directionality_data=VPOL_DIRECTIONALITY,
                         directionality_freqs=VPOL_FREQS,
                         orientation=(0,0,1),
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)
