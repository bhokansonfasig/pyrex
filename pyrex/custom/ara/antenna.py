"""
Module containing customized antenna classes for ARA.

Many of the methods here mirror methods used in the antennas in AraSim, to
ensure that AraSim results can be matched.

"""

import os.path
import numpy as np
import scipy.signal
from pyrex.internal_functions import normalize
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.detector import AntennaSystem
from pyrex.ice_model import IceModel


def _read_directionality_data(filename):
    """
    Gather antenna directionality data from a data file.

    The data file should have columns for theta, phi, dB gain, non-dB gain, and
    phase (in degrees). This should be divided into sections for each frequency
    with a header line "freq : X MHz", optionally followed by a second line
    "trans : Y".

    Parameters
    ----------
    filename : str
        Name of the data file.

    Returns
    -------
    dict
        Dictionary containing the data with keys (freq, theta, phi) and values
        (gain, phase).
    set
        Set of unique frequencies appearing in the data keys.

    """
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
    """
    Gather frequency-dependent filtering data from a data file.

    The data file should have columns for frequency, non-dB gain, and phase
    (in radians).

    Parameters
    ----------
    filename : str
        Name of the data file.

    Returns
    -------
    dict
        Dictionary containing the data with keys (freq) and values
        (gain, phase).

    """
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


ARA_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VPOL_DATA_FILE = os.path.join(ARA_DATA_DIR,
                              "ARA_bicone6in_output_MY_fixed.txt")
HPOL_DATA_FILE = os.path.join(ARA_DATA_DIR,
                              "ARA_dipoletest1_output_MY_fixed.txt")
FILT_DATA_FILE = os.path.join(ARA_DATA_DIR,
                              "ARA_Electronics_TotalGain_TwoFilters.txt")
VPOL_DIRECTIONALITY, VPOL_FREQS = _read_directionality_data(VPOL_DATA_FILE)
HPOL_DIRECTIONALITY, HPOL_FREQS = _read_directionality_data(HPOL_DATA_FILE)
ALL_FILTERS = _read_filter_data(FILT_DATA_FILE)


class ARAAntenna(Antenna):
    """
    Antenna class to be used for ARA antennas.

    Stores the attributes of an antenna as well as handling receiving,
    processing, and storing signals and adding noise.

    Parameters
    ----------
    position : array_like
        Vector position of the antenna.
    center_frequency : float
        Frequency (Hz) at the center of the antenna's frequency range.
    bandwidth : float
        Bandwidth (Hz) of the antenna.
    resistance : float
        The noise resistance (ohm) of the antenna. Used to calculate the RMS
        voltage of the antenna noise.
    orientation : array_like, optional
        Vector direction of the z-axis of the antenna.
    efficiency : float, optional
        Antenna efficiency applied to incoming signal values.
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.
    directionality_data : None or dict, optional
        Dictionary containing data on the directionality of the antenna. If
        ``None``, behavior is undefined.
    directionality_freqs : None or set, optional
        Set of frequencies in the directionality data ``dict`` keys. If
        ``None``, calculated automatically from `directionality_data`.

    Attributes
    ----------
    position : array_like
        Vector position of the antenna.
    z_axis : ndarray
        Vector direction of the z-axis of the antenna.
    x_axis : ndarray
        Vector direction of the x-axis of the antenna.
    antenna_factor : float
        Antenna factor used for converting fields to voltages.
    efficiency : float
        Antenna efficiency applied to incoming signal values.
    noisy : boolean
        Whether or not the antenna should add noise to incoming signals.
    unique_noises : int
        The number of expected noise waveforms needed for each received signal
        to have its own noise.
    freq_range : array_like
        The frequency band in which the antenna operates (used for noise
        production).
    temperature : float or None
        The noise temperature (K) of the antenna. Used in combination with
        `resistance` to calculate the RMS voltage of the antenna noise.
    resistance : float or None
        The noise resistance (ohm) of the antenna. Used in combination with
        `temperature` to calculate the RMS voltage of the antenna noise.
    noise_rms : float or None
        The RMS voltage (v) of the antenna noise. If not ``None``, this value
        will be used instead of the RMS voltage calculated from the values of
        `temperature` and `resistance`.
    signals : list of Signal
        The signals which have been received by the antenna.
    is_hit
    waveforms
    all_waveforms

    See Also
    --------
    pyrex.Antenna : Base class for antennas.

    """
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
        """
        Calculate the (complex) polarization gain of the antenna.

        Polarization gain is simply the dot product of the polarization
        with the antenna's z-axis.

        Parameters
        ----------
        polarization : array_like
            Vector polarization direction of the signal.

        Returns
        -------
        complex
            Complex gain in voltage for the given signal polarization.

        """
        return np.vdot(self.z_axis, polarization)


    def generate_directionality_gains(self, theta, phi):
        """
        Generate the (complex) frequency-dependent directional gains.

        For given angles, calculate arrays of frequencies and their
        corresponding gains and phases, based on the directionality data of the
        antenna.

        Parameters
        ----------
        theta : float
            Polar angle (radians) from which a signal is arriving.
        phi : float
            Azimuthal angle (radians) from which a signal is arriving.

        Returns
        -------
        freqs : array_like
            Frequencies over which the directionality was defined.
        gains : array_like
            Magnitudes of gains at the corrseponding frequencies.
        phases : array_like
            Phases of gains at the corresponding frequencies.

        """
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
        """
        Generate interpolated filter values for given frequencies.

        Calculate the interpolated values of the antenna's filter gain data
        for some frequencies.

        Parameters
        ----------
        frequencies : array_like
            1D array of frequencies (Hz) at which to calculate gains.

        Returns
        -------
        array_like
            Complex filter gain in voltage for the given `frequencies`.

        """
        freqs = sorted(self._filter_data.keys())
        gains = [self._filter_data[f][0] for f in freqs]
        phases = [self._filter_data[f][1] for f in freqs]
        interp_gains = np.interp(frequencies, freqs, gains, left=0, right=0)
        interp_phases = np.interp(frequencies, freqs, phases, left=0, right=0)
        return interp_gains * np.exp(1j * interp_phases)

    def response(self, frequencies):
        """
        Calculate the (complex) frequency response of the antenna.

        Frequency response of the antenna is based on the effective height
        calculation with some electronics gains thrown in. The frequency
        dependence of the directional gain is handled in the
        `generate_directionality_gains` method.

        Parameters
        ----------
        frequencies : array_like
            1D array of frequencies (Hz) at which to calculate gains.

        Returns
        -------
        array_like
            Complex gains in voltage for the given `frequencies`.

        """
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
        """
        Process and store an incoming signal.

        Processes the incoming signal according to the frequency response of
        the antenna, the efficiency, and the antenna factor. May also apply the
        directionality and the polarization gain depending on the provided
        parameters. Finally stores the processed signal to the signals list.

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

        """
        copy = Signal(signal.times, signal.values, value_type=Signal.Type.voltage)
        copy.filter_frequencies(self.response, force_real=force_real)

        if direction is not None:
            # Calculate theta and phi relative to the orientation
            origin = self.position - normalize(direction)
            r, theta, phi = self._convert_to_antenna_coordinates(origin)
            freq_data, gain_data, phase_data = self.generate_directionality_gains(theta, phi)
            def interpolate_directionality(frequencies):
                """
                Generate interpolated directionality for given frequencies.

                Parameters
                ----------
                frequencies : array_like
                    1D array of frequencies (Hz) at which to calculate gains.

                Returns
                -------
                array_like
                    Complex directional gain in voltage for the `frequencies`.

                """
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

        if signal.value_type==Signal.Type.voltage:
            pass
        elif signal.value_type==Signal.Type.field:
            signal_factor /= self.antenna_factor
        else:
            raise ValueError("Signal's value type must be either "
                             +"voltage or field. Given "+str(signal.value_type))

        copy.values *= signal_factor
        self.signals.append(copy)



class ARAAntennaSystem(AntennaSystem):
    """
    Antenna system extending base ARA antenna with front-end processing.

    Applies as the front end a filter representing the full ARA electronics
    chain (including amplification) and signal clipping. Additionally provides
    a method for passing a signal through the tunnel diode.

    Parameters
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    power_threshold : float
        Power threshold for trigger condition. Antenna triggers if a signal
        passed through the tunnel diode exceeds this threshold times the noise
        RMS of the tunnel diode.
    directionality_data : None or dict, optional
        Dictionary containing data on the directionality of the antenna. If
        ``None``, behavior is undefined.
    directionality_freqs : None or set, optional
        Set of frequencies in the directionality data ``dict`` keys. If
        ``None``, calculated automatically from `directionality_data`.
    orientation : array_like, optional
        Vector direction of the z-axis of the antenna.
    amplification : float, optional
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float, optional
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

    Attributes
    ----------
    antenna : Antenna
        ``Antenna`` object extended by the front end.
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    power_threshold : float
        Power threshold for trigger condition. Antenna triggers if a signal
        passed through the tunnel diode exceeds this threshold times the noise
        RMS of the tunnel diode.
    amplification : float
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    is_hit
    signals
    waveforms
    all_waveforms

    See Also
    --------
    pyrex.AntennaSystem : Base class for antenna system with front-end
                          processing.
    ARAAntenna : Antenna class to be used for ARA antennas.

    """
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
        """
        Setup the antenna by passing along its init arguments.

        Any arguments passed to this method are directly passed to the
        ``__init__`` methods of the ``antenna``'s class.

        Parameters
        ----------
        center_frequency : float, optional
            Frequency (Hz) at the center of the antenna's frequency range.
        bandwidth : float, optional
            Bandwidth (Hz) of the antenna.
        resistance : float, optional
            The noise resistance (ohm) of the antenna. Used to calculate the
            RMS voltage of the antenna noise.
        orientation : array_like, optional
            Vector direction of the z-axis of the antenna.
        directionality_data : None or dict, optional
            Dictionary containing data on the directionality of the antenna. If
            ``None``, behavior is undefined.
        directionality_freqs : None or set, optional
            Set of frequencies in the directionality data ``dict`` keys. If
            ``None``, calculated automatically from `directionality_data`.
        efficiency : float, optional
            Antenna efficiency applied to incoming signal values.
        noisy : boolean, optional
            Whether or not the antenna should add noise to incoming signals.
        unique_noise_waveforms : int, optional
            The number of expected noise waveforms needed for each received
            signal to have its own noise.

        """
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
        """
        Calculate a signal as processed by the tunnel diode.

        The given signal is convolved with the tunnel diodde response as in
        AraSim.

        Parameters
        ----------
        signal : Signal
            Signal to be processed by the tunnel diode.

        Returns
        -------
        Signal
            Signal output of the tunnel diode for the input `signal`.

        Raises
        ------
        ValueError
            If the input `signal` doesn't have a ``value_type`` of ``voltage``.

        """
        if signal.value_type!=Signal.Type.voltage:
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
        # varying dts (determined emperically, see ARVZAskaryanSignal comments)
        output = Signal(signal.times, conv*signal.dt,
                        value_type=Signal.Type.power)
        return output

    def front_end(self, signal):
        """
        Apply front-end processes to a signal and return the output.

        The front-end consists of the full ARA electronics chain (including
        amplification) and signal clipping.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to apply the front-end processes.

        Returns
        -------
        Signal
            Signal processed by the antenna front end.

        """
        copy = Signal(signal.times, signal.values)
        copy.filter_frequencies(self.antenna.interpolate_filter,
                                force_real=True)
        clipped_values = np.clip(copy.values,
                                 a_min=-self.amplifier_clipping,
                                 a_max=self.amplifier_clipping)
        return Signal(signal.times, clipped_values,
                      value_type=signal.value_type)

    def trigger(self, signal):
        """
        Check if the antenna system triggers on a given signal.

        Passes the signal through the tunnel diode. Then compares the maximum
        and minimum values to a tunnel diode noise signal. Triggers if one of
        the maximum or minimum values exceed the noise mean +/- the noise rms
        times the power threshold.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to test the trigger condition.

        Returns
        -------
        boolean
            Whether or not the antenna triggers on `signal`.

        """
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
        """
        Process and store an incoming signal.

        Processes the incoming signal according to the frequency response of
        the antenna, the efficiency, and the antenna factor. May also apply the
        directionality and the polarization gain depending on the provided
        parameters. Finally stores the processed signal to the signals list.

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

        """
        super().receive(signal=signal, direction=direction,
                        polarization=polarization,
                        force_real=force_real)



class HpolAntenna(ARAAntennaSystem):
    """
    ARA Hpol ("quad-slot") antenna system with front-end processing.

    Applies as the front end a filter representing the full ARA electronics
    chain (including amplification) and signal clipping. Additionally provides
    a method for passing a signal through the tunnel diode.

    Parameters
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    power_threshold : float
        Power threshold for trigger condition. Antenna triggers if a signal
        passed through the tunnel diode exceeds this threshold times the noise
        RMS of the tunnel diode.
    amplification : float, optional
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float, optional
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

    Attributes
    ----------
    antenna : Antenna
        ``Antenna`` object extended by the front end.
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    power_threshold : float
        Power threshold for trigger condition. Antenna triggers if a signal
        passed through the tunnel diode exceeds this threshold times the noise
        RMS of the tunnel diode.
    amplification : float
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    is_hit
    signals
    waveforms
    all_waveforms

    See Also
    --------
    ARAAntennaSystem : Antenna system extending base ARA antenna with front-end
                       processing.
    ARAAntenna : Antenna class to be used for ARA antennas.

    """
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
    """
    ARA Vpol ("bicone" or "birdcage") antenna system with front-end processing.

    Applies as the front end a filter representing the full ARA electronics
    chain (including amplification) and signal clipping. Additionally provides
    a method for passing a signal through the tunnel diode.

    Parameters
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    power_threshold : float
        Power threshold for trigger condition. Antenna triggers if a signal
        passed through the tunnel diode exceeds this threshold times the noise
        RMS of the tunnel diode.
    amplification : float, optional
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float, optional
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

    Attributes
    ----------
    antenna : Antenna
        ``Antenna`` object extended by the front end.
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    power_threshold : float
        Power threshold for trigger condition. Antenna triggers if a signal
        passed through the tunnel diode exceeds this threshold times the noise
        RMS of the tunnel diode.
    amplification : float
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    is_hit
    signals
    waveforms
    all_waveforms

    See Also
    --------
    ARAAntennaSystem : Antenna system extending base ARA antenna with front-end
                       processing.
    ARAAntenna : Antenna class to be used for ARA antennas.

    """
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
