"""
Module containing customized antenna classes for ARA.

Many of the methods here mirror methods used in the antennas in AraSim, to
ensure that AraSim results can be matched.

"""

import logging
import os.path
import pickle
import numpy as np
import scipy.constants
import scipy.signal
from pyrex.internal_functions import (normalize, complex_bilinear_interp,
                                      complex_interp)
from pyrex.signals import Signal, FunctionSignal
from pyrex.antenna import Antenna
from pyrex.detector import AntennaSystem

logger = logging.getLogger(__name__)


def _read_arasim_antenna_data(filename):
    """
    Gather antenna directionality data from an AraSim-formatted data file.

    The data file should have columns for theta, phi, dB gain, non-dB gain, and
    phase (in degrees). This should be divided into sections for each frequency
    with a header line "freq : X MHz", optionally followed by a second line
    "SWR : Y".

    Parameters
    ----------
    filename : str
        Name of the data file.

    Returns
    -------
    response : ndarray
        3-D array of complex-valued voltage gains as a function of frequency
        along axis 0, zenith along axis 1, and azimuth along axis 2.
    frequencies : ndarray
        Frequencies (Hz) corresponding to axis 0 of `response`.
    thetas : ndarray
        Zenith angles (degrees) corresponding to axis 1 of `response`.
    phis : ndarray
        Azimuth angles (degrees) corresponding to axis 2 of `response`.

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

    # Convert data dictionary into 3-D array of responses
    response = np.empty((len(freqs), len(thetas), len(phis)),
                        dtype=np.complex_)
    for i, freq in enumerate(sorted(freqs)):
        for j, theta in enumerate(sorted(thetas)):
            for k, phi in enumerate(sorted(phis)):
                gain, phase = data[(freq, theta, phi)]
                response[i, j, k] = gain * np.exp(1j*phase)

    response_data = (response, np.array(sorted(freqs)),
                     np.array(sorted(thetas)), np.array(sorted(phis)))
    return _fix_response_wrapping(response_data)


# If the antenna responses don't go all the way to phi=360, add the extra
# column for the sake of the interpolators
def _fix_response_wrapping(response_data):
    """
    Add phi=360 degrees column to antenna response data.

    The interpolators require that the full azimuth range of the antennas is
    described, so this function duplicates the phi=0 column of the antenna
    response into a phi=360 column, as long as it matches with the rest of
    the phi spacings.

    Parameters
    ----------
    response_data : tuple of ndarray
        Tuple containing the response data for the antenna. The first element
        should contain a 3-D array of the antenna response model as a function
        of frequency (axis 0), zenith (axis 1), and azimuth (axis 2). The
        remaining elements should be the values of the frequency, zenith, and
        azimuth axes, respectively.

    Returns
    -------
    response : ndarray
        Corrected 3-D array of antenna response values, including the phi=360
        degree column if possible.
    frequencies : ndarray
        Frequencies (Hz) corresponding to axis 0 of `response`.
    thetas : ndarray
        Zenith angles (degrees) corresponding to axis 1 of `response`.
    phis : ndarray
        Azimuth angles (degrees) corresponding to axis 2 of `response`.

    """
    response, freqs, thetas, phis = response_data
    if phis[-1]==360:
        return response_data
    if phis[0]==0 and phis[-1]==360-phis[1]:
        phis = np.concatenate((phis, [360]))
        response = np.concatenate((response, response[:, :, 0:1]), axis=2)
    return response, freqs, thetas, phis


def _read_arasim_antenna_pickle(filename):
    """
    Gather antenna directional response data from a pickled data file.

    The data file should be a pickled file containing the antenna directional
    response data from an AraSim-formatted data file as returned by the
    `_read_arasim_antenna_data` function.

    Parameters
    ----------
    filename : str
        Name of the data file without the ``.pkl`` extension.

    Returns
    -------
    response : ndarray
        3-D array of complex-valued voltage gains as a function of frequency
        along axis 0, zenith along axis 1, and azimuth along axis 2.
    frequencies : ndarray
        Frequencies (Hz) corresponding to axis 0 of `response`.
    thetas : ndarray
        Zenith angles (degrees) corresponding to axis 1 of `response`.
    phis : ndarray
        Azimuth angles (degrees) corresponding to axis 2 of `response`.

    See Also
    --------
    _read_arasim_antenna_data : Gather antenna directionality data from an
                                AraSim-formatted data file.

    """
    # Quick fix for filenames with one of the approved extensions already
    # (just strip it)
    if filename.endswith(".txt") or filename.endswith(".pkl"):
        filename = filename[:-4]

    # If there is no pickle file, read the response data using the
    # _read_arasim_antenna_data function, and then make a pickle file
    if not os.path.isfile(filename+".pkl"):
        logger.warning("Antenna model file %s.pkl not found. "+
                       "Generating a new file now", filename)
        response_data = _read_arasim_antenna_data(filename+".txt")
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(response_data, f)
        return response_data

    # Otherwise, read from the pickle file
    else:
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)


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
    gains : ndarray
        Complex-valued voltage gains as a function of frequency.
    frequencies : ndarray
        Frequencies (Hz) corresponding to the values of `gains`.

    """
    gains = []
    freqs = []
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
                freqs.append(freq)
                gains.append(gain * np.exp(1j*phase))

    return np.array(gains), np.array(freqs)


ARA_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VPOL_DATA_FILE = os.path.join(ARA_DATA_DIR,
                              "Vpol_original_CrossFeed_150mmHole_Ice_ARASim.txt")
HPOL_DATA_FILE = os.path.join(ARA_DATA_DIR,
                              "Hpol_original_150mmHole_Ice_ARASim.txt")
FILT_DATA_FILE = os.path.join(ARA_DATA_DIR,
                              "ARA_Electronics_TotalGain_TwoFilters.txt")
# Vpol data file contains only the theta responses
VPOL_THETA_RESPONSE_DATA = _read_arasim_antenna_pickle(VPOL_DATA_FILE)
VPOL_RESPONSE_DATA = (
    VPOL_THETA_RESPONSE_DATA[0],
    np.zeros(VPOL_THETA_RESPONSE_DATA[0].shape),
    *VPOL_THETA_RESPONSE_DATA[1:]
)
# Hpol data file contains only the phi responses
HPOL_PHI_RESPONSE_DATA = _read_arasim_antenna_pickle(HPOL_DATA_FILE)
HPOL_RESPONSE_DATA = (
    np.zeros(HPOL_PHI_RESPONSE_DATA[0].shape),
    *HPOL_PHI_RESPONSE_DATA
)
ALL_FILTERS_DATA = _read_filter_data(FILT_DATA_FILE)



class ARAAntenna(Antenna):
    """
    Antenna class to be used for ARA antennas.

    Stores the attributes of an antenna as well as handling receiving,
    processing, and storing signals and adding noise. Antenna response based on
    provided models.

    Parameters
    ----------
    response_data : tuple of array_like
        Tuple containing the response data for the antenna along the theta
        and phi polarization directions. The first and second elements should
        contain 3-D arrays of the antenna response model in the theta and phi
        polarizations, respectively, as a function of frequency (axis 0),
        zenith (axis 1), and azimuth (axis 2). The remaining elements should be
        the values of the frequency, zenith, and azimuth axes, respectively.
    position : array_like
        Vector position of the antenna.
    center_frequency : float
        Frequency (Hz) at the center of the antenna's frequency range.
    bandwidth : float
        Bandwidth (Hz) of the antenna.
    temperature : float
        The noise temperature (K) of the antenna. Used in combination with
        `resistance` to calculate the RMS voltage of the antenna noise.
    resistance : float
        The noise resistance (ohm) of the antenna. Used in combination with
        `temperature` to calculate the RMS voltage of the antenna noise.
    orientation : array_like, optional
        Vector direction of the z-axis of the antenna.
    efficiency : float, optional
        Antenna efficiency applied to incoming signal values.
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

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
        The RMS voltage (V) of the antenna noise. If not ``None``, this value
        will be used instead of the RMS voltage calculated from the values of
        `temperature` and `resistance`.
    signals : list of Signal
        The signals which have been received by the antenna.
    is_hit
    is_hit_mc_truth
    waveforms
    all_waveforms

    See Also
    --------
    pyrex.Antenna : Base class for antennas.

    """
    def __init__(self, response_data, position, center_frequency, bandwidth,
                 temperature, resistance, orientation=(0,0,1), efficiency=1,
                 noisy=True, unique_noise_waveforms=10):
        # Parse the response data
        self._theta_response = response_data[0]
        self._phi_response = response_data[1]
        self._response_freqs = response_data[2]
        self._response_zens = response_data[3]
        self._response_azis = response_data[4]

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
                         temperature=temperature, resistance=resistance,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)

    def directional_gain(self, theta, phi):
        raise NotImplementedError("Directional gain is not defined for "+
                                  self.__class__.__name__+". Use the "+
                                  "directional_response method instead.")

    def polarization_gain(self, polarization):
        raise NotImplementedError("Polarization gain is not defined for "+
                                  self.__class__.__name__+". Use the "+
                                  "directional_response method instead.")

    def directional_response(self, theta, phi, polarization):
        """
        Generate the (complex) frequency-dependent directional response.

        For given angles and polarization direction, use the model of the
        directional and polarization gains of the antenna to generate a
        function for the interpolated response of the antenna with respect to
        frequency. Used with the `frequency_response` method to calculate
        effective heights.

        Parameters
        ----------
        theta : float
            Polar angle (radians) from which a signal is arriving.
        phi : float
            Azimuthal angle (radians) from which a signal is arriving.
        polarization : array_like
            Normalized polarization vector in the antenna coordinate system.

        Returns
        -------
        function
            A function which returns complex-valued voltage gains for given
            frequencies, using the values of incoming angle and polarization.

        See Also
        --------
        ARAAntenna.frequency_response : Calculate the (complex) frequency
                                        response of the antenna.

        """
        e_theta = [np.cos(theta) * np.cos(phi),
                   np.cos(theta) * np.sin(phi),
                   -np.sin(theta)]
        e_phi = [-np.sin(phi), np.cos(phi), 0]
        theta_factor = np.dot(polarization, e_theta)
        phi_factor = np.dot(polarization, e_phi)
        theta_gains = complex_bilinear_interp(
            x=np.degrees(theta), y=np.degrees(phi),
            xp=self._response_zens,
            yp=self._response_azis,
            fp=self._theta_response,
            method='cartesian'
        )
        phi_gains = complex_bilinear_interp(
            x=np.degrees(theta), y=np.degrees(phi),
            xp=self._response_zens,
            yp=self._response_azis,
            fp=self._phi_response,
            method='cartesian'
        )
        freq_interpolator = lambda frequencies: complex_interp(
            x=frequencies, xp=self._response_freqs,
            fp=theta_factor*theta_gains + phi_factor*phi_gains,
            method='euler', outer=0
        )
        return freq_interpolator

    def frequency_response(self, frequencies):
        """
        Calculate the (complex) frequency response of the antenna.

        Rather than handling the entire frequency response of the antenna, this
        method is being used to convert the frequency-dependent gains from the
        `directional_response` method into effective heights.

        Parameters
        ----------
        frequencies : array_like
            1D array of frequencies (Hz) at which to calculate gains.

        Returns
        -------
        array_like
            Complex gains in voltage for the given `frequencies`.

        See Also
        --------
        ARAAntenna.directional_response : Generate the (complex) frequency
                                          dependent directional response.

        """
        # From AraSim GaintoHeight function, with gain calculation moved to
        # the directional_response method.
        # gain=4*pi*A_eff/lambda^2 and h_eff=2*sqrt(A_eff*Z_rx/Z_air)
        # Then 0.5 to calculate power with heff (cancels 2 above)
        heff = np.zeros(len(frequencies))
        # The index of refraction in this calculation should be the index of
        # the ice used in the production of the antenna model.
        n = 1.78
        heff[frequencies!=0] = np.sqrt((scipy.constants.c
                                        /frequencies[frequencies!=0]/n)**2
                                       * n*50/377 /(4*np.pi))
        return heff


    def apply_response(self, signal, direction=None, polarization=None,
                       force_real=True):
        """
        Process the complete antenna response for an incoming signal.

        Processes the incoming signal according to the frequency response of
        the antenna, the efficiency, and the antenna factor. May also apply the
        directionality and the polarization gain depending on the provided
        parameters. Subclasses may wish to overwrite this function if the
        full antenna response cannot be divided nicely into the described
        pieces.

        Parameters
        ----------
        signal : Signal
            Incoming ``Signal`` object to process.
        direction : array_like, optional
            Vector denoting the direction of travel of the signal as it reaches
            the antenna (in the global coordinate frame). If ``None`` no
            directional response will be applied.
        polarization : array_like, optional
            Vector denoting the signal's polarization direction (in the global
            coordinate frame). If ``None`` no polarization gain will be applied.
        force_real : boolean, optional
            Whether or not the frequency response should be redefined in the
            negative-frequency domain to keep the values of the filtered signal
            real.

        Returns
        -------
        Signal
            Processed ``Signal`` object after the complete antenna response has
            been applied. Should have a ``value_type`` of ``voltage``.

        Raises
        ------
        ValueError
            If the given `signal` does not have a ``value_type`` of ``voltage``
            or ``field``.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        new_signal = signal.copy()
        new_signal.value_type = Signal.Type.voltage
        freq_response = self.frequency_response

        if direction is not None and polarization is not None:
            # Calculate theta and phi relative to the orientation
            origin = self.position - normalize(direction)
            r, theta, phi = self._convert_to_antenna_coordinates(origin)
            # Calculate polarization vector in the antenna coordinates
            y_axis = np.cross(self.z_axis, self.x_axis)
            transformation = np.array([self.x_axis, y_axis, self.z_axis])
            ant_pol = np.dot(transformation, normalize(polarization))
            # Calculate directional response as a function of frequency
            directive_response = self.directional_response(theta, phi, ant_pol)
            freq_response = lambda f: (self.frequency_response(f)
                                       * directive_response(f))

        elif (direction is not None and polarization is None
              or direction is None and polarization is not None):
            raise ValueError("Direction and polarization must be specified together")

        # Apply (combined) frequency response
        new_signal.filter_frequencies(freq_response, force_real=force_real)

        signal_factor = self.efficiency

        if signal.value_type==Signal.Type.voltage:
            pass
        elif signal.value_type==Signal.Type.field:
            signal_factor /= self.antenna_factor
        else:
            raise ValueError("Signal's value type must be either "
                             +"voltage or field. Given "+str(signal.value_type))

        new_signal *= signal_factor

        return new_signal

    # Redefine receive method to use force_real as True by default
    def receive(self, signal, direction=None, polarization=None,
                force_real=True):
        """
        Process and store one or more incoming (polarized) signals.

        Processes the incoming signal(s) according to the ``apply_response``
        method, then stores the total processed signal to the signals list. If
        more than one signal is given, they should be logically connected as
        separately polarized portions of the same signal.

        Parameters
        ----------
        signal : Signal or array_like
            Incoming ``Signal`` object(s) to process and store. May be separate
            polarization representations, but therefore should have the same
            times.
        direction : array_like, optional
            Vector denoting the direction of travel of the signal(s) as they
            reach the antenna (in the global coordinate frame). If ``None`` no
            directional gain will be applied.
        polarization : array_like, optional
            Vector(s) denoting the signal's polarization direction (in the
            global coordinate frame). Number of vectors should match the number
            of elements in `signal` argument. If ``None`` no polarization gain
            will be applied.
        force_real : boolean, optional
            Whether or not the frequency response should be redefined in the
            negative-frequency domain to keep the values of the filtered signal
            real.

        Raises
        ------
        ValueError
            If the number of polarizations does not match the number of signals.
            Or if the signals do not have the same `times` array.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        super().receive(signal=signal, direction=direction,
                        polarization=polarization,
                        force_real=force_real)



class ARAAntennaSystem(AntennaSystem):
    """
    Antenna system extending base ARA antenna with front-end processing.

    Applies as the front end a filter representing the full ARA electronics
    chain (including amplification) and signal clipping. Additionally provides
    a method for passing a signal through the tunnel diode.

    Parameters
    ----------
    response_data : tuple of array_like
        Tuple containing the response data for the antenna along the theta
        and phi polarization directions. The first and second elements should
        contain 3-D arrays of the antenna response model in the theta and phi
        polarizations, respectively, as a function of frequency (axis 0),
        zenith (axis 1), and azimuth (axis 2). The remaining elements should be
        the values of the frequency, zenith, and azimuth axes, respectively.
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    power_threshold : float
        Power threshold for trigger condition. Antenna triggers if a signal
        passed through the tunnel diode exceeds this threshold times the noise
        RMS of the tunnel diode.
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
    lead_in_time : float
        Lead-in time (s) required for the front end to equilibrate.
        Automatically added in before calculation of signals and waveforms.
    is_hit
    is_hit_mc_truth
    signals
    waveforms
    all_waveforms

    See Also
    --------
    pyrex.AntennaSystem : Base class for antenna system with front-end
                          processing.
    ARAAntenna : Antenna class to be used for ARA antennas.

    """
    lead_in_time = 5e-9

    def __init__(self, response_data, name, position, power_threshold,
                 orientation=(0,0,1), amplification=1, amplifier_clipping=1,
                 noisy=True, unique_noise_waveforms=10,
                 **kwargs):
        super().__init__(ARAAntenna)

        self.name = str(name)
        self.position = position

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.setup_antenna(response_data=response_data,
                           orientation=orientation, noisy=noisy,
                           unique_noise_waveforms=unique_noise_waveforms,
                           **kwargs)

        self.power_threshold = power_threshold
        self._power_mean = None
        self._power_std = None

        self._filter_response = ALL_FILTERS_DATA[0]
        self._filter_freqs = ALL_FILTERS_DATA[1]

    @property
    def _metadata(self):
        """Metadata dictionary for writing `ARAAntennaSystem` information."""
        meta = super()._metadata
        meta.update({
            "name": self.name,
            "lead_in_time": self.lead_in_time,
            "amplification": self.amplification,
            "amplifier_clipping": self.amplifier_clipping,
            "power_threshold": self.power_threshold,
        })
        return meta

    def setup_antenna(self, response_data, center_frequency=500e6,
                      bandwidth=800e6, temperature=325, resistance=50,
                      orientation=(0,0,1), efficiency=1, noisy=True,
                      unique_noise_waveforms=10, **kwargs):
        """
        Setup the antenna by passing along its init arguments.

        Any arguments passed to this method are directly passed to the
        ``__init__`` methods of the ``antenna``'s class.

        Parameters
        ----------
        response_data : tuple of array_like
            Tuple containing the response data for the antenna along the theta
            and phi polarization directions. The first and second elements
            should contain 3-D arrays of the antenna response model in the
            theta and phi polarizations, respectively, as a function of
            frequency (axis 0), zenith (axis 1), and azimuth (axis 2). The
            remaining elements should be the values of the frequency, zenith,
            and azimuth axes, respectively.
        center_frequency : float, optional
            Frequency (Hz) at the center of the antenna's frequency range.
        bandwidth : float, optional
            Bandwidth (Hz) of the antenna.
        temperature : float, optional
            The noise temperature (K) of the antenna. Used in combination with
            `resistance` to calculate the RMS voltage of the antenna noise.
        resistance : float, optional
            The noise resistance (ohm) of the antenna. Used in combination with
            `temperature` to calculate the RMS voltage of the antenna noise.
        orientation : array_like, optional
            Vector direction of the z-axis of the antenna.
        efficiency : float, optional
            Antenna efficiency applied to incoming signal values.
        noisy : boolean, optional
            Whether or not the antenna should add noise to incoming signals.
        unique_noise_waveforms : int, optional
            The number of expected noise waveforms needed for each received
            signal to have its own noise.

        """
        # Noise rms should be about 40 mV (after filtering with gain of ~5000).
        # This is mostly satisfied by using the default noise temperature from
        # AraSim, 325 K, along with a 50 ohm resistance
        # Additionally, the bandwidth of the antenna is set slightly larger
        # than the nominal bandwidth of the true ARA antenna system (700 MHz),
        # but the extra frequencies should be killed by the front-end filter
        super().setup_antenna(response_data=response_data,
                              position=self.position,
                              center_frequency=center_frequency,
                              bandwidth=bandwidth,
                              temperature=temperature,
                              resistance=resistance,
                              orientation=orientation,
                              efficiency=efficiency,
                              noisy=noisy,
                              unique_noise_waveforms=unique_noise_waveforms,
                              **kwargs)

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

        The given signal is convolved with the tunnel diode response as in
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

        Notes
        -----
        The tunnel diode response is based on the response parameterized in
        AraSim, as developed by ANITA [1]_.

        References
        ----------
        .. [1] A. Connolly & R. Nichol, ANITA Note #411, "A Power-Based Time
            Domain Trigger Simulation."
            https://elog.phys.hawaii.edu/elog/anita_notes/080827_041639/powertrigger.pdf

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
        # varying dts (determined empirically, see ARZAskaryanSignal comments)
        output = Signal(signal.times, conv*signal.dt,
                        value_type=Signal.Type.power)
        return output

    def interpolate_filter(self, frequencies):
        """
        Generate interpolated filter values for given frequencies.

        Calculate the interpolated values of the antenna system's filter gain
        data for some frequencies.

        Parameters
        ----------
        frequencies : array_like
            1D array of frequencies (Hz) at which to calculate gains.

        Returns
        -------
        array_like
            Complex filter gain in voltage for the given `frequencies`.

        """
        return complex_interp(
            x=frequencies, xp=self._filter_freqs, fp=self._filter_response,
            method='euler', outer=0
        )

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
        base_signal = signal.copy()
        base_signal.filter_frequencies(self.interpolate_filter,
                                       force_real=True)
        # Apply sqrt(2) for 3dB splitter for TURF, SURF
        base_signal *= self.amplification / np.sqrt(2)
        clip_values = lambda times: np.clip(
            base_signal.with_times(times).values,
            a_min=-self.amplifier_clipping,
            a_max=self.amplifier_clipping
        )
        return FunctionSignal(signal.times, clip_values,
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
        if self._power_mean is None or self._power_std is None:
            # Prepare for antenna trigger by finding mean and standard
            # deviation of the full noise waveform convolved with the tunnel
            # diode response
            if len(self.antenna.signals)>0:
                times = self.antenna.signals[0].times
            else:
                times = signal.times
            n = len(times)
            dt = times[1]-times[0]
            duration = times[-1]-times[0] + dt
            full_times = np.linspace(0, duration*self.antenna.unique_noises,
                                     n*self.antenna.unique_noises)
            if self.antenna._noise_master is None:
                # Make sure the noise_master has the appropriate length
                # (automatically gets set to N*len(times) the first time it is
                # called, so make sure the first `times` is not the expanded
                # array but the single-signal array)
                self.antenna.make_noise(times)
            long_noise = self.antenna.make_noise(full_times)
            power_noise = self.tunnel_diode(self.front_end(long_noise))
            self._power_mean = np.mean(power_noise.values)
            self._power_std = np.std(power_noise.values)

        power_signal = self.tunnel_diode(signal)
        # Use the absolute value of the power_threshold value so that the value
        # can be specified as positive or negative (compatible with AraSim
        # which only works with negative values, resulting in some confusion)
        low_trigger = (self._power_mean -
                       self._power_std*np.abs(self.power_threshold))
        return np.min(power_signal.values)<low_trigger

    # Redefine receive method to use force_real as True by default
    def receive(self, signal, direction=None, polarization=None,
                force_real=True):
        """
        Process and store one or more incoming (polarized) signals.

        Processes the incoming signal(s) according to the ``apply_response``
        method, then stores the total processed signal to the signals list. If
        more than one signal is given, they should be logically connected as
        separately polarized portions of the same signal.

        Parameters
        ----------
        signal : Signal or array_like
            Incoming ``Signal`` object(s) to process and store. May be separate
            polarization representations, but therefore should have the same
            times.
        direction : array_like, optional
            Vector denoting the direction of travel of the signal(s) as they
            reach the antenna (in the global coordinate frame). If ``None`` no
            directional gain will be applied.
        polarization : array_like, optional
            Vector(s) denoting the signal's polarization direction (in the
            global coordinate frame). Number of vectors should match the number
            of elements in `signal` argument. If ``None`` no polarization gain
            will be applied.
        force_real : boolean, optional
            Whether or not the frequency response should be redefined in the
            negative-frequency domain to keep the values of the filtered signal
            real.

        Raises
        ------
        ValueError
            If the number of polarizations does not match the number of signals.
            Or if the signals do not have the same `times` array.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

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
    is_hit_mc_truth
    signals
    waveforms
    all_waveforms

    See Also
    --------
    ARAAntennaSystem : Antenna system extending base ARA antenna with front-end
                       processing.

    """
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=1, noisy=True,
                 unique_noise_waveforms=10):
        super().__init__(response_data=HPOL_RESPONSE_DATA,
                         name=name, position=position,
                         power_threshold=power_threshold,
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
    is_hit_mc_truth
    signals
    waveforms
    all_waveforms

    See Also
    --------
    ARAAntennaSystem : Antenna system extending base ARA antenna with front-end
                       processing.

    """
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=1, noisy=True,
                 unique_noise_waveforms=10):
        super().__init__(response_data=VPOL_RESPONSE_DATA,
                         name=name, position=position,
                         power_threshold=power_threshold,
                         orientation=(0,0,1),
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)
