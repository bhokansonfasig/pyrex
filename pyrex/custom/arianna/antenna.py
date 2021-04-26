"""
Module containing customized antenna classes for ARIANNA.

Based primarily on the LPDA implementation (and data) in NuRadioReco.

"""

import logging
import os.path
import pickle
import tarfile
import numpy as np
import scipy.constants
import scipy.interpolate
from pyrex.internal_functions import (normalize, complex_bilinear_interp,
                                      complex_interp)
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.detector import AntennaSystem

logger = logging.getLogger(__name__)



def _read_response_data(filename):
    """
    Gather antenna effective height data from a set of WIPLD data files.

    Data files should exist with names `filename`.ra1 and `filename`.ad1.
    The ``.ad1`` file should contain frequencies in the first column, real and
    imaginary parts of the impedance in the sixth and seventh columns, and
    S-parameter data in the eighth and ninth columns.
    The ``.ra1`` file should contain phi and theta in the first two columns,
    the real and imaginary parts of the phi field in the next two columns,
    and the real and imaginary parts of the theta field in the next two columns.
    This should be divided into sections for each frequency with a header line
    "  >  Gen. no.    1 X  GHz   73   91  Gain" where "X" is the frequency in
    GHz.

    Parameters
    ----------
    filename : str
        Name of the data files without extension. Extensions ``.ra1`` and
        ``.ad1`` will be added automatically.

    Returns
    -------
    theta_response : ndarray
        3-D array of complex-valued effective heights in the theta polarization
        as a function of frequency along axis 0, zenith along axis 1, and
        azimuth along axis 2.
    phi_response : ndarray
        3-D array of complex-valued effective heights in the phi polarization
        as a function of frequency along axis 0, zenith along axis 1, and
        azimuth along axis 2.
    frequencies : ndarray
        Frequencies (Hz) corresponding to axis 0 of `theta_response` and
        `phi_response`.
    thetas : ndarray
        Zenith angles (degrees) corresponding to axis 1 of `theta_response` and
        `phi_response`.
    phis : ndarray
        Azimuth angles (degrees) corresponding to axis 2 of `theta_response`
        and `phi_response`.

    Raises
    ------
    ValueError
        If the frequency values of the ``.ra1`` and ``.ad1`` files don't match.

    """
    # Quick fix for filenames with one of the approved extensions already
    # (just strip it)
    if filename.endswith(".ad1") or filename.endswith(".ra1"):
        filename = filename[:-4]

    # If the .ad1 or .ra1 files don't exist, check for an equivalent tar file
    # and unzip the files from the tar file
    if not(os.path.exists(filename+".ad1") and
           os.path.exists(filename+".ra1")):
        if os.path.exists(filename+".tar.gz"):
            with tarfile.open(filename+".tar.gz", "r:gz") as tar:
                tar.extract(os.path.basename(filename)+".ad1",
                            os.path.dirname(filename))
                tar.extract(os.path.basename(filename)+".ra1",
                            os.path.dirname(filename))

    # Get s-parameter data from .ad1 file
    freqs = set()
    s_params = {}
    with open(filename+".ad1") as f:
        for line in f:
            words = line.split()
            if words[0].startswith(">"):
                if words[1]=="Hz":
                    freq_scale = 1
                elif words[1]=="kHz":
                    freq_scale = 1e3
                elif words[1]=="MHz":
                    freq_scale = 1e6
                elif words[1]=="GHz":
                    freq_scale = 1e9
                else:
                    raise ValueError("Cannot parse line: '"+line+"'")
            elif len(words)==9:
                freq = float(words[0]) * freq_scale
                freqs.add(freq)
                # z = float(words[5]) + float(words[6])*1j
                s11 = float(words[7]) + float(words[8])*1j
                s_params[freq] = s11

    # Get directional/polarization gain from .ra1 file
    data = {}
    freqs_check = set()
    thetas = set()
    phis = set()
    freq = 0
    with open(filename+".ra1") as f:
        for line in f:
            words = line.split()
            if words[0].startswith(">"):
                freq = 1
                if words[5]=="Hz":
                    pass
                elif words[5]=="kHz":
                    freq *= 1e3
                elif words[5]=="MHz":
                    freq *= 1e6
                elif words[5]=="GHz":
                    freq *= 1e9
                else:
                    raise ValueError("Cannot parse line: '"+line+"'")
                freq *= float(words[4])
                freqs_check.add(freq)
            elif len(words)==8:
                phi = float(words[0])
                phis.add(phi)
                theta = float(words[1])
                thetas.add(theta)
                E_phi = float(words[2]) + float(words[3])*1j
                E_theta = float(words[4]) + float(words[5])*1j
                # gain = float(words[6])
                # db_gain = float(words[7])
                data[(freq, theta, phi)] = (E_theta, E_phi)

    if freqs!=freqs_check:
        raise ValueError("Frequency values of input files do not match")

    # Convert data dictionary into a 3-D arrays of responses, including
    # conversion of electric fields into effective heights
    theta_response = np.empty((len(freqs), len(thetas), len(phis)),
                              dtype=np.complex_)
    phi_response = np.empty((len(freqs), len(thetas), len(phis)),
                            dtype=np.complex_)
    for i, freq in enumerate(sorted(freqs)):
        for j, theta in enumerate(sorted(thetas)):
            for k, phi in enumerate(sorted(phis)):
                # gain, phase = data[(freq, theta, phi)]
                # response[i, j, k] = gain * np.exp(1j*phase)
                heff_factor = (scipy.constants.c/freq * (1+s_params[freq])
                               * 50/377j)
                E_theta, E_phi = data[(freq, theta, phi)]
                theta_response[i, j, k] = E_theta * heff_factor
                phi_response[i, j, k] = E_phi * heff_factor

    # WIPLD file defines thetas from -90 to 90 rather than 0 to 180, so add 90
    return (theta_response, phi_response, np.array(sorted(freqs)),
            np.array(sorted(thetas))+90, np.array(sorted(phis)))


def _read_response_pickle(filename):
    """
    Gather antenna effective height data from a pickled data file.

    The data file should be a pickled file containing the effective height
    data and axes as returned by the `_read_response_data` function.

    Parameters
    ----------
    filename : str
        Name of the data file without the ``.pkl`` extension.

    Returns
    -------
    theta_response : ndarray
        3-D array of complex-valued effective heights in the theta polarization
        as a function of frequency along axis 0, zenith along axis 1, and
        azimuth along axis 2.
    phi_response : ndarray
        3-D array of complex-valued effective heights in the phi polarization
        as a function of frequency along axis 0, zenith along axis 1, and
        azimuth along axis 2.
    frequencies : ndarray
        Frequencies (Hz) corresponding to axis 0 of `theta_response` and
        `phi_response`.
    thetas : ndarray
        Zenith angles (degrees) corresponding to axis 1 of `theta_response` and
        `phi_response`.
    phis : ndarray
        Azimuth angles (degrees) corresponding to axis 2 of `theta_response`
        and `phi_response`.

    See Also
    --------
    _read_response_data : Gather antenna effective height data from a set of
                          WIPLD data files.

    """
    # Quick fix for filenames with one of the approved extensions already
    # (just strip it)
    if filename.endswith(".pkl"):
        filename = filename[:-4]

    # If there is no pickle file, read the response data using the
    # _read_response_data function, and then make a pickle file
    if not os.path.isfile(filename+".pkl"):
        logger.warning("Antenna model file %s.pkl not found. "+
                       "Generating a new file now", filename)
        heff_data = _read_response_data(filename)
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(heff_data, f)
        return heff_data

    # Otherwise, read from the pickle file
    else:
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)


def _read_amplifier_data(gain_filename, phase_filename, gain_offset=0):
    """
    Gather frequency-dependent amplifier data from data files.

    Each data file should have columns for frequency, gain or phase data, and a
    third empty column. The gain should be in dB and the phase should be in
    degrees.

    Parameters
    ----------
    gain_filename : str
        Name of the data file containing gains (in dB).
    phase_filename : str
        Name of the data file containing phases (in degrees).
    gain_offset : float, optional
        Offset to apply to the gain values (in dB).

    Returns
    -------
    gains : ndarray
        Complex-valued voltage gains as a function of frequency.
    frequencies : ndarray
        Frequencies (Hz) corresponding to the values of `gains`.

    """
    gains = []
    freqs = []
    with open(gain_filename) as f:
        for line in f:
            words = line.split()
            if line.startswith('"'):
                continue
            elif len(words)==3 and words[0]!="Frequency":
                f, g, _ = line.split(",")
                freq = float(f)
                gain = 10**((float(g)+gain_offset)/20)
                freqs.append(freq)
                gains.append(gain)

    phases = []
    freqs2 = []
    with open(phase_filename) as f:
        for line in f:
            words = line.split()
            if line.startswith('"'):
                continue
            elif len(words)==3 and words[0]!="Frequency":
                f, p, _ = line.split(",")
                freq = float(f)
                phase = np.radians(float(p))
                freqs2.append(freq)
                phases.append(phase)

    if not np.array_equal(freqs, freqs2):
        raise ValueError("Frequency values of input files do not match")

    return np.array(gains) * np.exp(1j*np.array(phases)), np.array(freqs)


ARIANNA_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LPDA_DATA_FILE = os.path.join(ARIANNA_DATA_DIR,
                              "createLPDA_100MHz_InfFirn")
AMP_GAIN_FILE = os.path.join(ARIANNA_DATA_DIR,
                             "amp_300_gain.csv")
AMP_PHASE_FILE = os.path.join(ARIANNA_DATA_DIR,
                              "amp_300_phase.csv")
LPDA_RESPONSE_DATA = _read_response_pickle(LPDA_DATA_FILE)
AMPLIFIER_GAIN_DATA = _read_amplifier_data(AMP_GAIN_FILE, AMP_PHASE_FILE,
                                           gain_offset=40)
# Series 100 and 200 amps should have gain_offset=60,
# series 300 should have gain_offset=40



class ARIANNAAntenna(Antenna):
    """
    Antenna class to be used for ARIANNA antennas.

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
    z_axis : array_like, optional
        Vector direction of the z-axis of the antenna.
    x_axis : array_like, optional
        Vector direction of the x-axis of the antenna.
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
        The RMS voltage (v) of the antenna noise. If not ``None``, this value
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
                 temperature, resistance, z_axis=(0,0,1), x_axis=(1,0,0),
                 efficiency=1, noisy=True, unique_noise_waveforms=10):
        # Parse the response data
        self._theta_response = response_data[0]
        self._phi_response = response_data[1]
        self._response_freqs = response_data[2]
        self._response_zens = response_data[3]
        self._response_azis = response_data[4]

        # Get the critical frequencies in Hz
        f_low = center_frequency - bandwidth/2
        f_high = center_frequency + bandwidth/2

        super().__init__(position=position, z_axis=z_axis, x_axis=x_axis,
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
        frequency.

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
            Or if only one of `direction` and `polarization` is specified.

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


class ARIANNAAntennaSystem(AntennaSystem):
    """
    Antenna system extending base ARIANNA antenna with front-end processing.

    Applies as the front end a filter representing the ARIANNA amplifier and
    signal clipping.

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
    threshold : float
        Voltage sigma threshold for the trigger condition.
    trigger_window : float
        Time window (ns) for the trigger condition.
    z_axis : array_like, optional
        Vector direction of the z-axis of the antenna.
    x_axis : array_like, optional
        Vector direction of the x-axis of the antenna.
    amplification : float, optional
        Amplification to be applied to the signal pre-clipping.
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
    threshold : float
        Voltage sigma threshold for the trigger condition.
    trigger_window : float
        Time window (s) for the trigger condition.
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
    ARIANNAAntenna : Antenna class to be used for ARIANNA antennas.

    """
    lead_in_time = 25e-9

    def __init__(self, response_data, name, position, threshold,
                 trigger_window=5e-9, z_axis=(0,0,1), x_axis=(1,0,0),
                 amplification=1, amplifier_clipping=1, noisy=True,
                 unique_noise_waveforms=10, **kwargs):
        super().__init__(ARIANNAAntenna)

        self.name = str(name)
        self.position = position

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.setup_antenna(response_data=response_data,
                           z_axis=z_axis, x_axis=x_axis, noisy=noisy,
                           unique_noise_waveforms=unique_noise_waveforms,
                           **kwargs)

        self.threshold = threshold
        self.trigger_window = trigger_window

        self._noise_mean = None
        self._noise_std = None

        self._filter_response = AMPLIFIER_GAIN_DATA[0]
        self._filter_freqs = AMPLIFIER_GAIN_DATA[1]

    @property
    def _metadata(self):
        """Metadata dictionary for writing `ARIANNAAntennaSystem` information."""
        meta = super()._metadata
        meta.update({
            "name": self.name,
            "lead_in_time": self.lead_in_time,
            "amplification": self.amplification,
            "amplifier_clipping": self.amplifier_clipping,
            "threshold": self.threshold,
            "trigger_window": self.trigger_window
        })
        return meta

    def setup_antenna(self, center_frequency=350e6, bandwidth=600e6,
                      temperature=300, resistance=50, z_axis=(0,0,1),
                      x_axis=(1,0,0), efficiency=1, noisy=True,
                      unique_noise_waveforms=10,
                      response_data=None, response_freqs=None, **kwargs):
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
        z_axis : array_like, optional
            Vector direction of the z-axis of the antenna.
        x_axis : array_like, optional
            Vector direction of the x-axis of the antenna.
        efficiency : float, optional
            Antenna efficiency applied to incoming signal values.
        noisy : boolean, optional
            Whether or not the antenna should add noise to incoming signals.
        unique_noise_waveforms : int, optional
            The number of expected noise waveforms needed for each received
            signal to have its own noise.

        """
        # ARIANNA expects a noise rms of about 11 microvolts (before amps).
        # This is mostly satisfied by using the a noise temperature of 300 K,
        # along with a 50 ohm resistance
        super().setup_antenna(response_data=response_data,
                              position=self.position,
                              center_frequency=center_frequency,
                              bandwidth=bandwidth,
                              temperature=temperature,
                              resistance=resistance,
                              z_axis=z_axis,
                              x_axis=x_axis,
                              efficiency=efficiency,
                              noisy=noisy,
                              unique_noise_waveforms=unique_noise_waveforms,
                              **kwargs)

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
        return complex_interp(
            x=frequencies, xp=self._filter_freqs, fp=self._filter_response,
            method='euler', outer=0
        )

    def front_end(self, signal):
        """
        Apply front-end processes to a signal and return the output.

        The front-end consists of amplification according to data taken from
        NuRadioReco and signal clipping.

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
        base_signal *= self.amplification
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

        Compares the maximum and minimum values to a noise signal. Triggers if
        both the maximum and minimum values exceed the noise mean +/- the noise
        standard deviation times the threshold within the set trigger window.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to test the trigger condition.

        Returns
        -------
        boolean
            Whether or not the antenna triggers on `signal`.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        if self._noise_mean is None or self._noise_std is None:
            # Prepare for antenna trigger by finding mean and standard
            # deviation of the full noise waveform passed through the front-end
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
            processed_noise = self.front_end(long_noise)
            self._noise_mean = np.mean(processed_noise.values)
            self._noise_std = np.std(processed_noise.values)

        low_trigger = (self._noise_mean -
                       self._noise_std*np.abs(self.threshold))
        high_trigger = (self._noise_mean +
                        self._noise_std*np.abs(self.threshold))

        # Check whether low and high triggers occur within 5 ns
        low_bins = np.where(signal.values<low_trigger)[0]
        high_bins = np.where(signal.values>high_trigger)[0]
        # Find minimum bin distance between low and high
        i = 0
        j = 0
        min_diff = np.inf
        while i<len(low_bins) and j<len(high_bins):
            diff = np.abs(low_bins[i] - high_bins[j])
            if diff<min_diff:
                min_diff = diff
            if low_bins[i]<high_bins[j]:
                i += 1
            else:
                j += 1

        return min_diff*signal.dt<=self.trigger_window

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


class LPDA(ARIANNAAntennaSystem):
    """
    ARIANNA LPDA antenna system.

    Applies as the front end a filter representing the ARIANNA amplifier and
    signal clipping.

    Parameters
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    threshold : float
        Voltage sigma threshold for the trigger condition.
    trigger_window : float
        Time window (ns) for the trigger condition.
    z_axis : array_like, optional
        Vector direction of the z-axis of the antenna. The z-axis runs along
        the central "spine" of the antenna, with the positive direction
        pointing towards the longer "tines".
    x_axis : array_like, optional
        Vector direction of the x-axis of the antenna. The x-axis runs parallel
        to the "tines" of the antenna.
    amplification : float, optional
        Amplification to be applied to the signal pre-clipping.
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
    threshold : float
        Voltage sigma threshold for the trigger condition.
    trigger_window : float
        Time window (ns) for the trigger condition.
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
    ARIANNAAntennaSystem : Antenna system extending base ARIANNA antenna with
                           front-end processing.
    ARIANNAAntenna : Antenna class to be used for ARIANNA antennas.

    """
    def __init__(self, name, position, threshold, trigger_window=5e-9,
                 z_axis=(0,0,1), x_axis=(1,0,0), amplification=1,
                 amplifier_clipping=1, noisy=True, unique_noise_waveforms=10):
        super().__init__(response_data=LPDA_RESPONSE_DATA,
                         name=name, position=position, threshold=threshold,
                         trigger_window=trigger_window,
                         z_axis=z_axis, x_axis=x_axis,
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)
