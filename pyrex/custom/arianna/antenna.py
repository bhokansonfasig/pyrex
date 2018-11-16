"""
Module containing customized antenna classes for ARIANNA.

Based primarily on the LPDA implementation (and data) in NuRadioReco.

"""

import os.path
import pickle
import tarfile
import numpy as np
import scipy.interpolate
from pyrex.internal_functions import normalize
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.detector import AntennaSystem
from pyrex.ice_model import IceModel



def _read_response_data(filename):
    """
    Gather antenna directionality/polarization data from a WIPLD data file.

    Data files should exist with names `filename`.ra1 and `filename`.ad1.
    The ``.ad1`` file should contain frequencies in the first column, real and
    imaginary parts of the impedance in the sixth and seventh columns, and
    S-parameter data in the eigth and ninth columns.
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
    dict
        Dictionary containing the data with keys (freq, theta, phi) and values
        (theta E-field, phi E-field).
    set
        Set of unique frequencies appearing in the data keys.
    ndarray
        Array of S-parameter values corresponding to the frequencies.

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
    s_params = []
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
                s_params.append(s11)

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

    return data, freqs, np.array(s_params)


def _read_response_pickle(filename):
    """
    Gather antenna effective height data from a pickled data file.

    The data file should be a pickled file containing the effective height
    data dictionary and frequency set, calculated based on the data dictionary,
    frequency set, and impedance array typically returned by the
    `_read_response_data` function.

    Parameters
    ----------
    filename : str
        Name of the data file without the ``.pkl`` extension.

    Returns
    -------
    dict
        Dictionary containing the effective height data with keys
        (freq, theta, phi) and values (theta gain, phi gain).
    set
        Set of unique frequencies appearing in the data keys.

    See Also
    --------
    _read_response_data : Gather antenna directionality/polarization data from
                          a WIPLD data file.

    """
    # Quick fix for filenames with one of the approved extensions already
    # (just strip it)
    if filename.endswith(".pkl"):
        filename = filename[:-4]

    # If there is no pickle file, read the response data using the
    # _read_response_data function, and then make a pickle file
    if not os.path.isfile(filename+".pkl"):
        data, freqs, s_params = _read_response_data(filename)

        # Calculate effective height from the data
        heff_data = {}
        s11s = {f: s for f, s in zip(sorted(freqs), s_params)}
        for key, e_fields in data.items():
            freq = key[0]
            heff_factor = 3e8/freq * (1+s11s[freq]) * 50/377j
            heff_data[key] = (e_fields[0]*heff_factor, e_fields[1]*heff_factor)

        with open(filename+".pkl", 'wb') as f:
            pickle.dump((heff_data, freqs), f)
        return heff_data, freqs

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
    gain_offset : float
        Offset to apply to the gain values (in dB).

    Returns
    -------
    dict
        Dictionary containing the data with keys (freq) and values
        (gain, phase).

    """
    data = {}
    with open(gain_filename) as f:
        for line in f:
            words = line.split()
            if line.startswith('"'):
                continue
            elif len(words)==3 and words[0]!="Frequency":
                f, g, _ = line.split(",")
                freq = float(f)
                gain = 10**((float(g)+gain_offset)/20)
                data[freq] = (gain, 0)

    phase_offset = 0
    prev_phase = 0
    with open(phase_filename) as f:
        for line in f:
            words = line.split()
            if line.startswith('"'):
                continue
            elif len(words)==3 and words[0]!="Frequency":
                f, p, _ = line.split(",")
                freq = float(f)
                phase = np.radians(float(p))
                # In order to smoothly interpolate phases, don't allow the phase
                # to wrap from -pi to +pi, but instead apply an offset
                if phase-prev_phase>np.pi:
                    phase_offset -= 2*np.pi
                elif prev_phase-phase>np.pi:
                    phase_offset += 2*np.pi
                prev_phase = phase
                if freq not in data:
                    raise ValueError("Frequency values must match between "
                                     +"gain and phase files")
                gain = data[freq][0]
                data[freq] = (gain, phase+phase_offset)

    return data


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


ARIANNA_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LPDA_DATA_FILE = os.path.join(ARIANNA_DATA_DIR,
                              "createLPDA_100MHz_InfFirn")
AMP_GAIN_FILE = os.path.join(ARIANNA_DATA_DIR,
                             "amp_300_gain.csv")
AMP_PHASE_FILE = os.path.join(ARIANNA_DATA_DIR,
                              "amp_300_phase.csv")
ARA_FILT_DATA_FILE = os.path.join(ARIANNA_DATA_DIR,
                                  "ARA_Electronics_TotalGain_TwoFilters.txt")
LPDA_DIRECTIONALITY, LPDA_FREQS = _read_response_pickle(LPDA_DATA_FILE)
AMPLIFIER_GAIN = _read_amplifier_data(AMP_GAIN_FILE, AMP_PHASE_FILE,
                                      gain_offset=40)
# Series 100 and 200 amps should have gain_offset=60,
# series 300 should have gain_offset=40
ARA_FILTERS = _read_filter_data(ARA_FILT_DATA_FILE)



class ARIANNAAntenna(Antenna):
    """
    Antenna class to be used for ARIANNA antennas.

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
    response_data : None or dict, optional
        Dictionary containing data on the response of the antenna. If ``None``,
        behavior is undefined.
    response_freqs : None or set, optional
        Set of frequencies in the response data ``dict`` keys. If ``None``,
        calculated automatically from `response_data`.

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
                 z_axis=(0,0,1), x_axis=(1,0,0), efficiency=1, noisy=True,
                 unique_noise_waveforms=10,
                 response_data=None, response_freqs=None):
        # Get the critical frequencies in Hz
        f_low = center_frequency - bandwidth/2
        f_high = center_frequency + bandwidth/2

        super().__init__(position=position, z_axis=z_axis, x_axis=x_axis,
                         efficiency=efficiency, freq_range=(f_low, f_high),
                         temperature=IceModel.temperature(position[2]),
                         resistance=resistance, noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)

        self._resp_data = response_data
        self._resp_freqs = response_freqs
        # Just in case the frequencies don't get set, set them now
        if self._resp_freqs is None and self._resp_data is not None:
            self._resp_freqs = set()
            for key in self._resp_data:
                self._resp_freqs.add(key[0])

    def generate_response_gains(self, theta, phi):
        """
        Generate the (complex) frequency-dependent response gains.

        For given angles, calculate arrays of frequencies and their
        corresponding theta and phi (complex) gains, based on the response data
        of the antenna. These values are not technically gain in a traditional
        sense, but for lack of a better word we'll stick with gain for now.

        Parameters
        ----------
        theta : float
            Polar angle (radians) from which a signal is arriving.
        phi : float
            Azimuthal angle (radians) from which a signal is arriving.

        Returns
        -------
        freqs : array_like
            Frequencies over which the response was defined.
        theta_gains : array_like
            Complex theta gains at the corrseponding frequencies.
        phi_gains : array_like
            Complex phi gains at the corresponding frequencies.

        """
        if self._resp_data is None:
            return np.array([1]), np.array([1]), np.array([0])

        theta = np.degrees(theta)
        phi = np.degrees(phi)

        # Special case: if given exactly theta=180, don't take the modulus
        if theta!=180:
            theta %= 180
        phi %= 360
        theta_under = 2*int(theta/2)
        theta_over = 2*(int(theta/2)+1)
        phi_under = 5*int(phi/5)
        phi_over = 5*(int(phi/5)+1)
        t = (theta - theta_under) / (theta_over - theta_under)
        u = (phi - phi_under) / (phi_over - phi_under)

        theta_over = min(theta_over, 180)
        phi_over %= 360

        # WIPLD file defines thetas from -90 to 90 rather than 0 to 180
        theta_under -= 90
        theta_over -= 90

        nfreqs = len(self._resp_freqs)
        theta_gain_ij = np.zeros(nfreqs, dtype=np.complex_)
        phi_gain_ij = np.zeros(nfreqs, dtype=np.complex_)
        theta_gain_i1j = np.zeros(nfreqs, dtype=np.complex_)
        phi_gain_i1j = np.zeros(nfreqs, dtype=np.complex_)
        theta_gain_ij1 = np.zeros(nfreqs, dtype=np.complex_)
        phi_gain_ij1 = np.zeros(nfreqs, dtype=np.complex_)
        theta_gain_i1j1 = np.zeros(nfreqs, dtype=np.complex_)
        phi_gain_i1j1 = np.zeros(nfreqs, dtype=np.complex_)
        for f, freq in enumerate(sorted(self._resp_freqs)):
            theta_gain_ij[f] = self._resp_data[(freq, theta_under, phi_under)][0]
            phi_gain_ij[f] = self._resp_data[(freq, theta_under, phi_under)][1]
            theta_gain_i1j[f] = self._resp_data[(freq, theta_over, phi_under)][0]
            phi_gain_i1j[f] = self._resp_data[(freq, theta_over, phi_under)][1]
            theta_gain_ij1[f] = self._resp_data[(freq, theta_under, phi_over)][0]
            phi_gain_ij1[f] = self._resp_data[(freq, theta_under, phi_over)][1]
            theta_gain_i1j1[f] = self._resp_data[(freq, theta_over, phi_over)][0]
            phi_gain_i1j1[f] = self._resp_data[(freq, theta_over, phi_over)][1]

        freqs = np.array(sorted(self._resp_freqs))
        theta_gains = ((1-t)*(1-u)*theta_gain_ij + t*(1-u)*theta_gain_i1j +
                       (1-t)*u*theta_gain_ij1 + t*u*theta_gain_i1j1)
        phi_gains = ((1-t)*(1-u)*phi_gain_ij + t*(1-u)*phi_gain_i1j +
                     (1-t)*u*phi_gain_ij1 + t*u*phi_gain_i1j1)

        return freqs, theta_gains, phi_gains


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
            the antenna. If ``None`` no directional response will be applied,
            and `polarization` must be ``None`` as well.
        polarization : array_like, optional
            Vector denoting the signal's polarization direction. If ``None``
            no polarization gain will be applied, and `direction` must be
            ``None`` as well.
        force_real : boolean, optional
            Whether or not the frequency response should be redefined in the
            negative-frequency domain to keep the values of the filtered signal
            real.

        Raises
        ------
        ValueError
            If the given `signal` does not have a ``value_type`` of ``voltage``
            or ``field``.
            Or if only one of `direction` and `polarization` is specified.

        """
        copy = Signal(signal.times, signal.values, value_type=Signal.Type.voltage)
        copy.filter_frequencies(self.response, force_real=force_real)

        if direction is not None and polarization is not None:
            # Normalize the polarization
            polarization = normalize(polarization)
            # Calculate theta and phi relative to the orientation
            origin = self.position - normalize(direction)
            r, theta, phi = self._convert_to_antenna_coordinates(origin)
            # Calculate the e_theta and e_phi direction vectors in the antenna
            # coordinate system
            e_theta = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),
                       -np.sin(theta)]
            e_phi = [-np.sin(phi), np.cos(phi), 0]
            # Convert e_theta and e_phi back to the global coordinate system:
            # Matrix with antenna axes as rows in transformation provides the
            # forward transform into the antenna coordinate system, so invert
            # it to move back to global coordinates
            y_axis = np.cross(self.z_axis, self.x_axis)
            transformation = np.array([self.x_axis, y_axis, self.z_axis])
            inverse = np.linalg.inv(transformation)
            e_theta = np.dot(inverse, e_theta)
            e_phi = np.dot(inverse, e_phi)
            # The theta and phi gains should then be multiplied by the amount
            # of polarization in those respective directions
            theta_factor = np.dot(polarization, e_theta)
            phi_factor = np.dot(polarization, e_phi)
            freq_data, theta_gain_data, phi_gain_data = \
                self.generate_response_gains(theta, phi)
            def interpolate_response(frequencies):
                """
                Generate interpolated response for given frequencies.

                Parameters
                ----------
                frequencies : array_like
                    1D array of frequencies (Hz) at which to calculate gains.

                Returns
                -------
                array_like
                    Complex directional gain and polarization gain in voltage
                    for the `frequencies`.

                """
                interp_theta_gains = np.interp(frequencies, freq_data,
                                               theta_gain_data,
                                               left=0, right=0)
                interp_phi_gains = np.interp(frequencies, freq_data,
                                             phi_gain_data,
                                             left=0, right=0)
                return (interp_theta_gains * theta_factor +
                        interp_phi_gains * phi_factor)
            copy.filter_frequencies(interpolate_response,
                                    force_real=force_real)

        elif (direction is not None and polarization is None
              or direction is None and polarization is not None):
            raise ValueError("Direction and polarization must be specified together")

        signal_factor = self.efficiency

        if signal.value_type==Signal.Type.voltage:
            pass
        elif signal.value_type==Signal.Type.field:
            signal_factor /= self.antenna_factor
        else:
            raise ValueError("Signal's value type must be either "
                             +"voltage or field. Given "+str(signal.value_type))

        copy.values *= signal_factor
        self.signals.append(copy)


class ARIANNAAntennaSystem(AntennaSystem):
    """
    Antenna system extending base ARIANNA antenna with front-end processing.

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
    response_data : None or dict, optional
        Dictionary containing data on the response of the antenna. If ``None``,
        behavior is undefined.
    response_freqs : None or set, optional
        Set of frequencies in the response data ``dict`` keys. If ``None``,
        calculated automatically from `response_data`.

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

    def __init__(self, name, position, threshold, trigger_window=5e-9,
                 z_axis=(0,0,1), x_axis=(1,0,0), amplification=1,
                 amplifier_clipping=1, noisy=True, unique_noise_waveforms=10,
                 response_data=None, response_freqs=None, **kwargs):
        super().__init__(ARIANNAAntenna)

        self.name = str(name)
        self.position = position

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.setup_antenna(z_axis=z_axis, x_axis=x_axis, noisy=noisy,
                           unique_noise_waveforms=unique_noise_waveforms,
                           response_data=response_data,
                           response_freqs=response_freqs, **kwargs)

        self.threshold = threshold
        self.trigger_window = trigger_window

        self._noise_mean = None
        self._noise_std = None

        self._filter_data = AMPLIFIER_GAIN

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
                      resistance=16.5, z_axis=(0,0,1), x_axis=(1,0,0),
                      efficiency=1, noisy=True, unique_noise_waveforms=10,
                      response_data=None, response_freqs=None, **kwargs):
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
        response_data : None or dict, optional
            Dictionary containing data on the response of the antenna.
        response_freqs : None or set, optional
            Set of frequencies in the response data ``dict`` keys.

        """
        # ARIANNA expects a noise rms of about 11 microvolts (before amps).
        # This is satisfied for most ice temperatures by using an effective
        # resistance of ~16.5 Ohm
        # Additionally, the bandwidth of the antenna is set slightly larger
        # than the nominal bandwidth of the true ARA antenna system (700 MHz),
        # but the extra frequencies should be killed by the front-end filter
        super().setup_antenna(position=self.position,
                              center_frequency=center_frequency,
                              bandwidth=bandwidth,
                              resistance=resistance,
                              z_axis=z_axis,
                              x_axis=x_axis,
                              efficiency=efficiency,
                              noisy=noisy,
                              unique_noise_waveforms=unique_noise_waveforms,
                              response_data=response_data,
                              response_freqs=response_freqs,
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
        freqs = sorted(self._filter_data.keys())
        gains = [self._filter_data[f][0] for f in freqs]
        phases = [self._filter_data[f][1] for f in freqs]
        interp_gains = np.interp(frequencies, freqs, gains, left=0, right=0)
        interp_phases = np.interp(frequencies, freqs, phases, left=0, right=0)
        return interp_gains * np.exp(1j * interp_phases)

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
        copy = Signal(signal.times, signal.values)
        copy.filter_frequencies(self.interpolate_filter,
                                force_real=True)
        clipped_values = np.clip(copy.values * self.amplification,
                                 a_min=-self.amplifier_clipping,
                                 a_max=self.amplifier_clipping)
        return Signal(signal.times, clipped_values,
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
            # deviation of a long noise waveform (1 microsecond) passed through
            # the front-end
            long_noise = self.antenna.make_noise(np.linspace(0, 1e-6, 10001))
            processed_noise = self.front_end(long_noise)
            self._noise_mean = np.mean(processed_noise.values)
            self._noise_std = np.sqrt(np.mean((processed_noise.values
                                               -self._noise_mean)**2))

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
            the antenna. If ``None`` no directional response will be applied,
            and `polarization` must be ``None`` as well.
        polarization : array_like, optional
            Vector denoting the signal's polarization direction. If ``None``
            no polarization gain will be applied, and `direction` must be
            ``None`` as well.
        force_real : boolean, optional
            Whether or not the frequency response should be redefined in the
            negative-frequency domain to keep the values of the filtered signal
            real.

        Raises
        ------
        ValueError
            If the given `signal` does not have a ``value_type`` of ``voltage``
            or ``field``.
            Or if only one of `direction` and `polarization` is specified.

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
        Time window (ns) for the trigger condition.
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
    ARIANNAAntennaSystem : Antenna system extending base ARIANNA antenna with
                           front-end processing.
    ARIANNAAntenna : Antenna class to be used for ARIANNA antennas.

    """
    def __init__(self, name, position, threshold, trigger_window=5e-9,
                 z_axis=(0,0,1), x_axis=(1,0,0), amplification=1,
                 amplifier_clipping=1, noisy=True, unique_noise_waveforms=10):
        super().__init__(name=name, position=position, threshold=threshold,
                         trigger_window=trigger_window,
                         z_axis=z_axis, x_axis=x_axis,
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms,
                         response_data=LPDA_DIRECTIONALITY,
                         response_freqs=LPDA_FREQS)
