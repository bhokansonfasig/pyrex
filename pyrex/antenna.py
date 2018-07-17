"""
Module containing antenna classes responsible of receiving signals.

These classes are intended to model the properties of antennas including
how signals are received as well as the production of noise. A number of
attributes like directional gain, frequency response, and antenna factor
may be necessary to calculate how signals are manipulated upon reception by
an antenna.

"""

import logging
import numpy as np
import scipy.fftpack
import scipy.signal
from pyrex.internal_functions import normalize
from pyrex.signals import Signal, ThermalNoise, EmptySignal
from pyrex.ice_model import IceModel

logger = logging.getLogger(__name__)


class Antenna:
    """
    Base class for antennas.

    Stores the attributes of an antenna as well as handling receiving,
    processing, and storing signals and adding noise.

    Parameters
    ----------
    position : array_like
        Vector position of the antenna.
    z_axis : array_like, optional
        Vector direction of the z-axis of the antenna.
    x_axis : array_like, optional
        Vector direction of the x-axis of the antenna.
    antenna_factor : float, optional
        Antenna factor used for converting electric field values to voltages.
    efficiency : float, optional
        Antenna efficiency applied to incoming signal values.
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.
    freq_range : array_like, optional
        The frequency band in which the antenna operates (used for noise
        production).
    temperature : float, optional
        The noise temperature (K) of the antenna. Used in combination with
        `resistance` to calculate the RMS voltage of the antenna noise.
    resistance : float, optional
        The noise resistance (ohm) of the antenna. Used in combination with
        `temperature` to calculate the RMS voltage of the antenna noise.
    noise_rms : float, optional
        The RMS voltage (V) of the antenna noise. If specified, this value will
        be used instead of the RMS voltage calculated from the values of
        `temperature` and `resistance`.

    Attributes
    ----------
    position : array_like
        Vector position of the antenna.
    z_axis : ndarray
        Vector direction of the z-axis of the antenna.
    x_axis : ndarray
        Vector direction of the x-axis of the antenna.
    antenna_factor : float
        Antenna factor used for converting electric field values to voltages.
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

    """
    def __init__(self, position, z_axis=(0,0,1), x_axis=(1,0,0),
                 antenna_factor=1, efficiency=1, noisy=True,
                 unique_noise_waveforms=10, freq_range=None,
                 temperature=None, resistance=None, noise_rms=None):
        self.position = position
        self.set_orientation(z_axis=z_axis, x_axis=x_axis)
        self.antenna_factor = antenna_factor
        self.efficiency = efficiency
        self.noisy = noisy
        self.unique_noises = unique_noise_waveforms
        self.freq_range = freq_range
        self.noise_rms = noise_rms
        self.temperature = temperature
        self.resistance = resistance

        self.signals = []
        self._noise_master = None
        self._noises = []
        self._triggers = []

    def __str__(self):
        return self.__class__.__name__+"(position="+repr(self.position)+")"

    def set_orientation(self, z_axis=(0,0,1), x_axis=(1,0,0)):
        """
        Sets the orientation of the antenna.

        Sets up the z-axis and the x-axis of the antenna according to the given
        parameters. Fails if the z-axis and x-axis aren't perpendicular.

        Parameters
        ----------
        z_axis : array_like, optional
            Vector direction of the z-axis of the antenna.
        x_axis : array_like, optional
            Vector direction of the x-axis of the antenna.

        Raises
        ------
        ValueError
            If the z-axis and x-axis aren't perpendicular.

        """
        self.z_axis = normalize(z_axis)
        self.x_axis = normalize(x_axis)
        if np.dot(self.z_axis, self.x_axis)!=0:
            raise ValueError("Antenna's x_axis must be perpendicular to its "
                             +"z_axis")

    @property
    def is_hit(self):
        """Boolean of whether the antenna has been triggered."""
        return len(self.waveforms)>0

    def is_hit_during(self, times):
        """
        Check if the antenna is triggered in a time range.

        Generate the full waveform of the antenna over the given `times` array
        and check whether it triggers the antenna.

        Parameters
        ----------
        times : array_like
            1D array of times during which to check for a trigger.

        Returns
        -------
        boolean
            Whether or not the antenna triggered during the given `times`.

        """
        return self.trigger(self.full_waveform(times))

    def clear(self, reset_noise=False):
        """
        Reset the antenna to an empty state.

        Clears all signals, noises, and triggers from the antenna state. Can
        also optionally recalibrate the noise so that a new signal arriving
        at the same times as a previous signal will not have the same noise.

        Parameters
        ----------
        reset_noise : boolean, optional
            Whether or not to recalibrate the noise.

        """
        self.signals.clear()
        self._noises.clear()
        self._triggers.clear()
        if reset_noise:
            self._noise_master = None

    @property
    def waveforms(self):
        """Signal + noise (if ``noisy``) for each triggered antenna hit."""
        # Process any unprocessed triggers
        all_waves = self.all_waveforms
        while len(self._triggers)<len(all_waves):
            waveform = all_waves[len(self._triggers)]
            self._triggers.append(self.trigger(waveform))

        return [wave for wave, triggered in zip(all_waves, self._triggers)
                if triggered]

    @property
    def all_waveforms(self):
        """Signal + noise (if ``noisy``) for all antenna hits."""
        if not(self.noisy):
            return self.signals

        # Generate noise as necessary
        while len(self._noises)<len(self.signals):
            self._noises.append(
                self.make_noise(self.signals[len(self._noises)].times)
            )

        return [s + n for s, n in zip(self.signals, self._noises)]

    def full_waveform(self, times):
        """
        Signal + noise (if ``noisy``) for the given times.

        Creates the complete waveform of the antenna including noise and all
        received signals for the given `times` array.

        Parameters
        ----------
        times : array_like
            1D array of times during which to produce the full waveform.

        Returns
        -------
        Signal
            Complete waveform with noise and all signals.

        """
        if self.noisy:
            waveform = self.make_noise(times)
        else:
            waveform = EmptySignal(times)

        for signal in self.signals:
            waveform += signal.with_times(times)
        return waveform

    def make_noise(self, times):
        """
        Creates a noise signal over the given times.

        In order to add noise to signal to produce the waveforms of the
        antenna, this function is used to create the noise values at specific
        times. Makes use of the antenna's noise-related attributes.

        Parameters
        ----------
        times : array_like
            1D array of times during which to produce noise values.

        Returns
        -------
        ThermalNoise
            Noise values during the `times` array.

        Raises
        ------
        ValueError
            If not enough noise-related attributes are defined for the antenna.

        """
        if self._noise_master is None:
            if self.freq_range is None:
                raise ValueError("A frequency range is required to generate"
                                 +" antenna noise")
            elif (self.noise_rms is None and
                  (self.temperature is None or self.resistance is None)):
                raise ValueError("A noise rms value (or temperature and"
                                 +" resistance) are required to generate"
                                 +" antenna noise")

            # Calculate recommended number of frequencies for longest
            # signal length stored
            duration = 1e-7 if len(self.signals)==0 else 0
            for signal in self.signals:
                signal_duration = signal.times[-1] - signal.times[0]
                if signal_duration > duration:
                    duration = signal_duration
            n_freqs = (self.freq_range[1] - self.freq_range[0]) * duration

            # Multiply n_freqs by the number of unique noise waveforms needed
            # so that up to about that many signals can be stored without the
            # noise being obviously periodic
            n_freqs *= self.unique_noises

            if self.noise_rms is None:
                self._noise_master = ThermalNoise(times, f_band=self.freq_range,
                                                  temperature=self.temperature,
                                                  resistance=self.resistance,
                                                  n_freqs=n_freqs)
            else:
                self._noise_master = ThermalNoise(times, f_band=self.freq_range,
                                                  rms_voltage=self.noise_rms,
                                                  n_freqs=n_freqs)

        return self._noise_master.with_times(times)


    def trigger(self, signal):
        """
        Check if the antenna triggers on a given signal.

        This function defines the trigger condition for the antenna: Given a
        signal does the antenna trigger? It is expected to be overridden in
        subclasses, as for the base class it simply triggers on any signal.

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
        return True

    def _convert_to_antenna_coordinates(self, point):
        """
        Convert the given point to the coordinate system of the antenna.

        For the cartesian vector `point`, calculate the spherical coordinate
        values relative to the position and orientation of the antenna.

        Parameters
        ----------
        point : array_like
            Vector position of the point to convert.

        Returns
        -------
        r : float
            r-distance to the point from the antenna position.
        theta : float
            Polar angle (radians) of the point relative to the antenna's
            z-axis.
        phi : float
            Azimuthal angle (radians) of the point relative to the antenna's
            x-axis.

        """
        # Get cartesian point relative to antenna position
        rel_point = np.array(point) - np.array(self.position)
        # Matrix multiplication using antenna axes as rows in transformation
        # matrix to transform point into antenna coordinates
        y_axis = np.cross(self.z_axis, self.x_axis)
        transformation = np.array([self.x_axis, y_axis, self.z_axis])
        x, y, z = np.dot(transformation, rel_point)
        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        if r==0:
            return 0, 0, 0
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    def directional_gain(self, theta, phi):
        """
        Calculate the (complex) directional gain of the antenna.

        This function defines the directionality of the antenna: Given `theta`
        and `phi` in the antenna's coordinate system, what is the (complex)
        gain? It is expected to be overridden in subclasses, as for the base
        class it simply returns 1 for any `theta` and `phi`.

        Parameters
        ----------
        theta : float
            Polar angle (radians) from which a signal is arriving.
        phi : float
            Azimuthal angle (radians) from which a signal is arriving.

        Returns
        -------
        complex
            Complex gain in voltage for the given incoming angles.

        """
        logger.debug("Using default directional_gain from "+
                     "pyrex.antenna.Antenna")
        return 1

    def polarization_gain(self, polarization):
        """
        Calculate the (complex) polarization gain of the antenna.

        This function defines the gain of the antenna due to signal
        polarization: Given a vector signal `polarization`, what is the
        (complex) antenna gain? It is expected to be overridden in subclasses,
        as for the base class it simply returns 1 for any `polarization`.

        Parameters
        ----------
        polarization : array_like
            Vector polarization direction of the signal.

        Returns
        -------
        complex
            Complex gain in voltage for the given signal polarization.

        """
        logger.debug("Using default polarization_gain from "+
                     "pyrex.antenna.Antenna")
        return 1

    def response(self, frequencies):
        """
        Calculate the (complex) frequency response of the antenna.

        This function defines the frequency response of the antenna: Given
        some frequencies, what are the corresponding (complex) gains? It is
        expected to be overridden in subclasses, as for the base class it
        simply returns 1 for any frequency.

        Parameters
        ----------
        frequencies : array_like
            1D array of frequencies (Hz) at which to calculate gains.

        Returns
        -------
        array_like
            Complex gains in voltage for the given `frequencies`.

        """
        logger.debug("Using default response from "+
                     "pyrex.antenna.Antenna")
        return np.ones(len(frequencies))

    def receive(self, signal, direction=None, polarization=None,
                force_real=False):
        """
        Process and store an incoming signal.

        Processes the incoming signal according to the frequency response of
        the antenna, the efficiency, and the antenna factor. May also apply the
        directionality and the polarization gain depending on the provided
        parameters. Finally stores the processed signal to the signals list.
        Subclasses may extend this function, but likely should end with
        ``super().receive(signal)`` unless planning to fully reimplement the
        function.

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

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        copy = Signal(signal.times, signal.values,
                      value_type=Signal.Type.voltage)
        copy.filter_frequencies(self.response, force_real=force_real)

        if direction is None:
            d_gain = 1
        else:
            # Calculate theta and phi relative to the antenna's orientation
            origin = self.position - normalize(direction)
            r, theta, phi = self._convert_to_antenna_coordinates(origin)
            d_gain = self.directional_gain(theta=theta, phi=phi)

        if polarization is None:
            p_gain = 1
        else:
            p_gain = self.polarization_gain(normalize(polarization))

        signal_factor = d_gain * p_gain * self.efficiency

        if signal.value_type==Signal.Type.voltage:
            pass
        elif signal.value_type==Signal.Type.field:
            signal_factor /= self.antenna_factor
        else:
            raise ValueError("Signal's value type must be either "
                             +"voltage or field. Given "+str(signal.value_type))

        copy.values *= signal_factor
        self.signals.append(copy)



class DipoleAntenna(Antenna):
    """
    Class for half-wave dipole antennas.

    Stores the attributes of an antenna as well as handling receiving,
    processing, and storing signals and adding noise. Uses a first-order
    butterworth filter for the frequency response. Includes a simple threshold
    trigger.

    Parameters
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    center_frequency : float
        Tuned frequency (Hz) of the dipole.
    bandwidth : float
        Bandwidth (Hz) of the antenna.
    resistance : float
        The noise resistance (ohm) of the antenna. Used to calculate the RMS
        voltage of the antenna noise.
    orientation : array_like, optional
        Vector direction of the z-axis of the antenna.
    trigger_threshold : float, optional
        Voltage threshold (V) above which signals will trigger.
    effective_height : float, optional
        Effective length (m) of the antenna. By default calculated by the tuned
        `center_frequency` of the dipole.
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

    Attributes
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    z_axis : ndarray
        Vector direction of the z-axis of the antenna.
    x_axis : ndarray
        Vector direction of the x-axis of the antenna.
    antenna_factor : float
        Antenna factor used for converting electric field values to voltages.
    efficiency : float
        Antenna efficiency applied to incoming signal values.
    threshold : float, optional
        Voltage threshold (V) above which signals will trigger.
    effective_height : float, optional
        Effective length of the antenna. By default calculated by the tuned
        `center_frequency` of the dipole.
    filter_coeffs : tuple of ndarray
        Coefficients of the transfer function of the butterworth bandpass
        filter to be used for frequency response.
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
    waveforms
    all_waveforms

    See Also
    --------
    Antenna : Base class for antennas.

    """
    def __init__(self, name, position, center_frequency, bandwidth, resistance,
                 orientation=(0,0,1), trigger_threshold=0,
                 effective_height=None, noisy=True,
                 unique_noise_waveforms=10):
        self.name = name
        self.threshold = trigger_threshold
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
        # Note: ortho is not normalized, but will be normalized by Antenna init

        super().__init__(position=position, z_axis=orientation, x_axis=ortho,
                         antenna_factor=1/self.effective_height,
                         temperature=IceModel.temperature(position[2]),
                         freq_range=(f_low, f_high), resistance=resistance,
                         unique_noise_waveforms=unique_noise_waveforms,
                         noisy=noisy)

        # Build scipy butterworth filter to speed up response function
        b, a  = scipy.signal.butter(1, 2*np.pi*np.array(self.freq_range),
                                    btype='bandpass', analog=True)
        self.filter_coeffs = (b, a)


    def trigger(self, signal):
        """
        Check if the antenna triggers on a given signal.

        Antenna triggers if any single time bin has a voltage above the trigger
        threshold.

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
        return max(np.abs(signal.values)) > self.threshold

    def response(self, frequencies):
        """
        Calculate the (complex) frequency response of the antenna.

        Dipole antenna frequency response is a first order butterworth bandpass
        filter in the antenna's frequency range.

        Parameters
        ----------
        frequencies : array_like
            1D array of frequencies (Hz) at which to calculate gains.

        Returns
        -------
        array_like
            Complex gains in voltage for the given `frequencies`.

        """
        angular_freqs = np.array(frequencies) * 2*np.pi
        w, h = scipy.signal.freqs(self.filter_coeffs[0], self.filter_coeffs[1],
                                  angular_freqs)
        return h

    def directional_gain(self, theta, phi):
        """
        Calculate the (complex) directional gain of the antenna.

        Power gain of dipole antenna goes as sin(theta)^2, so electric field
        gain goes as sin(theta).

        Parameters
        ----------
        theta : float
            Polar angle (radians) from which a signal is arriving.
        phi : float
            Azimuthal angle (radians) from which a signal is arriving.

        Returns
        -------
        complex
            Complex gain in voltage for the given incoming angles.

        """
        return np.sin(theta)

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
