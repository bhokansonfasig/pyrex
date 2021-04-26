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
import scipy.constants
import scipy.signal
from pyrex.internal_functions import normalize
from pyrex.signals import Signal, ThermalNoise, EmptySignal

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
    is_hit_mc_truth
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
        self._all_waves = []
        self._triggers = []

    def __str__(self):
        return self.__class__.__name__+"(position="+repr(self.position)+")"

    @property
    def _metadata(self):
        """Metadata dictionary for writing `Antenna` information."""
        return {
            "class": str(type(self)),
            "position_x": self.position[0],
            "position_y": self.position[1],
            "position_z": self.position[2],
            "z_axis_x": self.z_axis[0],
            "z_axis_y": self.z_axis[1],
            "z_axis_z": self.z_axis[2],
            "x_axis_x": self.x_axis[0],
            "x_axis_y": self.x_axis[1],
            "x_axis_z": self.x_axis[2],
            "antenna_factor": self.antenna_factor,
            "efficiency": self.efficiency,
            "noisy": int(self.noisy),
            "unique_noises": self.unique_noises,
            "freq_range_min": (np.nan if self.freq_range is None
                               else self.freq_range[0]),
            "freq_range_max": (np.nan if self.freq_range is None
                               else self.freq_range[1]),
            "noise_rms": (np.nan if self.noise_rms is None
                          else self.noise_rms),
            "temperature": (np.nan if self.temperature is None
                            else self.temperature),
            "resistance": (np.nan if self.resistance is None
                           else self.resistance)
        }

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
        if not np.isclose(np.dot(self.z_axis, self.x_axis), 0, rtol=0):
            raise ValueError("Antenna's x_axis must be perpendicular to its "
                             +"z_axis")

    @property
    def is_hit(self):
        """Boolean of whether the antenna has been triggered."""
        return len(self.waveforms)>0

    @property
    def is_hit_mc_truth(self):
        """
        Boolean of whether the antenna has been triggered by signal.

        The decision is based on the Monte Carlo truth of whether noise would
        have triggered without the signal. If a signal triggered, but the noise
        alone in the same timeframe would have triggered as well, the trigger
        is not counted.
        """
        if not self.noisy:
            return self.is_hit
        for wave in self.waveforms:
            if not self.trigger(self.make_noise(wave.times)):
                return True
        return False

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
        self._all_waves.clear()
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
        # Process any unprocessed signals
        while len(self._all_waves)<len(self.signals):
            self._all_waves.append(
                self.full_waveform(self.signals[len(self._all_waves)].times)
            )
        return self._all_waves

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
        # Only include signals reasonably close to the times array
        # (i.e. within one extra signal length forwards and backwards)
        dt = times[1] - times[0]
        if len(self.signals)>0:
            signal_length = max(signal.times[-1] - signal.times[0]
                                for signal in self.signals)
        else:
            signal_length = 0
        n_pts = int(signal_length/dt)
        if signal_length%dt:
            n_pts += 1
        long_times = np.concatenate((
            times[0]+np.linspace(-n_pts*dt, 0, n_pts, endpoint=False),
            times,
            times[-1]+np.linspace(0, n_pts*dt, n_pts+1)[1:]
        ))

        if self.noisy:
            waveform = self.make_noise(long_times)
        else:
            waveform = EmptySignal(long_times)

        for signal in self.signals:
            if (signal.times[-1]<long_times[0]
                    or signal.times[0]>long_times[-1]):
                continue
            waveform += signal.with_times(long_times)

        return waveform.with_times(times)

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

            if self.noise_rms is None:
                self._noise_master = ThermalNoise(
                    times, f_band=self.freq_range,
                    temperature=self.temperature, resistance=self.resistance,
                    uniqueness_factor=self.unique_noises
                )
            else:
                self._noise_master = ThermalNoise(
                    times, f_band=self.freq_range,
                    rms_voltage=self.noise_rms,
                    uniqueness_factor=self.unique_noises
                )

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
        phi = np.arctan2(y, x) % (2*np.pi)
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

    def frequency_response(self, frequencies):
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

    def apply_response(self, signal, direction=None, polarization=None,
                       force_real=False):
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
        new_signal.filter_frequencies(self.frequency_response,
                                      force_real=force_real)

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

        new_signal *= signal_factor

        return new_signal

    def receive(self, signal, direction=None, polarization=None,
                force_real=False):
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
        if hasattr(signal, '__len__'):
            if (not hasattr(polarization, '__len__') or
                    len(signal)!=len(polarization)):
                raise ValueError("Must provide the same number of "+
                                 "polarizations as the number of signals")
        else:
            signal = [signal]
            polarization = [polarization]

        total_signal = sum([
            self.apply_response(signal=sig, direction=direction,
                                polarization=pol, force_real=force_real)
            for sig, pol in zip(signal, polarization)
        ])

        self.signals.append(total_signal)



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
    temperature : float
        The noise temperature (K) of the antenna. Used in combination with
        `resistance` to calculate the RMS voltage of the antenna noise.
    resistance : float
        The noise resistance (ohm) of the antenna. Used in combination with
        `temperature` to calculate the RMS voltage of the antenna noise.
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
    is_hit_mc_truth
    waveforms
    all_waveforms

    See Also
    --------
    Antenna : Base class for antennas.

    """
    def __init__(self, name, position, center_frequency, bandwidth,
                 temperature, resistance, orientation=(0,0,1),
                 trigger_threshold=0, effective_height=None, noisy=True,
                 unique_noise_waveforms=10):
        self.name = name
        self.threshold = trigger_threshold
        if effective_height is None:
            # Calculate length of half-wave dipole
            self.effective_height = scipy.constants.c / center_frequency / 2
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
                         freq_range=(f_low, f_high), temperature=temperature,
                         resistance=resistance, noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)

        # Build scipy butterworth filter to speed up response function
        b, a  = scipy.signal.butter(1, 2*np.pi*np.array(self.freq_range),
                                    btype='bandpass', analog=True)
        self.filter_coeffs = (b, a)

    @property
    def _metadata(self):
        """Metadata dictionary for writing `DipoleAntenna` information."""
        meta = super()._metadata
        meta.update({
            "name": self.name,
            "threshold": self.threshold,
            "effective_height": self.effective_height
        })
        return meta


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

    def frequency_response(self, frequencies):
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
