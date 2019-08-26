"""
Module containing classes for digital signal processing.

All classes in this module hold time-domain information about some signals,
and have methods for manipulating this data as it relates to digital signal
processing and general physics.

"""

from enum import Enum
import logging
import numpy as np
import scipy.signal
import scipy.fftpack
from pyrex.internal_functions import get_from_enum

logger = logging.getLogger(__name__)


class Signal:
    """
    Base class for time-domain signals.

    Stores the time-domain information for signal values. Supports adding
    between signals with the same times array and value type.

    Parameters
    ----------
    times : array_like
        1D array of times (s) for which the signal is defined.
    values : array_like
        1D array of values of the signal corresponding to the given `times`.
        Will be resized to the size of `times` by zero-padding or truncating
        as necessary.
    value_type
        Type of signal, representing the units of the values. Values should be
        from the ``Signal.Type`` enum, but integer or string values may
        work if carefully chosen. ``Signal.Type.undefined`` by default.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times (s) and corresponding values which define the signal.
    value_type : Signal.Type
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    dt
    frequencies
    spectrum
    envelope

    """
    class Type(Enum):
        """
        Enum containing possible types (units) for signal values.

        Attributes
        ----------
        voltage
        field
        power
        unknown, undefined

        """
        undefined = 0
        unknown = 0
        voltage = 1
        field = 2
        power = 3

    def __init__(self, times, values, value_type=None):
        self.times = np.array(times)
        len_diff = len(times)-len(values)
        if len_diff>0:
            self.values = np.concatenate((values, np.zeros(len_diff)))
        else:
            self.values = np.array(values[:len(times)])
        self.value_type = value_type

    def __add__(self, other):
        """
        Adds two signals by adding their values at each time.

        Adding ``Signal`` objects is only allowed when they have identical
        ``times`` arrays, and their ``value_type``s are compatible. This means
        that the ``value_type``s must be the same, or one must be ``undefined``
        which will be coerced to the other ``value_type``.

        Raises
        ------
        ValueError
            If the other ``Signal`` has different ``times`` or ``value_type``.

        """
        if not isinstance(other, Signal):
            return NotImplemented
        if not np.array_equal(self.times, other.times):
            raise ValueError("Can't add signals with different times")
        if (self.value_type!=self.Type.undefined and
                other.value_type!=self.Type.undefined and
                self.value_type!=other.value_type):
            raise ValueError("Can't add signals with different value types")

        if self.value_type==self.Type.undefined:
            value_type = other.value_type
        else:
            value_type = self.value_type

        return Signal(self.times, self.values+other.values,
                      value_type=value_type)

    def __radd__(self, other):
        """
        Allows for adding Signal object to 0.

        Since the python ``sum`` function starts by adding the first element
        to 0, to use ``sum`` with ``Signal`` objects we need to be able to add
        a ``Signal`` object to 0. If adding to anything else, raise the usual
        error.

        """
        if other==0:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiply signal values at all times by some value."""
        try:
            return Signal(self.times, self.values * other,
                          value_type=self.value_type)
        except TypeError:
            return NotImplemented

    def __rmul__(self, other):
        """Multiply signal values at all times by some value."""
        try:
            return Signal(self.times, other * self.values,
                          value_type=self.value_type)
        except TypeError:
            return NotImplemented

    def __imul__(self, other):
        """Multiply signal values at all times by some value in-place."""
        try:
            self.values *= other
        except TypeError:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """Divide signal values at all times by some value."""
        try:
            return Signal(self.times, self.values / other,
                          value_type=self.value_type)
        except TypeError:
            return NotImplemented

    def __itruediv__(self, other):
        """Divide signal values at all times by some value in-place."""
        try:
            self.values /= other
        except TypeError:
            return NotImplemented
        return self

    @property
    def value_type(self):
        """
        Type of signal, representing the units of the values.

        Should always be a value from the ``Signal.Type`` enum. Setting with
        integer or string values may work if carefully chosen.

        """
        return self._value_type

    @value_type.setter
    def value_type(self, val_type):
        if val_type is None:
            self._value_type = self.Type.undefined
        else:
            self._value_type = get_from_enum(val_type, self.Type)

    @property
    def dt(self):
        """The time spacing of the `times` array, or ``None`` if invalid."""
        try:
            return self.times[1]-self.times[0]
        except IndexError:
            return None

    @property
    def envelope(self):
        """The envelope of the signal by Hilbert transform."""
        analytic_signal = scipy.signal.hilbert(self.values)
        return np.abs(analytic_signal)

    def resample(self, n):
        """
        Resamples the signal into n points in the same time range, in-place.

        Parameters
        ----------
        n : int
            The number of points into which the signal should be resampled.

        """
        if n==len(self.times):
            return

        self.times = np.linspace(self.times[0], self.times[-1], n)
        self.values = scipy.signal.resample(self.values, n)

    def with_times(self, new_times):
        """
        Returns a representation of this signal over a different times array.

        Parameters
        ----------
        new_times : array_like
            1D array of times (s) for which to define the new signal.

        Returns
        -------
        Signal
            A representation of the original signal over the `new_times` array.

        Notes
        -----
        Interpolates the values of the ``Signal`` object across `new_times`,
        extrapolating with zero values on the left and right.

        """
        new_values = np.interp(new_times, self.times, self.values,
                               left=0, right=0)
        return Signal(new_times, new_values, value_type=self.value_type)


    @property
    def spectrum(self):
        """The FFT complex spectrum values of the signal."""
        return scipy.fftpack.fft(self.values)

    @property
    def frequencies(self):
        """The FFT frequencies of the signal."""
        return scipy.fftpack.fftfreq(n=len(self.values), d=self.dt)

    def filter_frequencies(self, freq_response, force_real=False):
        """
        Apply the given frequency response function to the signal, in-place.

        For the given response function, multiplies the response into the
        frequency domain of the signal. If the filtered signal is forced to be
        real, the positive-frequency response is mirrored into the negative
        frequencies by complex conjugation.

        Parameters
        ----------
        freq_response : function
            Response function taking a frequency (or array of frequencies) and
            returning the corresponding complex gain(s).
        force_real : boolean
            If ``True``, complex conjugation is used on the positive-frequency
            response to force the filtered signal to be real-valued. Otherwise
            the frequency response is left alone and any imaginary parts of the
            filtered signal are thrown out.

        Warns
        -----
        Raises a warning if the maximum value of the imaginary part of the
        filtered signal was greater than 1e-5 times the maximum value of the
        real part, indicating that there was significant signal lost when
        discarding the imaginary part.

        """
        # Zero-pad the signal so the filter doesn't cause the resulting
        # signal to wrap around the end of the time array
        vals = np.concatenate((self.values, np.zeros(len(self.values))))
        spectrum = scipy.fftpack.fft(vals)
        freqs = scipy.fftpack.fftfreq(n=2*len(self.values), d=self.dt)
        if force_real:
            true_freqs = np.array(freqs)
            freqs = np.abs(freqs)

        # Attempt to evaluate all responses in one function call
        try:
            responses = np.array(freq_response(freqs), dtype=np.complex_)
        # Otherwise evaluate responses one at a time
        except (TypeError, ValueError):
            logger.debug("Frequency response function %r could not be "+
                         "evaluated for multiple frequencies at once",
                         freq_response)
            responses = np.zeros(len(spectrum), dtype=np.complex_)
            for i, f in enumerate(freqs):
                responses[i] = freq_response(f)

        # To make the filtered signal real, mirror the positive frequency
        # response into the negative frequencies, making the real part even
        # (done above) and the imaginary part odd (below)
        if force_real:
            responses.imag[true_freqs<0] *= -1

        filtered_vals = scipy.fftpack.ifft(responses*spectrum)
        self.values = np.real(filtered_vals[:len(self.times)])

        # Issue a warning if there was significant signal in the (discarded)
        # imaginary part of the filtered values
        if np.any(np.imag(filtered_vals[:len(self.times)]) >
                  np.max(self.values) * 1e-5):
            msg = ("Significant signal amplitude was lost when forcing the "+
                   "signal values to be real after applying the frequency "+
                   "filter '%s'. This may be avoided by making sure the "+
                   "filter being used is properly defined for negative "+
                   "frequencies, or by passing force_real=True to the "+
                   "Signal.filter_frequencies function.")
            logger.warning(msg, freq_response.__name__)



class EmptySignal(Signal):
    """
    Class for signal with zero amplitude (all values = 0).

    Parameters
    ----------
    times : array_like
        1D array of times (s) for which the signal is defined.
    value_type
        Type of signal, representing the units of the values. Must be from the
        ``Signal.Type`` Enum.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times (s) and corresponding values which define the signal.
    value_type : Signal.Type
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    dt
    frequencies
    spectrum
    envelope

    See Also
    --------
    Signal : Base class for time-domain signals.

    """
    def __init__(self, times, value_type=None):
        super().__init__(times, np.zeros(len(times)), value_type=value_type)

    def with_times(self, new_times):
        """
        Returns a representation of this signal over a different times array.

        Parameters
        ----------
        new_times : array_like
            1D array of times (s) for which to define the new signal.

        Returns
        -------
        EmptySignal
            A representation of the original signal over the `new_times` array.

        Notes
        -----
        Since the ``EmptySignal`` always has zero values, the returned signal
        will also have all zero values.

        """
        return EmptySignal(new_times, value_type=self.value_type)


class FunctionSignal(Signal):
    """
    Class for signals generated by a function.

    Parameters
    ----------
    times : array_like
        1D array of times (s) for which the signal is defined.
    function : function
        Function which evaluates the corresponding value(s) for a given time or
        array of times.
    value_type
        Type of signal, representing the units of the values. Must be from the
        ``Signal.Type`` Enum.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times (s) and corresponding values which define the signal.
    value_type : Signal.Type
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    function : function
        Function to evaluate the signal values at given time(s).
    dt
    frequencies
    spectrum
    envelope

    See Also
    --------
    Signal : Base class for time-domain signals.
    EmptySignal : Class for signal with zero amplitude.

    """
    def __init__(self, times, function, value_type=None):
        self.times = np.array(times)
        self.function = function
        # Attempt to evaluate all values in one function call
        try:
            values = self.function(self.times)
        # Otherwise evaluate values one at a time
        except (ValueError, TypeError):
            values = []
            for t in self.times:
                values.append(self.function(t))

        super().__init__(times, values, value_type=value_type)

    def with_times(self, new_times):
        """
        Returns a representation of this signal over a different times array.

        Parameters
        ----------
        new_times : array_like
            1D array of times (s) for which to define the new signal.

        Returns
        -------
        FunctionSignal
            A representation of the original signal over the `new_times` array.

        Notes
        -----
        Leverages knowledge of the function that creates the signal to properly
        recalculate exact (not interpolated) values for the new times.

        """
        return FunctionSignal(new_times, self.function,
                              value_type=self.value_type)



class GaussianNoise(Signal):
    """
    Class for gaussian noise signals with standard deviation sigma.

    Calculates each time value independently from a normal distribution.

    Parameters
    ----------
    times : array_like
        1D array of times (s) for which the signal is defined.
    values : array_like
        1D array of values of the signal corresponding to the given `times`.
        Will be resized to the size of `times` by zero-padding or truncating.
    value_type
        Type of signal, representing the units of the values. Must be from the
        ``Signal.Type`` Enum.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times (s) and corresponding values which define the signal.
    value_type : Signal.Type.voltage
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    dt
    frequencies
    spectrum
    envelope

    See Also
    --------
    Signal : Base class for time-domain signals.

    """
    def __init__(self, times, sigma):
        self.sigma = sigma
        values = np.random.normal(0, self.sigma, size=len(times))
        super().__init__(times, values, value_type=self.Type.voltage)


class ThermalNoise(FunctionSignal):
    """
    Class for thermal Rayleigh noise signals.

    The Rayleigh thermal noise is calculated in a given frequency band with
    flat or otherwise specified amplitude and random phase at some number of
    frequencies. Values are scaled to a provided or calculated RMS voltage.

    Parameters
    ----------
    times : array_like
        1D array of times (s) for which the signal is defined.
    f_band : array_like
        Array of two elements denoting the frequency band (Hz) of the noise.
        The first element should be smaller than the second.
    f_amplitude : float or function, optional
        The frequency-domain amplitude of the noise. If ``float``, then all
        frequencies will have the same amplitude. If ``function``, then the
        function is evaluated at each frequency to determine its amplitude.
        By default, uses Rayleigh-distributed amplitudes.
    rms_voltage : float, optional
        The RMS voltage (V) of the noise. If specified, this value will be used
        instead of the RMS voltage calculated from the values of `temperature`
        and `resistance`.
    temperature : float, optional
        The thermal noise temperature (K). Used in combination with the value
        of `resistance` to calculate the RMS voltage of the noise.
    resistance : float, optional
        The resistance (ohm) for the noise. Used in combination with the value
        of `temperature` to calculate the RMS voltage of the noise.
    n_freqs : int, optional
        The number of frequencies within the frequency band to use to calculate
        the noise signal. By default determines the number of frequencies based
        on the FFT bin size of `times`.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times (s) and corresponding values which define the signal.
    value_type : Signal.Type.voltage
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    function : function
        Function to evaluate the signal values at given time(s).
    f_min : float
        Minimum frequency of the noise frequency band.
    f_max : float
        Maximum frequency of the noise frequency band.
    freqs, amps, phases : ndarray
        The frequencies used to define the noise signal and their corresponding
        amplitudes and phases.
    rms : float
        The RMS value of the noise signal.
    dt
    frequencies
    spectrum
    envelope

    Warnings
    --------
    Since this class inherits from ``FunctionSignal``, its ``with_times``
    method will properly extrapolate noise outside of the provided times. Be
    warned however that outside of the original signal times the noise signal
    will be highly periodic. Since the default number of frequencies used is
    based on the FFT bin size of `times`, the period of the noise signal is
    actually the length of `times`. As a result if you are planning on
    extrapolating the noise signal, increasing the number of frequencies used
    is strongly recommended.

    Raises
    ------
    ValueError
        If the RMS voltage cannot be calculated (i.e. `rms_voltage` or both
        `temperature` and `resistance` are ``None``).

    See Also
    --------
    FunctionSignal : Class for signals generated by a function.

    Notes
    -----
    Calculation of the noise signal is based on the Rayleigh noise model used
    by ANITA [1]_. Modifications have been made to the default to make the
    frequency-domain amplitudes Rayleigh-distributed, under the suggestion that
    this makes for more realistic noise traces.

    References
    ----------
    .. [1] A. Connolly et al, ANITA Note #76, "Thermal Noise Studies: Toward A
        Time-Domain Model of the ANITA Trigger."
        https://www.phys.hawaii.edu/elog/anita_notes/060228_110754/noise_simulation.ps

    """
    def __init__(self, times, f_band, f_amplitude=None, rms_voltage=None,
                 temperature=None, resistance=None, n_freqs=0):
        # Calculation based on Rician (Rayleigh) noise model for ANITA:
        # https://www.phys.hawaii.edu/elog/anita_notes/060228_110754/noise_simulation.ps

        self.f_min, self.f_max = f_band
        if self.f_min>=self.f_max:
            raise ValueError("Frequency band must have smaller frequency as "+
                             "first value and larger frequency as second value")
        # If number of frequencies is unspecified (or invalid),
        # determine based on the FFT bin size of the times array
        if n_freqs<1:
            n_freqs = (self.f_max - self.f_min) * (times[-1] - times[0])
            # Broken out into steps to ease understanding:
            #   duration = times[-1] - times[0]
            #   f_bin_size = 1 / duration
            #   n_freqs = (self.f_max - self.f_min) / f_bin_size

        # If number of frequencies is still zero (e.g. len(times)==1),
        # force it to 1
        if n_freqs<1:
            n_freqs = 1

        self.freqs = np.linspace(self.f_min, self.f_max, int(n_freqs),
                                 endpoint=False)

        if f_amplitude is None:
            f_amplitude = lambda f: np.random.rayleigh(1/np.sqrt(2),
                                                       size=f.shape)

        # Allow f_amplitude to be either a function or a single value
        if callable(f_amplitude):
            # Attempt to evaluate all amplitudes in one function call
            try:
                self.amps = np.array(f_amplitude(self.freqs))
                if len(self.amps)!=len(self.freqs):
                    raise ValueError("Amplitude calculation failed")
            # Otherwise evaluate responses one at a time
            except (TypeError, ValueError):
                logger.debug("Amplitude function %r could not be evaluated "+
                             "for multiple frequencies at once", f_amplitude)
                self.amps = np.array([f_amplitude(f) for f in self.freqs])
        else:
            self.amps = np.full(len(self.freqs), f_amplitude, dtype="float64")
            # If the frequency range extends to zero, force the zero-frequency
            # (DC) amplitude to zero
            if 0 in self.freqs:
                self.amps[np.where(self.freqs==0)[0]] = 0

        self.phases = np.random.rand(len(self.freqs)) * 2*np.pi

        if rms_voltage is not None:
            self.rms = rms_voltage
        elif temperature is not None and resistance is not None:
            # RMS voltage = sqrt(4 * kB * T * R * bandwidth)
            self.rms = np.sqrt(4 * 1.38e-23 * temperature * resistance
                               * (self.f_max - self.f_min))
        else:
            raise ValueError("Either RMS voltage or temperature and resistance"+
                             " must be provided to calculate noise amplitude")

        def f(ts):
            """Set the time-domain signal by adding sinusoidal signals of each
            frequency with the corresponding phase."""
            # This method is nicer than an inverse fourier transform because
            # results are consistant for differing ranges of ts around the same
            # time. The price payed is that the fft would be an order of
            # magnitude faster, but not reproducible for a slightly different
            # time array.
            values = sum(amp * np.cos(2*np.pi*freq * ts + phase)
                         for freq, amp, phase
                         in zip(self.freqs, self.amps, self.phases))

            # Normalization calculated by guess-and-check,
            # but seems to work fine
            # normalization = np.sqrt(2/len(self.freqs))
            values *= np.sqrt(2/len(self.freqs))

            # So far, the units of the values are V/V_rms, so multiply by the
            # rms voltage:
            values *= self.rms

            return values

        super().__init__(times, function=f, value_type=self.Type.voltage)
