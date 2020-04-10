"""
Module containing classes for digital signal processing.

All classes in this module hold time-domain information about some signals,
and have methods for manipulating this data as it relates to digital signal
processing and general physics.

"""

import copy
from enum import Enum
import logging
import numpy as np
import scipy.constants
import scipy.fft
import scipy.signal
from pyrex.internal_functions import (LazyMutableClass, lazy_property,
                                      get_from_enum)

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
    value_type : optional
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

    def copy(self):
        """
        Get a copy of the ``Signal`` object.

        Returns
        -------
        Signal
            A (deep) copy of the existing ``Signal`` object with identical
            ``times``, ``values``, and ``value_type``.

        """
        return Signal(self.times, self.values, self.value_type)

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

    def shift(self, dt):
        """
        Shifts the signal values in time by `dt`.

        Parameters
        ----------
        dt : float
            Time shift (s) to be applied to the signal.

        """
        self.times += dt

    @property
    def spectrum(self):
        """The FFT complex spectrum values of the signal."""
        return scipy.fft.fft(self.values)

    @property
    def frequencies(self):
        """The FFT frequencies of the signal."""
        return scipy.fft.fftfreq(n=len(self.values), d=self.dt)

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
        force_real : boolean, optional
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
        spectrum = scipy.fft.fft(vals)
        freqs = scipy.fft.fftfreq(n=2*len(self.values), d=self.dt)

        responses = self._get_filter_response(freqs, freq_response, force_real)
        filtered_vals = scipy.fft.ifft(responses*spectrum)
        self.values = np.real(filtered_vals[:len(self.times)])

        # Issue a warning if there was significant signal in the (discarded)
        # imaginary part of the filtered values
        if np.any(np.abs(np.imag(filtered_vals[:len(self.times)])) >
                  np.max(np.abs(self.values)) * 1e-5):
            msg = ("Significant signal amplitude was lost when forcing the "+
                   "signal values to be real after applying the frequency "+
                   "filter '%s'. This may be avoided by making sure the "+
                   "filter being used is properly defined for negative "+
                   "frequencies")
            if not force_real:
                msg += (", or by passing force_real=True to the "+
                        "Signal.filter_frequencies function")
            msg += "."
            logger.warning(msg, freq_response.__name__)

    @staticmethod
    def _get_filter_response(freqs, function, force_real=False):
        """
        Get the frequency response of a filter function.

        Parameters
        ----------
        freqs : ndarray
            Array of frequencies [Hz] over which to calculate the filter
            response.
        function : function
            Response function taking a frequency (or array of frequencies) and
            returning the corresponding complex gain(s).
        force_real : boolean, optional
            If ``True``, complex conjugation is used on the positive-frequency
            response to force the filtered signal to be real-valued.

        Returns
        -------
        response : ndarray
            Complex response of the filter at the given frequencies.

        """
        if force_real:
            true_freqs = np.array(freqs)
            freqs = np.abs(freqs)

        # Attempt to evaluate all responses in one function call
        try:
            responses = np.array(function(freqs), dtype=np.complex_)
        # Otherwise evaluate responses one at a time
        except (TypeError, ValueError):
            logger.debug("Frequency response function %r could not be "+
                         "evaluated for multiple frequencies at once",
                         function)
            responses = np.zeros(len(freqs), dtype=np.complex_)
            for i, f in enumerate(freqs):
                responses[i] = function(f)

        # To make the filtered signal real, mirror the positive frequency
        # response into the negative frequencies, making the real part even
        # (done above) and the imaginary part odd (below)
        if force_real:
            responses.imag[true_freqs<0] *= -1

        return responses



class EmptySignal(Signal):
    """
    Class for signal with zero amplitude (all values = 0).

    Parameters
    ----------
    times : array_like
        1D array of times (s) for which the signal is defined.
    value_type : optional
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

        # Adding an EmptySignal is essentially transparent (returns a copy
        # of the other Signal), except for the value_type coercion
        new_signal = other.copy()
        new_signal.value_type = value_type
        return new_signal

    def copy(self):
        """
        Get a copy of the ``EmptySignal`` object.

        Returns
        -------
        Signal
            A (deep) copy of the existing ``EmptySignal`` object with identical
            ``times`` and ``value_type``.

        """
        return EmptySignal(self.times, self.value_type)

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

    def filter_frequencies(self, freq_response, force_real=False):
        """
        Apply the given frequency response function to the signal, in-place.

        For the given response function, multiplies the response into the
        frequency domain of the signal. If the filtered signal is forced to be
        real, the positive-frequency response is mirrored into the negative
        frequencies by complex conjugation. For EmptySignal objects, all
        calculation is skipped and the EmptySignal is preserved.

        Parameters
        ----------
        freq_response : function
            Response function taking a frequency (or array of frequencies) and
            returning the corresponding complex gain(s).
        force_real : boolean, optional
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
        # All values of the signal are zero anyway, so filters will have no
        # effect. We can just skip all the calculation then.
        pass


class FunctionSignal(LazyMutableClass, Signal):
    """
    Class for signals generated by a function.

    Parameters
    ----------
    times : array_like
        1D array of times (s) for which the signal is defined.
    function : function
        Function which evaluates the corresponding value(s) for a given time or
        array of times.
    value_type : optional
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
    pyrex.internal_functions.LazyMutableClass : Class with lazy properties
                                                which may depend on other class
                                                attributes.

    """
    def __init__(self, times, function, value_type=None):
        self.times = np.array(times)
        self._functions = [function]
        self._t0s = [0]
        self._buffers = [[0, 0]]
        self._factors = [1]
        self._filters = [[]]
        self.value_type = value_type
        super().__init__(static_attributes=['times', '_functions', '_t0s',
                                            '_buffers', '_factors', '_filters'])

    def _full_times(self, index):
        """
        1D array of times including buffer time.

        Parameters
        ----------
        index : int
            Index of the function and buffer to calculate the times array for.

        Returns
        -------
        ndarray
            1D array of times for the function, including the buffer time.

        """
        # Number of points in the buffer arrays
        n_before = int(self._buffers[index][0]/self.dt)
        if self._buffers[index][0]%self.dt:
            n_before += 1
        n_after = int(self._buffers[index][1]/self.dt)
        if self._buffers[index][1]%self.dt:
            n_after += 1
        # Proper starting points of buffer arrays to preserve dt
        t_min = self.times[0] - n_before*self.dt
        t_max = self.times[-1] + n_after*self.dt
        return np.concatenate((
            np.linspace(t_min, self.times[0], n_before, endpoint=False),
            self.times,
            np.linspace(self.times[-1], t_max, n_after+1)[1:]
        ))

    def _value_window(self, index):
        """Window of `_full_times` values array corresponding to `times`."""
        # Number of points in the buffer arrays
        n_before = int(self._buffers[index][0]/self.dt)
        if self._buffers[index][0]%self.dt:
            n_before += 1
        # n_after = int(self._buffers[index][1]/self.dt)
        # if self._buffers[index][1]%self.dt:
        #     n_after += 1
        return slice(n_before, n_before+len(self.times))

    @lazy_property
    def values(self):
        """1D array of values which define the signal."""
        values = np.zeros(len(self.times))
        for i, function in enumerate(self._functions):
            # Attempt to evaluate all values in one function call
            try:
                func_vals = function(self._full_times(i) - self._t0s[i])
            # Otherwise evaluate values one at a time
            except (ValueError, TypeError):
                func_vals = [function(t) for t in
                             self._full_times(i)-self._t0s[i]]

            func_vals = np.asarray(func_vals) * self._factors[i]

            if len(self._filters[i])!=0:
                full_vals = self._apply_filters(func_vals, self._filters[i])
            else:
                full_vals = func_vals

            values += full_vals[self._value_window(i)]

        return values


    def __add__(self, other):
        """
        Adds two signals by adding their values at each time.

        Adding ``Signal`` objects is only allowed when they have identical
        ``times`` arrays, and their ``value_type``s are compatible. This means
        that the ``value_type``s must be the same, or one must be ``undefined``
        which will be coerced to the other ``value_type``. If two
        ``FunctionSignal`` objects are added, the result is another
        ``FunctionSignal``. If a ``FunctionSignal`` object is added to a
        ``Signal`` object, the result is a ``Signal`` object where the
        ``FunctionSignal`` has been evaluated over the ``Signal`` object's
        ``times``.

        Raises
        ------
        ValueError
            If the other ``Signal`` has a different ``value_type``.

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

        if isinstance(other, FunctionSignal):
            new_signal = self.copy()
            new_signal._functions += copy.deepcopy(other._functions)
            new_signal._t0s += copy.deepcopy(other._t0s)
            new_signal._buffers += copy.deepcopy(other._buffers)
            new_signal._factors += copy.deepcopy(other._factors)
            new_signal._filters += copy.deepcopy(other._filters)
            new_signal.value_type = value_type
            return new_signal
        elif isinstance(other, EmptySignal):
            # Adding an EmptySignal is essentially transparent (returns a copy
            # of the FunctionSignal), except for the value_type coercion
            new_signal = self.copy()
            new_signal.value_type = value_type
            return new_signal
        else:
            return Signal(self.times, self.values+other.values,
                          value_type=value_type)

    def __mul__(self, other):
        """Multiply signal values at all times by some value."""
        try:
            factors = [f * other for f in self._factors]
        except TypeError:
            return NotImplemented
        new_signal = self.copy()
        new_signal._factors = factors
        return new_signal

    def __rmul__(self, other):
        """Multiply signal values at all times by some value."""
        try:
            factors = [other * f for f in self._factors]
        except TypeError:
            return NotImplemented
        new_signal = self.copy()
        new_signal._factors = factors
        return new_signal

    def __imul__(self, other):
        """Multiply signal values at all times by some value in-place."""
        try:
            self._factors = [f * other for f in self._factors]
        except TypeError:
            return NotImplemented
        return self

    def __truediv__(self, other):
        """Divide signal values at all times by some value."""
        try:
            factors = [f / other for f in self._factors]
        except TypeError:
            return NotImplemented
        new_signal = self.copy()
        new_signal._factors = factors
        return new_signal

    def __itruediv__(self, other):
        """Divide signal values at all times by some value in-place."""
        try:
            self._factors = [f / other for f in self._factors]
        except TypeError:
            return NotImplemented
        return self

    def copy(self):
        """
        Get a copy of the ``FunctionSignal`` object.

        Returns
        -------
        Signal
            A (deep) copy of the existing ``FunctionSignal`` object with
            identical ``times``, ``value_type``, and internal function
            parameters.

        """
        new_signal = FunctionSignal(self.times, None, self.value_type)
        new_signal._functions = copy.deepcopy(self._functions)
        new_signal._t0s = copy.deepcopy(self._t0s)
        new_signal._buffers = copy.deepcopy(self._buffers)
        new_signal._factors = copy.deepcopy(self._factors)
        new_signal._filters = copy.deepcopy(self._filters)
        return new_signal

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

        Tries to interpret cases where `with_times` was used to incorporate
        effects of a leading (or trailing) signal outside of the `times` array
        by setting leading and trailing buffer values when `new_times` is fully
        contained by the previous `times` array.

        """
        new_signal = self.copy()
        new_signal.times = new_times
        # Check whether `new_times` is a subset of the previous `times`, and
        # set buffers accordingly
        if new_times[0]>=self.times[0] and new_times[-1]<=self.times[-1]:
            logger.debug("New times array is contained by previous times. "+
                         "Setting buffers to incorporate previous times.")
            new_signal.set_buffers(leading=new_times[0]-self.times[0],
                                   trailing=self.times[-1]-new_times[-1])
        return new_signal

    def shift(self, dt):
        """
        Shifts the signal values in time by `dt`.

        Parameters
        ----------
        dt : float
            Time shift (s) to be applied to the signal.

        """
        self.times += dt
        self._t0s = [t+dt for t in self._t0s]

    def set_buffers(self, leading=None, trailing=None, force=False):
        """
        Set leading and trailing buffers used in calculation of signal values.

        Parameters
        ----------
        leading : float or None
            Leading buffer time (s).
        trailing : float or None
            Trailing buffer time (s).
        force : boolean
            Whether the buffer times should be forced to the given values. If
            `False`, each buffer time is set to the maximum of the current and
            given buffer time. If `True`, each buffer time is set to the given
            buffer time regardless of the current buffer time (unless the given
            value is `None`).

        Raises
        ------
        ValueError
            If either buffer time is less than zero.

        """
        if leading is not None:
            if leading<0:
                raise ValueError("Buffer time cannot be less than zero")
            if force:
                for i, current in enumerate(self._buffers):
                    self._buffers[i][0] = leading
            else:
                for i, current in enumerate(self._buffers):
                    self._buffers[i][0] = max(leading, current[0])
        if trailing is not None:
            if trailing<0:
                raise ValueError("Buffer time cannot be less than zero")
            if force:
                for i, current in enumerate(self._buffers):
                    self._buffers[i][1] = trailing
            else:
                for i, current in enumerate(self._buffers):
                    self._buffers[i][1] = max(trailing, current[1])


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
        force_real : boolean, optional
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
        # Since we're using append instead of setting self._filters, need to
        # manually enforce the cache clearing
        self._clear_cache()
        for group in self._filters:
            group.append((freq_response, force_real))


    def _apply_filters(self, input_vals, filters):
        """
        Apply the given frequency response function to the signal, in-place.

        For each filter function, multiplies the response into the frequency
        domain of the signal. If a filtered signal is forced to be real, the
        positive-frequency response is mirrored into the negative frequencies
        by complex conjugation.

        Parameters
        ----------
        input_vals : array_like
            1D array of values for the unfiltered signal function.
        filters : list of tuple
            List of response functions and ``force_real`` parameters of filters
            to be applied to the unfiltered function values.

        Warns
        -----
        Raises a warning if the maximum value of the imaginary part of the
        filtered signal was greater than 1e-5 times the maximum value of the
        real part, indicating that there was significant signal lost when
        discarding the imaginary part.

        """
        freqs = scipy.fft.fftfreq(n=2*len(input_vals), d=self.dt)
        all_filters = np.ones(len(freqs), dtype=np.complex_)

        for freq_response, force_real in filters:
            all_filters *= self._get_filter_response(freqs, freq_response,
                                                     force_real)

        # Zero-pad the signal so the filter doesn't cause the resulting
        # signal to wrap around the end of the time array
        vals = np.concatenate((input_vals, np.zeros(len(input_vals))))
        spectrum = scipy.fft.fft(vals)

        filtered_vals = scipy.fft.ifft(all_filters*spectrum)
        output_vals = np.real(filtered_vals[:len(input_vals)])

        # Issue a warning if there was significant signal in the (discarded)
        # imaginary part of the filtered values
        if np.any(np.abs(np.imag(filtered_vals[:len(input_vals)])) >
                  np.max(np.abs(output_vals)) * 1e-5):
            msg = ("Significant signal amplitude was lost when forcing the "+
                   "signal values to be real after applying the frequency "+
                   "filters '%s'. This may be avoided by making sure the "+
                   "filters being used are properly defined for negative "+
                   "frequencies")
            if not np.all([force_real for _, force_real in filters]):
                msg += (", or by passing force_real=True to the "+
                        "Signal.filter_frequencies function")
            msg += "."
            logger.warning(msg, [name for name, _ in filters])

        return output_vals



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


class FullThermalNoise(FunctionSignal):
    """
    Class for thermal Rayleigh noise signals using exact functions.

    The Rayleigh thermal noise is calculated in a given frequency band with
    rayleigh-distributed or otherwise specified amplitudes and random phase.
    Values are calculated using a sum of cosine functions and then scaled to a
    provided or calculated RMS voltage.

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
    uniqueness_factor : int, optional
        The number of unique waveform traces that can be expected from this
        noise signal. This factor multiplies the length of the total trace to
        be calculated.

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
    will be periodic. Since the default number of frequencies used is based on
    the FFT bin size of `times`, the default period of the noise signal is
    actually the length of `times`. As a result if you are planning on
    extrapolating the noise signal, increasing the `uniqueness_factor` in order
    to increase the number of frequencies used is strongly recommended.

    Raises
    ------
    ValueError
        If the RMS voltage cannot be calculated (i.e. `rms_voltage` or both
        `temperature` and `resistance` are ``None``).

    See Also
    --------
    FunctionSignal : Class for signals generated by a function.
    FFTThermalNoise : Class for thermal Rayleigh noise signals using the FFT.

    Notes
    -----
    Calculation of the noise signal is based on the Rayleigh noise model used
    by ANITA [1]_. Modifications have been made to the default to make the
    frequency-domain amplitudes Rayleigh-distributed, under the suggestion that
    this makes for more realistic noise traces.

    The calculation of signal values is done using a sum of cosine functions.
    This method has the advantage that interpolated values (at times not given
    in the initial time trace) can be calculated exactly rather than linearly
    interpolated. The disadvantage is that this method is then slower than
    using an FFT-based strategy.

    References
    ----------
    .. [1] A. Connolly et al, ANITA Note #76, "Thermal Noise Studies: Toward A
        Time-Domain Model of the ANITA Trigger."
        https://www.phys.hawaii.edu/elog/anita_notes/060228_110754/noise_simulation.ps

    """
    def __init__(self, times, f_band, f_amplitude=None, rms_voltage=None,
                 temperature=None, resistance=None, uniqueness_factor=1):
        # Calculation based on Rician (Rayleigh) noise model for ANITA:
        # https://www.phys.hawaii.edu/elog/anita_notes/060228_110754/noise_simulation.ps

        self.f_min, self.f_max = f_band
        if self.f_min>=self.f_max:
            raise ValueError("Frequency band must have smaller frequency as "+
                             "first value and larger frequency as second value")
        # Determine the number of frequencies needed based on the FFT bin size
        # of the times array
        n_freqs = (self.f_max - self.f_min) * (times[-1] - times[0])
        # Broken out into steps to ease understanding:
        #   duration = times[-1] - times[0]
        #   f_bin_size = 1 / duration
        #   n_freqs = (self.f_max - self.f_min) / f_bin_size

        # If number of frequencies is zero (e.g. len(times)==1), force it to 1
        if n_freqs<1:
            n_freqs = 1

        # Force uniqueness_factor to at least 1
        if uniqueness_factor<1:
            uniqueness_factor = 1

        # Multiply number of frequencies by the uniqueness factor
        self._n_freqs = int(n_freqs * uniqueness_factor)

        self.freqs = np.linspace(self.f_min, self.f_max, self._n_freqs,
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
            self.amps = np.full(len(self.freqs), f_amplitude, dtype=np.float_)

        # If the frequency range includes zero, force the zero-frequency (DC)
        # amplitude to zero
        if 0 in self.freqs:
            self.amps[np.where(self.freqs==0)[0]] = 0

        self.phases = np.random.rand(len(self.freqs)) * 2*np.pi

        if rms_voltage is not None:
            self.rms = rms_voltage
        elif temperature is not None and resistance is not None:
            # RMS voltage = sqrt(kB * T * R * bandwidth)
            # Not using sqrt(4 * kB * T * R * bandwidth) because in the antenna
            # system only half the voltage is seen and the other half goes to
            # "ground" (changed under advisement by Cosmin Deaconu)
            self.rms = np.sqrt(scipy.constants.k * temperature * resistance
                               * (self.f_max - self.f_min))
        else:
            raise ValueError("Either RMS voltage or temperature and resistance"+
                             " must be provided to calculate noise amplitude")

        def f(ts):
            """Set the time-domain signal by adding sinusoidal signals of each
            frequency with the corresponding phase."""
            values = sum(amp * np.cos(2*np.pi*freq * ts + phase)
                         for freq, amp, phase
                         in zip(self.freqs, self.amps, self.phases))

            # Normalization calculated by guess-and-check; seems to work fine
            # normalization = np.sqrt(2/len(freqs))
            values *= np.sqrt(2/len(self.freqs))

            # So far, the units of the values are V/V_rms, so multiply by the
            # rms voltage:
            values *= self.rms

            return values

        super().__init__(times, function=f, value_type=self.Type.voltage)


class FFTThermalNoise(FunctionSignal):
    """
    Class for thermal Rayleigh noise signals using the FFT.

    The Rayleigh thermal noise is calculated in a given frequency band with
    rayleigh-distributed or otherwise specified amplitudes and random phase.
    Values are calculated using an inverse FFT and then scaled to a provided or
    calculated RMS voltage.

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
    uniqueness_factor : int, optional
        The number of unique waveform traces that can be expected from this
        noise signal. This factor multiplies the length of the total trace to
        be calculated.

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
    will be periodic. Since the default number of frequencies used is based on
    the FFT bin size of `times`, the default period of the noise signal is
    actually the length of `times`. As a result if you are planning on
    extrapolating the noise signal, increasing the `uniqueness_factor` in order
    to increase the number of frequencies used is strongly recommended.

    Raises
    ------
    ValueError
        If the RMS voltage cannot be calculated (i.e. `rms_voltage` or both
        `temperature` and `resistance` are ``None``).

    See Also
    --------
    FunctionSignal : Class for signals generated by a function.
    FullThermalNoise : Class for thermal Rayleigh noise signals using exact
                       functions.

    Notes
    -----
    Calculation of the noise signal is based on the Rayleigh noise model used
    by ANITA [1]_. Modifications have been made to the default to make the
    frequency-domain amplitudes Rayleigh-distributed, under the suggestion that
    this makes for more realistic noise traces.

    The calculation of signal values is done using an inverse fast Fourier
    transform. This method has the advantage of being an order of magnitude
    faster than directly summing cosine functions. The disadvantage is that
    interpolated values (at times not given in the initial time trace) cannot
    be calculated exactly and must be linearly interpolated, thereby losing
    some accuracy.

    References
    ----------
    .. [1] A. Connolly et al, ANITA Note #76, "Thermal Noise Studies: Toward A
        Time-Domain Model of the ANITA Trigger."
        https://www.phys.hawaii.edu/elog/anita_notes/060228_110754/noise_simulation.ps

    """
    def __init__(self, times, f_band, f_amplitude=None, rms_voltage=None,
                 temperature=None, resistance=None, uniqueness_factor=1):
        # Calculation based on Rician (Rayleigh) noise model for ANITA:
        # https://www.phys.hawaii.edu/elog/anita_notes/060228_110754/noise_simulation.ps

        self.f_min, self.f_max = f_band
        if self.f_min>=self.f_max:
            raise ValueError("Frequency band must have smaller frequency as "+
                             "first value and larger frequency as second value")

        # Force uniqueness_factor to at least 1
        if uniqueness_factor<1:
            uniqueness_factor = 1
        self._unique = int(uniqueness_factor)

        self._n_all_freqs = self._unique * len(times)
        self._dt = times[1] - times[0]

        all_freqs = scipy.fft.rfftfreq(self._n_all_freqs, self._dt)
        band = (all_freqs>=self.f_min) & (all_freqs<=self.f_max)
        self.freqs = all_freqs[band]
        self._n_freqs = len(self.freqs)

        if f_amplitude is None:
            f_amplitude = lambda f: np.random.rayleigh(1/np.sqrt(2),
                                                       size=f.shape)

        # Allow f_amplitude to be either a function or a single value
        if callable(f_amplitude):
            # Attempt to evaluate all amplitudes in one function call
            try:
                self.amps = np.array(f_amplitude(self.freqs))
            # Otherwise evaluate responses one at a time
            except (TypeError, ValueError):
                logger.debug("Amplitude function %r could not be evaluated "+
                             "for multiple frequencies at once", f_amplitude)
                self.amps = np.array([f_amplitude(f) for f in self.freqs])
        else:
            self.amps = np.full(self._n_freqs, f_amplitude, dtype=np.float_)

        # If the frequency range includes zero, force the zero-frequency (DC)
        # amplitude to zero
        if 0 in self.freqs:
            self.amps[np.where(self.freqs==0)[0]] = 0

        self.phases = np.random.rand(self._n_freqs) * 2*np.pi

        if rms_voltage is not None:
            self.rms = rms_voltage
        elif temperature is not None and resistance is not None:
            # RMS voltage = sqrt(kB * T * R * bandwidth)
            # Not using sqrt(4 * kB * T * R * bandwidth) because in the antenna
            # system only half the voltage is seen and the other half goes to
            # "ground" (changed under advisement by Cosmin Deaconu)
            self.rms = np.sqrt(scipy.constants.k * temperature * resistance
                               * (self.f_max - self.f_min))
        else:
            raise ValueError("Either RMS voltage or temperature and resistance"+
                             " must be provided to calculate noise amplitude")

        self._fft_start = times[0]
        self._fft_end = times[-1]

        def get_fft_values(ts):
            """Set the time-domain signal using the FFT."""
            # Return zeros if there are no frequencies in-band
            if self._n_freqs==0:
                return np.zeros(len(ts))

            # Get the complete times array
            length = ((self._fft_end-self._fft_start+self._dt) * self._unique
                      - self._dt)
            fft_times = np.linspace(self._fft_start, self._fft_start+length,
                                    self._n_all_freqs)

            # Get the complete values array
            all_freqs = scipy.fft.rfftfreq(self._n_all_freqs, self._dt)
            band = (all_freqs>=self.f_min) & (all_freqs<=self.f_max)
            amps = np.zeros(len(all_freqs))
            amps[band] = self.amps
            phases = np.zeros(len(all_freqs))
            phases[band] = self.phases
            fft_values = scipy.fft.irfft(amps * np.exp(-1j*phases),
                                         n=self._n_all_freqs)

            # Normalization calculated by guess-and-check; seems to work fine
            # normalization = len(all_freqs) * np.sqrt(1/(2*len(band_freqs)))
            fft_values *= self._n_all_freqs * np.sqrt(1/(2*self._n_freqs))

            # Interpolate values at the given ts based on the full arrays
            values = np.interp(ts, fft_times, fft_values, period=length)

            # So far, the units of the values are V/V_rms, so multiply by the
            # rms voltage:
            values *= self.rms

            return values

        super().__init__(times, function=get_fft_values,
                         value_type=self.Type.voltage)



# Preferred thermal noise model:
ThermalNoise = FFTThermalNoise
