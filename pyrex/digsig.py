"""Module containing classes for digital signal processing"""


import numpy as np
import scipy.signal

class Signal:
    """Base class for signals. Takes arrays of times and values
    (values array forced to size of times array by zero padding or slicing).
    Supports adding between signals with the same time values,
    resampling the signal, and calculating the signal's envelope."""
    def __init__(self, times, values):
        self.times = np.array(times)
        len_diff = len(times)-len(values)
        if len_diff>0:
            self.values = np.concatenate((values, np.zeros(len_diff)))
        else:
            self.values = np.array(values[:len(times)])

    def __add__(self, other):
        """Adds two signals by adding their values at each time."""
        if not(isinstance(other, Signal)):
            raise TypeError("Can't add object with type"
                            +str(type(other))+" to a signal")
        if not(np.array_equal(self.times, other.times)):
            raise ValueError("Can't add signals with different times")
        
        return Signal(self.times,self.values+other.values)

    def __radd__(self, other):
        """Allows for adding Signal object to 0.
        Useful when using sum() on a set of signals."""
        if other!=0:
            raise TypeError("unsupported operand type(s) for +: '"+
                            +str(type(other))+"' and 'Signal'")

        return self

    @property
    def dt(self):
        """Returns the spacing of the time array, or None if invalid."""
        try:
            return self.times[1]-self.times[0]
        except IndexError:
            return None

    @property
    def envelope(self):
        """Calculates envelope of the signal by Hilbert transform."""
        analytic_signal = scipy.signal.hilbert(self.values)
        return np.abs(analytic_signal)

    def resample(self, n):
        """Resamples the signal into n points in the same time range"""
        if n==len(self.times):
            return

        self.times = np.linspace(self.times[0], self.times[-1], n)
        self.values = scipy.signal.resample(self.values, n)
