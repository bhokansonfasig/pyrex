"""Module containing ice model. AntarcticIce class contains static and class
methods for easy swapping of models. IceModel class is set to the preferred
ice model."""

import numpy as np

class AntarcticIce:
    """Class containing characteristics of ice at the south pole.
    In all cases, depth z is given with negative values in the ice and positive
    values above the ice."""
    k = 0.438
    a = 0.0132
    n0 = 1.32
    thickness = 2850

    @classmethod
    def gradient(cls, z):
        """Returns the gradient of the index of refraction at depth z (m)."""
        return np.array([0.0, -cls.k * cls.a * np.exp(cls.a * z)])

    @classmethod
    def index(cls, z):
        """Returns the medium's index of refraction, n, at depth z (m).
        Supports passing a numpy array of depths."""
        try:
            indices = np.zeros(len(z))
        except TypeError:
            # z is a scalar, so just return one value
            if z>0:
                return 1
            else:
                return cls.n0 + cls.k * (1 - np.exp(cls.a * z))

        for i, depth in enumerate(z):
            if depth>0:
                indices[i] = 1
            else:
                indices[i] = cls.n0 + cls.k * (1 - np.exp(cls.a * depth))
        return indices

    @staticmethod
    def temperature(z):
        """Returns the temperature (K) of the ice at depth z (m).
        Supports passing a numpy array of depths."""
        km = -0.001 * z
        c_temp = -51.07 + km*(2.677 + km*(-0.01591 + km*1.83415))
        return c_temp + 273.15

    @staticmethod
    def __atten_coeffs(t, f):
        """Helper function to calculate a and b coefficients for attenuation
        length calculation at given temperature (K) and frequency (Hz).
        Supports passing numpy arrays for t and f."""
        t_C = t - 273.15
        w0 = np.log(1e-4)
        w1 = 0
        w2 = np.log(3.16)

        b0 = -6.7489 + t_C * (0.026709 - 8.84e-4 * t_C)
        b1 = -6.2212 - t_C * (0.070927 + 1.77e-3 * t_C)
        b2 = -4.0947 - t_C * (0.002213 + 3.32e-4 * t_C)

        if isinstance(t, np.ndarray) and isinstance(f, np.ndarray):
            # t and f are both arrays, so return 2-D array of coefficients
            # where each row is a single t and each column is a single f.
            a = np.zeros((len(t),len(f)))
            b = np.zeros((len(t),len(f)))
            # Use numpy slicing to calculate different values when
            # f<1e9 and f>=1e9. Transpose b0, b1, b2 into column vectors
            # so numpy multiplies properly
            a[:,f<1e9] += (b1[:,np.newaxis] * w0 - b0[:,np.newaxis] * w1) / (w0 - w1)
            b[:,f<1e9] += (b1[:,np.newaxis] - b0[:,np.newaxis]) / (w1 - w0)
            a[:,f>=1e9] += (b2[:,np.newaxis] * w1 - b1[:,np.newaxis] * w2) / (w1 - w2)
            b[:,f>=1e9] += (b2[:,np.newaxis] - b1[:,np.newaxis]) / (w2 - w1)
        elif isinstance(f, np.ndarray):
            # t is a scalar, so return an array of coefficients based
            # on the frequencies
            a = np.zeros(len(f))
            b = np.zeros(len(f))
            # Again use numpy slicing to differentiate f<1e9 and f>=1e9
            a[f<1e9] += (b1 * w0 - b0 * w1) / (w0 - w1)
            b[f<1e9] += (b1 - b0) / (w1 - w0)
            a[f>=1e9] += (b2 * w1 - b1 * w2) / (w1 - w2)
            b[f>=1e9] += (b2 - b1) / (w2 - w1)

        # Past this point, f must be a scalar
        # Then an array or single coefficient is returned based on the type of t
        elif f < 1e9:
            a = (b1 * w0 - b0 * w1) / (w0 - w1)
            b = (b1 - b0) / (w1 - w0)
        else:
            a = (b2 * w1 - b1 * w2) / (w1 - w2)
            b = (b2 - b1) / (w2 - w1)

        return a, b

    @classmethod
    def attenuation_length(cls, z, f):
        """Returns the attenuation length at depth z (m) and frequency f (Hz).
        Supports passing a numpy array of depths and/or frequencies.
        If both are passed as arrays, a 2-D array is returned where
        each row is a single depth and each column is a single frequency."""
        # Suppress RuntimeWarnings when f==0 temporarily
        with np.errstate(divide='ignore'):
            # w is log of frequency in GHz
            w = np.log(f*1e-9)

        # Temperature in kelvin
        t = cls.temperature(z)

        a, b = cls.__atten_coeffs(t, f)
        # a and b will be scalar, 1-D, or 2-D as necessary based on t and f

        return np.exp(-(a + b * w))



class NewcombIce(AntarcticIce):
    """Class inheriting from AntarcticIce, with new attenuation_length function
    based on Matt Newcomb's fit (DOESN'T CURRENTLY WORK - USE ANTARCTICICE)."""
    @classmethod
    def attenuation_length(cls, z, f):
        """Returns the attenuation length at depth z (m) and frequency f (MHz)
        by Matt Newcomb's fit (DOESN'T CURRENTLY WORK - USE BOGORODSKY)."""
        temp = cls.temperature(z)
        a = 5.03097 * np.exp(0.134806 * temp)
        b = 0.172082 + temp + 10.629
        c = -0.00199175 * temp - 0.703323
        return 1701 / (a + b * (0.001*f)**(c+1))



# Preferred ice model:
IceModel = AntarcticIce
