"""
Module containing ice model classes.

The ice model classes contain static and class methods for convenience,
and parameters of the ice model are set as class attributes.

"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class AntarcticIce:
    """Class containing characteristics of ice at the south pole.
    In all cases, depth z is given with negative values in the ice and positive
    values above the ice. Index of refraction goes as n(z)=n0-k*exp(az)."""
    # Based mostly on sources from http://icecube.wisc.edu/~mnewcomb/radio/
    k = 0.43
    a = 0.0132
    n0 = 1.78
    thickness = 2850

    @classmethod
    def gradient(cls, z):
        """Returns the gradient of the index of refraction at depth z (m)."""
        return np.array([0, 0, -cls.k * cls.a * np.exp(cls.a * z)])

    @classmethod
    def index(cls, z):
        """Returns the medium's index of refraction, n, at depth z (m).
        Supports passing a numpy array of depths."""
        try:
            indices = np.ones(len(z))
        except TypeError:
            # z is a scalar, so just return one value
            if z>0:
                return 1
            else:
                return cls.n0 - cls.k * np.exp(cls.a * z)

        indices[z<=0] = cls.n0 - cls.k * np.exp(cls.a * z[z<=0])

        return indices

    @classmethod
    def depth_with_index(cls, n):
        """Returns the depth z (m) at which the medium has the given index
        of refraction (inverse of index function, assumes index function is
        monotonic so only one solution exists).
        Supports passing a numpy array of indices."""
        n0 = cls.index(0)
        try:
            depths = np.zeros(len(n))
        except TypeError:
            # n is a scalar, so just return one value
            if n<=n0:
                return 0
            else:
                return np.log((cls.n0-n)/cls.k) / cls.a

        depths[n>n0] = np.log((cls.n0-n[n>n0])/cls.k) / cls.a

        return depths

    @staticmethod
    def temperature(z):
        """Returns the temperature (K) of the ice at depth z (m).
        Supports passing a numpy array of depths."""
        z_km = -0.001 * z
        c_temp = -51.07 + z_km*(2.677 + z_km*(-0.01591 + z_km*1.83415))
        return c_temp + 273.15

    @staticmethod
    def __atten_coeffs(t, f):
        """Helper function to calculate a and b coefficients for attenuation
        length calculation at given temperature (K) and frequency (Hz).
        Supports passing numpy arrays for t and f."""
        # Based on code from Besson, et.al. referenced at
        # http://icecube.wisc.edu/~mnewcomb/radio/atten/
        # but simplified significantly since w1=0

        t_C = t - 273.15
        w0 = np.log(1e-4)
        # w1 = 0
        w2 = np.log(3.16)

        b0 = -6.7489 + t_C * (0.026709 - 8.84e-4 * t_C)
        b1 = -6.2212 - t_C * (0.070927 + 1.773e-3 * t_C)
        b2 = -4.0947 - t_C * (0.002213 + 3.32e-4 * t_C)

        if isinstance(t, np.ndarray) and isinstance(f, np.ndarray):
            # t and f are both arrays, so return 2-D array of coefficients
            # where each row is a single t and each column is a single f.
            a = np.broadcast_to(b1[:,np.newaxis], (len(t), len(f)))
            b = np.zeros((len(t),len(f)))
            # Use numpy slicing to calculate different values for b when
            # f<1e9 and f>=1e9. Transpose b0, b1, b2 into column vectors
            # so numpy multiplies properly
            b[:,f<1e9] += (b0[:,np.newaxis] - b1[:,np.newaxis]) / w0
            b[:,f>=1e9] += (b2[:,np.newaxis] - b1[:,np.newaxis]) / w2

        elif isinstance(f, np.ndarray):
            # t is a scalar, so return an array of coefficients based
            # on the frequencies
            a = np.full(len(f), b1)
            b = np.zeros(len(f))
            # Again use numpy slicing to differentiate f<1e9 and f>=1e9
            b[f<1e9] += (b0 - b1) / w0
            b[f>=1e9] += (b2 - b1) / w2

        # Past this point, f must be a scalar
        # Then an array or single coefficient is returned based on the type of t
        elif f < 1e9:
            a = b1
            b = (b0 - b1) / w0
        else:
            a = b1
            b = (b2 - b1) / w2

        return a, b

    @classmethod
    def attenuation_length(cls, z, f):
        """Returns the attenuation length at depth z (m) and frequency f (Hz).
        Supports passing a numpy array of depths and/or frequencies.
        If both are passed as arrays, a 2-D array is returned where
        each row is a single depth and each column is a single frequency."""
        # Supress RuntimeWarnings when f==0 temporarily
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
    based on Matt Newcomb's fit (DOESN'T CURRENTLY WORK)."""
    k = 0.438
    a = 0.0132
    n0 = 1.32 + 0.438

    @classmethod
    def attenuation_length(cls, z, f):
        """Returns the attenuation length at depth z (m) and frequency f (MHz)
        by Matt Newcomb's fit (DOESN'T CURRENTLY WORK - USE BOGORODSKY)."""
        temp = cls.temperature(z)
        a = 5.03097 * np.exp(0.134806 * temp)
        b = 0.172082 + temp + 10.629
        c = -0.00199175 * temp - 0.703323
        return 1701 / (a + b * (0.001*f)**(c+1))



class ArasimIce(AntarcticIce):
    """Class containing characteristics of ice at the south pole.
    In all cases, depth z is given with negative values in the ice and positive
    values above the ice. Ice model index is the same as used in the ARA
    collaboration's AraSim package."""
    k = 0.43
    a = 0.0132
    n0 = 1.78

    atten_depths = [
        72.7412, 76.5697, 80.3982, 91.8836, 95.7121, 107.198, 118.683,
        133.997, 153.139, 179.939, 206.738, 245.023, 298.622, 356.049,
        405.819, 470.904, 516.845, 566.616, 616.386, 669.985, 727.412,
        784.839, 838.438, 899.694, 949.464, 1003.06, 1060.49, 1121.75,
        1179.17, 1236.6,  1297.86, 1347.63, 1405.05, 1466.31, 1516.08,
        1565.85, 1611.79, 1657.73, 1699.85, 1745.79, 1791.73, 1833.84,
        1883.61, 1929.56, 1990.81, 2052.07, 2109.49, 2170.75, 2232.01,
        2304.75, 2362.17, 2431.09, 2496.17
    ]

    atten_lengths = [
        1994.67, 1952,    1896,    1842.67, 1797.33, 1733.33, 1680,
        1632,    1586.67, 1552,    1522.67, 1501.33, 1474.67, 1458.67,
        1437.33, 1416,    1392,    1365.33, 1344,    1312,    1274.67,
        1242.67, 1205.33, 1168,    1128,    1090.67, 1048,    1008,
        965.333, 920,     874.667, 834.667, 797.333, 752,     714.667,
        677.333, 648,     616,     589.333, 557.333, 530.667, 506.667,
        477.333, 453.333, 418.667, 389.333, 362.667, 333.333, 309.333,
        285.333, 264,     242.667, 221.333
    ]

    @classmethod
    def attenuation_length(cls, z, f):
        """Returns the attenuation length at depth z (m) and frequency f (Hz).
        Attenuation length not actually frequency dependent; according to
        AraSim always uses the 300 MHz value.
        Supports passing a numpy array of depths and/or frequencies.
        If both are passed as arrays, a 2-D array is returned where
        each row is a single depth and each column is a single frequency."""
        lengths = np.interp(-z, cls.atten_depths, cls.atten_lengths)

        if isinstance(z, np.ndarray) and isinstance(f, np.ndarray):
            # z and f are both arrays, so return 2-D array where each
            # row is a single z and each column is a single f
            return np.broadcast_to(lengths[:,np.newaxis], (len(z), len(f)))

        elif isinstance(f, np.ndarray):
            # z is a scalar, so return an array of coefficients based
            # on the frequencies
            return np.broadcast_to(lengths, len(f))

        else:
            # Past this point, f must be a scalar, so an array or
            # single coefficient is returned based on the type of z
            return lengths



# Preferred ice model:
IceModel = AntarcticIce
