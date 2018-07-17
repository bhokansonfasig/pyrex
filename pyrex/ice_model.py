"""
Module containing ice model classes.

The ice model classes contain static and class methods for convenience,
and parameters of the ice model are set as class attributes.

"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class AntarcticIce:
    """
    Class describing the ice at the south pole.

    For convenience, consists of static methods and class methods, so creating
    an instance of the class may not be necessary. In all methods, the depth
    z should be given as a negative value if it is below the surface of the
    ice.

    Attributes
    ----------
    k, a, n0 : float
        Parameters of the index of refraction of the ice.
    thickness : float
        Thickness of the ice sheet.

    Notes
    -----
    Mostly based on ice characteristics outlined by Matt Newcomb.

    """
    # Based mostly on sources from http://icecube.wisc.edu/~mnewcomb/radio/
    k = 0.43
    a = 0.0132
    n0 = 1.78
    thickness = 2850

    @classmethod
    def gradient(cls, z):
        """
        Calculates the gradient of the index of refraction at a given depth.

        Parameters
        ----------
        z : float
            (Negative-valued) depth (m) in the ice.

        Returns
        -------
        float
            Gradient of the index of refraction at `depth`.

        """
        return np.array([0, 0, -cls.k * cls.a * np.exp(cls.a * z)])

    @classmethod
    def index(cls, z):
        """
        Calculates the index of refraction of the ice at a given depth.

        Index of refraciton goes as n(z)=n0-k*exp(az). Supports passing an
        array of depths.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.

        Returns
        -------
        array_like
            Indices of refraction at `depth` values.

        """
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
        """
        Calculates the corresponding depth for a given index of refraction.

        Assumes that the function for the index of refraction is invertible.
        Index of refraciton goes as n(z)=n0-k*exp(az), so the inversion goes as
        z(n)=log((n0-n)/k)/a. Supports passing an array of indices.

        Parameters
        ----------
        n : array_like
            Indices of refraction.

        Returns
        -------
        array_like
            (Negative-valued) depths corresponding to the given `n` values.

        """
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
        """
        Calculates the temperature of the ice at a given depth.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.

        Returns
        -------
        array_like
            Temperatures (K) at `depth` values.

        """
        z_km = -0.001 * z
        c_temp = -51.07 + z_km*(2.677 + z_km*(-0.01591 + z_km*1.83415))
        return c_temp + 273.15

    @staticmethod
    def __atten_coeffs(t, f):
        """
        Calculates attenuation coefficients for temperature and frequency.

        Helper function to calculate ``a`` and ``b`` coefficients for use in
        the attenuation length calculation.

        Parameters
        ----------
        t : array_like
            Temperatures (K) of the ice.
        f : array_like
            Frequencies (Hz) of the signal.

        Returns
        -------
        a : array_like
            Array of ``a`` coefficients.
        b : array_like
            Array of ``b`` coefficients.

        Notes
        -----
        The shape of the output arrays is determined by the shapes of the input
        arrays. If both inputs are scalar, the outputs will be scalar. If one
        input is scalar and the other is a 1D array, the outputs will be 1D
        arrays. If both inputs are 1D arrays, the outputs will be 2D arrays
        where each row corresponds to a single temperature and each column
        corresponds to a single frequency.

        """
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
        """
        Calculates attenuation lengths for given depths and frequencies.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.
        f : array_like
            Frequencies (Hz) of the signal.

        Returns
        -------
        array_like
            Attenuation lengths for the given parameters.

        Notes
        -----
        The shape of the output array is determined by the shapes of the input
        arrays. If both inputs are scalar, the output will be scalar. If one
        input is scalar and the other is a 1D array, the output will be a 1D
        array. If both inputs are 1D arrays, the output will be a 2D array
        where each row corresponds to a single depth and each column
        corresponds to a single frequency.

        """
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
    """
    Class describing the ice at the south pole.

    Uses an attenuation length based on Matt Newcomb's fit. For convenience,
    consists of static methods and class methods, so creating an instance of
    the class may not be necessary. In all methods, the depth z should be given
    as a negative value if it is below the surface of the ice.

    Attributes
    ----------
    k, a, n0 : float
        Parameters of the index of refraction of the ice.
    thickness : float
        Thickness of the ice sheet.

    Warnings
    --------
    The `attenuation_length` method if this class does not currently work
    properly. This class should not be used until it is fixed.

    Notes
    -----
    Mostly based on ice characteristics outlined by Matt Newcomb.

    """
    k = 0.438
    a = 0.0132
    n0 = 1.32 + 0.438

    @classmethod
    def attenuation_length(cls, z, f):
        """
        Calculates attenuation lengths for given depths and frequencies.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.
        f : array_like
            Frequencies (MHz) of the signal.

        Returns
        -------
        array_like
            Attenuation lengths for the given parameters.

        Warnings
        --------
        This method does not currently work properly. Instead the Bogorodsky
        attenuation in the `AntarcticIce` class should be used.

        This method does not currently support passing both inputs as 1D arrays
        the way `AntarcticIce.attenuation_length` does.

        Notes
        -----
        The shape of the output array is determined by the shapes of the input
        arrays. If both inputs are scalar, the output will be scalar. If one
        input is scalar and the other is a 1D array, the output will be a 1D
        array.

        """
        temp = cls.temperature(z)
        a = 5.03097 * np.exp(0.134806 * temp)
        b = 0.172082 + temp + 10.629
        c = -0.00199175 * temp - 0.703323
        return 1701 / (a + b * (0.001*f)**(c+1))



class ArasimIce(AntarcticIce):
    """
    Class describing the ice at the south pole.

    Designed to match ice model used in AraSim. For convenience, consists of
    static methods and class methods, so creating an instance of the class may
    not be necessary. In all methods, the depth z should be given as a negative
    value if it is below the surface of the ice.

    Attributes
    ----------
    k, a, n0 : float
        Parameters of the index of refraction of the ice.
    thickness : float
        Thickness of the ice sheet.
    atten_depths, atten_lengths : list
        Depths and corresponding attenuation lengths to be interpolated in the
        `attenuation_length` calculation.

    """
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
        """
        Calculates attenuation lengths for given depths and frequencies.

        Attenuation length not actually frequency dependent. According to
        AraSim the attenuation lengths are for 300 MHz.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.
        f : array_like
            Frequencies (Hz) of the signal.

        Returns
        -------
        array_like
            Attenuation lengths for the given parameters.

        Notes
        -----
        The shape of the output array is determined by the shapes of the input
        arrays. If both inputs are scalar, the output will be scalar. If one
        input is scalar and the other is a 1D array, the output will be a 1D
        array. If both inputs are 1D arrays, the output will be a 2D array
        where each row corresponds to a single depth and each column
        corresponds to a single frequency.

        """
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
