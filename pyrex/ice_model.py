"""
Module containing ice model classes.

Contains models for ice at the South Pole and at Summit Station in Greenland,
plus a model for ice with uniform index of refraction.

"""

import logging
import numpy as np
import scipy.constants
import scipy.interpolate

logger = logging.getLogger(__name__)


class AntarcticIce:
    """
    Class describing the ice at the south pole.

    In all methods, the depth z should be given as a negative value if it is
    below the surface of the ice.

    Parameters
    ----------
    n0 : float, optional
        Asymptotic index of refraction of the deep ice.
    k : float, optional
        Multiplicative factor for the index of refraction parameterization.
    a : float, optional
        Exponential factor for the index of refraction parameterization with
        units of 1/m.
    valid_range : array_like of float, optional
        Range of depths over which the uniform index of refraction applies.
        Assumed to have two elements where the first value is lower (deeper,
        more negative) than the second.
    index_above : float or None, optional
        Index of refraction above the ice region. If `None`, uses the same
        index of refraction as the top of the ice.
    index_below : float or None, optional
        Index of refraction below the ice region. If `None`, uses the same
        index of refraction as the bottom of the ice.

    Attributes
    ----------
    n0, k, a : float
        Parameters of the index of refraction of the ice.
    valid_range : tuple
        Range of depths over which the ice model is valid. Consists of two
        elements where the first value is lower (deeper, more negative) than
        the second.
    index_above
    index_below

    Notes
    -----
    Parameterizations mostly based on South Pole ice characteristics outlined
    by Matt Newcomb [1]_.

    References
    ----------
    .. [1] M. Newcomb (2013) http://icecube.wisc.edu/~araproject/radio/

    """
    def __init__(self, n0=1.78, k=0.43, a=0.0132, valid_range=(-2850, 0),
                 index_above=1, index_below=None):
        self.n0 = n0
        self.k = k
        self.a = a
        self.valid_range = tuple(sorted(valid_range))
        self._index_above = index_above
        self._index_below = index_below

    @property
    def index_above(self):
        """The index of refraction above the ice's valid range."""
        if self._index_above is None:
            return self.index(self.valid_range[1])
        else:
            return self._index_above

    @index_above.setter
    def index_above(self, index):
        self._index_above = index

    @property
    def index_below(self):
        """The index of refraction below the ice's valid range."""
        if self._index_below is None:
            return self.index(self.valid_range[0])
        else:
            return self._index_below

    @index_below.setter
    def index_below(self, index):
        self._index_below = index

    def contains(self, point):
        """
        Determines if the given point is within the ice's valid range.

        Parameters
        ----------
        point : array_like
            Point to be tested.

        Returns
        -------
        bool
            Whether `point` is cointained within the ice model.

        """
        return self.valid_range[0]<=point[2]<=self.valid_range[1]

    def index(self, z):
        """
        Calculates the index of refraction of the ice at a given depth.

        Index of refraction goes as n(z)=n0-k*exp(az). Supports passing an
        array of depths.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.

        Returns
        -------
        array_like
            Indices of refraction at `z` values.

        """
        try:
            indices = np.ones(len(z))
        except TypeError:
            if z<self.valid_range[0]:
                return self.index_below
            elif z>self.valid_range[1]:
                return self.index_above
            else:
                return self.n0 - self.k * np.exp(self.a * z)

        indices = self.n0 - self.k * np.exp(self.a * np.asarray(z))
        indices[z<self.valid_range[0]] = self.index_below
        indices[z>self.valid_range[1]] = self.index_above
        return indices

    def gradient(self, z):
        """
        Calculates the gradient of the index of refraction at a given depth.

        Parameters
        ----------
        z : float
            (Negative-valued) depth (m) in the ice.

        Returns
        -------
        ndarray
            Gradient of the index of refraction at `z`.

        """
        return np.array([0, 0, -self.k * self.a * np.exp(self.a * z)])


    def depth_with_index(self, n):
        """
        Calculates the corresponding depth for a given index of refraction.

        Assumes that the function for the index of refraction is invertible.
        Index of refraction goes as n(z)=n0-k*exp(az), so the inversion goes as
        z(n)=log((n0-n)/k)/a. Supports passing an array of indices.

        Parameters
        ----------
        n : array_like
            Indices of refraction.

        Returns
        -------
        array_like
            (Negative-valued) depths corresponding to the given `n` values.

        Notes
        -----
        For indices of refraction outside of the range of indices in the ice,
        returns the bounds of the ice. For example, if given an index of
        refraction less than the minimum in the valid range, the upper boundary
        depth will be returned.

        """
        try:
            depths = np.zeros(len(n))
        except TypeError:
            # n is a scalar, so just return one value
            if n<self.index(self.valid_range[1]):
                return self.valid_range[1]
            elif n>self.index(self.valid_range[0]):
                return self.valid_range[0]
            else:
                return np.log((self.n0-n)/self.k) / self.a

        depths = np.log((self.n0-np.asarray(n))/self.k) / self.a
        depths[n<self.index(self.valid_range[1])] = self.valid_range[1]
        depths[n>self.index(self.valid_range[0])] = self.valid_range[0]
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
            Temperatures (K) at `z` values.

        Notes
        -----
        Based on a polynomial fit by Matt Newcomb [1]_.

        References
        ----------
        .. [1] M. Newcomb (2013)
            https://icecube.wisc.edu/~araproject/radio/temp/

        """
        z_km = -0.001 * z
        c_temp = -51.07 + z_km*(2.677 + z_km*(-0.01591 + z_km*1.83415))
        return c_temp + scipy.constants.zero_Celsius

    @staticmethod
    def _atten_coeffs(t, f):
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
        Based on code from Besson et al as reported by Matt Newcomb [1]_.
        The shape of the output arrays is determined by the shapes of the input
        arrays. If both inputs are scalar, the outputs will be scalar. If one
        input is scalar and the other is a 1D array, the outputs will be 1D
        arrays. If both inputs are 1D arrays, the outputs will be 2D arrays
        where each row corresponds to a single temperature and each column
        corresponds to a single frequency.

        References
        ----------
        .. [1] M. Newcomb (2013)
            http://icecube.wisc.edu/~araproject/radio/atten/


        """
        # Based on the code from Besson, et.al.
        # but simplified significantly since w1=0

        t_C = t - scipy.constants.zero_Celsius
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

    def attenuation_length(self, z, f):
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
        Based on code from Besson et al as reported by Matt Newcomb [1]_.
        The shape of the output array is determined by the shapes of the input
        arrays. If both inputs are scalar, the output will be scalar. If one
        input is scalar and the other is a 1D array, the output will be a 1D
        array. If both inputs are 1D arrays, the output will be a 2D array
        where each row corresponds to a single depth and each column
        corresponds to a single frequency.

        References
        ----------
        .. [1] M. Newcomb (2013)
            http://icecube.wisc.edu/~araproject/radio/atten/

        """
        # Suppress RuntimeWarnings when f==0 temporarily
        with np.errstate(divide='ignore'):
            # w is log of frequency in GHz
            w = np.log(f*1e-9)

        # Temperature in kelvin
        t = self.temperature(z)

        a, b = self._atten_coeffs(t, f)
        # a and b will be scalar, 1-D, or 2-D as necessary based on t and f

        return np.exp(-(a + b * w))



class UniformIce:
    """
    Class describing ice with a uniform index of refraction.

    In all methods, the depth z should be given as a negative value if it is
    below the surface of the ice.

    Parameters
    ----------
    index : float
        Index of refraction of the ice.
    valid_range : array_like of float
        Range of depths over which the uniform index of refraction applies.
        Assumed to have two elements where the first value is lower (deeper,
        more negative) than the second.
    index_above : float or None, optional
        Index of refraction above the ice region. If `None`, uses the same
        index of refraction.
    index_below : float or None, optional
        Index of refraction below the ice region. If `None`, uses the same
        index of refraction.

    Attributes
    ----------
    n : float
        Index of refraction of the ice.
    valid_range : tuple
        Range of depths over which the ice model is valid. Consists of two
        elements where the first value is lower (deeper, more negative) than
        the second.
    index_above
    index_below

    """
    def __init__(self, index, valid_range=(-2850, 0), index_above=1,
                 index_below=None):
        self.n = index
        self.valid_range = tuple(sorted(valid_range))
        self._index_above = index_above
        self._index_below = index_below

    @property
    def index_above(self):
        """The index of refraction above the ice's valid range."""
        if self._index_above is None:
            return self.n
        else:
            return self._index_above

    @index_above.setter
    def index_above(self, index):
        self._index_above = index

    @property
    def index_below(self):
        """The index of refraction below the ice's valid range."""
        if self._index_below is None:
            return self.n
        else:
            return self._index_below

    @index_below.setter
    def index_below(self, index):
        self._index_below = index

    def contains(self, point):
        """
        Determines if the given point is within the ice's valid range.

        Parameters
        ----------
        point : array_like
            Point to be tested.

        Returns
        -------
        bool
            Whether `point` is cointained within the ice model.

        """
        return self.valid_range[0]<=point[2]<=self.valid_range[1]

    def index(self, z):
        """
        Calculates the index of refraction of the ice at a given depth.

        Supports passing an array of depths. For depths outside of the ice
        range, returns `index_below` or `index_above`.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.

        Returns
        -------
        array_like
            Indices of refraction at `z` values.

        """
        try:
            indices = np.ones(len(z)) * self.n
        except TypeError:
            if z<self.valid_range[0]:
                return self.index_below
            elif z>self.valid_range[1]:
                return self.index_above
            else:
                return self.n

        indices[z<self.valid_range[0]] = self.index_below
        indices[z>self.valid_range[1]] = self.index_above
        return indices

    def gradient(self, z):
        """
        Calculates the gradient of the index of refraction at a given depth.

        Parameters
        ----------
        z : float
            (Negative-valued) depth (m) in the ice.

        Returns
        -------
        ndarray
            Gradient of the index of refraction at `z`.

        """
        return np.array([0, 0, 0])

    def depth_with_index(self, n):
        """
        Calculates the corresponding depth for a given index of refraction.

        Invalid for uniform ice as all depths have the same index.

        Parameters
        ----------
        n : array_like
            Indices of refraction.

        Returns
        -------
        array_like
            (Negative-valued) depths corresponding to the given `n` values.

        """
        raise NotImplementedError("depth_with_index is invalid for "+
                                  self.__class__.__name__)

    # Use some of the same methods from AntarcticIce without being a subclass
    temperature = staticmethod(AntarcticIce.temperature)
    _atten_coeffs = staticmethod(AntarcticIce._atten_coeffs)
    attenuation_length = AntarcticIce.attenuation_length



class ArasimIce(AntarcticIce):
    """
    Class describing the ice at the south pole.

    Designed to match the ice model used in AraSim. In all methods, the depth z
    should be given as a negative value if it is below the surface of the ice.

    Parameters
    ----------
    n0 : float, optional
        Asymptotic index of refraction of the deep ice.
    k : float, optional
        Multiplicative factor for the index of refraction parameterization.
    a : float, optional
        Exponential factor for the index of refraction parameterization with
        units of 1/m.
    valid_range : array_like of float, optional
        Range of depths over which the uniform index of refraction applies.
        Assumed to have two elements where the first value is lower (deeper,
        more negative) than the second.
    index_above : float or None, optional
        Index of refraction above the ice region. If `None`, uses the same
        index of refraction as the top of the ice.
    index_below : float or None, optional
        Index of refraction below the ice region. If `None`, uses the same
        index of refraction as the bottom of the ice.

    Attributes
    ----------
    n0, k, a : float
        Parameters of the index of refraction of the ice.
    valid_range : tuple
        Range of depths over which the ice model is valid. Consists of two
        elements where the first value is lower (deeper, more negative) than
        the second.
    index_above
    index_below

    """
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

    def attenuation_length(self, z, f):
        """
        Calculates attenuation lengths for given depths and frequencies.

        Attenuation lengths not actually frequency dependent. According to
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
        interp = scipy.interpolate.interp1d(self.atten_depths,
                                            self.atten_lengths,
                                            fill_value="extrapolate",
                                            assume_sorted=True)
        lengths = interp(-z)

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



class GreenlandIce(AntarcticIce):
    """
    Class describing the ice at Summit Station in Greenland.

    In all methods, the depth z should be given as a negative value if it is
    below the surface of the ice.

    Parameters
    ----------
    n0 : float, optional
        Asymptotic index of refraction of the deep ice.
    k : float, optional
        Multiplicative factor for the index of refraction parameterization.
    a : float, optional
        Exponential factor for the index of refraction parameterization with
        units of 1/m.
    valid_range : array_like of float, optional
        Range of depths over which the uniform index of refraction applies.
        Assumed to have two elements where the first value is lower (deeper,
        more negative) than the second.
    index_above : float or None, optional
        Index of refraction above the ice region. If `None`, uses the same
        index of refraction as the top of the ice.
    index_below : float or None, optional
        Index of refraction below the ice region. If `None`, uses the same
        index of refraction as the bottom of the ice.

    Attributes
    ----------
    n0, k, a : float
        Parameters of the index of refraction of the ice.
    valid_range : tuple
        Range of depths over which the ice model is valid. Consists of two
        elements where the first value is lower (deeper, more negative) than
        the second.
    index_above
    index_below

    Notes
    -----
    Index of refraction parameterization based on a slightly altered version
    of the density parameterization at Summit Station [1]_. The altered version
    ignores the break at small depths in order to have a uniform index of
    refraction parameterization matching the form of the Antarctic index.
    The temperature and attenuation length parameterizations are also based on
    parameterizations defined for Summit Station [2]_.

    References
    ----------
    .. [1] C. Deaconu et al, "Measurements and modeling of near-surface radio
        propagation in glacial ice and implications for neutrino experiments."
        Physical Review D **98**, 043010 (2018). :arxiv:`1805.12576`
        :doi:`10.1103/PhysRevD.98.043010`
    .. [2] J. Avva et al, "An in Situ Measurement of the Radio-Frequency
        Attenuation in Ice at Summit Station, Greenland." Journal of Glaciology
        **61**, no. 229, 1005-1011 (2015). :doi:`10.3189/2015JoG15J057`

    """
    def __init__(self, n0=1.775, k=0.448, a=0.0247, valid_range=(-3000, 0),
                 index_above=1, index_below=None):
        super().__init__(n0=n0, k=k, a=a,
                         valid_range=valid_range,
                         index_above=index_above,
                         index_below=index_below)

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
            Temperatures (K) at `z` values.

        Notes
        -----
        Based on a polynomial fit of the GRIP borehole data [1]_.

        References
        ----------
        .. [1] Greenland Ice Core Project (GRIP) (1994)
            ftp://ftp.ncdc.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/physical/griptemp.txt

        """
        z_km = -0.001 * z
        c_temp = -31.771 + z_km*(-0.32485 + z_km*(6.7427 + z_km*(-11.471 + z_km*(5.9122 - 0.84945*z_km))))
        return c_temp + scipy.constants.zero_Celsius

    def attenuation_length(self, z, f):
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
        Attenuation length based on the measurement at 75 MHz [1]_, then
        extrapolated in depth based on the temperature profile of the ice and
        extrapolated in frequency based on a linear slope (as outlined in the
        reference paper).
        The shape of the output array is determined by the shapes of the input
        arrays. If both inputs are scalar, the output will be scalar. If one
        input is scalar and the other is a 1D array, the output will be a 1D
        array. If both inputs are 1D arrays, the output will be a 2D array
        where each row corresponds to a single depth and each column
        corresponds to a single frequency.

        References
        ----------
        .. [1] J. Avva et al, "An in Situ Measurement of the Radio-Frequency
            Attenuation in Ice at Summit Station, Greenland." Journal of
            Glaciology **61**, no. 229, 1005-1011 (2015).

        """
        # Temperature at the relevant depths
        t = self.temperature(z)
        t_C = t - scipy.constants.zero_Celsius

        # Parameterization of attenuation length at 75 MHz based on the ice
        # temperature profile
        alen_75 = 10**(-1.736e-2*t_C+2.5134)

        # Attenuation lengths at different frequencies are linearly
        # extrapolated from the 75 MHz value down to some minimum length
        min_alen = 1

        if isinstance(z, np.ndarray) and isinstance(f, np.ndarray):
            # z and f are both arrays, so return 2-D array of lengths
            # where each row is a single z and each column is a single f
            # alen = np.zeros((len(t), len(f)))
            alen = -0.55e-6 * (f - 75e6) + alen_75[:, np.newaxis]
            alen[alen<min_alen] = min_alen

        else:
            # If either z and/or f is a scalar, numpy will return an array or
            # scalar to match the dimensionality
            alen = -0.55e-6 * (f - 75e6) + alen_75

        # Enforce the minimum attenuation length
        if isinstance(alen, np.ndarray):
            alen[alen<min_alen] = min_alen
        elif alen<min_alen:
            alen = min_alen

        return alen



# Preferred ice model:
ice = AntarcticIce()
