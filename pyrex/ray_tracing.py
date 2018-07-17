"""
Module containing classes for ray tracing through the ice.

Ray tracer classes correspond to ray trace path classes, where the ray
tracer is responsible for calculating the existance and launch angle of
paths between points, and the ray tracer path objects are responsible for
returning information about propagation along their respective path.

"""

import logging
import numpy as np
import scipy.optimize
from pyrex.internal_functions import normalize, LazyMutableClass, lazy_property
from pyrex.ice_model import AntarcticIce, IceModel

logger = logging.getLogger(__name__)


class BasicRayTracePath(LazyMutableClass):
    """
    Class for representing a single ray-trace solution between points.

    Stores parameters of the ray path with calculations performed by
    integrating z-steps of size ``dz``. Most properties are lazily evaluated
    to save on computation time. If any attributes of the class instance are
    changed, the lazily-evaluated properties will be cleared.

    Parameters
    ----------
    parent_tracer : BasicRayTracer
        Ray tracer for which this path is a solution.
    launch_angle : float
        Launch angle (radians) of the ray path.
    direct : boolean
        Whether the ray path is direct. If ``True`` this means the path does
        not "turn over". If ``False`` then the path does "turn over" by either
        reflection or refraction after reaching some maximum depth.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    theta0 : float
        The launch angle of the ray path at `from_point`.
    ice
        The ice model used for the ray tracer.
    dz : float
        The z-step (m) to be used for integration of the ray path attributes.
    direct : boolean
        Whether the ray path is direct. If ``True`` this means the path does
        not "turn over". If ``False`` then the path does "turn over" by either
        reflection or refraction after reaching some maximum depth.
    emitted_direction
    received_direction
    path_length
    tof
    coordinates

    See Also
    --------
    pyrex.internal_functions.LazyMutableClass : Class with lazy properties
                                                which may depend on other class
                                                attributes.
    BasicRayTracer : Class for calculating the ray-trace solutions between
                     points.

    Notes
    -----
    Even more attributes than those listed are available for the class, but
    are mainly for internal use. These attributes can be found by exploring
    the source code.

    """
    def __init__(self, parent_tracer, launch_angle, direct):
        self.from_point = parent_tracer.from_point
        self.to_point = parent_tracer.to_point
        self.theta0 = launch_angle
        self.ice = parent_tracer.ice
        self.dz = parent_tracer.dz
        self.direct = direct
        super().__init__()

    @property
    def z_turn_proximity(self):
        """
        Parameter for how closely path approaches z_turn.

        Necessary to avoid diverging integrals which occur at z_turn.

        """
        # Best value of dz/10 determined empirically by checking errors
        return self.dz/10

    @property
    def z0(self):
        """Depth (m) of the launching point."""
        return self.from_point[2]

    @property
    def z1(self):
        """Depth (m) of the receiving point."""
        return self.to_point[2]

    @lazy_property
    def n0(self):
        """Index of refraction of the ice at the launching point."""
        return self.ice.index(self.z0)

    @lazy_property
    def rho(self):
        """Radial distance (m) between the endpoints."""
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def phi(self):
        """Azimuthal angle (radians) between the endpoints."""
        u = self.to_point - self.from_point
        return np.arctan2(u[1], u[0])

    @lazy_property
    def beta(self):
        """Launching beta parameter (n(z0) * sin(theta0))."""
        return self.n0 * np.sin(self.theta0)

    @lazy_property
    def z_turn(self):
        """Turning depth (m) of the path."""
        return self.ice.depth_with_index(self.beta)

    # @property
    # def exists(self):
    #     """Boolean of whether the path between the points with the
    #     given launch angle exists."""
    #     return True

    @lazy_property
    def emitted_direction(self):
        """Direction in which ray is emitted."""
        return np.array([np.sin(self.theta0) * np.cos(self.phi),
                         np.sin(self.theta0) * np.sin(self.phi),
                         np.cos(self.theta0)])

    @lazy_property
    def received_direction(self):
        """Direction ray is travelling when it is received."""
        if self.direct:
            sign = np.sign(np.cos(self.theta0))
            return np.array([np.sin(self.theta(self.z1)) * np.cos(self.phi),
                             np.sin(self.theta(self.z1)) * np.sin(self.phi),
                             sign*np.cos(self.theta(self.z1))])
        else:
            return np.array([np.sin(self.theta(self.z1)) * np.cos(self.phi),
                             np.sin(self.theta(self.z1)) * np.sin(self.phi),
                             -np.cos(self.theta(self.z1))])

    def theta(self, z):
        """
        Polar angle of the ray at the given depths.

        Calculates the polar angle of the ray's direction at the given depth
        in the ice. Note that the ray could be travelling upward or downward
        at this polar angle.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.

        Returns
        -------
        array_like
            Polar angle at the given values of `z`.

        """
        return np.arcsin(np.sin(self.theta0) * self.n0/self.ice.index(z))


    # Log-scaled zs (commented out below and in z_integral method) seemed
    # like a good idea for reducing dimensionality, but didn't work out.
    # Kept here in case it works out better in the future

    # @lazy_property
    # def dn(self):
    #     return np.abs(self.ice.gradient(-10)[2])*self.dz

    # def _log_scale_zs(self, z0, z1):
    #     # Base dn on dz at 10 meter depth
    #     n0 = self.ice.index(z0)
    #     n1 = self.ice.index(z1)
    #     n_steps = int(np.abs(n1-n0)/self.dn)
    #     ns = np.linspace(n0, n1, n_steps+2)
    #     return self.ice.depth_with_index(ns)


    def z_integral(self, integrand):
        """
        Calculate the numerical integral of the given integrand.

        For the integrand as a function of z, the numerical integral is
        calculated along the ray path.

        Parameters
        ----------
        integrand : function
            Function returning the values of the integrand at a given array of
            values for the depth z.

        Returns
        -------
        float
            The value of the numerical integral along the ray path.

        """
        if self.direct:
            n_zs = int(np.abs(self.z1-self.z0)/self.dz)
            zs, dz = np.linspace(self.z0, self.z1, n_zs+1, retstep=True)
            return np.trapz(integrand(zs), dx=np.abs(dz), axis=0)
            # zs = self._log_scale_zs(self.z0, self.z1)
            # return np.trapz(integrand(zs), x=zs, axis=0)
        else:
            n_zs_1 = int(np.abs(self.z_turn-self.z_turn_proximity-self.z0)/self.dz)
            zs_1, dz_1 = np.linspace(self.z0, self.z_turn-self.z_turn_proximity,
                                     n_zs_1+1, retstep=True)
            n_zs_2 = int(np.abs(self.z_turn-self.z_turn_proximity-self.z1)/self.dz)
            zs_2, dz_2 = np.linspace(self.z_turn-self.z_turn_proximity, self.z1,
                                     n_zs_2+1, retstep=True)
            return (np.trapz(integrand(zs_1), dx=np.abs(dz_1), axis=0) +
                    np.trapz(integrand(zs_2), dx=np.abs(dz_2), axis=0))
            # zs_1 = self._log_scale_zs(self.z0, self.z_turn-self.z_turn_proximity)
            # zs_2 = self._log_scale_zs(self.z1, self.z_turn-self.z_turn_proximity)
            # return (np.trapz(integrand(zs_1), x=zs_1, axis=0) +
            #         np.trapz(integrand(zs_2), x=zs_2, axis=0))

    @lazy_property
    def path_length(self):
        """Length (m) of the ray path."""
        return self.z_integral(lambda z: 1/np.cos(self.theta(z)))

    @lazy_property
    def tof(self):
        """Time of flight (s) along the ray path."""
        return self.z_integral(lambda z: self.ice.index(z) / 3e8 /
                               np.cos(self.theta(z)))

    @lazy_property
    def fresnel(self):
        """
        Fresnel factor for reflection off the ice surface.

        The fresnel reflectance calculated is the square root (ratio of
        amplitudes, not powers) for reflection off ice surface (1 if doesn't
        reach surface).

        """
        if self.direct or self.z_turn<0:
            return 1
        else:
            n_1 = self.ice.index(0)
            n_2 = 1 # air
            theta_1 = self.theta(0)
            if theta_1>=np.arcsin(n_2/n_1):
                # Total internal reflection
                return 1
            else:
                # Askaryan signal is p-polarized
                # n_1 * cos(theta_2):
                n_1_cos_2 = n_1 * np.sqrt(1 - (n_1/n_2*np.sin(theta_1))**2)
                # n_2 * cos(theta_1):
                n_2_cos_1 = n_2 * np.cos(theta_1)
                # TODO: Confirm sign convention here
                return (n_1_cos_2 - n_2_cos_1) / (n_1_cos_2 + n_2_cos_1)

    def attenuation(self, f):
        """
        Calculate the attenuation factor for signal frequencies.

        Calculates the attenuation factor to be multiplied by the signal
        amplitude at the given frequencies.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz) at which to calculate signal attenuation.

        Returns
        -------
        array_like
            Attenuation factors for the signal at the frequencies `f`.

        """
        fa = np.abs(f)
        def integrand(z):
            partial_integrand = 1 / np.cos(self.theta(z))
            alen = self.ice.attenuation_length(z, fa)
            if alen.ndim<2:
                return np.vstack(partial_integrand / alen)
            else:
                return np.vstack(partial_integrand) / alen

        return np.exp(-np.abs(self.z_integral(integrand))) * self.fresnel

    def propagate(self, signal):
        """
        Propagate the signal along the ray path, in-place.

        Applies the frequency-dependent signal attenuation along the ray path
        and shifts the times according to the ray time of flight.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object to propagate.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        signal.filter_frequencies(self.attenuation)
        signal.times += self.tof

    @lazy_property
    def coordinates(self):
        """
        x, y, and z-coordinates along the path (using dz step).

        Coordinates are provided for plotting purposes only, and are not vetted
        for use in calculations.

        """
        if self.direct:
            n_zs = int(np.abs(self.z1-self.z0)/self.dz)
            zs, dz = np.linspace(self.z0, self.z1, n_zs+1, retstep=True)
            integrand = np.tan(self.theta(zs))

            rs = np.zeros(len(integrand))
            trap_areas = (integrand[:-1] + np.diff(integrand)/2) * dz
            rs[1:] += np.abs(np.cumsum(trap_areas))

        else:
            n_zs_1 = int(np.abs(self.z_turn-self.z_turn_proximity-self.z0) /
                         self.dz)
            zs_1, dz_1 = np.linspace(self.z0, self.z_turn-self.z_turn_proximity,
                                     n_zs_1+1, retstep=True)
            integrand_1 = np.tan(self.theta(zs_1))
            n_zs_2 = int(np.abs(self.z_turn-self.z_turn_proximity-self.z1) /
                         self.dz)
            zs_2, dz_2 = np.linspace(self.z_turn-self.z_turn_proximity, self.z1,
                                     n_zs_2+1, retstep=True)
            integrand_2 = np.tan(self.theta(zs_2))

            rs_1 = np.zeros(len(integrand_1))
            trap_areas = ((integrand_1[:-1] + np.diff(integrand_1)/2) *
                          np.abs(dz_1))
            rs_1[1:] += np.cumsum(trap_areas)

            rs_2 = np.zeros(len(integrand_2)) + rs_1[-1]
            trap_areas = ((integrand_2[:-1] + np.diff(integrand_2)/2) *
                          np.abs(dz_2))
            rs_2[1:] += np.cumsum(trap_areas)

            rs = np.concatenate((rs_1, rs_2[1:]))
            zs = np.concatenate((zs_1, zs_2[1:]))

        xs = self.from_point[0] + rs*np.cos(self.phi)
        ys = self.from_point[1] + rs*np.sin(self.phi)

        return xs, ys, zs



class SpecializedRayTracePath(BasicRayTracePath):
    """
    Class for representing a single ray-trace solution between points.

    Stores parameters of the ray path with calculations performed analytically
    (with the exception of attenuation). These calculations require the index
    of refraction of the ice to be of the form n(z)=n0-k*exp(a*z). However this
    restriction allows for most of the integrations to be performed
    analytically. The attenuation is the only attribute which is still
    calculated by numerical integration with z-steps of size ``dz``. Most
    properties are lazily evaluated to save on computation time. If any
    attributes of the class instance are changed, the lazily-evaluated
    properties will be cleared.

    Parameters
    ----------
    parent_tracer : SpecializedRayTracer
        Ray tracer for which this path is a solution.
    launch_angle : float
        Launch angle (radians) of the ray path.
    direct : boolean
        Whether the ray path is direct. If ``True`` this means the path does
        not "turn over". If ``False`` then the path does "turn over" by either
        reflection or refraction after reaching some maximum depth.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    theta0 : float
        The launch angle of the ray path at `from_point`.
    ice
        The ice model used for the ray tracer.
    dz : float
        The z-step (m) to be used for integration of the ray path attributes.
    direct : boolean
        Whether the ray path is direct. If ``True`` this means the path does
        not "turn over". If ``False`` then the path does "turn over" by either
        reflection or refraction after reaching some maximum depth.
    uniformity_factor : float
        Factor (<1) of the base index of refraction (n0 in the ice model)
        beyond which calculations start to break down numerically.
    beta_tolerance : float
        ``beta`` value (near 0) below which calculations start to break down
        numerically.
    emitted_direction
    received_direction
    path_length
    tof
    coordinates

    See Also
    --------
    pyrex.internal_functions.LazyMutableClass : Class with lazy properties
                                                which may depend on other class
                                                attributes.
    SpecializedRayTracer : Class for calculating the ray-trace solutions
                           between points.

    Notes
    -----
    Even more attributes than those listed are available for the class, but
    are mainly for internal use. These attributes can be found by exploring
    the source code.

    The requirement that the ice model go as n(z)=n0-k*exp(a*z) is implemented
    by requiring the ice model to inherit from `AntarcticIce`. Obviously this
    is not fool-proof, but likely the ray tracing will obviously fail if the
    index follows a very different functional form.

    """
    # Factor of index of refraction at which calculations may break down
    uniformity_factor = 0.99999
    # Beta value below which calculations may break down
    beta_tolerance = 0.005

    @lazy_property
    def valid_ice_model(self):
        """Whether the ice model being used supports this specialization."""
        return ((isinstance(self.ice, type) and
                 issubclass(self.ice, AntarcticIce))
                or isinstance(self.ice, AntarcticIce))

    @lazy_property
    def z_uniform(self):
        """
        Depth (m) beyond which the ice should be treated as uniform.

        Calculated based on the ``uniformity_factor``. Necessary due to
        numerical rounding issues at indices close to the index limit.

        """
        return self.ice.depth_with_index(self.ice.n0 * self.uniformity_factor)

    @staticmethod
    def _z_int_uniform_correction(z0, z1, z_uniform, beta, ice, integrand,
                                  derivative_special_case=False):
        """
        Function to perform a z-integration with a uniform ice correction.

        Can be an analytic or numerical integration. Takes into account the
        effect of treating the ice as uniform beyond some depth.

        Parameters
        ----------
        z0 : float
            (Negative-valued) depth (m) of the left limit of the integral.
        z1 : float
            (Negative-valued) depth (m) of the right limit of the integral.
        z_uniform : float
            (Negative-valued) depth (m) below which the ice is assumed to have
            a uniform index.
        beta : float
            ``beta`` value of the ray path.
        ice
            Ice model to be used for ray tracing.
        integrand : function
            Function returning the values of the integrand at a given array of
            values for the depth z.
        derivative_special_case : boolean, optional
            Boolean controlling whether the special case of doing the distance
            integral beta derivative should be used.

        Returns
        -------
        Integral of the given `integrand` along the path from `z0` to `z1`.

        """
        # Suppress numpy RuntimeWarnings
        with np.errstate(divide='ignore', invalid='ignore'):
            int_z0 = integrand(z0, beta, ice, deep=z0<z_uniform)
            int_z1 = integrand(z1, beta, ice, deep=z1<z_uniform)
            if not derivative_special_case:
                if (z0<z_uniform)==(z1<z_uniform):
                    # z0 and z1 on same side of z_uniform
                    return int_z1 - int_z0
                else:
                    int_diff = (integrand(z_uniform, beta, ice, deep=True) -
                                integrand(z_uniform, beta, ice, deep=False))
                    if z0<z1:
                        # z0 below z_uniform, z1 above z_uniform
                        return int_z1 - int_z0 + int_diff
                    else:
                        # z0 above z_uniform, z1 below z_uniform
                        return int_z1 - int_z0 - int_diff
            else:
                # Deal with special case of doing distance integral beta derivative
                # which includes two bounds instead of just giving indef. integral
                # FIXME: Somewhat inaccurate, should probably be done differently
                z_turn = np.log((ice.n0-beta)/ice.k)/ice.a
                if (z0<z_uniform)==(z1<z_uniform)==(z_turn<z_uniform):
                    # All on same side of z_uniform
                    return int_z0 + int_z1
                else:
                    int_diff = (integrand(z_uniform, beta, ice, deep=True) -
                                integrand(z_uniform, beta, ice, deep=False))
                    if (z0<z_uniform)==(z1<z_uniform):
                        # z0 and z1 below z_uniform, but z_turn above
                        return int_z0 + int_z1 - 2*int_diff
                    else:
                        # z0 or z1 below z_uniform, others above
                        return int_z0 + int_z1 - int_diff



    def z_integral(self, integrand, numerical=False, x_func=lambda x: x):
        """
        Calculate the integral of the given integrand.

        For the integrand as a function of z, the analytic or numerical
        integral is calculated along the ray path.

        Parameters
        ----------
        integrand : function
            Function returning the values of the integrand at a given array of
            values for the depth z.
        numerical : boolean, optional
            Whether to use the numerical integral instead of an analytic one.
            If ``False`` the analytic integral is calculated. If ``True`` the
            numerical integral is calculated.
        x_func : function, optional
            A function returning x values corresponding to given z values. By
            default just matches the z values.

        Returns
        -------
        float
            The value of the integral along the ray path.

        Raises
        ------
        TypeError
            If the ice model is not valid for the specialized analytic
            integrations.

        """
        if not numerical:
            if not self.valid_ice_model:
                raise TypeError("Ice model must inherit methods from "+
                                "pyrex.AntarcticIce")
            if self.direct:
                return self._z_int_uniform_correction(self.z0, self.z1,
                                                      self.z_uniform,
                                                      self.beta, self.ice,
                                                      integrand)
            else:
                int_1 = self._z_int_uniform_correction(self.z0, self.z_turn,
                                                       self.z_uniform,
                                                       self.beta, self.ice,
                                                       integrand)
                int_2 = self._z_int_uniform_correction(self.z1, self.z_turn,
                                                       self.z_uniform,
                                                       self.beta, self.ice,
                                                       integrand)
                return int_1 + int_2
        else:
            if self.direct:
                n_zs = int(np.abs(self.z1-self.z0)/self.dz)
                zs = np.linspace(self.z0, self.z1, n_zs+1)
                # zs = self._log_scale_zs(self.z0, self.z1)
                return np.trapz(integrand(zs), x=x_func(zs), axis=0)
            else:
                n_zs_1 = int(np.abs(self.z_turn-self.z0)/self.dz)
                zs_1 = np.linspace(self.z0, self.z_turn, n_zs_1+1)
                n_zs_2 = int(np.abs(self.z_turn-self.z1)/self.dz)
                zs_2 = np.linspace(self.z1, self.z_turn, n_zs_2+1)
                # zs_1 = self._log_scale_zs(self.z0, self.z_turn)
                # zs_2 = self._log_scale_zs(self.z1, self.z_turn)
                return (np.trapz(integrand(zs_1), x=x_func(zs_1), axis=0) +
                        np.trapz(integrand(zs_2), x=x_func(zs_2), axis=0))

    @staticmethod
    def _int_terms(z, beta, ice):
        """
        Useful pre-calculated substitutions for integrations.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depth (m) in the ice.
        beta : float
            ``beta`` value of the ray path.
        ice
            Ice model to be used for ray tracing.

        Returns
        -------
        alpha : float
            ``n0``^2 - `beta`^2
        n_z : float
            Index at depth `z`.
        gamma : float
            `n_z`^2 - `beta`^2
        log_term_1 : float
            ``n0``*`n_z` - `beta`^2 - sqrt(`alpha`*`gamma`)
        log_term_2 : float
            `n_z` + sqrt(`gamma`)

        """
        alpha = ice.n0**2 - beta**2
        n_z = ice.n0 - ice.k*np.exp(ice.a*z)
        gamma = n_z**2 - beta**2
        # Prevent errors when gamma is a very small negative number due to
        # numerical rounding errors. This could cause other problems for cases
        # where a not-tiny negative gamma would have meant nans but now leads to
        # non-nan values. It appears this only occurs when the launch angle
        # is greater than the maximum value allowed in the ray tracer however,
        # so it's likely alright. If problems arise, replace with gamma<0 and
        # np.isclose(gamma, 0) or similar
        gamma = np.where(gamma<0, 0, gamma)
        log_term_1 = ice.n0*n_z - beta**2 - np.sqrt(alpha*gamma)
        log_term_2 = -n_z - np.sqrt(gamma)
        return alpha, n_z, gamma, log_term_1, -log_term_2

    @classmethod
    def _distance_integral(cls, z, beta, ice, deep=False):
        """
        Indefinite z-integral for calculating radial distance.

        Calculates the indefinite z-integral of tan(arcsin(beta/n(z))), which
        between two z values gives the radial distance of the direct path
        between the z values.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depth (m) in the ice.
        beta : float
            ``beta`` value of the ray path.
        ice
            Ice model to be used for ray tracing.
        deep : boolean, optional
            Whether or not the integral is calculated in deep (uniform) ice.

        Returns
        -------
        array_like
            The value of the indefinite integral at `z`.

        """
        alpha, n_z, gamma, log_1, log_2 = cls._int_terms(z, beta, ice)
        if deep:
            return beta * z / np.sqrt(alpha)
        else:
            return np.where(np.isclose(beta, 0, atol=cls.beta_tolerance),
                            0,
                            beta / np.sqrt(alpha) * (-z + np.log(log_1)/ice.a))

    @classmethod
    def _distance_integral_derivative(cls, z, beta, ice, deep=False):
        """
        Beta derivative of indefinite z-integral for radial distance.

        Calculates the beta derivative of the indefinite z-integral of
        tan(arcsin(beta/n(z))), which is used for finding the maximum distance
        integral value as a function of launch angle. This function actually
        gives the integral from z to the turning point ``z_turn``, since that
        is what's needed for finding the peak angle.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depth (m) in the ice.
        beta : float
            ``beta`` value of the ray path.
        ice
            Ice model to be used for ray tracing.
        deep : boolean, optional
            Whether or not the integral is calculated in deep (uniform) ice.

        Returns
        -------
        array_like
            The value of the indefinite integral derivatve at `z`.

        """
        alpha, n_z, gamma, log_1, log_2 = cls._int_terms(z, beta, ice)
        z_turn = np.log((ice.n0-beta)/ice.k)/ice.a
        # print("z_turn:", z_turn)
        if deep:
            if z_turn<0:
                return ((np.log((ice.n0-beta)/ice.k)/ice.a - z -
                         beta/(ice.a*(ice.n0-beta))) / np.sqrt(alpha))
            else:
                return -z / np.sqrt(alpha)
        else:
            if z_turn<0:
                term_1 = ((1+beta**2/alpha)/np.sqrt(alpha) * 
                          (z + np.log(beta*ice.k/log_1) / ice.a))
                term_2 = -(beta**2+ice.n0*n_z) / (ice.a*alpha*np.sqrt(gamma))
            else:
                term_1 = -(1+beta**2/alpha)/np.sqrt(alpha)*(-z + np.log(log_1) /
                           ice.a)
                term_2 = -((beta*(np.sqrt(alpha)-np.sqrt(gamma)))**2 /
                           (ice.a*alpha*np.sqrt(gamma)*log_1))
                alpha, n_z, gamma, log_1, log_2 = cls._int_terms(0, beta, ice)
                term_1 += (1+beta**2/alpha)/np.sqrt(alpha)*(np.log(log_1) /
                           ice.a)
                term_2 += ((beta*(np.sqrt(alpha)-np.sqrt(gamma)))**2 /
                           (ice.a*alpha*np.sqrt(gamma)*log_1))
        return np.where(np.isclose(beta, 0, atol=cls.beta_tolerance),
                        np.inf,
                        term_1+term_2)

        # If the value of the integral just at z is needed (e.g. you want the
        # correct values when reflecting off the surface of the ice),
        # then use the terms below instead
        # Be warned, however, that this gives the wrong value when turning over
        # below the surface of the ice. The values get closer if only term_1
        # is returned in cases where gamma==0 (turning over in ice),
        # though the values are still slightly off
        # if deep:
        #     return z / np.sqrt(alpha)
        # term_1 = (1+beta**2/alpha)/np.sqrt(alpha)*(-z + np.log(log_1) / ice.a)
        # term_2 = ((beta*(np.sqrt(alpha)-np.sqrt(gamma)))**2 /
        #           (ice.a*alpha*np.sqrt(gamma)*log_1))
        # return np.where(gamma==0, term_1, term_1+term_2)

    @classmethod
    def _pathlen_integral(cls, z, beta, ice, deep=False):
        """
        Indefinite z-integral for calculating path length.

        Calculates the indefinite z-integral of sec(arcsin(beta/n(z))), which
        between two z values gives the radial distance of the direct path
        between the z values.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depth (m) in the ice.
        beta : float
            ``beta`` value of the ray path.
        ice
            Ice model to be used for ray tracing.
        deep : boolean, optional
            Whether or not the integral is calculated in deep (uniform) ice.

        Returns
        -------
        array_like
            The value of the indefinite integral at `z`.

        """
        alpha, n_z, gamma, log_1, log_2 = cls._int_terms(z, beta, ice)
        if deep:
            return ice.n0 * z / np.sqrt(alpha)
        else:
            return np.where(np.isclose(beta, 0, atol=cls.beta_tolerance),
                            z,
                            (ice.n0/np.sqrt(alpha) * (-z + np.log(log_1)/ice.a)
                             + np.log(log_2) / ice.a))

    @classmethod
    def _tof_integral(cls, z, beta, ice, deep=False):
        """
        Indefinite z-integral for calculating time of flight.

        Calculates the indefinite z-integral of n(z)/c*sec(arcsin(beta/n(z))),
        which between two z values gives the radial distance of the direct path
        between the z values.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depth (m) in the ice.
        beta : float
            ``beta`` value of the ray path.
        ice
            Ice model to be used for ray tracing.
        deep : boolean, optional
            Whether or not the integral is calculated in deep (uniform) ice.

        Returns
        -------
        array_like
            The value of the indefinite integral at `z`.

        """
        alpha, n_z, gamma, log_1, log_2 = cls._int_terms(z, beta, ice)
        if deep:
            return ice.n0*(n_z+ice.n0*(ice.a*z-1)) / (ice.a*np.sqrt(alpha)*3e8)
        else:
            return np.where(np.isclose(beta, 0, atol=cls.beta_tolerance),
                            ((n_z-ice.n0)/ice.a + ice.n0*z) / 3e8,
                            (((np.sqrt(gamma) + ice.n0*np.log(log_2) +
                               ice.n0**2*np.log(log_1)/np.sqrt(alpha))/ice.a) -
                             z*ice.n0**2/np.sqrt(alpha)) / 3e8)

    @lazy_property
    def path_length(self):
        """Length (m) of the ray path."""
        return np.abs(self.z_integral(self._pathlen_integral))

    @lazy_property
    def tof(self):
        """Time of flight (s) along the ray path."""
        return np.abs(self.z_integral(self._tof_integral))

    def attenuation(self, f):
        """
        Calculate the attenuation factor for signal frequencies.

        Calculates the attenuation factor to be multiplied by the signal
        amplitude at the given frequencies. Uses numerical integration since
        frequency dependence causes there to be no analytic form.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz) at which to calculate signal attenuation.

        Returns
        -------
        array_like
            Attenuation factors for the signal at the frequencies `f`.

        """
        fa = np.abs(f)

        def xi(z):
            return np.sqrt(1 - (self.beta/self.ice.index(z))**2)

        def xi_integrand(z):
            partial_integrand = (self.ice.index(z)**3 / self.beta**2 /
                                 (-self.ice.k*self.ice.a*np.exp(self.ice.a*z)))
            alen = self.ice.attenuation_length(z, fa)
            if alen.ndim<2:
                return np.vstack(partial_integrand / alen)
            else:
                return np.vstack(partial_integrand) / alen

        def z_integrand(z):
            partial_integrand = 1 / np.cos(np.arcsin(self.beta /
                                                     self.ice.index(z)))
            alen = self.ice.attenuation_length(z, fa)
            if alen.ndim<2:
                return np.vstack(partial_integrand / alen)
            else:
                return np.vstack(partial_integrand) / alen

        # If beta close to zero, just do the regular integral
        if np.isclose(self.beta, 0, atol=self.beta_tolerance):
            return (np.exp(-np.abs(self.z_integral(z_integrand, numerical=True)))
                    * self.fresnel)
        # Otherwise, do xi integral designed to avoid singularity at z_turn
        else:
            return (np.exp(-np.abs(self.z_integral(xi_integrand, numerical=True, x_func=xi)))
                    * self.fresnel)

    @lazy_property
    def coordinates(self):
        """
        x, y, and z-coordinates along the path (using dz step).

        Coordinates are provided for plotting purposes only, and are not vetted
        for use in calculations.

        """
        def r_int(z0, z1s):
            return np.array([self._z_int_uniform_correction(
                                z0, z, self.z_uniform, self.beta, self.ice,
                                self._distance_integral
                             )
                             for z in z1s])

        if self.direct:
            n_zs = int(np.abs(self.z1-self.z0)/self.dz)
            zs = np.linspace(self.z0, self.z1, n_zs+1)
            rs = r_int(self.z0, zs)
            rs *= np.sign(np.cos(self.theta0))

        else:
            n_zs_1 = int(np.abs(self.z_turn-self.z0)/self.dz)
            zs_1 = np.linspace(self.z0, self.z_turn, n_zs_1, endpoint=False)
            rs_1 = r_int(self.z0, zs_1)

            r_turn = r_int(self.z0, np.array([self.z_turn]))[0]

            n_zs_2 = int(np.abs(self.z_turn-self.z1)/self.dz)
            zs_2 = np.linspace(self.z_turn, self.z1, n_zs_2+1)
            rs_2 = r_turn - r_int(self.z_turn, zs_2)

            rs = np.concatenate((rs_1, rs_2))
            zs = np.concatenate((zs_1, zs_2))

        xs = self.from_point[0] + rs*np.cos(self.phi)
        ys = self.from_point[1] + rs*np.sin(self.phi)

        return xs, ys, zs





class BasicRayTracer(LazyMutableClass):
    """
    Class for calculating the ray-trace solutions between points.

    Calculations performed by integrating z-steps of size ``dz``. Most
    properties are lazily evaluated to save on computation time. If any
    attributes of the class instance are changed, the lazily-evaluated
    properties will be cleared.

    Parameters
    ----------
    from_point : array_like
        Vector starting point of the ray path.
    to_point : array_like
        Vector ending point of the ray path.
    ice_model
        The ice model used for the ray tracer.
    dz : float
        The z-step (m) to be used for integration of the ray path attributes.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    ice
        The ice model used for the ray tracer.
    dz : float
        The z-step (m) to be used for integration of the ray path attributes.
    solution_class
        Class to be used for each ray-trace solution path.
    exists
    expected_solutions
    solutions

    See Also
    --------
    pyrex.internal_functions.LazyMutableClass : Class with lazy properties
                                                which may depend on other class
                                                attributes.
    BasicRayTracePath : Class for representing a single ray-trace solution
                        between points.

    Notes
    -----
    Even more attributes than those listed are available for the class, but
    are mainly for internal use. These attributes can be found by exploring
    the source code.

    """
    solution_class = BasicRayTracePath

    def __init__(self, from_point, to_point, ice_model=IceModel, dz=1):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model
        self.dz = dz
        super().__init__()

    @property
    def z_turn_proximity(self):
        """
        Parameter for how closely path approaches z_turn.

        Necessary to avoid diverging integrals which occur at z_turn.

        """
        # Best value of dz/10 determined empirically by checking errors
        return self.dz/10

    # Calculations performed as if launching from low to high
    @property
    def z0(self):
        """
        Depth (m) of the lower endpoint.

        Ray tracing performed as if launching from lower point to higher point,
        since the only difference in the paths produced is a time reversal.
        This is then depth of the assumed launching point.

        """
        return min([self.from_point[2], self.to_point[2]])

    @property
    def z1(self):
        """
        Depth (m) of the higher endpoint.

        Ray tracing performed as if launching from lower point to higher point,
        since the only difference in the paths produced is a time reversal.
        This is then depth of the assumed receiving point.

        """
        return max([self.from_point[2], self.to_point[2]])

    @lazy_property
    def n0(self):
        """Index of refraction of the ice at the lower endpoint."""
        return self.ice.index(self.z0)

    @lazy_property
    def rho(self):
        """Radial distance between the endpoints."""
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def max_angle(self):
        """Maximum possible launch angle that could connect the endpoints."""
        return np.arcsin(self.ice.index(self.z1)/self.n0)

    @lazy_property
    def peak_angle(self):
        """
        Angle at which the indirect solutions curve (in r vs angle) peaks.

        This angle separates the angle intervals to be used for indirect
        solution root-finding.

        """
        for tolerance in np.logspace(-12, -4, num=3):
            for angle_step in np.logspace(-3, 0, num=4):
                r_func = (lambda angle, brent_arg:
                          self._indirect_r_prime(angle, brent_arg,
                                                 d_angle=angle_step))
                try:
                    peak_angle = self.angle_search(0, r_func,
                                                   angle_step, self.max_angle,
                                                   tolerance=tolerance)
                except (RuntimeError, ValueError):
                    # Failed to converge
                    continue
                else:
                    if peak_angle>np.pi/2:
                        peak_angle = np.pi - peak_angle
                    return peak_angle
        # If all else fails, just use the max_angle
        return self.max_angle

    @lazy_property
    def direct_r_max(self):
        """Maximum r value of direct ray solutions."""
        z_turn = self.ice.depth_with_index(self.n0 * np.sin(self.max_angle))
        return self._direct_r(self.max_angle,
                              force_z1=z_turn-self.z_turn_proximity)

    @lazy_property
    def indirect_r_max(self):
        """Maximum r value of indirect ray solutions."""
        return self._indirect_r(self.peak_angle)

    @lazy_property
    def exists(self):
        """Boolean of whether any paths exist between the endpoints."""
        return True in self.expected_solutions

    @lazy_property
    def expected_solutions(self):
        """
        List of which types of solutions are expected to exist.

        The first element of the list represents the direct path, the second
        element represents the indirect path with a launch angle greater than
        the peak angle, and the third element represents the indirect path with
        a launch angle less than the peak angle.

        """
        if self.from_point[2]>0 or self.to_point[2]>0:
            return [False, False, False]
        if self.rho<self.direct_r_max:
            return [True, False, True]
        elif self.rho<self.indirect_r_max:
            return [False, True, True]
        else:
            return [False, False, False]

    @lazy_property
    def solutions(self):
        """
        List of existing rays between the two points.

        This list should have zero elements if there are no possible paths
        between the endpoints or two elements otherwise, representing the
        more direct and the less direct paths, respectively.

        """
        angles = [
            self.direct_angle,
            self.indirect_angle_1,
            self.indirect_angle_2
        ]

        return [self.solution_class(self, angle, direct=(i==0))
                for i, angle, exists in zip(range(3), angles,
                                            self.expected_solutions)
                if exists and angle is not None]


    def _direct_r(self, angle, brent_arg=0, force_z1=None):
        """
        Calculate the r distance of the direct ray for a given launch angle.

        Parameters
        ----------
        angle : float
            Launch angle (radians) of a direct ray.
        brent_arg : float, optional
            Argument to subtract from the return value. Used for the brentq
            root finder to find a value other than zero.
        force_z1 : float or None, optional
            Value to use for the ``z1`` receiving depth. If ``None``, the
            ``z1`` property of the class will be used. Useful for changing the
            integration limits to integrate to the turning point instead.

        Returns
        -------
        float
            Value of the radial distance integral minus the `brent_arg`.

        """
        if force_z1 is not None:
            z1 = force_z1
        else:
            z1 = self.z1
        n_zs = int(np.abs((z1-self.z0)/self.dz))
        zs, dz = np.linspace(self.z0, z1, n_zs+1, retstep=True)
        integrand = np.tan(np.arcsin(np.sin(angle) *
                                     self.n0/self.ice.index(zs)))
        return np.trapz(integrand, dx=dz) - brent_arg

    def _indirect_r(self, angle, brent_arg=0):
        """
        Calculate the r distance of the indirect ray for a given launch angle.

        Parameters
        ----------
        angle : float
            Launch angle (radians) of an indirect ray.
        brent_arg : float, optional
            Argument to subtract from the return value. Used for the brentq
            root finder to find a value other than zero.

        Returns
        -------
        float
            Value of the radial distance integral minus the `brent_arg`.

        """
        z_turn = self.ice.depth_with_index(self.n0 * np.sin(angle))
        n_zs_1 = int(np.abs((z_turn-self.z_turn_proximity-self.z0)/self.dz))
        zs_1, dz_1 = np.linspace(self.z0, z_turn-self.z_turn_proximity,
                                 n_zs_1+1, retstep=True)
        integrand_1 = np.tan(np.arcsin(np.sin(angle) *
                                       self.n0/self.ice.index(zs_1)))
        n_zs_2 = int(np.abs((z_turn-self.z_turn_proximity-self.z1)/self.dz))
        zs_2, dz_2 = np.linspace(z_turn-self.z_turn_proximity, self.z1,
                                 n_zs_2+1, retstep=True)
        integrand_2 = np.tan(np.arcsin(np.sin(angle) *
                                       self.n0/self.ice.index(zs_2)))
        return (np.trapz(integrand_1, dx=dz_1) +
                np.trapz(integrand_2, dx=-dz_2)) - brent_arg

    def _indirect_r_prime(self, angle, brent_arg=0, d_angle=0.001):
        """
        Calculate the r distance derivative of the indirect ray.

        Parameters
        ----------
        angle : float
            Launch angle (radians) of an indirect ray.
        brent_arg : float, optional
            Argument to subtract from the return value. Used for the brentq
            root finder to find a value other than zero.
        d_angle : float, optional
            Difference in angle to use for calculation of the derivative.

        Returns
        -------
        float
            Value of the numerical derivative of the radial distance integral,
            minus the `brent_arg`.

        """
        return ((self._indirect_r(angle) - self._indirect_r(angle-d_angle))
                / d_angle) - brent_arg


    def _get_launch_angle(self, r_function, min_angle=0, max_angle=90):
        """
        Calculates the launch angle for a ray with the given r_function.

        Finds the root of the given r function as a function of angle to
        determine the corresponding launch angle.

        Parameters
        ----------
        r_function : function
            Function to calculate the radial distance for a given launch angle.
        min_angle : float, optional
            Minimum allowed angle for the `r_function`'s root.
        max_angle : float, optional
            Maximum allowed angle for the `r_function`'s root.

        Returns
        -------
        float or None
            True launch angle (radians) of the path corresponding to the
            `r_function`. True launch angle means launches from ``from_point``
            rather than from ``z0``. Value is ``None`` if the root finder was
            unable to converge.

        """
        try:
            launch_angle = self.angle_search(self.rho, r_function,
                                             min_angle, max_angle)
        except RuntimeError:
            # Failed to converge
            launch_angle = None
        except ValueError:
            logger.error("Error calculating launch angle between %s and %s",
                         self.from_point, self.to_point)
            raise

        # Convert to true launch angle from self.from_point
        # rather than from lower point (self.z0)
        return np.arcsin(np.sin(launch_angle) *
                         self.n0 / self.ice.index(self.from_point[2]))


    @lazy_property
    def direct_angle(self):
        """Launch angle (radians) of the direct ray."""
        if self.expected_solutions[0]:
            launch_angle = self._get_launch_angle(self._direct_r,
                                                  max_angle=self.max_angle)
            if self.from_point[2] > self.to_point[2]:
                launch_angle = np.pi - launch_angle
            return launch_angle
        else:
            return None

    @lazy_property
    def indirect_angle_1(self):
        """
        Launch angle (radians) of the first indirect ray.

        The first indirect ray is the indirect ray where the launch angle is
        greater than the peak angle.

        """
        if self.expected_solutions[1]:
            return self._get_launch_angle(self._indirect_r,
                                          min_angle=self.peak_angle,
                                          max_angle=self.max_angle)
        else:
            return None

    @lazy_property
    def indirect_angle_2(self):
        """
        Launch angle (radians) of the second indirect ray.

        The first indirect ray is the indirect ray where the launch angle is
        less than the peak angle.

        """
        if self.expected_solutions[2]:
            if self.expected_solutions[1]:
                max_angle = self.peak_angle
            else:
                max_angle = self.max_angle
            return self._get_launch_angle(self._indirect_r,
                                          max_angle=max_angle)
        else:
            return None

    @staticmethod
    def angle_search(true_r, r_function, min_angle, max_angle,
                     tolerance=1e-12, max_iterations=100):
        """
        Calculates the angle where `r_function` (angle) == `true_r`.

        Runs the brentq root-finding algorithm on `r_function` with an offset
        of `true_r` to find the angle at which they are equal.

        Parameters
        ----------
        true_r : float
            Desired value for the radial distance.
        r_function : function
            Function to calculate the radial distance for a given launch angle.
        min_angle : float
            Minimum allowed angle for the `r_function`.
        max_angle : float
            Maximum allowed angle for the `r_function`.
        tolerance : float, optional
            Tolerance in the root value for convergence.
        max_iterations : int, optional
            Maximum number of iterations the root finder will attempt.

        Returns
        -------
        float
            The launch angle which will satisfy the condition
            `r_function` (angle) == `true_r`.

        Raises
        ------
        RuntimeError
            If the root finder doesn't converge.

        """
        return scipy.optimize.brentq(r_function, min_angle, max_angle,
                                     args=(true_r), xtol=tolerance,
                                     maxiter=max_iterations)



class SpecializedRayTracer(BasicRayTracer):
    """
    Class for calculating the ray-trace solutions between points.

    Calculations in this class require the index of refraction of the ice to be
    of the form n(z)=n0-k*exp(a*z). However this restriction allows for most of
    the integrations to be performed analytically. Most properties are lazily
    evaluated to save on computation time. If any attributes of the class
    instance are changed, the lazily-evaluated properties will be cleared.

    Parameters
    ----------
    from_point : array_like
        Vector starting point of the ray path.
    to_point : array_like
        Vector ending point of the ray path.
    ice_model
        The ice model used for the ray tracer.
    dz : float
        The z-step (m) to be used for integration of the ray path attributes.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    ice
        The ice model used for the ray tracer.
    dz : float
        The z-step (m) to be used for integration of the ray path attributes.
    solution_class
        Class to be used for each ray-trace solution path.
    exists
    expected_solutions
    solutions

    See Also
    --------
    pyrex.internal_functions.LazyMutableClass : Class with lazy properties
                                                which may depend on other class
                                                attributes.
    SpecializedRayTracePath : Class for representing a single ray-trace
                              solution between points.

    Notes
    -----
    Even more attributes than those listed are available for the class, but
    are mainly for internal use. These attributes can be found by exploring
    the source code.

    The requirement that the ice model go as n(z)=n0-k*exp(a*z) is implemented
    by requiring the ice model to inherit from `AntarcticIce`. Obviously this
    is not fool-proof, but likely the ray tracing will obviously fail if the
    index follows a very different functional form.

    """
    solution_class = SpecializedRayTracePath

    @lazy_property
    def valid_ice_model(self):
        """Whether the ice model being used supports this specialization."""
        return ((isinstance(self.ice, type) and
                 issubclass(self.ice, AntarcticIce))
                or isinstance(self.ice, AntarcticIce))

    @lazy_property
    def z_uniform(self):
        """
        Depth (m) beyond which the ice should be treated as uniform.

        Calculated based on the ``uniformity_factor`` of the
        ``solution_class``. Necessary due to numerical rounding issues at
        indices close to the index limit.

        """
        return self.ice.depth_with_index(self.ice.n0 *
                                         self.solution_class.uniformity_factor)

    @lazy_property
    def direct_r_max(self):
        """Maximum r value of direct ray solutions."""
        z_turn = self.ice.depth_with_index(self.n0 * np.sin(self.max_angle))
        return self._direct_r(self.max_angle, force_z1=z_turn)

    def _r_distance(self, theta, z0, z1):
        """
        Calculate the r distance between depths for a given launch angle.

        Parameters
        ----------
        theta : float
            Launch angle (radians) of a ray path.
        z0 : float
            (Negative-valued) first depth (m) in the ice.
        z1 : float
            (Negative-valued) second depth (m) in the ice.

        Returns
        -------
        float
            Value of the radial distance integral between `z0` and `z1`.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must inherit methods from "+
                            "pyrex.AntarcticIce")
        beta = np.sin(theta) * self.n0
        return self.solution_class._z_int_uniform_correction(
            z0, z1, self.z_uniform, beta, self.ice,
            self.solution_class._distance_integral
        )

    def _r_distance_derivative(self, theta, z0, z1):
        """
        Calculate the derivative of the r distance between depths for an angle.

        Parameters
        ----------
        theta : float
            Launch angle (radians) of a ray path.
        z0 : float
            (Negative-valued) first depth (m) in the ice.
        z1 : float
            (Negative-valued) second depth (m) in the ice.

        Returns
        -------
        float
            Value of the derivative of the radial distance integral between
            `z0` and `z1`.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must inherit methods from "+
                            "pyrex.AntarcticIce")
        beta = np.sin(theta) * self.n0
        beta_prime = np.cos(theta) * self.n0
        return beta_prime * self.solution_class._z_int_uniform_correction(
            z0, z1, self.z_uniform, beta, self.ice,
            self.solution_class._distance_integral_derivative,
            derivative_special_case=True
        )

    def _direct_r(self, angle, brent_arg=0, force_z1=None):
        """
        Calculate the r distance of the direct ray for a given launch angle.

        Parameters
        ----------
        angle : float
            Launch angle (radians) of a direct ray.
        brent_arg : float, optional
            Argument to subtract from the return value. Used for the brentq
            root finder to find a value other than zero.
        force_z1 : float or None, optional
            Value to use for the ``z1`` receiving depth. If ``None``, the
            ``z1`` property of the class will be used. Useful for changing the
            integration limits to integrate to the turning point instead.

        Returns
        -------
        float
            Value of the radial distance integral minus the `brent_arg`.

        """
        if force_z1 is not None:
            z1 = force_z1
        else:
            z1 = self.z1
        return self._r_distance(angle, self.z0, z1) - brent_arg

    def _indirect_r(self, angle, brent_arg=0):
        """
        Calculate the r distance of the indirect ray for a given launch angle.

        Parameters
        ----------
        angle : float
            Launch angle (radians) of an indirect ray.
        brent_arg : float, optional
            Argument to subtract from the return value. Used for the brentq
            root finder to find a value other than zero.

        Returns
        -------
        float
            Value of the radial distance integral minus the `brent_arg`.

        """
        z_turn = self.ice.depth_with_index(self.n0 * np.sin(angle))
        return (self._r_distance(angle, self.z0, z_turn) +
                self._r_distance(angle, self.z1, z_turn)) - brent_arg

    def _indirect_r_prime(self, angle, brent_arg=0):
        """
        Calculate the r distance derivative of the indirect ray.

        Parameters
        ----------
        angle : float
            Launch angle (radians) of an indirect ray.
        brent_arg : float, optional
            Argument to subtract from the return value. Used for the brentq
            root finder to find a value other than zero.

        Returns
        -------
        float
            Value of the derivative of the radial distance integral minus the
            `brent_arg`.

        """
        return self._r_distance_derivative(angle, self.z0, self.z1) - brent_arg

    @lazy_property
    def peak_angle(self):
        """
        Angle at which the indirect solutions curve (in r vs angle) peaks.

        This angle separates the angle intervals to be used for indirect
        solution root-finding.

        """
        try:
            peak_angle = self.angle_search(0, self._indirect_r_prime,
                                           0, self.max_angle)
        except ValueError:
            # _indirect_r_prime(0) and _indirect_r_prime(max_angle) have the
            # same sign -> no true peak angle
            return self.max_angle
        except RuntimeError:
            # Failed to converge
            return None
        else:
            if peak_angle>np.pi/2:
                peak_angle = np.pi - peak_angle
            return peak_angle



# Set preferred ray tracer and path to specialized classes
RayTracer = SpecializedRayTracer
RayTracePath = SpecializedRayTracePath



class PathFinder:
    """
    Class for pseudo ray tracing. Just uses straight-line paths.

    Parameters
    ----------
    ice_model
        The ice model used for the ray tracer.
    from_point : array_like
        Vector starting point of the ray path.
    to_point : array_like
        Vector ending point of the ray path.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    ice
        The ice model used for the ray tracer.
    exists
    emitted_ray
    received_ray
    path_length
    tof

    """
    def __init__(self, ice_model, from_point, to_point):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model

    @property
    def exists(self):
        """
        Boolean of whether the path exists between the endpoints.

        Path existance is determined by a total internal refleaciton
        calculation.

        """
        ni = self.ice.index(self.from_point[2])
        nf = self.ice.index(self.to_point[2])
        nr = nf / ni
        # If relative index is greater than 1, total internal reflection
        # is impossible
        if nr > 1:
            return True
        # Check z-component of emitted ray against normalized z-component
        # of critical ray for total internal reflection
        tir = np.sqrt(1 - nr**2)
        return self.emitted_ray[2] > tir

    @property
    def emitted_ray(self):
        """Direction in which the ray is emitted."""
        return normalize(self.to_point - self.from_point)

    @property
    def received_ray(self):
        """Direction from which the ray is received."""
        return self.emitted_ray

    @property
    def path_length(self):
        """Length (m) of the path."""
        return np.linalg.norm(self.to_point - self.from_point)

    @property
    def tof(self):
        """
        Time of flight (s) for a particle along the path.

        Calculated using default values of self.time_of_flight()

        """
        return self.time_of_flight()

    def time_of_flight(self, n_steps=100):
        """
        Time of flight (s) for a particle along the path.

        Calculated by integrating the time durations of steps along the path.

        Parameters
        ----------
        n_steps : int, optional
            Number of z-steps to divide the path into.

        Returns
        -------
        float
            The approximate time of flight (s) along the path.

        """
        z0 = self.from_point[2]
        z1 = self.to_point[2]
        zs = np.linspace(z0, z1, n_steps, endpoint=True)
        u = self.to_point - self.from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        integrand = self.ice.index(zs)
        t = np.trapz(integrand, zs) / 3e8 * np.sqrt(1 + (rho / (z1 - z0))**2)
        return np.abs(t)

    def attenuation(self, f, n_steps=100):
        """
        Calculate the attenuation factor for signal frequencies.

        Calculates the attenuation factor to be multiplied by the signal
        amplitude at the given frequencies by mutliplying the attenuation
        factors of each step along the path.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz) at which to calculate signal attenuation.
        n_steps : int, optional
            Number of z-steps to divide the path into.

        Returns
        -------
        array_like
            Attenuation factors for the signal at the frequencies `f`.

        """
        fa = np.abs(f)
        z0 = self.from_point[2]
        z1 = self.to_point[2]
        zs, dz = np.linspace(z0, z1, n_steps, endpoint=False, retstep=True)
        u = self.to_point - self.from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dr = rho / (z1 - z0) * dz
        dp = np.sqrt(dz**2 + dr**2)
        alens = self.ice.attenuation_length(zs, fa)
        attens = np.exp(-dp/alens)
        return np.prod(attens, axis=0)

    def propagate(self, signal):
        """
        Propagate the signal along the ray path, in-place.

        Applies the frequency-dependent signal attenuation along the ray path
        and shifts the times according to the ray time of flight.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object to propagate.

        Raises
        ------
        RuntimeError
            If the path does not exist.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        if not self.exists:
            raise RuntimeError("Cannot propagate signal along a path that "+
                               "doesn't exist")
        signal.filter_frequencies(self.attenuation)
        signal.times += self.tof


class ReflectedPathFinder:
    """
    Class for pseudo ray tracing of a reflected ray. Uses straight-line paths.

    Parameters
    ----------
    ice_model
        The ice model used for the ray tracer.
    from_point : array_like
        Vector starting point of the ray path.
    to_point : array_like
        Vector ending point of the ray path.
    reflection_depth : float, optional
        (Negative-valued) depth (m) at which the ray reflects.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    ice
        The ice model used for the ray tracer.
    bounce_point : ndarray
        The point at which the ray path is reflected.
    path_1 : PathFinder
        The path from `from_point` to `bounce_point`.
    path_2 : PathFinder
        The path from `bounce_point` to `to_point`.
    exists
    emitted_ray
    received_ray
    path_length
    tof

    """
    def __init__(self, ice_model, from_point, to_point, reflection_depth=0):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model

        self.bounce_point = self.get_bounce_point(reflection_depth)

        self.path_1 = PathFinder(ice_model=self.ice,
                                 from_point=self.from_point,
                                 to_point=self.bounce_point)
        self.path_2 = PathFinder(ice_model=self.ice,
                                 from_point=self.bounce_point,
                                 to_point=self.to_point)

    def get_bounce_point(self, reflection_depth=0):
        """
        Calculates the point at which the ray is reflected.

        Parameters
        ----------
        reflection_depth : float, optional
            (Negative-valued) depth (m) at which the ray reflects.

        Returns
        -------
        ndarray
            Vector point at the `reflection_depth` where the ray is reflected.

        """
        z0 = self.from_point[2] - reflection_depth
        z1 = self.to_point[2] - reflection_depth
        u = self.to_point - self.from_point
        # x-y distance between points
        rho = np.sqrt(u[0]**2 + u[1]**2)
        # x-y distance to bounce point based on geometric arguments
        distance = z0*rho / (z0+z1)
        # x-y direction vector
        u_xy = np.array([u[0], u[1], 0])
        direction = normalize(u_xy)
        bounce_point = self.from_point + distance*direction
        bounce_point[2] = reflection_depth
        return bounce_point

    @property
    def exists(self):
        """
        Boolean of whether the path exists between the endpoints.

        Path existance is determined by whether its sub-paths exist and whether
        it could reflect off the ice surface (or ice layer at the reflection
        depth).

        """
        # nr = nf / ni = 1 / ni
        ni = self.ice.index(self.from_point[2])
        nf = self.ice.index(self.bounce_point[2]) if self.bounce_point[2]<0 else 1
        nr = nf / ni
        # For completeness, check that ice index isn't less than 1
        if nr>1:
            surface_reflection = False
        else:
            # Check z-component of emitted ray against normalized z-component
            # of critical ray for total internal reflection
            tir = np.sqrt(1 - nr**2)
            surface_reflection = self.emitted_ray[2] <= tir
        return self.path_1.exists and self.path_2.exists and surface_reflection

    @property
    def emitted_ray(self):
        """Direction in which the ray is emitted."""
        return normalize(self.bounce_point - self.from_point)

    @property
    def received_ray(self):
        """Direction from which the ray is received."""
        return normalize(self.to_point - self.bounce_point)

    @property
    def path_length(self):
        """Length of the path (m)."""
        return self.path_1.path_length + self.path_2.path_length

    @property
    def tof(self):
        """
        Time of flight (s) for a particle along the path.

        Calculated using default values of self.time_of_flight()

        """
        return self.path_1.tof + self.path_2.tof

    def time_of_flight(self, n_steps=100):
        """
        Time of flight (s) for a particle along the path.

        Calculated by integrating the time durations of steps along the path.

        Parameters
        ----------
        n_steps : int, optional
            Number of z-steps to divide the path into. Each sub-path divided
            into this many steps.

        Returns
        -------
        float
            The approximate time of flight (s) along the path.

        """
        return (self.path_1.time_of_flight(n_steps) +
                self.path_2.time_of_flight(n_steps))

    def attenuation(self, f, n_steps=100):
        """
        Calculate the attenuation factor for signal frequencies.

        Calculates the attenuation factor to be multiplied by the signal
        amplitude at the given frequencies by mutliplying the attenuation
        factors of each step along the path.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz) at which to calculate signal attenuation.
        n_steps : int, optional
            Number of z-steps to divide the path into. Each sub-path divided
            into this many steps.

        Returns
        -------
        array_like
            Attenuation factors for the signal at the frequencies `f`.

        """
        return (self.path_1.attenuation(f, n_steps) *
                self.path_2.attenuation(f, n_steps))

    def propagate(self, signal):
        """
        Propagate the signal along the ray path, in-place.

        Applies the frequency-dependent signal attenuation along the ray path
        and shifts the times according to the ray time of flight.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object to propagate.

        Raises
        ------
        RuntimeError
            If the path does not exist.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        self.path_1.propagate(signal)
        self.path_2.propagate(signal)
