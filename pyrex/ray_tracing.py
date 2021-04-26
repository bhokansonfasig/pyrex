"""
Module containing classes for ray tracing through the ice.

Ray tracer classes correspond to ray trace path classes, where the ray
tracer is responsible for calculating the existence and launch angle of
paths between points, and the ray tracer path objects are responsible for
returning information about propagation along their respective path.

"""

import logging
import numpy as np
import scipy.constants
import scipy.fft
import scipy.optimize
from pyrex.internal_functions import normalize, LazyMutableClass, lazy_property
from pyrex.ice_model import AntarcticIce, UniformIce, ice

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
    def _metadata(self):
        """Metadata dictionary for writing `BasicRayTracePath` information."""
        return {
            "n0": self.n0,
            "dz": self.dz,
            "emitted_x": self.emitted_direction[0],
            "emitted_y": self.emitted_direction[1],
            "emitted_z": self.emitted_direction[2],
            "received_x": self.received_direction[0],
            "received_y": self.received_direction[1],
            "received_z": self.received_direction[2],
            "launch_angle": np.arccos(self.emitted_direction[2]),
            "receiving_angle": np.pi-np.arccos(self.received_direction[2]),
            "path_length": self.path_length,
            "tof": self.tof
        }

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
        return self.z_integral(lambda z: self.ice.index(z) / scipy.constants.c
                               / np.cos(self.theta(z)))

    @lazy_property
    def fresnel(self):
        """
        Fresnel factors for reflection off the ice surface.

        The fresnel reflectance calculated is the square root (ratio of
        amplitudes, not powers) for reflection off ice surface (1 if doesn't
        reach surface). Stores the s and p polarized reflectances, respectively.

        """
        if self.direct or self.z_turn<self.ice.valid_range[1]:
            return 1, 1
        else:
            n_1 = self.ice.index(self.ice.valid_range[1])
            n_2 = self.ice.index_above
            theta_1 = self.theta(self.ice.valid_range[1])
            cos_1 = np.cos(theta_1)
            sin_2 = n_1/n_2*np.sin(theta_1)
            if sin_2<=1:
                # Plain reflection with real coefficients
                cos_2 = np.sqrt(1 - (sin_2)**2)
            else:
                # Total internal reflection off the surface, results in complex
                # fresnel factors encoding the phase data
                cos_2 = np.sqrt((sin_2)**2 - 1)*1j
            # TODO: Confirm sign convention here
            r_s = (n_1*cos_1 - n_2*cos_2) / (n_1*cos_1 + n_2*cos_2)
            r_p = (n_2*cos_1 - n_1*cos_2) / (n_2*cos_1 + n_1*cos_2)
            return r_s, r_p

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
            return (partial_integrand / alen.T).T

        return np.exp(-np.abs(self.z_integral(integrand)))

    def propagate(self, signal=None, polarization=None,
                  attenuation_interpolation=None):
        """
        Propagate the signal with optional polarization along the ray path.

        Applies the frequency-dependent signal attenuation along the ray path
        and shifts the times according to the ray time of flight. Additionally
        provides the s and p polarization directions.

        Parameters
        ----------
        signal : Signal, optional
            ``Signal`` object to propagate.
        polarization : array_like, optional
            Vector representing the linear polarization of the `signal`.
        attenuation_interpolation: float, optional
            Logarithmic (base 10) interpolation step to be used for
            interpolating attenuation along the ray path. If `None`, no
            interpolation is applied and the attenuation is pre-calculated at
            the expected signal frequencies.

        Returns
        -------
        tuple of Signal
            Tuple of ``Signal`` objects representing the s and p polarizations
            of the original `signal` attenuated along the ray path. Only
            returned if `signal` was not ``None``.
        tuple of ndarray
            Tuple of polarization vectors representing the s and p polarization
            directions of the `signal` at the end of the ray path. Only
            returned if `polarization` was not ``None``.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        if polarization is None:
            if signal is None:
                return

            else:
                new_signal = signal.copy()
                new_signal.shift(self.tof)
                # Pre-calculate attenuation at the designated frequencies to
                # save on heavy computation time of the attenuation method
                freqs = scipy.fft.fftfreq(2*len(signal.times), d=signal.dt)
                if attenuation_interpolation is None:
                    freqs.sort()
                else:
                    logf_min = np.log10(np.min(freqs[freqs>0]))
                    logf_max = np.log10(np.max(freqs))
                    n_steps = int((logf_max - logf_min)
                                  / attenuation_interpolation)
                    if (logf_max-logf_min)%attenuation_interpolation:
                        n_steps += 1
                    logf = np.logspace(logf_min, logf_max, n_steps+1)
                    freqs = np.concatenate((-np.flipud(logf), [0], logf))
                atten_vals = self.attenuation(freqs)
                attenuation = lambda f: np.interp(f, freqs, atten_vals)
                new_signal.filter_frequencies(attenuation)
                return new_signal

        else:
            # Unit vectors perpendicular and parallel to plane of incidence
            # at the launching point
            u_s0 = normalize(np.cross(self.emitted_direction, [0, 0, 1]))
            u_p0 = normalize(np.cross(u_s0, self.emitted_direction))
            # Unit vector parallel to plane of incidence at the receiving point
            # (perpendicular vector stays the same)
            u_p1 = normalize(np.cross(u_s0, self.received_direction))

            if signal is None:
                return (u_s0, u_p1)

            else:
                # Amplitudes of s and p components
                pol_s = np.dot(polarization, u_s0)
                pol_p = np.dot(polarization, u_p0)
                # Fresnel reflectances of s and p components
                r_s, r_p = self.fresnel
                # Pre-calculate attenuation at the designated frequencies to
                # save on heavy computation time of the attenuation method
                freqs = scipy.fft.fftfreq(2*len(signal.times), d=signal.dt)
                if attenuation_interpolation is None:
                    freqs.sort()
                else:
                    logf_min = np.log10(np.min(freqs[freqs>0]))
                    logf_max = np.log10(np.max(freqs))
                    n_steps = int((logf_max - logf_min)
                                  / attenuation_interpolation)
                    if (logf_max-logf_min)%attenuation_interpolation:
                        n_steps += 1
                    logf = np.logspace(logf_min, logf_max, n_steps+1)
                    freqs = np.concatenate((-np.flipud(logf), [0], logf))
                atten_vals = self.attenuation(freqs)
                # Apply fresnel s and p coefficients in addition to attenuation
                attenuation_s = lambda f: np.interp(f, freqs, atten_vals) * r_s
                attenuation_p = lambda f: np.interp(f, freqs, atten_vals) * r_p
                signal_s = signal * pol_s
                signal_p = signal * pol_p
                signal_s.shift(self.tof)
                signal_p.shift(self.tof)
                signal_s.filter_frequencies(attenuation_s, force_real=True)
                signal_p.filter_frequencies(attenuation_p, force_real=True)
                return (signal_s, signal_p), (u_s0, u_p1)

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
                                  integrand_kwargs={}, numerical=False, dz=None,
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
        integrand_kwargs : dict, optional
            A dictionary of keyword arguments to be passed into the `integrand`
            function.
        numerical : boolean, optional
            Whether to use the numerical integral instead of an analytic one.
            If ``False`` the analytic integral is calculated. If ``True`` the
            numerical integral is calculated.
        dz : float, optional
            The z-step to use for numerical integration. Only needed when
            `numerical` is ``True``.
        derivative_special_case : boolean, optional
            Boolean controlling whether the special case of doing the distance
            integral beta derivative should be used.

        Returns
        -------
        Integral of the given `integrand` along the path from `z0` to `z1`.

        """
        # Suppress numpy RuntimeWarnings
        with np.errstate(divide='ignore', invalid='ignore'):
            if numerical:
                if dz is None:
                    raise ValueError("Argument dz must be specified for "+
                                     "numerical integrals")
                if (z0<z_uniform)==(z1<z_uniform):
                    # z0 and z1 on same side of z_uniform
                    n_zs = int(np.abs(z1-z0)/dz)
                    if n_zs<10:
                        n_zs = 10
                    zs = np.linspace(z0, z1, n_zs+1)
                    return integrand(zs, beta=beta, ice=ice, deep=z0<z_uniform,
                                     **integrand_kwargs)
                else:
                    n_zs_1 = int(np.abs(z_uniform-z0)/dz)
                    if n_zs_1<10:
                        n_zs_1 = 10
                    zs_1 = np.linspace(z0, z_uniform, n_zs_1+1)
                    n_zs_2 = int(np.abs(z1-z_uniform)/dz)
                    if n_zs_2<10:
                        n_zs_2 = 10
                    zs_2 = np.linspace(z_uniform, z1, n_zs_2+1)
                    return (integrand(zs_1, beta=beta, ice=ice,
                                      deep=z0<z_uniform,
                                      **integrand_kwargs) +
                            integrand(zs_2, beta=beta, ice=ice,
                                      deep=z1<z_uniform,
                                      **integrand_kwargs))

            # Analytic integrals
            int_z0 = integrand(z0, beta, ice, deep=z0<z_uniform,
                                **integrand_kwargs)
            int_z1 = integrand(z1, beta, ice, deep=z1<z_uniform,
                                **integrand_kwargs)
            if not derivative_special_case:
                if (z0<z_uniform)==(z1<z_uniform):
                    # z0 and z1 on same side of z_uniform
                    return int_z1 - int_z0
                else:
                    int_diff = (
                        integrand(z_uniform, beta, ice, deep=True,
                                  **integrand_kwargs) -
                        integrand(z_uniform, beta, ice, deep=False,
                                  **integrand_kwargs)
                    )
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
                    int_diff = (
                        integrand(z_uniform, beta, ice, deep=True,
                                  **integrand_kwargs) -
                        integrand(z_uniform, beta, ice, deep=False,
                                  **integrand_kwargs)
                    )
                    if (z0<z_uniform)==(z1<z_uniform):
                        # z0 and z1 below z_uniform, but z_turn above
                        return int_z0 + int_z1 - 2*int_diff
                    else:
                        # z0 or z1 below z_uniform, others above
                        return int_z0 + int_z1 - int_diff



    def z_integral(self, integrand, integrand_kwargs={}, numerical=False):
        """
        Calculate the integral of the given integrand.

        For the integrand as a function of z, the analytic or numerical
        integral is calculated along the ray path.

        Parameters
        ----------
        integrand : function
            Function returning the values of the integrand at a given array of
            values for the depth z.
        integrand_kwargs : dict, optional
            A dictionary of keyword arguments to be passed into the `integrand`
            function.
        numerical : boolean, optional
            Whether to use the numerical integral instead of an analytic one.
            If ``False`` the analytic integral is calculated. If ``True`` the
            numerical integral is calculated.

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
        if not self.valid_ice_model:
            raise TypeError("Ice model must inherit methods from "+
                            "pyrex.AntarcticIce")
        if self.direct:
            return self._z_int_uniform_correction(self.z0, self.z1,
                                                  self.z_uniform,
                                                  self.beta, self.ice,
                                                  integrand, integrand_kwargs,
                                                  numerical, self.dz)
        else:
            int_1 = self._z_int_uniform_correction(self.z0, self.z_turn,
                                                   self.z_uniform,
                                                   self.beta, self.ice,
                                                   integrand, integrand_kwargs,
                                                   numerical, self.dz)
            int_2 = self._z_int_uniform_correction(self.z1, self.z_turn,
                                                   self.z_uniform,
                                                   self.beta, self.ice,
                                                   integrand, integrand_kwargs,
                                                   numerical, self.dz)
            return int_1 + int_2

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
            The value of the indefinite integral derivative at `z`.

        """
        alpha, n_z, gamma, log_1, log_2 = cls._int_terms(z, beta, ice)
        z_turn = np.log((ice.n0-beta)/ice.k)/ice.a
        if deep:
            if z_turn<ice.valid_range[1]:
                return ((np.log((ice.n0-beta)/ice.k)/ice.a - z -
                         beta/(ice.a*(ice.n0-beta))) / np.sqrt(alpha))
            else:
                return -z / np.sqrt(alpha)
        else:
            if z_turn<ice.valid_range[1]:
                term_1 = ((1+beta**2/alpha)/np.sqrt(alpha) * 
                          (z + np.log(beta*ice.k/log_1) / ice.a))
                term_2 = -(beta**2+ice.n0*n_z) / (ice.a*alpha*np.sqrt(gamma))
            else:
                term_1 = -(1+beta**2/alpha)/np.sqrt(alpha)*(-z + np.log(log_1) /
                           ice.a)
                term_2 = -((beta*(np.sqrt(alpha)-np.sqrt(gamma)))**2 /
                           (ice.a*alpha*np.sqrt(gamma)*log_1))
                alpha, n_z, gamma, log_1, log_2 = cls._int_terms(ice.valid_range[1], beta, ice)
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
        between two z values gives the path length of the direct path between
        the z values.

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
        which between two z values gives the time of flight of the direct path
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
            return (ice.n0*(n_z+ice.n0*(ice.a*z-1))
                    / (ice.a*np.sqrt(alpha)*scipy.constants.c))
        else:
            return np.where(np.isclose(beta, 0, atol=cls.beta_tolerance),
                            ((n_z-ice.n0)/ice.a + ice.n0*z) / scipy.constants.c,
                            (((np.sqrt(gamma) + ice.n0*np.log(log_2) +
                               ice.n0**2*np.log(log_1)/np.sqrt(alpha))/ice.a) -
                             z*ice.n0**2/np.sqrt(alpha)) / scipy.constants.c)

    @classmethod
    def _attenuation_integral_def(cls, zs, f, beta, ice, deep=False):
        """
        Definite z-integral for calculating attenuation.

        Calculates the definite z-integral of sec(arcsin(beta/n(z)))/A(z,f),
        which between two z values gives the path length over attenuation length
        of the direct path between the z values.

        Parameters
        ----------
        zs : array_like
            (Negative-valued) depths (m) in the ice.
        f : array_like
            Frequencies (Hz) at which to calculate signal attenuation.
        beta : float
            ``beta`` value of the ray path.
        ice
            Ice model to be used for ray tracing.
        deep : boolean, optional
            Whether or not the integral is calculated in deep (uniform) ice.

        Returns
        -------
        array_like
            The value of the definite integral along `zs`.

        """
        fa = np.abs(f)

        if deep or np.isclose(beta, 0, atol=cls.beta_tolerance):
            int_var = zs
            partial_integrand = 1 / np.cos(np.arcsin(beta/ice.index(zs)))
        else:
            # When approaching z_turn, the usual integrand approaches infinity.
            # In that case make the change of variables below to fix it.
            # The assumption now is that z_turn is always above z_uniform,
            # which is valid for most realistic detector configurations.
            int_var = np.sqrt(1 - (beta/ice.index(zs))**2)
            partial_integrand = (ice.index(zs)**3 / beta**2 /
                                 (-ice.k*ice.a*np.exp(ice.a*zs)))

        alen = ice.attenuation_length(zs, fa)
        integrand = (partial_integrand / alen.T).T

        return np.trapz(integrand, x=int_var, axis=0)


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
        return np.exp(-np.abs(self.z_integral(
            self._attenuation_integral_def,
            integrand_kwargs={'f': f},
            numerical=True
        )))

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
    ice_model : optional
        The ice model used for the ray tracer.
    dz : float, optional
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

    def __init__(self, from_point, to_point, ice_model=ice, dz=1):
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
        This is the depth of the assumed launching point.

        """
        return min([self.from_point[2], self.to_point[2]])

    @property
    def z1(self):
        """
        Depth (m) of the higher endpoint.

        Ray tracing performed as if launching from lower point to higher point,
        since the only difference in the paths produced is a time reversal.
        This is the depth of the assumed receiving point.

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
        if not(self.ice.contains(self.from_point) and
               self.ice.contains(self.to_point)):
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


    def _get_launch_angle(self, r_function, min_angle=0, max_angle=np.pi/2):
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

        The second indirect ray is the indirect ray where the launch angle is
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
    ice_model : optional
        The ice model used for the ray tracer.
    dz : float, optional
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
        return self._direct_r(self.max_angle)

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

    def _indirect_r(self, angle, brent_arg=0, link_range=1e-6):
        """
        Calculate the r distance of the indirect ray for a given launch angle.

        Parameters
        ----------
        angle : float
            Launch angle (radians) of an indirect ray.
        brent_arg : float, optional
            Argument to subtract from the return value. Used for the brentq
            root finder to find a value other than zero.
        link_range : float, optional
            Angular range from `max_angle` over which the indirect ray distance
            is adjusted so that it linearly approaches the maximum direct ray
            distance at `max_angle`.

        Returns
        -------
        float
            Value of the radial distance integral minus the `brent_arg`.

        """
        z_turn = self.ice.depth_with_index(self.n0 * np.sin(angle))
        link_angle = self.max_angle - link_range
        if angle>link_angle:
            link_dist = (self._r_distance(link_angle, self.z0, z_turn) +
                         self._r_distance(link_angle, self.z1, z_turn))
            slope = (link_dist - self.direct_r_max) / link_range
            dist = self.direct_r_max + slope * (self.max_angle - angle)
            return dist - brent_arg
        else:
            dist = (self._r_distance(angle, self.z0, z_turn) +
                    self._r_distance(angle, self.z1, z_turn))
            return dist - brent_arg

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





class UniformRayTracePath(LazyMutableClass):
    """
    Class for representing a single ray solution in uniform ice.

    Stores parameters of the ray path through uniform ice. Most properties are
    lazily evaluated to save on computation time. If any attributes of the
    class instance are changed, the lazily-evaluated properties will be
    cleared.

    Parameters
    ----------
    parent_tracer : UniformRayTracer
        Ray tracer for which this path is a solution.
    launch_angle : float
        Launch angle (radians) of the ray path.
    reflections : int
        Number of reflections made by the ray path at boundaries of the ice.

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
    direct : boolean
        Whether the ray path is direct (does not reflect).
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
    UniformRayTracer : Class for calculating ray solutions in uniform ice.

    Notes
    -----
    Even more attributes than those listed are available for the class, but
    are mainly for internal use. These attributes can be found by exploring
    the source code.

    """
    def __init__(self, parent_tracer, launch_angle, reflections):
        self.from_point = parent_tracer.from_point
        self.to_point = parent_tracer.to_point
        self.theta0 = launch_angle
        self.ice = parent_tracer.ice
        self.direct = reflections==0
        self._reflections = reflections
        super().__init__()

    @lazy_property
    def _points(self):
        """Relevant points along the path."""
        if self.direct:
            return np.asarray([self.from_point, self.to_point])
        else:
            points = np.zeros((self._reflections+2, 3))
            points[0] = self.from_point
            dzs = []
            if self.theta0>0:
                initial_direction = 1
            elif self.theta0<0:
                initial_direction = -1
            else:
                raise ValueError("Invalid initial direction")
            if initial_direction==1:
                dzs.append(self.ice.valid_range[1]-self.z0)
            else:
                dzs.append(self.z0-self.ice.valid_range[0])
            size = self.ice.valid_range[1] - self.ice.valid_range[0]
            dzs.extend([size]*(self._reflections-1))
            final_direction = initial_direction * (-1)**self._reflections
            if final_direction==1:
                dzs.append(self.z1-self.ice.valid_range[0])
            else:
                dzs.append(self.ice.valid_range[1]-self.z1)
            drs = self.rho * np.asarray(dzs)/np.sum(dzs)
            rs = np.cumsum(drs)
            points[1:, 0] = rs * np.cos(self.phi)
            points[1:, 1] = rs * np.sin(self.phi)
            for i in range(self._reflections):
                dirn = ((initial_direction * (-1)**i)+1)//2
                points[i+1, 2] = self.ice.valid_range[dirn]
            points[-1] = self.to_point
            return points

    @property
    def valid_ice_model(self):
        """Whether the ice model being used is supported."""
        return isinstance(self.ice, UniformIce)

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
    def emitted_direction(self):
        """Direction in which ray is emitted."""
        if self.direct and np.array_equal(self.from_point, self.to_point):
            return np.array([0, 0, 1])
        return normalize(self._points[1] - self._points[0])

    @lazy_property
    def received_direction(self):
        """Direction ray is travelling when it is received."""
        if self.direct and np.array_equal(self.from_point, self.to_point):
            return np.array([0, 0, 1])
        return normalize(self._points[-1] - self._points[-2])

    @lazy_property
    def path_length(self):
        """Length (m) of the ray path."""
        if not self.valid_ice_model:
            raise TypeError("Ice model must be uniform ice")
        return np.sum([np.sqrt(np.sum((p2-p1)**2))
                       for p1, p2 in zip(self._points[:-1], self._points[1:])])

    @lazy_property
    def tof(self):
        """Time of flight (s) along the ray path."""
        return self.n0 * self.path_length / scipy.constants.c

    @lazy_property
    def fresnel(self):
        """
        Fresnel factors for reflections off the ice boundaries.

        The fresnel reflectances are calculated as the square root (ratio of
        amplitudes, not powers). Stores the s and p polarized factors,
        respectively.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must be uniform ice")
        r_s = 1
        r_p = 1
        n_1 = self.n0
        if len(self._points)<3:
            return r_s, r_p
        for p1, p2 in zip(self._points[:-2], self._points[1:-1]):
            if p2[2]==self.ice.valid_range[0]:
                n_2 = self.ice.index_below
            elif p2[2]==self.ice.valid_range[1]:
                n_2 = self.ice.index_above
            else:
                raise ValueError("Intermediate points don't reflect off the "+
                                 "ice boundaries")
            dr = np.sqrt(np.sum((p2[:2]-p1[:2])**2))
            dz = np.abs(p2[2]-p1[2])
            theta_1 = np.arctan(dr/dz)
            cos_1 = np.cos(theta_1)
            sin_2 = n_1/n_2*np.sin(theta_1)
            if sin_2<=1:
                cos_2 = np.sqrt(1 - (sin_2)**2)
            else:
                cos_2 = np.sqrt((sin_2)**2 - 1)*1j
            # TODO: Confirm sign convention here
            r_s *= (n_1*cos_1 - n_2*cos_2) / (n_1*cos_1 + n_2*cos_2)
            r_p *= (n_2*cos_1 - n_1*cos_2) / (n_2*cos_1 + n_1*cos_2)
        return r_s, r_p

    def attenuation(self, f, dz=1):
        """
        Calculate the attenuation factor for signal frequencies.

        Calculates the attenuation factor to be multiplied by the signal
        amplitude at the given frequencies.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz) at which to calculate signal attenuation.
        dz : float, optional
            Step size in z to divide the ice. Actual step size will not be
            exactly this value, but is guaranteed to be less than the given
            value.

        Returns
        -------
        array_like
            Attenuation factors for the signal at the frequencies `f`.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must be uniform ice")
        fa = np.abs(f)
        attens = np.ones(fa.shape)
        for p1, p2 in zip(self._points[:-1], self._points[1:]):
            if p1[2]==p2[2]:
                dp = np.sqrt(np.sum((p2-p1)**2))
                zs = np.array([p1[2]])
            else:
                dpdz = (p2-p1)/(p2[2]-p1[2])
                n_steps = int(np.abs(p2[2]-p1[2]) / dz) + 2
                zs, dz_true = np.linspace(p1[2], p2[2], n_steps,
                                          endpoint=False, retstep=True)
                dp = np.sqrt(np.sum((dpdz*dz_true)**2))
            alens = self.ice.attenuation_length(zs, fa)
            attens *= np.prod(np.exp(-dp/alens), axis=0)
        return attens

    def propagate(self, signal=None, polarization=None):
        """
        Propagate the signal with optional polarization along the ray path.

        Applies the frequency-dependent signal attenuation along the ray path
        and shifts the times according to the ray time of flight. Additionally
        provides the s and p polarization directions.

        Parameters
        ----------
        signal : Signal, optional
            ``Signal`` object to propagate.
        polarization : array_like, optional
            Vector representing the linear polarization of the `signal`.

        Returns
        -------
        tuple of Signal
            Tuple of ``Signal`` objects representing the s and p polarizations
            of the original `signal` attenuated along the ray path. Only
            returned if `signal` was not ``None``.
        tuple of ndarray
            Tuple of polarization vectors representing the s and p polarization
            directions of the `signal` at the end of the ray path. Only
            returned if `polarization` was not ``None``.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        if polarization is None:
            if signal is None:
                return

            else:
                new_signal = signal.copy()
                new_signal.shift(self.tof)
                new_signal.filter_frequencies(self.attenuation)
                return new_signal

        else:
            # Unit vectors perpendicular and parallel to plane of incidence
            # at the launching point
            u_s0 = normalize(np.cross(self.emitted_direction, [0, 0, 1]))
            u_p0 = normalize(np.cross(u_s0, self.emitted_direction))
            # Unit vector parallel to plane of incidence at the receiving point
            # (perpendicular vector stays the same)
            u_p1 = normalize(np.cross(u_s0, self.received_direction))

            if signal is None:
                return (u_s0, u_p1)

            else:
                # Amplitudes of s and p components
                pol_s = np.dot(polarization, u_s0)
                pol_p = np.dot(polarization, u_p0)
                # Fresnel reflectances of s and p components
                r_s, r_p = self.fresnel
                # Apply fresnel s and p coefficients in addition to attenuation
                attenuation_s = lambda freqs: self.attenuation(freqs) * r_s
                attenuation_p = lambda freqs: self.attenuation(freqs) * r_p
                signal_s = signal * pol_s
                signal_p = signal * pol_p
                signal_s.shift(self.tof)
                signal_p.shift(self.tof)
                signal_s.filter_frequencies(attenuation_s, force_real=True)
                signal_p.filter_frequencies(attenuation_p, force_real=True)
                return (signal_s, signal_p), (u_s0, u_p1)

    @lazy_property
    def coordinates(self):
        """
        x, y, and z-coordinates along the path.

        Coordinates are only calculated at ice layer boundaries, as the path
        is assumed to be straight within an ice layer.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must be uniform ice")
        xs = np.array([p[0] for p in self._points])
        ys = np.array([p[1] for p in self._points])
        zs = np.array([p[2] for p in self._points])
        return xs, ys, zs


class UniformRayTracer(LazyMutableClass):
    """
    Class for calculating ray solutions in uniform ice.

    Calculations performed using straight-line paths. Most properties are
    lazily evaluated to save on computation time. If any attributes of the
    class instance are changed, the lazily-evaluated properties will be
    cleared.

    Parameters
    ----------
    from_point : array_like
        Vector starting point of the ray path.
    to_point : array_like
        Vector ending point of the ray path.
    ice_model
        The ice model used for the ray tracer.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    ice
        The ice model used for the ray tracer.
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
    UniformRayTracePath : Class for representing a single ray solution in
                          uniform ice.

    Notes
    -----
    Even more attributes than those listed are available for the class, but
    are mainly for internal use. These attributes can be found by exploring
    the source code.

    """
    solution_class = UniformRayTracePath
    max_reflections = 0

    def __init__(self, from_point, to_point, ice_model):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model
        super().__init__()

    @property
    def valid_ice_model(self):
        """Whether the ice model being used is supported."""
        return isinstance(self.ice, UniformIce)

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
        """Index of refraction of the ice at the starting endpoint."""
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
    def exists(self):
        """
        Boolean of whether any paths exist between the endpoints.

        Paths are deemed invalid if at least one of the endpoints is outside of
        the allowed ice range.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must be uniform ice")
        return (self.ice.valid_range[0]<=self.z0<=self.ice.valid_range[1] and
                self.ice.valid_range[0]<=self.z1<=self.ice.valid_range[1])

    def _reflected_path(self, reflections, initial_direction):
        """
        Generate reflected path for given parameters.

        Path will have the given number of reflections and the given initial
        direction (+1 for upward, -1 for downward).

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must be uniform ice")
        if reflections<1:
            raise ValueError("Number of reflections must be one or larger")
        if (self.ice._index_above is None and
                (reflections>1 or initial_direction==1)):
            raise TypeError("Reflections not allowed off the upper surface")
        if (self.ice._index_below is None and
                (reflections>1 or initial_direction==-1)):
            raise TypeError("Reflections not allowed off the lower surface")
        points = np.zeros((reflections+2, 3))
        points[0] = self.from_point
        dzs = []
        if initial_direction==1:
            dzs.append(self.ice.valid_range[1]-self.z0)
        elif initial_direction==-1:
            dzs.append(self.z0-self.ice.valid_range[0])
        else:
            raise ValueError("Invalid initial direction")
        size = self.ice.valid_range[1] - self.ice.valid_range[0]
        dzs.extend([size]*(reflections-1))
        final_direction = initial_direction * (-1)**reflections
        if final_direction==1:
            dzs.append(self.z1-self.ice.valid_range[0])
        else:
            dzs.append(self.ice.valid_range[1]-self.z1)
        theta = np.arctan2(initial_direction*np.sum(dzs), self.rho)
        return self.solution_class(self, theta, reflections)


    @lazy_property
    def solutions(self):
        """
        List of existing rays between the two points.

        This list should have zero elements if there are no possible paths
        between the endpoints or up to 2N+1 elements otherwise, where N is the
        number of reflections allowed. If the ice model has no index above or
        below, the number of solutions may be reduced since reflections off
        no index will be disallowed.

        """
        if not self.exists:
            return []
        # Direct path
        sols = [
            self.solution_class(self, np.arctan2(self.z1-self.z0, self.rho), 0)
        ]
        # Reflected paths
        for ref in range(1, self.max_reflections+1):
            for direction in [1, -1]:
                try:
                    sols.append(self._reflected_path(ref, direction))
                except TypeError as e:
                    logger.info("Reflected path solution skipped due to %s",
                                str(e).lower())
        return sols





# Set preferred ray tracer and path to specialized classes
RayTracer = SpecializedRayTracer
RayTracePath = SpecializedRayTracePath
