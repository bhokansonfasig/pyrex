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
    """Class for storing a single ray-trace solution betwen points.
    Calculations preformed by integrating z-steps of size dz.
    Most properties lazily evaluated to save on re-computation time."""
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
        """Parameter for how closely path approaches z_turn.
        Necessary to avoid diverging integrals."""
        # Best value of dz/10 determined empirically by checking errors
        return self.dz/10

    @property
    def z0(self):
        """Depth of the launching point."""
        return self.from_point[2]

    @property
    def z1(self):
        """Depth of the receiving point."""
        return self.to_point[2]

    @lazy_property
    def n0(self):
        """Index of refraction of the ice at the launching point."""
        return self.ice.index(self.z0)

    @lazy_property
    def rho(self):
        """Radial distance between the launching and receiving points."""
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def phi(self):
        """Azimuthal angle between the launching and receiving points."""
        u = self.to_point - self.from_point
        return np.arctan2(u[1], u[0])

    @lazy_property
    def beta(self):
        """Launching beta parameter (n(z0) * sin(theta0))."""
        return self.n0 * np.sin(self.theta0)

    @lazy_property
    def z_turn(self):
        """Turning depth of the path."""
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
        """Polar angle of the ray at given depth or array of depths."""
        return np.arcsin(np.sin(self.theta0) * self.n0/self.ice.index(z))


    # Log-scaled zs (commented out below and in z_integral method) seemed
    # like a good idea for reducing dimentionality, but didn't work out.
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
        """Returns the integral of the integrand (a function of z) along
        the path."""
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
        """Length of the path (m)."""
        return self.z_integral(lambda z: 1/np.cos(self.theta(z)))

    @lazy_property
    def tof(self):
        """Time of flight (s) along the path."""
        return self.z_integral(lambda z: self.ice.index(z) / 3e8 /
                               np.cos(self.theta(z)))

    @lazy_property
    def fresnel(self):
        """Fresnel reflectance square root (ratio of amplitudes, not powers)
        for reflection off ice surface (1 if doesn't reach surface)."""
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
        """Returns the attenuation factor for a signal of frequency f (Hz)
        traveling along the path. Supports passing a list of frequencies."""
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
        """Applies attenuation to the signal along the path."""
        signal.filter_frequencies(self.attenuation)
        signal.times += self.tof

    @lazy_property
    def coordinates(self):
        """x, y, and z-coordinates along the path (using dz step)."""
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
    """Class for storing a single ray-trace solution betwen points,
    specifically for ice model with index of refraction
    n(z) = n0 - k*exp(a*z). Calculations performed using true integral
    evaluation (except attenuation). Ice model must use methods inherited from
    pyrex.AntarcticIce"""
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
        """Depth beyond which the ice should be treated as uniform.
        Necessary due to numerical rounding issues."""
        return self.ice.depth_with_index(self.ice.n0 * self.uniformity_factor)

    @staticmethod
    def _z_int_uniform_correction(z0, z1, z_uniform, beta, ice, integrand,
                                  derivative_special_case=False):
        """Function for performing z-integration, taking into account the
        effect of treating the ice as uniform beyond some depth."""
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
        """Function for integrating a given integrand along the depths of
        the path."""
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
        """Useful pre-calculated substitutions for integrations."""
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
        """Indefinite z-integral of tan(arcsin(beta/n(z))), which between
        two z values gives the radial distance of the direct path between the
        z values."""
        alpha, n_z, gamma, log_1, log_2 = cls._int_terms(z, beta, ice)
        if deep:
            return beta * z / np.sqrt(alpha)
        else:
            return np.where(np.isclose(beta, 0, atol=cls.beta_tolerance),
                            0,
                            beta / np.sqrt(alpha) * (-z + np.log(log_1)/ice.a))

    @classmethod
    def _distance_integral_derivative(cls, z, beta, ice, deep=False):
        """Beta derivative of the distance integral for finding maximum
        distance integral value as a function of launch angle.
        This function actually gives the integral from z to the turning point
        z_turn=ice.depth_with_index(beta), since that is what's needed for
        finding the peak angle."""
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
        """Indefinite z-integral of sec(arcsin(beta/n(z))), which between
        two z values gives the path length of the direct path between the
        z values."""
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
        """Indefinite z-integral of n(z)/c*sec(arcsin(beta/n(z))), which between
        two z values gives the time of flight of the direct path between the
        z values."""
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
        """Length of the path (m)."""
        return np.abs(self.z_integral(self._pathlen_integral))

    @lazy_property
    def tof(self):
        """Time of flight (s) along the path."""
        return np.abs(self.z_integral(self._tof_integral))

    def attenuation(self, f):
        """Returns the attenuation factor for a signal of frequency f (Hz)
        traveling along the path. Supports passing a list of frequencies."""
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
        """x, y, and z-coordinates along the path (using dz step)."""
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
    """Class for proper ray tracing. Calculations performed by integrating
    z-steps with size dz. Most properties lazily evaluated to save on
    re-computation time."""
    solution_class = BasicRayTracePath

    def __init__(self, from_point, to_point, ice_model=IceModel, dz=1):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model
        self.dz = dz
        super().__init__()

    @property
    def z_turn_proximity(self):
        """Parameter for how closely path approaches z_turn.
        Necessary to avoid diverging integrals."""
        # Best value of dz/10 determined empirically by checking errors
        return self.dz/10

    # Calculations performed as if launching from low to high
    @property
    def z0(self):
        """Depth of lower point. Ray tracing performed as if launching
        from lower point to higher point."""
        return min([self.from_point[2], self.to_point[2]])

    @property
    def z1(self):
        """Depth of higher point. Ray tracing performed as if launching
        from lower point to higher point."""
        return max([self.from_point[2], self.to_point[2]])

    @lazy_property
    def n0(self):
        """Index of refraction of the ice at the lower point."""
        return self.ice.index(self.z0)

    @lazy_property
    def rho(self):
        """Radial distance between the launching and receiving points."""
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def max_angle(self):
        """Maximum possible launch angle for rays between the points."""
        return np.arcsin(self.ice.index(self.z1)/self.n0)

    @lazy_property
    def peak_angle(self):
        """Angle at which indirect solutions curve (in r vs angle) peaks.
        Separates angle intervals for indirect solution root-finding."""
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
        """Boolean of whether any paths exist between the points."""
        return True in self.expected_solutions

    @lazy_property
    def expected_solutions(self):
        """List of which types of solutions are expected to exist.
        0: direct path, 1: indirect path > peak, 2: indirect path < peak."""
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
        """List of existing rays between the two points.
        Should have zero or two elements."""
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
        """Returns the r distance of the direct ray for the given launch
        angle."""
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
        """Returns the r distance of the indirect ray for the given launch
        angle."""
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
        """Returns the derivative of the r distance of the indirect ray for the
        given launch angle."""
        return ((self._indirect_r(angle) - self._indirect_r(angle-d_angle))
                / d_angle) - brent_arg


    def _get_launch_angle(self, r_function, min_angle=0, max_angle=90):
        """Calculates the launch angle by finding the root of the given
        r function."""
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
        """Launch angle of the direct ray."""
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
        """Launch angle of the indirect ray (where the launch angle is greater
        than the peak angle)."""
        if self.expected_solutions[1]:
            return self._get_launch_angle(self._indirect_r,
                                          min_angle=self.peak_angle,
                                          max_angle=self.max_angle)
        else:
            return None

    @lazy_property
    def indirect_angle_2(self):
        """Launch angle of the indirect ray (where the launch angle is less
        than the peak angle)."""
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
        """Root-finding algorithm."""
        return scipy.optimize.brentq(r_function, min_angle, max_angle,
                                     args=(true_r), xtol=tolerance,
                                     maxiter=max_iterations)



class SpecializedRayTracer(BasicRayTracer):
    """Ray tracer specifically for ice model with index of refraction
    n(z) = n0 - k*exp(a*z). Calculations performed using true integral
    evaluation. Ice model must use methods inherited from pyrex.AntarcticIce"""
    solution_class = SpecializedRayTracePath

    @lazy_property
    def valid_ice_model(self):
        """Whether the ice model being used supports this specialization."""
        return ((isinstance(self.ice, type) and
                 issubclass(self.ice, AntarcticIce))
                or isinstance(self.ice, AntarcticIce))

    @lazy_property
    def z_uniform(self):
        """Depth beyond which the ice should be treated as uniform.
        Necessary due to numerical rounding issues."""
        return self.ice.depth_with_index(self.ice.n0 *
                                         self.solution_class.uniformity_factor)

    @lazy_property
    def direct_r_max(self):
        """Maximum r value of direct ray solutions."""
        z_turn = self.ice.depth_with_index(self.n0 * np.sin(self.max_angle))
        return self._direct_r(self.max_angle, force_z1=z_turn)

    def _r_distance(self, theta, z0, z1):
        """Returns the r distance between given depths for given launch
        angle."""
        if not self.valid_ice_model:
            raise TypeError("Ice model must inherit methods from "+
                            "pyrex.AntarcticIce")
        beta = np.sin(theta) * self.n0
        return self.solution_class._z_int_uniform_correction(
            z0, z1, self.z_uniform, beta, self.ice,
            self.solution_class._distance_integral
        )

    def _r_distance_derivative(self, theta, z0, z1):
        """Returns the derivative of the r distance between given depths for
        given launch angle."""
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
        """Returns the r distance of the direct ray for the given launch
        angle."""
        if force_z1 is not None:
            z1 = force_z1
        else:
            z1 = self.z1
        return self._r_distance(angle, self.z0, z1) - brent_arg

    def _indirect_r(self, angle, brent_arg=0):
        """Returns the r distance of the indirect ray for the given launch
        angle."""
        z_turn = self.ice.depth_with_index(self.n0 * np.sin(angle))
        return (self._r_distance(angle, self.z0, z_turn) +
                self._r_distance(angle, self.z1, z_turn)) - brent_arg

    def _indirect_r_prime(self, angle, brent_arg=0):
        """Returns the derivative of the r distance of the indirect ray for the
        given launch angle."""
        return self._r_distance_derivative(angle, self.z0, self.z1) - brent_arg

    @lazy_property
    def peak_angle(self):
        """Angle at which indirect solutions curve (in r vs angle) peaks.
        Separates angle intervals for indirect solution root-finding."""
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
    """Class for pseudo ray tracing. Just uses straight-line paths."""
    def __init__(self, ice_model, from_point, to_point):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model

    @property
    def exists(self):
        """Boolean of whether path exists based on basic total internal
        reflection calculation."""
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
        """Direction in which ray is emitted."""
        return normalize(self.to_point - self.from_point)

    @property
    def received_ray(self):
        """Direction from which ray is received."""
        return self.emitted_ray

    @property
    def path_length(self):
        """Length of the path (m)."""
        return np.linalg.norm(self.to_point - self.from_point)

    @property
    def tof(self):
        """Time of flight (s) for a particle along the path.
        Calculated using default values of self.time_of_flight()"""
        return self.time_of_flight()

    def time_of_flight(self, n_steps=100):
        """Time of flight (s) for a particle along the path."""
        z0 = self.from_point[2]
        z1 = self.to_point[2]
        zs = np.linspace(z0, z1, n_steps, endpoint=True)
        u = self.to_point - self.from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        integrand = self.ice.index(zs)
        t = np.trapz(integrand, zs) / 3e8 * np.sqrt(1 + (rho / (z1 - z0))**2)
        return np.abs(t)

    def attenuation(self, f, n_steps=100):
        """Returns the attenuation factor for a signal of frequency f (Hz)
        traveling along the path. Supports passing a list of frequencies."""
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
        """Applies attenuation to the signal along the path."""
        if not self.exists:
            raise RuntimeError("Cannot propagate signal along a path that "+
                               "doesn't exist")
        signal.filter_frequencies(self.attenuation)
        signal.times += self.tof


class ReflectedPathFinder:
    """Class for pseudo ray tracing of ray reflected off ice surface.
    Just uses straight-line paths."""
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
        """Calculation of point at which signal is reflected by the ice surface
        (z=0)."""
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
        """Boolean of whether path exists based on whether its sub-paths
        exist and whether it could reflect off the ice surface."""
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
        """Direction in which ray is emitted."""
        return normalize(self.bounce_point - self.from_point)

    @property
    def received_ray(self):
        """Direction from which ray is received."""
        return normalize(self.to_point - self.bounce_point)

    @property
    def path_length(self):
        """Length of the path (m)."""
        return self.path_1.path_length + self.path_2.path_length

    @property
    def tof(self):
        """Time of flight (s) for a particle along the path.
        Calculated using default values of self.time_of_flight()"""
        return self.path_1.tof + self.path_2.tof

    def time_of_flight(self, n_steps=100):
        """Time of flight (s) for a particle along the path."""
        return (self.path_1.time_of_flight(n_steps) +
                self.path_2.time_of_flight(n_steps))

    def attenuation(self, f, n_steps=100):
        """Returns the attenuation factor for a signal of frequency f (Hz)
        traveling along the path. Supports passing a list of frequencies."""
        return (self.path_1.attenuation(f, n_steps) *
                self.path_2.attenuation(f, n_steps))

    def propagate(self, signal):
        """Applies attenuation to the signal along the path."""
        self.path_1.propagate(signal)
        self.path_2.propagate(signal)
