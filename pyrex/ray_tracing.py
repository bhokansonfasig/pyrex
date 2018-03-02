"""Module containing class for ray tracing through the ice."""

import numpy as np
from pyrex.internal_functions import (normalize, ConvergenceError,
                                      LazyMutableClass, lazy_property)
from pyrex.ice_model import IceModel



class RayTracePath(LazyMutableClass):
    """Class for storing a single ray-trace solution betwen points."""
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
        return self.dz/10

    @property
    def z0(self):
        return self.from_point[2]

    @property
    def z1(self):
        return self.to_point[2]

    @lazy_property
    def rho(self):
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def phi(self):
        u = self.to_point - self.from_point
        return np.arctan2(u[1], u[0])

    @lazy_property
    def z_turn(self):
        return self.ice.depth_with_index(self.ice.index(self.z0) *
                                         np.sin(self.theta0))

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
            return np.array([np.sin(self.theta(self.z1)) * np.cos(self.phi),
                             np.sin(self.theta(self.z1)) * np.sin(self.phi),
                             np.cos(self.theta(self.z1))])
        else:
            return np.array([np.sin(self.theta(self.z1)) * np.cos(self.phi),
                             np.sin(self.theta(self.z1)) * np.sin(self.phi),
                             -np.cos(self.theta(self.z1))])

    def theta(self, z):
        """Polar angle of the ray at given depth or array of depths."""
        return np.arcsin(np.sin(self.theta0) *
                         self.ice.index(self.z0)/self.ice.index(z))


    def z_integral(self, integrand):
        """Returns the integral of the integrand (a function of z) along
        the path."""
        if self.direct:
            n_zs = int(np.abs(self.z1-self.z0)/self.dz)
            zs, dz = np.linspace(self.z0, self.z1, n_zs+1, retstep=True)
            return np.trapz(integrand(zs), dx=np.abs(dz))
        else:
            n_zs_1 = int(np.abs(self.z_turn-self.z_turn_proximity-self.z0)/self.dz)
            zs_1, dz_1 = np.linspace(self.z0, self.z_turn-self.z_turn_proximity,
                                     n_zs_1+1, retstep=True)
            n_zs_2 = int(np.abs(self.z_turn-self.z_turn_proximity-self.z1)/self.dz)
            zs_2, dz_2 = np.linspace(self.z_turn-self.z_turn_proximity, self.z1,
                                     n_zs_2+1, retstep=True)
            return (np.trapz(integrand(zs_1), dx=np.abs(dz_1)) +
                    np.trapz(integrand(zs_2), dx=np.abs(dz_2)))

    @lazy_property
    def path_length(self):
        """Length of the path (m)."""
        return self.z_integral(lambda z: 1/np.cos(self.theta(z)))

    @lazy_property
    def tof(self):
        """Time of flight (s) along the path."""
        return self.z_integral(lambda z: self.ice.index(z) / 3e8 /
                               np.cos(self.theta(z)))

    def attenuation(self, f):
        """Returns the attenuation factor for a signal of frequency f (Hz)
        traveling along the path. Supports passing a list of frequencies."""
        fa = np.abs(f)
        return np.exp(-self.z_integral(lambda z: 1 / np.cos(self.theta(z)) /
                                       self.ice.attenuation_length(z, fa)))

    def propagate(self, signal):
        """Applies attenuation to the signal along the path."""
        signal.filter_frequencies(self.attenuation)
        signal.times += self.tof

    @lazy_property
    def coordinates(self):
        if self.direct:
            n_zs = int(np.abs(self.z1-self.z0)/self.dz)
            zs, dz = np.linspace(self.z0, self.z1, n_zs+1, retstep=True)
            integrand = np.tan(self.theta(zs))

            rs = np.zeros(len(integrand))
            trap_areas = (integrand[:-1] + np.diff(integrand)/2) * dz
            rs[1:] += np.abs(np.cumsum(trap_areas))

        else:
            n_zs_1 = int(np.abs(self.z_turn-self.z_turn_proximity-self.z0)/self.dz)
            zs_1, dz_1 = np.linspace(self.z0, self.z_turn-self.z_turn_proximity,
                                     n_zs_1+1, retstep=True)
            integrand_1 = np.tan(self.theta(zs_1))
            n_zs_2 = int(np.abs(self.z_turn-self.z_turn_proximity-self.z1)/self.dz)
            zs_2, dz_2 = np.linspace(self.z_turn-self.z_turn_proximity, self.z1,
                                     n_zs_2+1, retstep=True)
            integrand_2 = np.tan(self.theta(zs_2))

            rs_1 = np.zeros(len(integrand_1))
            trap_areas = (integrand_1[:-1] + np.diff(integrand_1)/2) * np.abs(dz_1)
            rs_1[1:] += np.cumsum(trap_areas)

            rs_2 = np.zeros(len(integrand_2)) + rs_1[-1]
            trap_areas = (integrand_2[:-1] + np.diff(integrand_2)/2) * np.abs(dz_2)
            rs_2[1:] += np.cumsum(trap_areas)

            rs = np.concatenate((rs_1, rs_2[1:]))
            zs = np.concatenate((zs_1, zs_2[1:]))

        xs = self.from_point[0] + rs*np.cos(self.phi)
        ys = self.from_point[1] + rs*np.sin(self.phi)

        return xs, ys, zs



class RayTracer(LazyMutableClass):
    """Class for proper ray tracing. Most properties lazily evaluated to save
    on re-computation time."""
    solution_class = RayTracePath

    def __init__(self, from_point, to_point, ice_model=IceModel, dz=0.001):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model
        self.dz = dz
        super().__init__()

    @property
    def z_turn_proximity(self):
        return self.dz/10

    # Calculations performed launching from low to high
    # (better for numerical error)
    @property
    def z0(self):
        return min([self.from_point[2], self.to_point[2]])

    @property
    def z1(self):
        return max([self.from_point[2], self.to_point[2]])

    @lazy_property
    def n0(self):
        return self.ice.index(self.z0)

    @lazy_property
    def rho(self):
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def max_angle(self):
        return np.arcsin(self.ice.index(self.z1)/self.n0)

    @lazy_property
    def peak_angle(self):
        peak_angle = None
        tolerance = self.dz
        while peak_angle is None:
            for angle_step in np.logspace(-3, 0, num=4):
                r_func = lambda angle: self._indirect_r_prime(angle, angle_step)
                try:
                    # FIXME: Is min_angle=0.1 necessary?
                    peak_angle = self.angle_search(0, r_func,
                                                   0.1, self.max_angle,
                                                   tolerance=tolerance)
                except ConvergenceError:
                    continue
                else:
                    if peak_angle>np.pi/2:
                        peak_angle = np.pi - peak_angle
                    return peak_angle
            tolerance *= 10
            if tolerance>np.abs(self.z1-self.z0):
                raise ConvergenceError("peak_angle calculation failed to converge even for exceptionally high tolerance")

    @lazy_property
    def direct_r_max(self):
        z_turn = self.ice.depth_with_index(self.n0 * np.sin(self.max_angle))
        return self._direct_r(self.max_angle,
                              force_z1=z_turn-self.z_turn_proximity)

    @lazy_property
    def indirect_r_max(self):
        return self._indirect_r(self.peak_angle)

    @lazy_property
    def exists(self):
        return np.any(self.expected_solutions)

    @lazy_property
    def expected_solutions(self):
        if self.rho<self.direct_r_max:
            return [True, False, True]
        elif self.rho<self.indirect_r_max:
            return [False, True, True]
        else:
            return [False, False, False]

    @lazy_property
    def solutions(self):
        """Calculate existing rays between the two points."""
        angles = [
            self.direct_angle,
            self.indirect_angle_1,
            self.indirect_angle_2
        ]

        return [self.solution_class(self, angle, direct=(i==0))
                for i, angle, exists in zip(range(3), angles,
                                            self.expected_solutions)
                if exists and angle is not None]


    def _direct_r(self, angle, force_z1=None):
        if force_z1 is not None:
            z1 = force_z1
        else:
            z1 = self.z1
        n_zs = int(np.abs((z1-self.z0)/self.dz))
        zs, dz = np.linspace(self.z0, z1, n_zs+1, retstep=True)
        integrand = np.tan(np.arcsin(np.sin(angle) *
                                     self.n0/self.ice.index(zs)))
        return np.trapz(integrand, dx=dz)

    def _indirect_r(self, angle):
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
                np.trapz(integrand_2, dx=-dz_2))

    def _indirect_r_prime(self, angle, d_angle=0.001):
        return (self._indirect_r(angle+d_angle) -
                self._indirect_r(angle)) / d_angle


    def _get_launch_angle(self, r_function, min_angle=0, max_angle=90):
        launch_angle = None
        tolerance = self.dz
        while launch_angle is None:
            try:
                launch_angle = self.angle_search(self.rho, r_function,
                                                 min_angle, max_angle,
                                                 tolerance=tolerance)
            except ConvergenceError:
                tolerance *= 10
                if tolerance>np.abs(self.z1-self.z0):
                    raise ConvergenceError("launch_angle calculation failed to converge even for exceptionally high tolerance")

        # Convert to true launch angle from self.from_point
        # rather than from lower point (self.z0)
        return np.arcsin(np.sin(launch_angle) *
                         self.n0 / self.ice.index(self.from_point[2]))


    @lazy_property
    def direct_angle(self):
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
        if self.expected_solutions[1]:
            return self._get_launch_angle(self._indirect_r,
                                          min_angle=self.peak_angle,
                                          max_angle=self.max_angle)
        else:
            return None

    @lazy_property
    def indirect_angle_2(self):
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
    def binary_angle_search(true_r, r_function, min_angle, max_angle,
                            angle_guess=None, tolerance=0.001,
                            max_iterations=100):
        angle = angle_guess if angle_guess is not None else (max_angle + min_angle)/2

        r = r_function(angle) - true_r
        r_min = r_function(min_angle) - true_r

        i = 0
        prev_angle = None
        while not np.isclose(r, 0, atol=tolerance, rtol=0) and angle!=prev_angle:
            prev_angle = angle
            i += 1
            if i>=max_iterations:
                raise ConvergenceError("Didn't converge fast enough")

            if (r<=0)==(r_min<=0):
                min_angle = angle
                r_min = r
            else:
                max_angle = angle

            angle = (max_angle + min_angle) / 2
            r = r_function(angle) - true_r

        return angle

    # Default angle search method
    angle_search = binary_angle_search



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
