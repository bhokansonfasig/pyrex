"""Module containing class for ray tracking through the ice.
Ray tracing not yet implemented."""

import numpy as np
from pyrex.internal_functions import normalize

class PathFinder:
    """Class for ray tracking."""
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
    """Class for ray tracking of ray reflected off ice surface."""
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
