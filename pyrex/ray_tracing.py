"""Module containing class for ray tracking through the ice.
Ray tracing not yet implemented."""

import numpy as np

class PathFinder:
    """Class for ray tracking."""
    def __init__(self, ice_model, from_point, to_point):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model

    @property
    def exists(self):
        """Boolean of whether path exists."""
        ni = self.ice.index(self.from_point[2])
        nf = self.ice.index(self.to_point[2])
        nr = nf / ni
        if nr > 1:
            return True
        tir = np.sqrt(1 - nr**2)
        return self.emitted_ray[2] > tir

    @property
    def emitted_ray(self):
        """Direction in which ray is emitted."""
        r = self.to_point - self.from_point
        return r / np.linalg.norm(r)

    @property
    def path_length(self):
        """Length of the path (m)."""
        r = self.to_point - self.from_point
        return np.linalg.norm(r)

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
        zs, dz = np.linspace(z0, z1, n_steps, endpoint=True, retstep=True)
        u = self.to_point - self.from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dr = rho / (z1 - z0) * dz
        dp = np.sqrt(dz**2 + dr**2)
        alens = self.ice.attenuation_length(zs, fa*1e-6)
        attens = np.exp(-dp/alens)
        return np.prod(attens, axis=0)

    def propagate(self, signal):
        """Applies attenuation to the signal along the path."""
        if not self.exists:
            raise RuntimeError("Cannot propagate signal along a path that "+
                               "doesn't exist")
        signal.filter_frequencies(self.attenuation)
        signal.times += self.tof
