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

    def time_of_flight(self, n_steps=10):
        """Time of flight (s) for a particle along the path."""
        t = 0
        z0 = self.from_point[2]
        z1 = self.to_point[2]
        u = self.to_point - self.from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dz = z1 - z0
        drdz = rho / dz
        dz /= n_steps
        for i in range(n_steps):
            z = z0 + (i+0.5)*dz
            dr = drdz * dz
            p = np.sqrt(dr**2 + dz**2)
            t += p / 3e8 * self.ice.index(z)
        return t

    def attenuation(self, f, n_steps=10):
        """Returns the attenuation factor for a signal of frequency f (MHz)
        traveling along the path."""
        atten = 1
        z0 = self.from_point[2]
        z1 = self.to_point[2]
        u = self.to_point - self.from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dz = z1 - z0
        drdz = rho / dz
        dz /= n_steps
        for i in range(n_steps):
            z = z0 + (i+0.5)*dz
            dr = drdz * dz
            p = np.sqrt(dr**2 + dz**2)
            alen = self.ice.attenuation_length(z, f)
            atten *= np.exp(-p/alen)
        return atten

    def propagate(self, signal):
        """Applies attenuation to the signal along the path."""
        signal *= 1 / self.path_length
        signal.filter_frequencies(self.attenuation)
        signal.times += self.tof
