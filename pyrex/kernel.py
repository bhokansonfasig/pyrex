"""Module for the simulation kernel. Includes neutrino generation,
ray tracking (no raytracing yet), and hit generation."""

import numpy as np
import scipy.fftpack
from pyrex.digsig import Signal, AskaryanSignal

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

    def propagate_ray(self, f, n_step=10):
        """Returns the attenuation factor and time of flight (ns) for a signal
        of frequency f (MHz) traveling along the path."""
        atten = 1
        tof = 0
        z0 = self.from_point[2]
        z1 = self.to_point[2]
        u = self.to_point - self.from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dz = z1 - z0
        drdz = rho / dz
        dz /= n_step
        for i in range(n_step):
            z = z0 + (i+0.5)*dz
            dr = drdz * dz
            p = np.sqrt(dr**2 + dz**2)
            alen = self.ice.attenuation_length(z, f)
            atten *= np.exp(-p/alen)
            tof += p / .3 / self.ice.index(z)
        return atten, tof


def pulse_at_antenna(energy, angle, path, n_ice=1.78):
    """Function to propagate Askaryan pulse from particle with given energy
    (GeV) along the path (at specified angle from particle direction).
    Takes dispersion of ice into account based on n_ice."""
    if not(path.exists()):
        raise ValueError("Path to antenna does not exist")

    t0 = 75e-9
    tmax = 250e-9
    times = np.linspace(0,tmax,2048,endpoint=False)
    try:
        pulse = AskaryanSignal(times=times, theta=angle, energy=energy*1e-3,
                               n=n_ice, t0=t0)
    except ValueError:
        raise
    # Apply 1/R effect
    pulse.values /= path.getPathLength()

    famps = scipy.fftpack.fft(pulse.values)
    n_fft = len(famps)
    freqs = scipy.fftpack.fftfreq(n=n_fft, d=pulse.dt)
    # Apply attenuation
    for i, f in enumerate(freqs):
        if f==0:
            continue
        elif f>0:
            atten, tof = path.propagateRay(f*1e-6)
        else:
            # Special case when n is even, the frequency at the halfway point
            # doesn't have a match
            if i==n_fft/2:
                atten, tof = path.propagateRay(-f*1e-6)
                famps[i] *= atten
            break
        famps[i] *= atten
        # Do the same for the negative frequency
        # (saves time instead of calculating same atten twice)
        famps[n_fft-i] *= atten
    vals = np.real(scipy.fftpack.ifft(famps))

    # Transform the times array to the proper time values
    times += tof*1e-9 - t0
    return Signal(times,vals), tof


class EventKernel:
    """Kernel for generation of events with a given particle generator,
    ice model, and list of antennas."""
    def __init__(self, generator, ice_model, antennas):
        self.gen = generator
        self.ice = ice_model
        self.ant_array = antennas

    def event(self):
        """Generate particle, propagate signal through ice to antennas,
        process signal at antennas, and return the original particle."""
        p = self.gen.create_particle()
        n = self.ice.index(p.vertex[2])
        for ant in self.ant_array:
            pf = PathFinder(self.ice, p.vertex, ant.pos)
            if not(pf.exists):
                continue
            k = pf.emitted_ray
            epol = np.vdot(k, p.direction) * k - p.direction
            epol = epol / np.linalg.norm(epol)
            # p.direction and k should both be unit vectors
            psi = np.arccos(np.vdot(p.direction, k))

            try:
                askaryan_pulse, hit_time = pulse_at_antenna(p.energy, psi, pf, n)
            except ValueError:
                pass
            else:
                ant.receive(askaryan_pulse, hit_time, epol)
        return p

