"""Module for the simulation kernel. Includes neutrino generation,
ray tracking (no raytracing yet), and hit generation."""

import numpy as np
import scipy.fftpack
from pyrex.signals import Signal, AskaryanSignal
from pyrex.ray_tracing import PathFinder


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
        pulse = AskaryanSignal(times=times, energy=energy*1e-3, theta=angle,
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

            times = np.linspace(-10e-9, 40e-9, 2048, endpoint=False)
            pulse = AskaryanSignal(times=times, energy=p.energy, theta=psi, n=n)

            pf.propagate(pulse)

            ant.receive(pulse, epol)

        return p
