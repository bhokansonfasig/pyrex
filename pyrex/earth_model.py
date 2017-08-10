"""Module containing earth model. Uses PREM for density as a function of radius
and a simple integrator for calculation of the slant depth as a function of
nadir angle."""

import numpy as np

EARTH_RADIUS = 6371e3

def prem_density(r):
    """Returns the earth's density (g/cm^3) for a given radius r (m).
    Calculated by the Preliminary Earth Model (PREM)."""
    x = r/EARTH_RADIUS
    if (r < 1221.5E3):
        return 13.0885 - 8.8381 * (x**2)
    elif (r < 3480.0E3):
        return 12.5815 - x * (1.2638 + x * (3.6426 + x * 5.5281))
    elif (r < 5701.0E3):
        return  7.9565 - x * (6.4761 - x * (5.5283 - x * 3.0807))
    elif (r < 5771.0E3):
        return  5.3197 - x * 1.4836
    elif (r < 5971.0E3):
        return 11.2494 - x * 8.0298
    elif (r < 6151.0E3):
        return  7.1089 - x * 3.8045
    elif (r < 6346.6E3):
        return  2.691  + x * 0.6924
    elif (r < 6356.0E3):
        return 2.9
    elif (r < 6368.0E3):
        return 2.6
    elif (r < EARTH_RADIUS):
        return 1.02
    else:
        return 0.0

def slant_depth(angle, depth, step=5000):
    """Returns the  material thickness (g/cm^2) for a chord cutting through
    earth at Nadir angle and starting at depth (m)."""
    p = np.array((0.0, EARTH_RADIUS - depth), 'd')
    d = np.array((np.sin(angle), -np.cos(angle)), 'd')
    t = 0.0
    r = np.hypot(*p)
    while r <= EARTH_RADIUS:
        rho = prem_density(r)
        ds  = step * rho * 100.
        t  += ds
        p  += step * d
        r   = np.hypot(*p)
    return t