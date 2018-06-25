"""
Module containing earth model functions.

The earth model uses the Preliminary Earth Model (PREM) for density as a
function of radius and a simple integrator for calculation of the slant
depth along a straight path through the Earth.

"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

EARTH_RADIUS = 6371.0e3 # meters


def prem_density(r):
    """Returns the earth's density (g/cm^3) for a given radius r (m).
    Calculated by the Preliminary Earth Model (PREM).
    Supports passing a list of radii."""
    r = np.array(r)
    return np.piecewise(
        r/EARTH_RADIUS,
        (r < 1221.5e3,
         (1221.5e3 <= r) & (r < 3480.0e3),
         (3480.0e3 <= r) & (r < 5701.0e3),
         (5701.0e3 <= r) & (r < 5771.0e3),
         (5771.0e3 <= r) & (r < 5971.0e3),
         (5971.0e3 <= r) & (r < 6151.0e3),
         (6151.0e3 <= r) & (r < 6346.6e3),
         (6346.6e3 <= r) & (r < 6356.0e3),
         (6356.0e3 <= r) & (r < 6368.0e3),
         (6368.0e3 <= r) & (r < EARTH_RADIUS)),
        (lambda x: 13.0885 - 8.8381 * (x**2),
         lambda x: 12.5815 - x * (1.2638 + x * (3.6426 + x * 5.5281)),
         lambda x: 7.9565 - x * (6.4761 - x * (5.5283 - x * 3.0807)),
         lambda x: 5.3197 - x * 1.4836,
         lambda x: 11.2494 - x * 8.0298,
         lambda x: 7.1089 - x * 3.8045,
         lambda x: 2.691 + x * 0.6924,
         2.9,
         2.6,
         1.02,
         0) # Last value is used if no conditions are met (r >= EARTH_RADIUS)
    )


def slant_depth(angle, depth, step=500):
    """Returns the material thickness (g/cm^2) for a chord cutting through
    earth at Nadir angle and starting at (positive-valued) depth (m).
    Can optionally specify the step size (m)."""
    # Starting point (x0, z0)
    x0 = 0
    z0 = EARTH_RADIUS - depth
    # Find exit point (x1, z1)
    if angle==0:
        x1 = 0
        z1 = -EARTH_RADIUS
    else:
        m = -np.cos(angle) / np.sin(angle)
        a = z0-m*x0
        b = 1+m**2
        if angle<0:
            x1 = -m*a/b - np.sqrt(m**2*a**2/b**2 - (a**2 - EARTH_RADIUS**2)/b)
        else:
            x1 = -m*a/b + np.sqrt(m**2*a**2/b**2 - (a**2 - EARTH_RADIUS**2)/b)
        z1 = z0 + m*(x1-x0)

    # Parametrize line integral with t from 0 to 1, with steps just under the
    # given step size (in meters)
    l = np.sqrt((x1-x0)**2 + (z1-z0)**2)
    ts = np.linspace(0, 1, int(l/step)+2)
    xs = x0 + (x1-x0)*ts
    zs = z0 + (z1-z0)*ts
    rs = np.sqrt(xs**2 + zs**2)
    rhos = prem_density(rs)
    x_int = np.trapz(rhos*(x1-x0), ts)
    z_int = np.trapz(rhos*(z1-z0), ts)
    return 100 * np.sqrt(x_int**2 + z_int**2)
