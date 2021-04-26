"""
Module containing earth model classes.

The earth models define density as a function of radius and provide a simple
integrator for calculation of the column density along a straight path through
the Earth.

"""

import logging
import numpy as np
from pyrex.internal_functions import normalize

logger = logging.getLogger(__name__)


class PREM:
    """
    Class describing the Earth's density.

    Uses densities from the Preliminary reference Earth Model (PREM).

    Attributes
    ----------
    earth_radius : float
        Mean radius of the Earth (m).
    radii : tuple
        Boundary radii (m) at which the functional form of the density of the
        Earth changes. The density function in `densities` at index `i`
        corresponds to the radius range from radius at index `i-1` to radius
        at index `i`.
    densities : tuple
        Functions which calculate the density of the Earth (g/cm^3) in a
        specific radius range as described by `radii`. The parameter of each
        function is the fractional radius, e.g. radius divided by
        `earth_radius`. Scalar values denote constant density over the range of
        radii.

    Notes
    -----
    The density calculation is based on the Preliminary reference Earth Model
    [1]_.

    References
    ----------
    .. [1] A. Dziewonski & D. Anderson, "Preliminary reference Earth model."
        Physics of the Earth and Planetary Interiors **25**, 297–356 (1981). :doi:`10.1016/0031-9201(81)90046-7`

    """
    earth_radius = 6.3710e6

    radii = (1.2215e6, 3.4800e6, 5.7010e6, 5.7710e6, 5.9710e6,
             6.1510e6, 6.3466e6, 6.3560e6, 6.3680e6, earth_radius)

    densities = (
        lambda x: 13.0885 - 8.8381*x**2,
        lambda x: 12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3,
        lambda x: 7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3,
        lambda x: 5.3197 - 1.4836*x,
        lambda x: 11.2494 - 8.0298*x,
        lambda x: 7.1089 - 3.8045*x,
        lambda x: 2.691 + 0.6924*x,
        2.9,
        2.6,
        1.02
    )

    def density(self, r):
        """
        Calculates the Earth's density at a given radius.

        Supports passing an array of radii or a single radius.

        Parameters
        ----------
        r : array_like
            Radius (m) at which to calculate density.

        Returns
        -------
        array_like
            Density (g/cm^3) of the Earth at the given radii.

        """
        r = np.array(r)
        radius_bounds = np.concatenate(([0], self.radii))
        conditions = list((lower<=r) & (r<upper) for lower, upper in
                          zip(radius_bounds[:-1], radius_bounds[1:]))
        return np.piecewise(r/self.earth_radius, conditions, self.densities)


    def slant_depth(self, endpoint, direction, step=500):
        """
        Calculates the column density of a chord cutting through Earth.

        Integrates the Earth's density along the chord, resulting in a column
        density (or material thickness) with units of mass per area.

        Parameters
        ----------
        endpoint : array_like
            Vector position (m) of the chord endpoint, in a coordinate system
            centered on the surface of the Earth (e.g. a negative third
            coordinate represents the depth below the surface).
        direction : array_like
            Vector direction of the chord, in a coordinate system
            centered on the surface of the Earth (e.g. a negative third
            coordinate represents the chord pointing into the Earth).
        step : float, optional
            Step size (m) for the density integration.

        Returns
        -------
        float
            Column density (g/cm^2) along the chord starting at `endpoint` and
            passing through the Earth at the given `direction`.

        See Also
        --------
        PREM.density : Calculates the Earth's density at a given radius.

        """
        # Convert to Earth-centric coordinate system (e.g. center of the Earth
        # is at (0, 0, 0))
        endpoint = np.array([endpoint[0], endpoint[1],
                             endpoint[2]+self.earth_radius])
        direction = normalize(direction)
        dot_prod = np.dot(endpoint, direction)
        # Check for intersection of line and sphere
        discriminant = dot_prod**2 - np.sum(endpoint**2) + self.earth_radius**2
        if discriminant<=0:
            return 0
        # Calculate the distance at which the line intersects the sphere
        distance = -dot_prod + np.sqrt(discriminant)
        if distance<=0:
            return 0
        # Parameterize line integral with ts from 0 to 1, with steps just under
        # the given step size (in meters)
        n_steps = int(distance/step)
        if distance%step:
            n_steps += 1
        ts = np.linspace(0, 1, n_steps)
        xs = endpoint[0] + ts * distance * direction[0]
        ys = endpoint[1] + ts * distance * direction[1]
        zs = endpoint[2] + ts * distance * direction[2]
        rs = np.sqrt(xs**2 + ys**2 + zs**2)
        rhos = self.density(rs)
        # Integrate the density times the distance along the chord
        return 100 * np.trapz(rhos*distance, ts)



class CoreMantleCrustModel(PREM):
    """
    Class describing the Earth's density.

    Uses densities from the Core-Mantle-Crust model as implemented in AraSim.

    Attributes
    ----------
    earth_radius : float
        Mean radius of the Earth (m).
    radii : tuple
        Boundary radii (m) at which the functional form of the density of the
        Earth changes. The density function in `densities` at index `i`
        corresponds to the radius range from radius at index `i-1` to radius
        at index `i`.
    densities : tuple
        Functions which calculate the density of the Earth (g/cm^3) in a
        specific radius range as described by `radii`. The parameter of each
        function is the fractional radius, e.g. radius divided by
        `earth_radius`. Scalar values denote constant density over the range of
        radii.

    """
    earth_radius = 6.378140e6

    radii = (np.sqrt(1.2e13), earth_radius-4e4, earth_radius)

    densities = (14, 3.4, 2.9)



# Preferred earth model:
earth = PREM()
