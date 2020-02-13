"""
Module containing earth model classes.

The earth models define density as a function of radius and provide a simple
integrator for calculation of the column density along a straight path through
the Earth.

"""

import logging
import numpy as np

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


    def slant_depth(self, angle, depth, step=500):
        """
        Calculates the column density of a chord cutting through Earth.

        Integrates the Earth's density along the chord, resulting in a column
        density (or material thickness) with units of mass per area.

        Parameters
        ----------
        angle : float
            Nadir angle (radians) of the chord's direction.
        depth : float
            (Positive-valued) depth (m) of the chord endpoint.
        step : float, optional
            Step size (m) for the integration.

        Returns
        -------
        float
            Column density (g/cm^2) along the chord starting from `depth` and
            passing through the Earth at `angle`.

        See Also
        --------
        PREM.density : Calculates the Earth's density at a given radius.

        """
        # Starting point (x0, z0)
        x0 = 0
        z0 = self.earth_radius - depth
        # Find exit point (x1, z1)
        if angle==0:
            x1 = 0
            z1 = -self.earth_radius
        else:
            m = -np.cos(angle) / np.sin(angle)
            a = z0-m*x0
            b = 1+m**2
            if angle<0:
                x1 = -m*a/b - np.sqrt(m**2*a**2/b**2
                                      - (a**2 - self.earth_radius**2)/b)
            else:
                x1 = -m*a/b + np.sqrt(m**2*a**2/b**2
                                      - (a**2 - self.earth_radius**2)/b)
            z1 = z0 + m*(x1-x0)

        # Parameterize line integral with t from 0 to 1, with steps just under the
        # given step size (in meters)
        l = np.sqrt((x1-x0)**2 + (z1-z0)**2)
        ts = np.linspace(0, 1, int(l/step)+2)
        xs = x0 + (x1-x0)*ts
        zs = z0 + (z1-z0)*ts
        rs = np.sqrt(xs**2 + zs**2)
        rhos = self.density(rs)
        x_int = np.trapz(rhos*(x1-x0), ts)
        z_int = np.trapz(rhos*(z1-z0), ts)
        return 100 * np.sqrt(x_int**2 + z_int**2)



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
