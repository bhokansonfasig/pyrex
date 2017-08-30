"""Module containing ice model. AntarcticIce class contains static and class
methods for easy swapping of models. IceModel class is set to the preferred
ice model."""

import numpy as np

class AntarcticIce:
    """Class containing characteristics of ice at the south pole."""
    k = 0.438
    a = 0.0132
    n0 = 1.32
    thickness = 2850

    @classmethod
    def gradient(cls, z):
        """Returns the gradient of the index of refraction at depth z (m)."""
        return np.array([0.0, -cls.k * cls.a * np.exp(cls.a * z)])

    @classmethod
    def index(cls, z):
        """Returns the medium's index of refraction, n, at depth z (m)."""
        if z>0:
            return 1
        else:
            return cls.n0 + cls.k * (1 - np.exp(cls.a * z))

    @staticmethod
    def temperature(z):
        """Returns the temperature (K) of the ice at depth z (m)"""
        km = -0.001 * z
        c_temp = -51.07 + km*(2.677 + km*(-0.01591 + km*1.83415))
        return c_temp + 273.15

    @classmethod
    def attenuation_length(cls, z, f):
        """Returns the attenuation length at depth z (m) and frequency f (MHz)."""
        w = np.log(f*0.001)
        w0 = np.log(1e-4)
        w1 = 0
        w2 = np.log(3.16)
        t = cls.temperature(z) - 273.15
        b0 = -6.7489 + t * (0.026709 - 8.84e-4 * t)
        b1 = -6.2212 - t * (0.070927 + 1.77e-3 * t)
        b2 = -4.0947 - t * (0.002213 + 3.32e-4 * t)
        if f < 1000.0:
            a = (b1 * w0 - b0 * w1) / (w0 - w1)
            b = (b1 - b0) / (w1 - w0)
        else:
            a = (b2 * w1 - b1 * w2) / (w1 - w2)
            b = (b2 - b1) / (w2 - w1)
        return np.exp(-(a + b * w))


class NewcombIce(AntarcticIce):
    """Class inheriting from AntarcticIce, with new attenuation_length function
    based on Matt Newcomb's fit (DOESN'T CURRENTLY WORK - USE ANTARCTICICE)."""
    @classmethod
    def attenuation_length(cls, z, f):
        """Returns the attenuation length at depth z (m) and frequency f (MHz)
        by Matt Newcomb's fit (DOESN'T CURRENTLY WORK - USE BOGORODSKY)."""
        temp = cls.temperature(z)
        a = 5.03097 * np.exp(0.134806 * temp)
        b = 0.172082 + temp + 10.629
        c = -0.00199175 * temp - 0.703323
        return 1701 / (a + b * (0.001*f)**(c+1))



# Preferred ice model:
IceModel = AntarcticIce
