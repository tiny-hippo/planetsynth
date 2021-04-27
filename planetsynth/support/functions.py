""" This module contains some useful support functions. """
import numpy as np
from .constants import sigma_b


def get_L(R, Teff) -> np.ndarray:
    """Calculates the luminosity of a black body given its
    radius and effective temperature.

    Args:
        R (array_like): The radius in cm.
        Teff (array_like): The effective temperature in K.

    Returns:
        array_like: The luminosity of a black body in erg/s.
    """
    return 4 * np.pi * sigma_b * Teff ** 4 * R ** 2


def get_R(L, Teff) -> np.ndarray:
    """Calculates the radius of a black body given its
    luminosity and effective temperature.

    Args:
        L (array_like): The luminosity in cgs.
        Teff (array_like): The effective temperature in K.

    Returns:
        array_like: The radius of a black body in cm.
    """
    return np.sqrt(L / (4 * np.pi * sigma_b * Teff ** 4))


def get_F(L, d) -> np.ndarray:
    """Calculates the incident day-side flux given the stellar luminosity L
    and planetary semi-major axis d.

    Args:
        L (array_like): Stellar luminosity in erg/s.
        d (array_like): Planetary semi-major axis in cm.

    Returns:
        np.ndarray: The incident day-side flux in erg/s/cm2
    """
    return L / (4 * np.pi * d ** 2)


def get_Teq_from_L(L, d, A) -> np.ndarray:
    """Calculates the equilibrium temperature of a planet
    given the stellar luminosity L, planetary semi-major axis d
    and surface albedo A:

    Args:
        L (array_like): Stellar luminosity in erg/s.
        d (array_like): Planetary semi-major axis in cm.
        A (array_like): Planetary albedo.

    Returns:
        np.ndarray: The planetary equilibrium temperature in K.
    """
    return ((L * (1 - A)) / (16 * sigma_b * np.pi * d ** 2)) ** 0.25


def get_Teq_from_T(T, R, d, A) -> np.ndarray:
    """Calculates the equilibrium temperature of a planet
    given the stellar effective temperature and temperature,
    planetary semi-major axis d and surface albedo A:

    Args:
        T (array_like): Stellar effective temperature in K.
        R (array_like): Stellar radius in cm.
        d (array_like): Planetary semi-major axis in cm.
        A (array_like): Planetary albedo.

    Returns:
        np.ndarray: The planetary equilibrium temperature in K.
    """
    return T * np.sqrt(R / (2 * d)) * (1 - A) ** 0.25
