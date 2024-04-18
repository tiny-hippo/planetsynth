import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import sigma_b


def get_L(R: ArrayLike, Teff: ArrayLike) -> NDArray:
    """Calculates the luminosity of a black body given its
    radius and effective temperature.

    Args:
        R (ArrayLike): The radius in cm.
        Teff (ArrayLike): The effective temperature in K.

    Returns:
        ArrayLike: The luminosity of a black body in erg/s.
    """
    return 4 * np.pi * sigma_b * Teff**4 * R**2


def get_R(L: ArrayLike, Teff: ArrayLike) -> NDArray:
    """Calculates the radius of a black body given its
    luminosity and effective temperature.

    Args:
        L (ArrayLike): The luminosity in cgs.
        Teff (ArrayLike): The effective temperature in K.

    Returns:
        ArrayLike: The radius of a black body in cm.
    """
    return np.sqrt(L / (4 * np.pi * sigma_b * Teff**4))


def get_F_from_L(L: ArrayLike, d: ArrayLike) -> NDArray:
    """Calculates the incident day-side flux given the stellar luminosity L
    and planetary semi-major axis d.

    Args:
        L (ArrayLike): Stellar luminosity in erg/s.
        d (ArrayLike): Planetary semi-major axis in cm.

    Returns:
        np.ndarray: The incident day-side flux in erg/s/cm2.
    """
    return L / (4 * np.pi * d**2)


def get_F_from_Teq(Teq: ArrayLike, d: ArrayLike, A: ArrayLike) -> NDArray:
    """Calculates the incident day-side flux given the
    given the planetary equilibrium temperature Teq,
    semi-major axis d and surface albedo A:

    Args:
        L (ArrayLike): Stellar luminosity in erg/s.
        d (ArrayLike): Planetary semi-major axis in cm.
        A (ArrayLike): Planetary albedo.

    Returns:
        np.ndarray: The incident day-side flux in erg/s/cm2.
    """
    L_star = get_L_from_Teq(Teq, d, A)
    return get_F_from_L(L_star, d)


def get_Teq_from_L(L: ArrayLike, d: ArrayLike, A: ArrayLike) -> NDArray:
    """Calculates the equilibrium temperature of a planet
    given the stellar luminosity L, planetary semi-major axis d
    and surface albedo A:

    Args:
        L (ArrayLike): Stellar luminosity in erg/s.
        d (ArrayLike): Planetary semi-major axis in cm.
        A (ArrayLike): Planetary albedo.

    Returns:
        np.ndarray: The planetary equilibrium temperature in K.
    """
    return ((L * (1 - A)) / (16 * sigma_b * np.pi * d**2)) ** 0.25


def get_Teq_from_T(T: ArrayLike, R: ArrayLike, d: ArrayLike, A: ArrayLike) -> NDArray:
    """Calculates the equilibrium temperature of a planet
    given the stellar effective temperature and temperature,
    planetary semi-major axis d and surface albedo A:

    Args:
        T (ArrayLike): Stellar effective temperature in K.
        R (ArrayLike): Stellar radius in cm.
        d (ArrayLike): Planetary semi-major axis in cm.
        A (ArrayLike): Planetary albedo.

    Returns:
        np.ndarray: The planetary equilibrium temperature in K.
    """
    return T * np.sqrt(R / (2 * d)) * (1 - A) ** 0.25


def get_L_from_Teq(Teq: ArrayLike, d: ArrayLike, A: ArrayLike) -> NDArray:
    """Calculates the stellar luminosity
    given the planetary equilibrium temperature Teq,
    semi-major axis d and surface albedo A:

    Args:
        L (ArrayLike): Stellar luminosity in erg/s.
        d (ArrayLike): Planetary semi-major axis in cm.
        A (ArrayLike): Planetary albedo.

    Returns:
        np.ndarray: The stellar luminosity in erg/s.
    """
    return 16 * sigma_b * np.pi * d**2 * Teq**4 / (1 - A)
