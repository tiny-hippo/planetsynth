"""This module contains some useful constants in cgs units."""

# physical constants
avo = 6.02214076e23  # avogadro constant
k_b = 1.380649e-16  # boltzmann constant
sigma_b = 5.6704e-5  # stefan-boltzmann constant
G = 6.67430e-8  # gravitational constant
m_n = 1.674927471e-24  # neutron mass
m_p = 1.67262192369e-24  # proton mass
m_e = 9.10938356e-28  # electron mass

# astronomical units
au = 1.49597870700e13  # astronomical unit
mu_sun = 1.3271244e26  # solar gravitational parameter
mu_jup = 1.2668653e23  # jupiter gravitational parameter
mu_earth = 3.986004e20  # earth gravitational parameter

tau_sun = 4.57e9  # solar age
M_sun = mu_sun / G  # solar mass
R_sun = 6.957e10  # solar radius
L_sun = 3.828e33  # solar luminosity

M_jup = mu_jup / G  # jupiter mass
R_jup_eq = 7.1492e9  # jupiter equatorial radius
R_jup_polar = 6.6854e9  # jupiter polar radius
R_jup_vol = 6.9911e9  # jupiter volumetric mean radius
R_jup = R_jup_vol

M_earth = mu_earth / G  # earth mass
R_earth_eq = 6.3781e8  # earth equatorial radius
R_earth_polar = 6.3568e8  # earth polar radius
R_earth_vol = 6.371e8  # earth volumetric mean radius
R_earth = R_earth_vol
F_earth = 1.361e6  # earth flux
