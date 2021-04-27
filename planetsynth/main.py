import os
import pickle
import zipfile
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from .support.constants import M_jup, R_jup, L_sun, sigma_b, G


class PlanetSynth:
    """
     Synthetic cooling track generation for giant planets.

     Args:
        verbose (bool, optional): Print out warnings if input parameters
        are out of range or the interpolation has to fall back on
        nearest-neighbour interpolation. Defaults to True.

    Attributes:
        # to-do: list attributes
    """

    def __init__(self, verbose=True) -> None:
        self.interpolant_path = Path(__file__).parent / "interpolators"
        self.verbose = verbose

        # boundaries of the suite of models
        self.min_M = 0.1
        self.max_M = 30
        self.min_Z = 0.00
        self.max_Z = 0.80
        self.min_Ze = 0.00
        self.max_Ze = 0.10
        self.min_logF = 1
        self.max_logF = 9
        self.max_M1 = 1
        self.max_Z1 = 0.8
        self.max_M2 = 3
        self.max_Z2 = 0.5
        self.max_M3 = 5
        self.max_Z3 = 0.2
        self.max_M4 = 30
        self.max_Z4 = 0.036

        # input array
        self.i_M = 0  # index for the mass in M_j
        self.i_Z = 1  # index for the bulk metallicity
        self.i_Zatm = 2  # index for the atmospheric metallicity
        self.i_logF = 3  # index for the log10 stellar flux in lerg/s/cm2
        self.num_features = 4
        self.num_timesteps = 16
        self.min_logage = 7  # in yrs
        self.max_logage = 10  # in yrs
        self.log_time = np.linspace(
            self.min_logage, self.max_logage, self.num_timesteps
        )

        # output array
        self.i_R = 0  # index for the radius in R_jup
        self.i_logL = 1  # index for the log10 luminosity in Lsun
        self.i_Teff = 2  # index for the effective temperature in K
        self.i_logg = 3  # index for the log10 gravitational acceleration
        self.num_targets = 4
        self.which_targets = [self.i_R, self.i_logL]

        self.__load_interpolator()

    @staticmethod
    def __unzip(zipPath):
        """Extracts the interpolator file from the split zip files.
        Credits to Guven Degirmenci on StackOverflow."""
        zips = os.listdir(zipPath)
        for zipName in zips:
            if "zip" not in zipName:
                continue
            with open(os.path.join(zipPath, "tmp.zip"), "ab") as f:
                with open(os.path.join(zipPath, zipName), "rb") as z:
                    f.write(z.read())

        with zipfile.ZipFile(os.path.join(zipPath, "tmp.zip"), "r") as zipObj:
            zipObj.extractall(zipPath)
        os.remove(os.path.join(zipPath, "tmp.zip"))

    def __load_interpolator(self) -> None:
        """ Loads the RegularGridInterpolator object. """
        fname = "RegularGridInterpolator.pkl"
        src = os.path.join(self.interpolant_path, fname)

        if not os.path.isfile(src):
            self.__unzip(self.interpolant_path)

        with open(src, "rb") as file:
            self.reg = pickle.load(file)

    def __check_input(self, planet_params) -> tuple:
        """Checks whether the input is within the interpolation range.

        Args:
            planet_params: (n_planets, 3), array_like
                Array of floats of the planet parameters.

        Raises:
            ValueError: Raised if planet_params has the wrong shape.

        Returns:
            tuple (array_like, bool or boolean array_like)
        """

        if not isinstance(planet_params, np.ndarray):
            planet_params = np.asarray(planet_params)

        self.input_dim = planet_params.ndim

        if self.input_dim == 1:
            input_check = False
            if planet_params[self.i_M] <= self.max_M1:
                max_Z = self.max_Z1
            elif planet_params[self.i_M] <= self.max_M2:
                max_Z = self.max_Z2
            elif planet_params[self.i_M] <= self.max_M3:
                max_Z = self.max_Z3
            else:
                max_Z = self.max_Z4
            if (
                planet_params[self.i_M] > self.max_M
                or planet_params[self.i_M] < self.min_M
            ):
                if self.verbose:
                    print(f"M = {planet_params[self.i_M]:.2f} out of range")
            elif (
                planet_params[self.i_Z] > max_Z or planet_params[self.i_Z] < self.min_Z
            ):
                if self.verbose:
                    print(f"Z = {planet_params[self.i_Z]:.2f} out of range")
            elif (
                planet_params[self.i_logF] > self.max_logF
                or planet_params[self.i_logF] < self.min_logF
            ):
                if self.verbose:
                    print(f"logF = {planet_params[self.i_logF]:.2f} out of range")
            elif (
                planet_params[self.i_Zatm] > self.max_Ze
                or planet_params[self.i_Zatm] < self.min_Ze
            ):
                if self.verbose:
                    print(f"Zatm = {planet_params[self.i_Zatm]:.2f} out of range")
            elif planet_params[self.i_Zatm] > planet_params[self.i_Z]:
                if self.verbose:
                    print(
                        f"Zatm = {planet_params[self.i_Zatm]:.2f} > Z = {planet_params[self.i_Z]:.2f}"
                    )
            else:
                input_check = True

            return (planet_params, input_check)

        elif self.input_dim == 2:
            i1 = np.logical_and(
                planet_params[:, self.i_M] >= self.min_M,
                planet_params[:, self.i_M] <= self.max_M,
            )
            iM1 = np.logical_and(
                planet_params[:, self.i_M] <= self.max_M1,
                planet_params[:, self.i_Z] <= self.max_Z1,
            )
            iM2 = np.logical_and(
                planet_params[:, self.i_M] > self.max_M1,
                planet_params[:, self.i_M] <= self.max_M2,
            )
            iM2 = np.logical_and(iM2, planet_params[:, self.i_Z] <= self.max_Z2)
            iM3 = np.logical_and(
                planet_params[:, self.i_M] > self.max_M2,
                planet_params[:, self.i_M] <= self.max_M3,
            )
            iM3 = np.logical_and(iM3, planet_params[:, self.i_Z] <= self.max_Z3)
            iM4 = np.logical_and(
                planet_params[:, self.i_M] > self.max_M3,
                planet_params[:, self.i_M] <= self.max_M4,
            )
            iM4 = np.logical_and(iM4, planet_params[:, self.i_Z] <= self.max_Z4)
            i2 = np.logical_or(iM1, iM2)
            i2 = np.logical_or(i2, iM3)
            i2 = np.logical_or(i2, iM4)
            i3 = np.logical_and(
                planet_params[:, self.i_logF] >= self.min_logF,
                planet_params[:, self.i_logF] <= self.max_logF,
            )
            i4 = np.logical_and(
                planet_params[:, self.i_Zatm] >= self.min_Ze,
                planet_params[:, self.i_Zatm] <= self.max_Ze,
            )

            i = np.logical_and(i1, i2)
            i = np.logical_and(i, i3)
            i = np.logical_and(i, i4)
            if np.any(~i) and self.verbose:
                print(f"Warning: bad input parameters at", np.where(~i)[0])
            return (planet_params, i)
        else:
            raise ValueError("Failed in __check_input: Input has the wrong shape.")

    def __get_time_interpolant(self, prediction, kind="cubic") -> interp1d:
        """Helper function to create a 1d interpolant in time.

        Args:
            prediction (array_like): Prediction generated by
                the synthesize method.
            kind (str, optional): Order of interpolation. Defaults to "cubic".

        Returns:
            interp1d: Returns the interp1d object
        """
        return interp1d(self.log_time, prediction.T, axis=1, kind=kind)

    def __get_Teff(self, prediction) -> np.ndarray:
        """Calculates the effective temperature of a black body given its
        radius and luminosity.

        Args:
            prediction (array_like): Prediction generated by
                the synthesize method.

        Raises:
            ValueError: Raised if the input has an incompatible shape.

        Returns:
            array_like: The effective temperature of a black body in K.
        """
        if prediction.ndim > 2:
            R = prediction[:, :, self.i_R] * R_jup
            L = 10 ** prediction[:, :, self.i_logL] * L_sun
        elif prediction.ndim == 2:
            R = prediction[:, self.i_R] * R_jup
            L = 10 ** prediction[:, self.i_logL] * L_sun
        elif prediction.ndim == 1:
            R = prediction[self.i_R] * R_jup
            L = 10 ** prediction[self.i_logL] * L_sun
        else:
            raise ValueError("Failed in __get_Teff: Input has the wrong shape.")
        return (L / (4 * np.pi * sigma_b * R ** 2)) ** 0.25

    def __get_logg(self, planet_params, prediction) -> np.ndarray:
        """Calculates the log10 of the gravitational acceleration
        at photosphere.

        Args:
            planet_params: (n_planets, 3), array_like
            prediction (array_like): Prediction generated by the
            synthesize method.

        Raises:
            ValueError: Raisd if the input has an incompatible shape.

        Returns:
            array_like: The log10 of the gravitational acceleration in cm/s^2.
        """
        if prediction.ndim > 2:
            R = prediction[:, :, self.i_R] * R_jup
        elif prediction.ndim == 2:
            R = prediction[:, self.i_R] * R_jup
        elif prediction.ndim == 1:
            R = prediction[self.i_R] * R_jup
        else:
            raise ValueError("Failed in __get_logg: Input has the wrong shape.")

        if self.input_dim == 1:
            M = planet_params[self.i_M] * M_jup
        elif self.input_dim == 2:
            M = planet_params[:, self.i_M] * M_jup
            # broadcasting to match shape of R
            M = M[:, None]
        else:
            raise ValueError("Failed in __get_log: Input has the wrong shape.")

        return np.log10(G * M / R ** 2)

    def synthesize(self, planet_params) -> np.ndarray:
        """Synthesizes a cooling track for a set of planetary parameters in
        terms of planetary mass M [in Jupiter masses], metallicity Z
        and the log of the incident stellar irradiation logF [in erg/s/cm2].

        Args:
            planet_params: (n_planets, 3), array_like
                Array of floats of the planet parameters in the
                following order [Mass, Metallicity, log(Irradiation)].
                The units are Jupiter masses, mass-fraction and erg/s/cm2.
                The shape of planet_params can either be (3,) to calculate the
                cooling of a single planet, or (n_planets, 3) for several.

                The following input ranges are supported:
                Mass: 0.1 - 10
                Metallicity: Depending on the mass
                    * M = 0.1 - 1: Z = 0 - 0.8
                    * M = 1 - 3: Z = 0 - 0.5
                    * M = 3 - 5: Z = 0 - 0.2
                    * M > 5: Z = 0 - 0.036
                Atmospheric metallicity: Depending on mass
                    * M = 0.1 - 5: Zatm = 0 - 0.1
                    * M > 5: Zatm = 0 - 0.036
                log(Irradiation): 1 - 9

        Returns:
            array_like: Array of the synthetic cooling track calculated
                for the times defined in log_time. Shape (16, 4) for
                a single planet, or (n_planets, 16, 4) for several.
                The output quantities are radius [Jupiter radius],
                log(luminosity) [solar luminosity] and effective temperature [K]
        """
        planet_params, self.check_params = self.__check_input(planet_params)
        if self.input_dim == 1:
            res = np.zeros((self.num_timesteps, self.num_targets))
            if not self.check_params:
                res[:] = np.nan
                return res
            else:

                prediction = self.reg(planet_params)[0]
        else:
            res = np.zeros(
                (planet_params.shape[0], self.num_timesteps, self.num_targets)
            )
            res[~self.check_params, :, :] = np.nan
            pp = planet_params[self.check_params]
            prediction = self.reg(pp)

        if np.any(np.isnan(prediction)):
            if self.input_dim == 1:
                if self.verbose:
                    print(
                        "Failed in synthesize ",
                        f"for M = {planet_params[self.i_M]:.3f}",
                        f"Z = {planet_params[self.i_Z]:.3f}",
                        f"logF = {planet_params[self.i_logF]:.3f}",
                        f"Z_atm = {planet_params[self.i_Zatm]:.3f}",
                    )
            else:
                k = np.unique(np.where(np.isnan(prediction))[0])
                if self.verbose:
                    print(f"Failed in synthesize for input parameters at {k}")

        if self.input_dim == 1:
            res[:, self.which_targets] = prediction
            res[:, self.i_Teff] = self.__get_Teff(prediction)
            res[:, self.i_logg] = self.__get_logg(planet_params, prediction)

        else:
            res[self.check_params, :, : self.i_Teff] = prediction
            res[self.check_params, :, self.i_Teff] = self.__get_Teff(prediction)
            res[self.check_params, :, self.i_logg] = self.__get_logg(pp, prediction)
        return res

    def predict(self, logt, planet_params, kind="cubic") -> np.ndarray:
        """Predicts the radius, log(luminosity) and effective temperature
        at a specific log(time) [yr] for a set of planetary parameters
        in terms of planetary mass M [in Jupiter masses], metallicity Z
        and the log of the incident stellar irradiation logF [in erg/s/cm2].

        Args:
            logt (array_like): log of the time [yr] at which
                to calculate the prediction. Can be ndarray or float.
            planet_params: (n_planets, 3), array_like
                Array of floats of the planet parameters in the
                following order [Mass, Metallicity, log(Irradiation)].
                The units are Jupiter masses, mass-fraction and erg/s/cm2.
                The shape of planet_params can either be (3,) to calculate the
                cooling of a single planet, or (n_planets, 3) for several.

                The following input ranges are supported:
                Mass: 0.1 - 10
                Metallicity: Depending on the mass
                    * M = 0.1 - 1: Z = 0 - 0.8
                    * M = 1 - 3: Z = 0 - 0.5
                    * M = 3 - 5: Z = 0 - 0.2
                    * M > 5: Z = 0 - 0.036
                Atmospheric metallicity: Depending on mass
                    * M = 0.1 - 5: Zatm = 0 - 0.1
                    * M > 5: Zatm = 0 - 0.036
                log(Irradiation): 1 - 9
        kind (str, optional): Order of interpolation. Defaults to "cubic".

        Returns:
            array_like: Array of the synthetic cooling track calculated
                for the time(s) given in the input. Shape (n_times, 4) for
                a single planet, or (n_planets, n_times, 4) for several.
                The output quantities are radius [Jupiter radius],
                log(luminosity) [solar luminosity] and
                effective temperature [K].
        """
        if not isinstance(logt, np.ndarray):
            logt = np.asarray(logt)

        if np.any(logt < self.min_logage) or np.any(logt > self.max_logage):
            raise ValueError("Failed in predict because logt is out of range.")

        prediction = self.synthesize(planet_params)
        if np.any(np.isnan(prediction)):
            if self.input_dim == 1:
                result = np.empty(self.num_targets)
                result.fill(np.nan)
            else:
                if np.isscalar(logt):
                    result = np.empty((prediction.shape[0], self.num_targets))
                else:
                    result = np.empty(
                        (prediction.shape[0], logt.shape[0], self.num_targets)
                    )
                result.fill(np.nan)
                i = np.where(np.isfinite(prediction))
                i = np.unique(i[0])
                f = self.__get_time_interpolant(prediction[i], kind=kind)
                result[i] = np.transpose(f(logt))
            return result
        else:
            f = self.__get_time_interpolant(prediction, kind=kind)
            return np.transpose(f(logt))
