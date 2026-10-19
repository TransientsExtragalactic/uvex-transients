import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.core import (
    ComposedSpectralModel, Lightcurve, LogNormalPrior, NormalPrior, Parameter, Spectrum,
)
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._utils import _BOL_LUM_UNIT


class RampDecayLightcurve(Lightcurve):
    _DEFAULT_PARAMETERS = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5), scale=1e43 * _BOL_LUM_UNIT,
        ),
        "t_peak": Parameter(prior=LogNormalPrior(mean=0.0, sigma=0.5), scale=1.0 * u.day),
        "tau_decay": Parameter(prior=LogNormalPrior(mean=0.0, sigma=0.5), scale=5.0 * u.day),
    }

    @classmethod
    def _eval(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        amplitude, t_peak, tau_decay = (
            parameters["amplitude"], parameters["t_peak"], parameters["tau_decay"],
        )
        with np.errstate(divide="ignore"):
            log_rise = np.log(t / t_peak)
        log_decay = -(t - t_peak) / tau_decay
        return np.log(amplitude) + np.where(t <= t_peak, log_rise, log_decay)


class ExponentialCutoffPowerLawSpectrum(Spectrum):
    _DEFAULT_PARAMETERS = {
        "spectral_index": Parameter(prior=NormalPrior(mean=1.0, sigma=0.3), scale=1.0),
        "cutoff_frequency": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3), scale=1e15 * u.Hz,
        ),
    }

    @classmethod
    def _eval(cls, nu: FloatArray, *, spectral_index, cutoff_frequency) -> FloatArray:
        x = nu / cutoff_frequency
        with np.errstate(divide="ignore"):
            log_shape = spectral_index * np.log(x) - x
        return log_shape - np.log(cutoff_frequency)


class ToyFlareSED(ComposedSpectralModel):
    _LIGHTCURVE_CLASS = RampDecayLightcurve
    _SPECTRUM_CLASS = ExponentialCutoffPowerLawSpectrum

rng = np.random.default_rng(20260911)
n_samples = 200

params = ToyFlareSED().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.linspace(0, 30, 200) * u.day
L_bol = ToyFlareSED.eval_bolometric(t, **params_grid)

plt.plot(t.to_value(u.day), L_bol.to_value(u.erg / u.s).T, color="C0", lw=0.5, alpha=0.15)
plt.yscale("log")
plt.xlabel("Time since explosion [days]")
plt.ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
plt.title(f"{n_samples} simulated ToyFlareSED light curves")