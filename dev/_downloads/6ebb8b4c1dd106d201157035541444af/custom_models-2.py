import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.core import NormalPrior, LogNormalPrior, Parameter, Spectrum
from uvex_transients.models._typing import CGSParameterValue, FloatArray


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

spec = ExponentialCutoffPowerLawSpectrum()
spec["cutoff_frequency"].fix(3000 * u.AA.to(u.Hz, equivalencies=u.spectral()) * u.Hz)
params = spec.sample_parameters(rng=1)

nu = np.geomspace(1e13, 1e17, 300) * u.Hz
S = spec.eval(nu, **params)

plt.plot(nu.to_value(u.Hz), S.to_value(1 / u.Hz))
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Frequency [Hz]")
plt.ylabel(r"$S(\nu)$ [Hz$^{-1}$]")
plt.title("ExponentialCutoffPowerLawSpectrum, one realization")
plt.axvline(
    spec["cutoff_frequency"].fixed_value.value, color="k", ls="--", lw=1, label="cutoff_frequency",
)
plt.legend()