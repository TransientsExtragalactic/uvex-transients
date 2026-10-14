import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.core import Lightcurve, LogNormalPrior, Parameter
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

lc = RampDecayLightcurve()
params = lc.sample_parameters(rng=0)

t = np.linspace(0, 30, 300) * u.day
L_bol = lc.eval(t, **params)

plt.plot(t.to_value(u.day), L_bol.to_value(u.erg / u.s))
plt.xlabel("Time since explosion [days]")
plt.ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
plt.title("RampDecayLightcurve, one realization")