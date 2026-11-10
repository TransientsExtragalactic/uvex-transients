r"""
Analytic shock cooling emission models.

* :mod:`~uvex_transients.models.shock_cooling.morag` implements the SEDs of :footcite:t:`2024MNRAS.528.7137M`
  for shock cooling from the extended hydrogen envelopes of red supergiants (Type II SNe), including deviations
  from blackbody, and a Type IIb SED that adds nickel decay to them.
* :mod:`~uvex_transients.models.shock_cooling.piro` implements the bolometric luminosity, photospheric radius,
  and temperature of :footcite:t:`2021ApJ...909..209P` for a shocked, extended low-mass envelope (e.g., Type IIb
  SNe), as functions and as an SED.
* :mod:`~uvex_transients.models.shock_cooling.khatami` implements a shock breakout SED: the two-parameter
  :class:`~uvex_transients.models.lightcurves.generic.KhatamiSBOLightcurve` with a blackbody whose temperature
  follows the instantaneous luminosity and the time since breakout.
* :mod:`~uvex_transients.models.shock_cooling.inverse_gamma` implements a shock breakout SED: the
  :class:`~uvex_transients.models.lightcurves.generic.InverseGammaBrokenPowerLawLightcurve` with a blackbody on a
  photosphere that expands linearly from a breakout radius :math:`c\,t_\mathrm{pk}`.

References
----------
.. footbibliography::
"""

from .inverse_gamma import InverseGammaBlackbodySED
from .khatami import KhatamiSED
from .morag import MoragShockCoolingArnettSED, MoragShockCoolingBlackbodySED, MoragShockCoolingSED
from .piro import PiroShockCoolingSED

__all__ = [
    "InverseGammaBlackbodySED",
    "KhatamiSED",
    "MoragShockCoolingArnettSED",
    "MoragShockCoolingBlackbodySED",
    "MoragShockCoolingSED",
    "PiroShockCoolingSED",
]
