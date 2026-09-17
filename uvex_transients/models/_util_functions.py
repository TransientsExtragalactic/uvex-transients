import numpy as np

from uvex_transients.models._typing import FloatArray


def _softplus(x: FloatArray) -> FloatArray:
    r"""Numerically stable :math:`\sigma(x) = \ln(1+e^x)`."""
    return np.logaddexp(0.0, x)


def _log_sigmoid(x: FloatArray) -> FloatArray:
    r"""Numerically stable :math:`\ln[\mathrm{sigmoid}(x)] = -\sigma(-x)`."""
    return -_softplus(-x)
