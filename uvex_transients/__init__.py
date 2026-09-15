"""End-to-end simulations of transients from UVEX all-sky surveys."""

import warnings

# Handle warnings.
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.filterwarnings("ignore", ".*dubious year.*")
warnings.filterwarnings("ignore", "Tried to get polar motions for times after IERS data is valid.*")

__all__ = [
    "models",
    "surveys",
    "simulation",
    "utils",
    "transients",
]

from . import models
from .models import *

__all__.extend(models.__all__)

from . import surveys
from .surveys import *

__all__.extend(surveys.__all__)

from . import simulation
from .simulation import *

__all__.extend(simulation.__all__)

from . import transients
from .transients import *

__all__.extend(transients.__all__)

from . import utils
from .utils import *

__all__.extend(utils.__all__)
