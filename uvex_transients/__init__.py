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
    "missions",
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

from . import missions, utils
from .utils import *

__all__.extend(utils.__all__)

# Warn (without blocking the import) if a newer release exists; see `utils.updates`.
from .utils.updates import check_for_updates

check_for_updates()
