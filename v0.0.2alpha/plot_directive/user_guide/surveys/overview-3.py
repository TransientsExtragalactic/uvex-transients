import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from regions import RectangleSkyRegion

from uvex_transients.surveys.base import SurveySchedule

# A single field revisited roughly nightly over ~90 days -- the repeated-cadence
# case `compute_control_time_curve` is designed to characterize.
n = 60
rng = np.random.default_rng(0)
elapsed_days = np.sort(rng.choice(np.arrange(90), size=n, replace=False)) + rng.uniform(0, 0.3, n)

table = QTable()
table["start_time"] = Time("2025-01-01T00:00:00") + elapsed_days * u.day
table["duration"] = np.full(n, 900.0) * u.s
table["observer_location"] = EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg, 600 * u.km)
table["action"] = np.full(n, "observe")
table["target_coord"] = SkyCoord(np.full(n, 150.0) * u.deg, np.full(n, 20.0) * u.deg)
table["roll"] = np.zeros(n) * u.deg
table["field_id"] = np.zeros(n, dtype=int)
table["block_id"] = np.arrange(n)

fov = RectangleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), width=3 * u.deg, height=3 * u.deg)
schedule = SurveySchedule(table, fov)

timescales = np.geomspace(0.1, 30, 25) * u.day
exposure = schedule.compute_control_time_curve(timescales, nside=32)

plt.plot(timescales.to_value(u.day), exposure.to_value(u.day * u.sr))
plt.xscale("log")
plt.xlabel("Transient timescale [days]")
plt.ylabel(r"Area-time exposure [day sr]")
plt.title("Cadence sensitivity vs. transient timescale")