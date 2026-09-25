import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from regions import RectangleSkyRegion

from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.transients.TDEs import TidalDisruptionEvent

n = 400
rng = np.random.default_rng(0)

table = QTable()
table["start_time"] = Time("2025-01-01T00:00:00") + np.sort(rng.uniform(0, 180, n)) * u.day
table["duration"] = np.full(n, 900.0) * u.s
table["observer_location"] = EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg, 600 * u.km)
table["action"] = np.full(n, "observe")
table["target_coord"] = SkyCoord(
    rng.uniform(0, 360, n) * u.deg,
    np.degrees(np.arcsin(rng.uniform(-1, 1, n))) * u.deg,
)
table["roll"] = np.zeros(n) * u.deg
table["field_id"] = np.arrange(n)
table["block_id"] = np.zeros(n, dtype=int)

fov = RectangleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), width=3 * u.deg, height=3 * u.deg)
schedule = SurveySchedule(table, fov)

tde = TidalDisruptionEvent()
simulator = SurveySimulator(schedule, transients={"tde": tde}, simulation_seed=0)

catalog = simulator.generate_events(time_bins=6, nside=32)

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection="aitoff")
ax.grid(True)
ra = catalog.coord.ra.wrap_at(180 * u.deg).radian
ax.scatter(ra, catalog.coord.dec.radian, s=4, color="C0")
ax.set_title(f"{len(catalog)} sampled TDEs across the example schedule")