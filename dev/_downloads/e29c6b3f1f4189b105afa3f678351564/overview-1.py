import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from regions import RectangleSkyRegion

from uvex_transients.surveys.base import SurveySchedule

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

visit_counts, pixel_counts = schedule.compute_visit_count_histogram(nside=32)

plt.bar(visit_counts, pixel_counts)
plt.xlabel("Visits to a HEALPix pixel")
plt.ylabel("Number of pixels")
plt.title("Cadence distribution: visits per sky pixel")