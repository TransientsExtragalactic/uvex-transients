import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

rng = np.random.default_rng(0)
n = 200_000
coords = SkyCoord(
    rng.uniform(0, 360, n) * u.deg,
    np.degrees(np.arcsin(rng.uniform(-1, 1, n))) * u.deg,  # uniform on the sphere
)

in_ztf = ztf.contains_skycoord(coords)
print(f"{in_ztf.mean():.1%} of the sky lies in ZTF's footprint")