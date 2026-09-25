import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.transients.TDEs import TidalDisruptionEvent

tde = TidalDisruptionEvent()

print(tde.sed)                  # the VanVelzenTDESED SED instance
print(tde.duration_limit)       # 200.0 d
print(tde.redshift_limit)       # 2

z = np.linspace(0, tde.redshift_limit, 200)
rate = tde.event_rate(z)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(z, rate)
ax.set_xlabel("Redshift")
ax.set_ylabel(r"Event rate [Mpc$^{-3}$ yr$^{-1}$]")
ax.set_title("TDE volumetric event rate")