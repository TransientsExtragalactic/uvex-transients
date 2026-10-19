from astropy.time import Time

events = tde.sample_events_on_healpix_grid(
    nside=32,
    t_start=Time("2025-01-01"),
    duration=180 * u.day,
    seed=0,
)
print(f"Sampled {len(events)} events.")

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection="aitoff")
ax.grid(True)
ra = events["coord"].ra.wrap_at(180 * u.deg).radian
ax.scatter(ra, events["coord"].dec.radian, s=4, c=events["redshift"], cmap="viridis")
ax.set_title(f"{len(events)} sampled TDEs, whole sky, 180 days")