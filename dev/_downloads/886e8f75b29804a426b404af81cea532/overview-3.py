event = detected.get_events(19, {"tde": tde}, schedule)

phot = event.simulate_photometry(mission)
t_since_explosion = (phot["obs_time"] - event.t_explosion).to(u.day)
t_theory = np.linspace(0, tde.duration_limit.to_value(u.day), 300) * u.day

fig, ax = plt.subplots(figsize=(7, 4))
for band, color in {"FUV": "#4C72B0", "NUV": "#DD8452"}.items():
    ax.plot(t_theory.value, event.mag(t_theory, mission, band=band).value, color=color, lw=1.5, alpha=0.6)

    in_band = np.isfinite(phot["ab_mag"]) & (phot["band"] == band)
    if np.any(in_band):
        ax.errorbar(
            t_since_explosion[in_band].value,
            phot["ab_mag"][in_band],
            yerr=phot["mag_err"][in_band],
            marker="s", mfc=color, mec="k", ecolor=color, linestyle="none", label=band,
        )

ax.invert_yaxis()
ax.set_xlabel("Days since explosion")
ax.set_ylabel("AB magnitude")
ax.set_title(f"Event {event.event_id} (z={event.redshift:.2f}, {event.n_observations} observations)")
ax.legend()