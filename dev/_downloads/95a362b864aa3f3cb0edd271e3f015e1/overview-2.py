z_samples = tde.sample_event_redshift(20000, rng=0)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(z_samples, bins=60, density=True, color="C0", alpha=0.7)
ax.set_xlabel("Redshift")
ax.set_ylabel("Probability density")
ax.set_title("TDE redshift distribution (rate-weighted, inverse-transform sampled)")