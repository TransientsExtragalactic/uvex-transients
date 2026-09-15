from m4opt.missions import uvex

mission = uvex
mag_filtered = simulator.filter_by_limiting_magnitude(catalog, mission, mag_limit=25.0)
detected = simulator.filter_by_snr(mag_filtered, mission, snr_threshold=5.0)

stages = ["Sampled", "Mag < 25", "SNR > 5"]
counts = [len(catalog), len(mag_filtered), len(detected)]

fig, ax = plt.subplots()
ax.bar(stages, counts, color=["#888888", "#4C72B0", "#55A868"])
for i, count in enumerate(counts):
    ax.text(i, count, f"{count:,}", ha="center", va="bottom")
ax.set_ylabel("Number of TDEs")
ax.set_title("TDE detection funnel")