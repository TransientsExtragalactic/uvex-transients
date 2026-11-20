from uvex_transients.surveys.footprints import combine_footprints

overlap = combine_footprints(
    "demo:ztf_and_sole",
    "intersection",
    ["ztf:main", "ls4:sole"],
    persist=False,
)
ztf_only = combine_footprints(
    "demo:ztf_minus_sole",
    "difference",
    [ztf, sole],
    persist=False,
)

plot_footprints(
    {"ZTF minus SOLE": ztf_only, "ZTF and SOLE": overlap},
    title="Combining footprints",
)
plt.show()