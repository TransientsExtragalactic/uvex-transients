from functools import reduce

from mocpy import MOC

def circular_fields(*, max_order, centers, radius):
    """Union of circles (centers in (RA, Dec) degrees, radius in degrees)."""
    mocs = [
        MOC.from_cone(lon=ra * u.deg, lat=dec * u.deg, radius=radius * u.deg, max_depth=max_order)
        for ra, dec in centers
    ]
    return reduce(MOC.union, mocs)

fields = SurveyFootprint(
    name="demo:fields:three",
    generator=circular_fields,
    params={"centers": [(30, 20), (150, -10), (260, 50)], "radius": 12.0},
    description="Three 12 degree circular fields.",
)

print(f"{fields.moc.sky_fraction * 41253:,.0f} deg^2")
plot_footprints({"Three fields": fields}, title="A custom generator")
plt.show()