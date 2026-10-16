from uvex_transients.surveys.footprints import SurveyFootprint
from uvex_transients.surveys.footprints.utils import dec_band_MOC

polar = SurveyFootprint(
    name="demo:polar:north",
    generator=dec_band_MOC,
    params={"min_dec": 60.0},
    description="A made-up polar survey: everything above Dec = +60 deg.",
)

plot_footprints({"Polar survey": polar, "ZTF": ztf}, title="A custom footprint")
plt.show()