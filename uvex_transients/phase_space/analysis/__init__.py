"""
Read-only analysis of finished recovery maps.

:mod:`~uvex_transients.phase_space.analysis.overlay` places the shipped transient classes on a map by measuring its two
axes on their SEDs, and :mod:`~uvex_transients.phase_space.analysis.compare` differences two maps. Neither runs a survey
simulation or needs scikit-learn, and the package imports neither, so that importing one does not load the other.
"""
