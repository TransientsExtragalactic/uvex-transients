.. _report_infant_sne_gallery:

*******************************
Infant Supernovae
*******************************

.. card::
   :class-card: sd-shadow-sm gallery-banner

   *Run:* ``configs/infant_sne.yaml`` · *Data:* ``get_results(..., simulation="infant_sne")``

   The rate of internal triggers for early supernova spectroscopy. An infant supernova has at least one detection
   brighter than 20 AB within 48 hours of its last non-detection. These events are extremely rare, and the aim is
   to get on target as quickly as possible.

   :ref:`Back to the report <report_page>`

The sample
==========

Every supernova class of the Standard Run (Ia, Ib, Ic, Ic-BL, IIP, IIP with an early excess, IIb and the SLSNe) is
sampled over **only the 72 hours after explosion**, since an infant supernova is caught in its first days. Events
are generated at their true rate, with no downsampling, and observed against the UVEX survey schedule. Two screens
reduce them to the detected sample:

- a loose **23 AB** magnitude screen on the peak, which discards events too faint for UVEX to see in the window,
- a **5 sigma** detection in at least one observation in the window, ignoring an event whose only detection is the
  first-ever visit to its field.

The photometry of the events that survive covers the **30 days before** the explosion (the survey's observations of
the position before there was a source, which give the non-detections and limits that bracket the rise) and the
**100 days after** it, so every light curve shows its full evolution. The brighter selection that makes an event an
infant is applied afterwards, to these tables, so the definition can be changed without regenerating the run.

What is an infant supernova?
============================

To make sure spectroscopic follow-up is possible, the event must reach **20 AB** or brighter in a 5 sigma detection
within **48 hours** of the last non-detection in the same band. To trigger on one early, the alert stream needs a
cut that can be applied to the first detection, and an event is **triggered** if either of two conditions holds on
a 5 sigma detection within 48 hours of the last non-detection:

- (a) it has **brightened** by at least a given number of magnitudes since the limit of that non-detection,
- (b) it is **brightening** at at least a given rate, in magnitudes per day, since that limit.

The analysis below counts the infants and the triggers, and shows the light curves of triggered events.
