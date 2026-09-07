"""Reference implementation of the aperiodic-exponent analysis.

A reading of the paper's Methods in code: every step that turns a recording
into a number, and every model that turns those numbers into a claim.

The package holds no data, no file paths, and no patient information. It is
written to be read as an account of the computation, and it runs end to end on
synthetic data (see ``demo/walkthrough.py``).

Layout, in pipeline order::

    config       every fixed threshold, annotated with its Methods section
    preprocess   reference, filter, stage, epoch, adaptive artefact rejection
    spectral     Welch PSD, specparam fit, raw log-log slope, quality control
    geometry     contact parsing, SOZ centre, along-electrode distance, regions
    discharges   Teager-Kaiser detection and the discharge controls
    scalp        lesion-locked frame, regions, channel classes, change measures
    stats        mixed models, per-electrode slopes, ANCOVA, permutation, FDR
    reporting    the manuscript's p-value rounding rule
    recording    optional adapter from recordings on disk (needs MNE and YASA)
"""

from __future__ import annotations

__version__ = "3.0.0"

from . import (config, discharges, geometry, preprocess, reporting, scalp,
               spectral, stats)

__all__ = ["config", "discharges", "geometry", "preprocess", "reporting",
           "scalp", "spectral", "stats", "__version__"]
