# Aperiodic exponent in focal epilepsy — reference implementation

Analysis code accompanying a study of the **aperiodic (1/f) exponent** of the EEG and
SEEG power spectrum as a marker of cortical excitation/inhibition balance in focal
epilepsy treated with SEEG-guided radiofrequency thermocoagulation.

**This repository contains the computation, not the study data.** It is a readable
implementation of the paper's Methods: every step that turns a recording into a number,
and every model that turns those numbers into a claim. It holds no recordings, no
derived result files, no file paths, and no information about individual patients.

It runs end to end on synthetic data, so that each step can be executed and inspected
rather than only read:

```bash
pip install -r requirements.txt
python demo/walkthrough.py
```

The walkthrough prints every intermediate quantity — filter output, epoch counts,
rejection thresholds, spectra, fitted exponents, model coefficients, confidence
intervals — against a known ground truth. Because the synthetic data are generated with
parameters chosen for legibility, the numbers it prints are properties of the
estimators. **They are not study results and do not reproduce any figure in the paper.**

## What is here

```
aperiodic/            the computation, in pipeline order
  config.py           every fixed threshold, annotated with its Methods section
  preprocess.py       average reference, band-pass, N2 mask, epoching, artefact rejection
  spectral.py         Welch PSD, specparam fit, raw log-log slope, quality control
  geometry.py         contact-label parsing, SOZ centre, along-electrode distance, regions
  discharges.py       Teager-Kaiser detection (7 settings) and the discharge controls
  scalp.py            lesion-locked frame, regions, channel classes, ΔExponent, SCI
  stats.py            mixed models, per-electrode slopes, ANCOVA, permutation, FDR
  reporting.py        the manuscript's p-value rounding rule
  recording.py        optional adapter from recordings on disk (needs MNE and YASA)

demo/
  synthetic.py        generators with known ground truth
  walkthrough.py      the end-to-end run

docs/
  METHODS_MAP.md      Methods section → function, both directions
```

`aperiodic/` depends only on NumPy, SciPy, pandas, statsmodels and specparam, and every
function takes plain arrays or DataFrames. Reading recordings off disk and automated
sleep staging need MNE and YASA and are confined to `recording.py`, which is imported
nowhere else — so the computation can be read, tested and run without either.

## The method in one paragraph

Per recording, identically for scalp and intracranial data: average reference → FIR
band-pass 0.5–45 Hz → automated sleep staging, keep stage 2 NREM only → 5 s
fixed-length epochs → adaptive peak-to-peak artefact rejection → Welch PSD (4 s window)
→ specparam fit in fixed aperiodic mode over 1–45 Hz → per-channel exponent, offset and
R². Intracranial contacts are then mapped to their clinically defined seizure onset zone
status and to along-electrode distance from the SOZ centre, and group statistics are
computed with the unit of analysis stated explicitly — mixed models with a random
intercept per patient for contact-level quantities, ordinary tests for patient-level
ones.

`docs/METHODS_MAP.md` maps each Methods section of the paper to the functions that
implement it.

## Data

Under the ethics approval governing these clinical recordings, the recordings are not
distributed, and no derived per-patient file is included here either. De-identified
individual-level data are available as described in the paper's Data availability
statement.

`aperiodic/` never reads a path. `recording.py` reads the path you give it. There is no
`DATA_ROOT`, no expected folder layout, and no dataset-specific file naming, because
none of that is part of the method.

To run the pipeline on your own recordings, `recording.py` states the contract: give it
an EDF and the channels to keep, and it returns the array the rest of the package takes.

## Reproducibility notes

- Every threshold is in `aperiodic/config.py`, fixed before analysis and applied
  identically to every recording. Nothing in the package is fitted to data.
- Bootstrap and permutation procedures take an explicit seed (`config.BOOTSTRAP_SEED`),
  so a published interval is reproducible digit for digit.
- Where a choice could have gone either way — the discharge detector's operating point,
  the two threshold families — the code runs all of them rather than one, and the
  docstring says what each buys and costs.
- `reporting.py` implements the manuscript's p-value rounding as a function of the value
  produced by the analysis, not of a number already rounded once. Rounding twice is not
  the same as rounding once, and this is where that is enforced.

## How far this has been checked

The signal-processing chain and the primary model are verified by direct numerical
comparison against the pipeline that produced the reported analyses, not by inspection:
the band-pass and the Welch spectrum against the same operations in MNE (the spectra
agree to machine precision), the discharge detectors against the source formulas
event for event, the seizure-onset-zone mixed model against the same model specified
directly, and the p-value rounding across thousands of values including every boundary
case.

Three defaults were caught this way and are worth naming, because each of them changes
a number by about as much as the effects the study reports: filtering forward and
backward rather than once, tapering the Welch segments with Hann rather than Hamming,
and thresholding the envelope detector against a robust centre rather than the mode it
is defined against. The last of these produced conspicuously tidier output while being
the wrong algorithm.

The remaining statistical procedures -- the ANCOVA, the FDR correction, the partial
correlations, the effect-size intervals -- have no single counterpart in the source
project to compare against, and have been checked by reading the formulas rather than
by running them side by side. That is the current limit of the evidence.

## Requirements

Python ≥ 3.10. Versions in `requirements.txt` are those the reported analyses were run
under. Note that `specparam==2.0.0rc6` is a pre-release, so install with `--pre` or keep
the exact pin:

```bash
pip install --pre -r requirements.txt
```

## Citation

Please cite the associated paper; details will be added on publication.

## License

MIT — see [LICENSE](LICENSE).
