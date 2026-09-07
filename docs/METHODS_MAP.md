# Methods → code

Each numbered section of the paper's Methods, and the functions that implement it.
Thresholds are all in `aperiodic/config.py`; the constant names are given so a value in
the text can be traced to the line that fixes it.

## 2.4 EEG acquisition and sleep staging

| What the paper says | Where it lives |
|---|---|
| average reference | `preprocess.average_reference` |
| band-pass 0.5–45 Hz | `preprocess.bandpass` · `L_FREQ_HZ`, `H_FREQ_HZ` |
| automated staging, N2 only | `recording.stage` · `preprocess.stage_mask_to_samples` |
| 5 s fixed epochs | `preprocess.epoch` · `EPOCH_SECONDS` |
| per-recording peak-to-peak rejection | `preprocess.reject_adaptive` · `PTP_PERCENTILE`, `PTP_FLOOR_V_SCALP`, `PTP_FLOOR_V_INTRACRANIAL` |
| relative breach-band power, pre vs post | `spectral.relative_band_power` · `BREACH_BAND_HZ` |

The 45 Hz upper edge is below the 50 Hz mains frequency of the recording site. The
rejection threshold is `max(90th percentile, floor)` and has no ceiling — the reasoning
is in the `reject_adaptive` docstring.

## 2.5 Spectral parameterization

| What the paper says | Where it lives |
|---|---|
| Welch PSD, 4 s window | `spectral.compute_psd` · `WELCH_WINDOW_SECONDS`, `WELCH_OVERLAP` |
| specparam, fixed mode, 1–45 Hz | `spectral.fit_aperiodic` · `APERIODIC_MODE`, `FIT_RANGE_HZ`, `PEAK_WIDTH_LIMITS_HZ`, `MIN_PEAK_HEIGHT`, `MAX_N_PEAKS_*` |
| quality control | `spectral.AperiodicFit.passes_qc` · `MIN_R_SQUARED`, `MIN_EXPONENT` |
| raw spectral slope, no peak model | `spectral.ols_log_log_slope` |

The raw slope is the sensitivity analysis showing the SOZ effect does not depend on peak
removal, not a fallback for when the fit fails.

## 2.6 SOZ delineation and spatial-decay modeling

| What the paper says | Where it lives |
|---|---|
| SOZ contrast = mean(SOZ) − mean(non-SOZ) | `scalp.soz_contrast` |
| along-electrode distance to the SOZ centre | `geometry.Electrode.distance_mm` · `CONTACT_SPACING_MM` |
| gradient set: SOZ trajectories, within 63 mm | `geometry.gradient_set` · `MAX_DISTANCE_MM` |
| per-electrode slope control | `stats.per_electrode_slopes` · `MIN_CONTACTS_FOR_SLOPE` |
| within-patient class means, compared | `stats.compare_trajectory_classes` |
| channel-label parsing | `geometry.parse_contact_label` |

The distance metric is contact index times spacing — a property of the hardware, which
is why it needs no imaging, and why every distance analysis is specified *within* a
trajectory. SOZ-free electrodes have no SOZ centre, so the slope control regresses both
classes on the electrode's own axis (`geometry.electrode_axis_positions`).

## 2.7 Control for interictal epileptiform discharges

| What the paper says | Where it lives |
|---|---|
| Teager-Kaiser operator on 20–60 Hz | `discharges.teager_kaiser_energy` · `IED_BAND_HZ`, `NEO_SMOOTH_MS` |
| seven detector settings in parallel | `discharges.detector_settings` · `NEO_C_PER_CHANNEL`, `NEO_C_SHARED`, `ENVELOPE_K` |
| event counting with a refractory period | `discharges.count_events` · `REFRACTORY_MS` |
| discharge rate as a gradient covariate | `stats.distance_gradient_model(covariates=("Discharge_rate",))` |
| re-estimation on discharge-free epochs | `discharges.DischargeDetection.discharge_free_mask` · `MIN_DISCHARGE_FREE_EPOCHS` |
| equally sized random subset | `discharges.matched_random_subset` |

The 20 Hz lower edge, rather than the 10 Hz of the source method, is because all
analysis is on N2 and a 10 Hz edge fires on sleep spindles. The matched random subset
exists because dropping discharges also drops epochs, and fewer epochs alone move a
spectral estimate.

## 2.8 Anatomical-baseline correction

| What the paper says | Where it lives |
|---|---|
| region from the target structure, one keyword rule | `geometry.classify_region` · `REGION_KEYWORD_RULE` |
| region-wise model, contact capped at 18 | `stats.distance_gradient_model` · `MAX_CONTACT_OFFSET` |
| ≥ 3 electrodes of each class per region | `MIN_ELECTRODES_PER_REGION` |

The rule is ordered: specific keywords precede general ones, so "orbitofrontal" is not
swallowed by "frontal". It classifies the deep end of the trajectory, where contact 1
sits, not the entry point. Targets that fall in no grouped region return `None` and the
electrode is excluded rather than forced into a category.

## 2.9 Scalp topography, cross-modal correlation, outcome

| What the paper says | Where it lives |
|---|---|
| mirror-flip to a lesion-hemisphere frame | `scalp.mirror_to_lesion_frame` · `MIRROR_MAP` |
| three regions exhausting the montage | `scalp.region_channels` · `LEFT_CHANNELS`, `RIGHT_CHANNELS`, `MIDLINE_CHANNELS` |
| whole-scalp and regional exponents | `scalp.whole_scalp`, `scalp.region_mean` |
| ΔExponent = post − pre | `scalp.delta` |
| three cross-modal channel classes | `scalp.channel_classes` · `REGION_TO_LEADS` |
| the midline / region-mapped overlap | `ChannelClasses.overlap`, `.region_mapped_strict` |
| selective contrast index | `scalp.selective_contrast_index` |

The regions and the classes are different groupings of the same 19 channels, for
different questions. The regions partition the montage 8 + 8 + 3; the classes do not —
region-mapped can overlap midline, which is why the overlap is recorded explicitly and
`region_mapped_strict` exists for the sensitivity analysis.

## 2.11 Statistical analysis

| What the paper says | Where it lives |
|---|---|
| `Exponent ~ Is_SOZ + (1 \| Subject)` | `stats.soz_contrast_model` |
| gradient with nested random intercepts | `stats.distance_gradient_model` |
| paired t, Wilcoxon sensitivity, BF₀₁ | `stats.paired_change`, `stats.bayes_factor_01` |
| Cohen's d and its interval | `stats.cohens_d_paired`, `stats.cohens_d_ci` |
| Spearman with bootstrap CI | `stats.spearman_bootstrap` · `BOOTSTRAP_N`, `BOOTSTRAP_SEED` |
| partial correlation | `stats.partial_spearman` |
| one-sample tests under FDR | `stats.fdr` · `FDR_METHOD` |
| ANCOVA with permutation null | `stats.ancova_outcome` · `PERMUTATION_N` |
| two-sample permutation test | `stats.permutation_test` |

Two conventions run through all of it:

- **The unit of analysis is declared.** Contact-level quantities go into mixed models
  with a random intercept per patient; patient-level quantities go into ordinary tests.
- **Every effect estimate carries an interval.** `d` uses the analytic large-sample
  interval rather than a bootstrap, because it is monotone in the same statistic as the
  paired t test and so cannot disagree with its own p value; `ρ` uses a percentile
  bootstrap over patients, with a fixed seed.

## Reporting

`reporting.round_p` implements the manuscript's p-value rule, including the deliberate
exception that keeps three decimals between 0.045 and 0.055 so that results either side
of the conventional threshold do not print identically. It takes the value the analysis
produced — applying it to an already-rounded number gives a different and wrong answer,
which `demo/walkthrough.py` step 9 demonstrates.

## Not in this repository

Figure rendering, supplementary table generation, and the study's own driver scripts.
They are specific to one dataset's layout and to one journal's figure specification, and
they describe no part of the method. Everything the Methods section states is here.
