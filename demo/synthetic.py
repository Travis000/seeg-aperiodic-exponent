"""Synthetic recordings and cohorts, so that the computation can be run.

Nothing here is study data, and none of it is fitted to study data. The
generating parameters are round numbers chosen to make each step legible: a
seizure onset zone sits half an exponent below its surround, the surround
recovers linearly with distance, and the good-outcome group loses a tenth of an
exponent at the midline after surgery. Real effects are smaller and noisier.

The point of the generator is that the ground truth is known. When the
walkthrough recovers an exponent of about 1.5 from a signal built with an
exponent of 1.5, that is a statement about the estimator, and it is the only
kind of statement a synthetic dataset can support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aperiodic import config as cfg

SFREQ = 512.0

# ── Ground truth, all invented ───────────────────────────────────────────────
BASELINE_EXPONENT = 1.50
SOZ_DEPRESSION = -0.50          # exponent units at the SOZ centre
GRADIENT_RECOVERY_MM = 40.0     # distance over which the depression fades
CONTACT_NOISE = 0.06
SCALP_BASELINE = 1.30
MIDLINE_DROP_GOOD = -0.10       # post minus pre, good-outcome group
LATERAL_DROP_GOOD = -0.03
SUBJECT_SPREAD = 0.12


def pink_noise(exponent: float, n_samples: int, sfreq: float = SFREQ,
               rng: np.random.Generator | None = None,
               peak_hz: float = 10.0, peak_amplitude: float = 3.0,
               peak_width_hz: float = 1.5) -> np.ndarray:
    """A signal whose power spectrum is ``1/f**exponent`` plus one rhythm.

    Built in the frequency domain: white noise is scaled by
    ``f ** (-exponent / 2)`` in *amplitude*, which is ``f ** -exponent`` in
    power, then a Gaussian bump is added at ``peak_hz``.

    The peak is not decoration. It is there so that the two estimators in
    :mod:`aperiodic.spectral` can be told apart -- the parameterised fit should
    ignore it, the raw log-log slope cannot.
    """
    rng = rng or np.random.default_rng(0)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sfreq)

    amplitude = np.zeros_like(freqs)
    nonzero = freqs > 0
    amplitude[nonzero] = freqs[nonzero] ** (-exponent / 2.0)
    amplitude += peak_amplitude * np.exp(
        -((freqs - peak_hz) ** 2) / (2.0 * peak_width_hz ** 2))

    phase = rng.uniform(0, 2 * np.pi, freqs.size)
    spectrum = amplitude * np.exp(1j * phase)
    spectrum[0] = 0.0

    signal = np.fft.irfft(spectrum, n_samples)
    return signal / np.std(signal) * 20e-6      # scale to plausible EEG volts


def inject_discharges(signal: np.ndarray, rate_per_minute: float,
                      sfreq: float = SFREQ,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Add sharp biphasic transients at a given rate.

    Shaped as the derivative of a Gaussian, about 40 ms wide: brief and sharp,
    so the Teager-Kaiser operator responds to it far more than to a slow wave
    of the same amplitude.
    """
    rng = rng or np.random.default_rng(0)
    signal = signal.copy()

    duration_min = signal.size / sfreq / 60.0
    n_events = rng.poisson(rate_per_minute * duration_min)
    if n_events == 0:
        return signal

    width = int(0.02 * sfreq)
    t = np.arange(-2 * width, 2 * width + 1)
    shape = -t * np.exp(-(t ** 2) / (2.0 * (width / 2.0) ** 2))
    shape = shape / np.abs(shape).max() * 15.0 * np.std(signal)

    for onset in rng.integers(len(shape), signal.size - len(shape), n_events):
        signal[onset:onset + len(shape)] += shape
    return signal


def make_recording(exponent: float, minutes: float = 5.0,
                   discharge_rate: float = 0.0, sfreq: float = SFREQ,
                   seed: int = 0, n_channels: int = 1
                   ) -> tuple[np.ndarray, float]:
    """Synthetic N2 signal, shape ``(n_channels, n_samples)``.

    Channels are independent draws with the same target exponent. Independence
    is deliberate: it means the average reference cannot manufacture or destroy
    a gradient across them, so what the walkthrough recovers is attributable to
    the estimator rather than to the montage.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(minutes * 60 * sfreq)

    channels = []
    for _ in range(n_channels):
        signal = pink_noise(exponent, n_samples, sfreq, rng)
        if discharge_rate > 0:
            signal = inject_discharges(signal, discharge_rate, sfreq, rng)
        channels.append(signal)
    return np.stack(channels), sfreq


def make_contact_table(n_subjects: int = 16, n_electrodes: int = 8,
                       n_soz_electrodes: int = 3, n_contacts: int = 12,
                       seed: int = 7) -> pd.DataFrame:
    """A synthetic intracranial dataset with a known SOZ depression and gradient.

    Several electrodes per patient carry the SOZ and the rest do not, which the
    analysis needs in both directions: SOZ-free trajectories are the comparison
    class for the slope control, and more than one SOZ trajectory per patient
    is what makes patient and electrode distinguishable groupings. With one
    SOZ electrode each, the nested random intercepts of the gradient model
    describe the same partition and cannot both be estimated.

    The exponent of a contact is

        baseline + subject offset + depression * exp(-distance / recovery)

    so the depression is deepest at the SOZ centre and fades with distance --
    a gradient present on SOZ-bearing electrodes only, and flat elsewhere.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for s in range(n_subjects):
        subject = f"S{s + 1:02d}"
        subject_offset = rng.normal(0, SUBJECT_SPREAD)

        for e in range(n_electrodes):
            electrode = chr(ord("A") + e)
            contains_soz = e < n_soz_electrodes
            # Near the deep end, where contact 1 sits at the target structure
            # -- the same convention the region keyword rule assumes. The
            # position varies between electrodes so that the gradient is not
            # the same curve repeated.
            first = int(rng.integers(2, 5))
            soz_contacts = {first, first + 1}
            soz_center = float(np.mean(sorted(soz_contacts)))

            for contact in range(1, n_contacts + 1):
                if contains_soz:
                    distance = abs(contact - soz_center) * cfg.CONTACT_SPACING_MM
                    effect = SOZ_DEPRESSION * np.exp(-distance / GRADIENT_RECOVERY_MM)
                else:
                    distance = np.nan
                    effect = 0.0

                rows.append({
                    "Subject": subject,
                    "Electrode": f"{subject}-{electrode}",
                    "Contact": contact,
                    "Contains_SOZ": contains_soz,
                    "Is_SOZ": contains_soz and contact in soz_contacts,
                    "Distance_mm": distance,
                    "Exponent": (BASELINE_EXPONENT + subject_offset + effect
                                 + rng.normal(0, CONTACT_NOISE)),
                    # Discharges are denser near the SOZ -- the confound the
                    # controls in Methods 2.7 exist to rule out. The rate is
                    # deliberately noisy rather than a clean function of
                    # distance: a deterministic one would be perfectly
                    # collinear with distance, and the two coefficients in the
                    # adjusted model would then split arbitrarily between them,
                    # which says nothing about whether the control works.
                    #
                    # Note also that the exponent above does *not* depend on
                    # the discharge rate. In this synthetic cohort the gradient
                    # is real and the discharges merely accompany it, so the
                    # covariate should leave the slope roughly where it was.
                    "Discharge_rate": max(0.1, (
                        8.0 * np.exp(-distance / 25.0) * rng.lognormal(0, 0.5)
                        if contains_soz else abs(rng.normal(0.5, 0.3)))),
                })

    return pd.DataFrame(rows)


def make_scalp_table(contacts: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Paired pre/post scalp exponents, one row per subject and channel.

    Built so that the good-outcome group falls at the midline after surgery and
    the poor-outcome group does not. Outcome is assigned by the synthetic SOZ
    contrast -- deeper contrast, better outcome -- which is a relationship the
    walkthrough then recovers, and a reminder that the demo can only ever find
    what was written into it.
    """
    rng = np.random.default_rng(seed)
    contrast = (contacts.groupby("Subject")
                .apply(lambda g: (g.loc[g.Is_SOZ, "Exponent"].mean()
                                  - g.loc[~g.Is_SOZ, "Exponent"].mean()),
                       include_groups=False))

    median_contrast = contrast.median()
    rows = []
    for subject, value in contrast.items():
        outcome = "Good" if value < median_contrast else "Poor"
        level = SCALP_BASELINE + rng.normal(0, SUBJECT_SPREAD)
        side = "Left" if rng.random() < 0.5 else "Right"
        # Patients differ in how much they change, not only in where they
        # start. Without this term every patient in a group would move by
        # exactly the same amount and the effect sizes would be absurd.
        responsiveness = rng.normal(1.0, 0.5)

        for channel in cfg.MONTAGE_1020:
            pre = level + rng.normal(0, 0.05)
            if outcome == "Good":
                change = responsiveness * (
                    MIDLINE_DROP_GOOD if channel in cfg.MIDLINE_CHANNELS
                    else LATERAL_DROP_GOOD)
            else:
                change = 0.0
            rows.append({
                "Subject": subject, "Channel": channel, "Outcome": outcome,
                "Lesion_side": side, "Contrast": value,
                "Pre": pre, "Post": pre + change + rng.normal(0, 0.04),
            })
    return pd.DataFrame(rows)
