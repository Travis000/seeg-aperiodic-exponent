"""Interictal discharge detection, and the control it exists to support.

The concern this addresses is specific. Interictal discharges are denser near
the seizure onset zone and inject broadband power. A detector-free analysis
could therefore produce a distance-dependent exponent gradient that is nothing
but a discharge-density gradient.

Two controls follow, in increasing severity (Methods 2.7):

1. discharge rate as a covariate in the gradient model;
2. re-estimating the exponent on only those epochs in which the contact carried
   no detected discharge -- and, because that also removes epochs, on an
   equally sized random subset of the retained epochs, so that the effect of
   removing discharges is separated from the effect of having fewer epochs.

Seven detector settings run in parallel. No single operating point is
privileged: a control that holds only at one threshold is not a control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from . import config as cfg
from .preprocess import bandpass


def teager_kaiser_energy(data: np.ndarray, sfreq: float,
                         smooth_ms: float = cfg.NEO_SMOOTH_MS) -> np.ndarray:
    """Smoothed Teager-Kaiser nonlinear energy operator.

    ``psi(x[n]) = x[n]^2 - x[n-1] * x[n+1]``, which for a narrowband signal
    tracks ``A^2 * omega^2``. Because it is quadratic in frequency as well as
    amplitude it responds to *sharpness*, not amplitude alone -- which is what
    separates a discharge from a large slow wave.

    Edges are held rather than zero-padded, negatives clipped (the operator can
    go negative on noise, and only positive excursions are events), then a
    moving average turns a spiky instantaneous trace into a usable envelope.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    energy = np.empty_like(data)
    energy[:, 1:-1] = data[:, 1:-1] ** 2 - data[:, :-2] * data[:, 2:]
    energy[:, 0] = energy[:, 1]
    energy[:, -1] = energy[:, -2]
    np.maximum(energy, 0, out=energy)

    width = max(1, int(round(smooth_ms * sfreq / 1000.0)))
    return uniform_filter1d(energy, width, axis=-1, mode="nearest")


def envelope(data: np.ndarray) -> np.ndarray:
    """Analytic-signal amplitude envelope -- the second detector family.

    Deliberately a different principle from the energy operator: amplitude
    only, no frequency weighting. Agreement between the two families is
    evidence that the control does not hinge on one detector's quirks.
    """
    return np.abs(hilbert(np.atleast_2d(np.asarray(data, dtype=float)), axis=-1))


def lognormal_mode(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mode of a lognormal fitted to ``values``: ``exp(mean(log) - var(log))``.

    This is the background level the envelope detector thresholds against, and
    it must be estimated **per window**, not once over the whole recording.
    Pooling windows mixes different sleep states into one fit, which inflates
    the variance; since the mode falls as the variance rises, the threshold
    then sinks into the background and the detector fires everywhere. The
    source implementation carries a note that its first version did exactly
    this.

    Note what this threshold is *not*: a robust centre like the median. The
    mode of a lognormal sits below its median by ``exp(-variance)``, so on a
    bursty window the threshold drops rather than rises. Substituting a median
    produces a detector that looks better behaved on quiet data and is not the
    published one.
    """
    # Guard against an exact zero in the envelope, which would send log to -inf
    # and take the whole window's mean with it.
    log_values = np.log(np.maximum(np.asarray(values, dtype=float), 1e-18))
    return np.exp(log_values.mean(axis=axis) - log_values.var(axis=axis))


def count_events(above: np.ndarray, refractory_samples: int) -> np.ndarray:
    """Count events in a supra-threshold mask, merging what falls inside one
    refractory period.

    Without this, one discharge lasting 60 ms at 512 Hz would be counted as
    thirty. Implemented as max-pooling into refractory-length bins followed by
    counting rising edges, which is linear in signal length and gives the same
    answer as walking the mask.
    """
    above = np.atleast_2d(np.asarray(above, dtype=bool))
    n_bins = above.shape[-1] // refractory_samples
    if n_bins < 2:
        return above.any(-1).astype(int)

    trimmed = above[..., :n_bins * refractory_samples]
    binned = trimmed.reshape(above.shape[:-1] + (n_bins, refractory_samples)).any(-1)
    rising = (np.diff(binned.astype(np.int8), axis=-1) == 1).sum(-1)
    return (rising + binned[..., 0]).astype(int)


def detector_settings() -> list[tuple[str, str, float]]:
    """The seven settings, as ``(name, family, coefficient)``.

    ``neo*``  per-channel threshold: coefficient x that channel's own median
              energy. Controls for amplitude differences due to impedance and
              to grey versus white matter -- but also normalises away "this
              contact is intrinsically spikier", which is the signal under test.
    ``neoS*`` shared threshold: coefficient x a median pooled across channels.
              Keeps between-contact differences, admits amplitude bias.
    ``env*``  envelope family, as a cross-method check.

    The two threshold families err in opposite directions, so both are run and
    a conclusion is only accepted if it holds under both.
    """
    settings = [(f"neo{c:g}", "neo_per_channel", c) for c in cfg.NEO_C_PER_CHANNEL]
    settings += [(f"neoS{c:g}", "neo_shared", c) for c in cfg.NEO_C_SHARED]
    settings += [(f"env{k:g}", "envelope", k) for k in cfg.ENVELOPE_K]
    return settings


@dataclass(frozen=True)
class DischargeDetection:
    """Per-channel discharge counts and the epoch mask they imply."""

    setting: str
    counts_per_epoch: np.ndarray        # (n_channels, n_epochs)
    threshold: np.ndarray               # (n_channels,)

    @property
    def rate_per_minute(self) -> np.ndarray:
        """Mean discharges per minute per channel."""
        n_epochs = self.counts_per_epoch.shape[1]
        if n_epochs == 0:
            return np.full(self.counts_per_epoch.shape[0], np.nan)
        minutes = n_epochs * cfg.EPOCH_SECONDS / 60.0
        return self.counts_per_epoch.sum(axis=1) / minutes

    def discharge_free_mask(self) -> np.ndarray:
        """``(n_channels, n_epochs)``: True where that channel had no discharge.

        Per channel, not per epoch. An epoch is discarded for a contact that
        discharged in it, while remaining available to every other contact --
        so the re-estimation is not gated by the noisiest contact in the array.
        """
        return self.counts_per_epoch == 0


def detect(epochs: np.ndarray, sfreq: float, setting: str = "neo6"
           ) -> DischargeDetection:
    """Run one detector setting over epoched data.

    ``epochs`` is ``(n_epochs, n_channels, n_times)`` -- the same epochs the
    exponents were computed from, which is the point: the control has to speak
    about the data actually analysed, not a separate pass over the recording.
    """
    epochs = np.asarray(epochs, dtype=float)
    n_epochs, n_channels, n_times = epochs.shape

    match = [s for s in detector_settings() if s[0] == setting]
    if not match:
        raise ValueError(f"unknown setting {setting!r}; "
                         f"expected one of {[s[0] for s in detector_settings()]}")
    _, family, coefficient = match[0]

    # Channels x time, so that a threshold can be taken over the whole
    # recording rather than epoch by epoch.
    continuous = epochs.transpose(1, 0, 2).reshape(n_channels, -1)
    filtered = bandpass(continuous, sfreq, *cfg.IED_BAND_HZ)

    if family == "envelope":
        trace = envelope(filtered)
    else:
        trace = teager_kaiser_energy(filtered, sfreq)

    # (n_channels, n_epochs, n_times), so a threshold can be taken per epoch
    # where the family calls for it.
    windowed = trace.reshape(n_channels, n_epochs, n_times)

    if family == "envelope":
        # Per epoch and channel: the mode of a lognormal fitted to that window.
        threshold = coefficient * lognormal_mode(windowed, axis=-1)
    elif family == "neo_shared":
        # One number for the whole array: differences between contacts survive.
        threshold = np.full((n_channels, n_epochs),
                            coefficient * float(np.median(trace)))
    else:
        # Per channel over the whole recording: amplitude differences between
        # contacts are divided out.
        threshold = np.repeat((coefficient * np.median(trace, axis=-1))[:, None],
                              n_epochs, axis=1)

    above = windowed > threshold[:, :, None]
    refractory = max(1, int(round(cfg.REFRACTORY_MS * sfreq / 1000.0)))

    counts = np.stack([count_events(above[:, i, :], refractory)
                       for i in range(n_epochs)], axis=1)

    return DischargeDetection(setting=setting, counts_per_epoch=counts,
                              threshold=threshold)


def matched_random_subset(keep: np.ndarray, rng: np.random.Generator
                          ) -> np.ndarray:
    """A random epoch subset of the same size as ``keep``, from all epochs.

    The second half of the stringent control. Re-estimating on discharge-free
    epochs changes two things at once: it removes discharges, and it leaves
    fewer epochs, which alone makes a spectral estimate noisier. Matching the
    count isolates the first.
    """
    n_epochs = keep.size
    n_selected = int(keep.sum())
    subset = np.zeros(n_epochs, dtype=bool)
    if n_selected:
        subset[rng.choice(n_epochs, size=n_selected, replace=False)] = True
    return subset
