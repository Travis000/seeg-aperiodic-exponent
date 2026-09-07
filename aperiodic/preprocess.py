"""From a continuous recording to the epochs an exponent is computed on.

The order is fixed and identical for scalp and intracranial data:

    average reference -> band-pass -> keep N2 only -> 5 s epochs
    -> adaptive peak-to-peak rejection

Everything here operates on plain arrays of shape ``(n_channels, n_samples)``
in volts. Reading recordings off disk and automated sleep staging need MNE and
YASA and live in :mod:`aperiodic.recording`; they are kept separate so that the
computation can be read, tested and run without either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, oaconvolve

from . import config as cfg


def average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the instantaneous mean across channels from every channel.

    Applied to both modalities. It removes the (unrecorded) reference from the
    estimate, at the cost of making channels slightly dependent -- acceptable
    here because the exponent is read per channel, not from between-channel
    covariance.

    With independent channels of similar variance the subtraction scales every
    channel's power by roughly ``1 - 1/n_channels`` at all frequencies, so it
    shifts the aperiodic offset and leaves the exponent alone.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError(
            "average reference needs at least two channels; with one channel "
            "it subtracts the channel from itself and returns zeros")
    return data - data.mean(axis=0, keepdims=True)


def bandpass(data: np.ndarray, sfreq: float,
             l_freq: float = cfg.L_FREQ_HZ,
             h_freq: float = cfg.H_FREQ_HZ) -> np.ndarray:
    """Zero-phase FIR band-pass, single pass with the delay compensated.

    Zero-phase matters twice over. The discharge detector in
    :mod:`aperiodic.discharges` locates events in time on the same signal, and
    a filter with group delay would shift them relative to the epochs the
    exponents came from. And the exponent itself is read from the shape of the
    spectrum, so anything the filter does to the band edges is read as signal.

    Two details are what they are because the reported analyses used MNE's
    ``raw.filter(..., fir_design="firwin")``, and a reference implementation
    that quietly filtered differently would not describe those analyses:

    * The filter is applied **once**, with its linear-phase delay removed by
      taking the centre of the convolution. Forward-backward filtering
      (``filtfilt``) is also zero-phase but applies the response twice, which
      squares it -- and since the fit range ends exactly at the upper edge, a
      squared roll-off there is read as a steeper spectrum. Measured on
      synthetic 1/f against the reference pipeline, filtering twice inflates
      the recovered exponent by 0.013; what this function does instead leaves
      a residual of 0.0005.
    * The -6 dB points sit half a transition band *outside* the requested
      passband, not on it. Placing them on the edges instead attenuates inside
      the fit range and tilts the estimate the same way.
    """
    data = np.asarray(data, dtype=float)
    nyq = sfreq / 2.0

    # Transition bandwidths: a quarter of the edge frequency, floored at 2 Hz,
    # but never wider than the edge itself (at 0.5 Hz that last clause binds).
    # The narrower of the two sets the filter length; odd length keeps the
    # filter exactly linear phase.
    l_trans = min(max(l_freq * 0.25, 2.0), l_freq)
    h_trans = min(max(h_freq * 0.25, 2.0), nyq - h_freq)
    n_taps = int(round(3.3 * sfreq / min(l_trans, h_trans)))
    n_taps = min(n_taps | 1, (data.shape[-1] // 3) | 1)

    cutoff = [(l_freq - l_trans / 2.0) / nyq,
              min(h_freq + h_trans / 2.0, nyq * 0.999) / nyq]
    taps = firwin(n_taps, cutoff, pass_zero=False, window="hamming")

    delay = (n_taps - 1) // 2

    # Pad by reflecting about each endpoint before convolving. Convolving the
    # raw array instead implicitly pads it with zeros, and a filter this long
    # then corrupts the first and last few seconds -- measured against the
    # reference pipeline, the first and last 3,379 samples came out with errors
    # nearly three times the signal's own RMS, which is one or two whole epochs
    # at each end. Reflection is what the reference pipeline does, and it
    # brings those edges into line with the rest of the trace.
    pad = min(delay, data.shape[-1] - 1)
    padded = np.pad(data, ((0, 0), (pad, pad)), mode="reflect", reflect_type="odd")

    filtered = oaconvolve(padded, taps[None, :], mode="full", axes=-1)
    start = delay + pad
    return filtered[:, start:start + data.shape[-1]]


def stage_mask_to_samples(stages: list[str], n_samples: int, sfreq: float,
                          keep: str = "N2",
                          stage_seconds: float = cfg.STAGING_EPOCH_SECONDS
                          ) -> np.ndarray:
    """Expand a per-30-s hypnogram into a boolean mask over samples."""
    mask = np.zeros(n_samples, dtype=bool)
    step = int(round(stage_seconds * sfreq))
    for i, stage in enumerate(stages):
        if stage == keep:
            start = i * step
            mask[start:min(start + step, n_samples)] = True
    return mask


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` sample ranges of the True runs in ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = (np.flatnonzero(edges == 1) + 1).tolist()
    stops = (np.flatnonzero(edges == -1) + 1).tolist()
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(mask.size)
    return list(zip(starts, stops))


def epoch(data: np.ndarray, sfreq: float, mask: np.ndarray | None = None,
          seconds: float = cfg.EPOCH_SECONDS) -> np.ndarray:
    """Cut fixed-length, non-overlapping epochs.

    With a ``mask``, epochs are cut *within* each retained segment and each
    segment's remainder is dropped, so no epoch straddles a discontinuity.

    The reported analyses reached the same requirement by a different route:
    the retained segments were concatenated and epochs cut along the joined
    timeline, with those overlapping a join discarded afterwards. Both refuse
    to average across a break in the recording, but they do not select the same
    epochs -- one aligns each epoch grid to its own segment, the other to the
    concatenation. Expect epoch counts to differ by up to one per segment.

    Returns an array of shape ``(n_epochs, n_channels, n_times)``.
    """
    data = np.asarray(data, dtype=float)
    n_times = int(round(seconds * sfreq))
    segments = ([(0, data.shape[-1])] if mask is None
                else contiguous_segments(mask))

    out = []
    for start, stop in segments:
        n_full = (stop - start) // n_times
        for k in range(n_full):
            a = start + k * n_times
            out.append(data[:, a:a + n_times])
    if not out:
        return np.empty((0, data.shape[0], n_times))
    return np.stack(out)


@dataclass(frozen=True)
class RejectionResult:
    """What the adaptive rejection did, kept for the per-recording log."""

    epochs: np.ndarray
    n_created: int
    n_clean: int
    threshold_v: float
    percentile_v: float

    @property
    def n_rejected(self) -> int:
        return self.n_created - self.n_clean


def reject_adaptive(epochs: np.ndarray, floor_v: float,
                    percentile: float = cfg.PTP_PERCENTILE) -> RejectionResult:
    """Drop epochs whose worst channel swings further than the threshold.

    The threshold is ``max(percentile of the per-epoch peak-to-peak amplitudes,
    a fixed floor)``. Two properties are wanted at once and neither term gives
    both:

    * A *relative* threshold adapts to recordings that differ in overall
      amplitude, which they do across patients and between the two modalities.
      On its own it would always discard the noisiest tenth, including in a
      recording that is clean throughout.
    * The *floor* stops that: in a clean recording the percentile falls below
      the floor, the floor wins, and nothing is rejected.

    There is deliberately no ceiling. A recording noisy enough that its 90th
    percentile is very high is not silently truncated; it keeps its epochs and
    is caught later by the spectral quality criteria instead.
    """
    epochs = np.asarray(epochs, dtype=float)
    n_created = int(epochs.shape[0])
    if n_created == 0:
        return RejectionResult(epochs, 0, 0, float(floor_v), float("nan"))

    ptp = epochs.max(axis=2) - epochs.min(axis=2)      # (n_epochs, n_channels)
    worst = ptp.max(axis=1)                            # (n_epochs,)

    pct = float(np.percentile(worst, percentile))
    threshold = max(pct, float(floor_v))
    keep = worst <= threshold

    return RejectionResult(
        epochs=epochs[keep],
        n_created=n_created,
        n_clean=int(keep.sum()),
        threshold_v=threshold,
        percentile_v=pct,
    )
