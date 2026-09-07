"""Power spectrum -> aperiodic exponent.

The exponent is the study's only measured quantity; everything downstream is a
comparison between exponents. Two estimators are implemented:

``fit_aperiodic``
    The parameterised fit (specparam / FOOOF). Rhythmic peaks are modelled
    explicitly and removed, so the exponent describes the broadband background
    rather than being dragged around by whatever alpha or spindle activity
    happens to sit in the band.

``ols_log_log_slope``
    Ordinary least squares of log power on log frequency over the same range,
    with no peak model at all. This is *not* a fallback -- it is the
    sensitivity analysis reported in Methods 2.5, there to show that the
    SOZ/non-SOZ difference does not depend on peak removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from . import config as cfg


def compute_psd(epochs: np.ndarray, sfreq: float,
                fmin: float = cfg.PSD_FMIN_HZ,
                fmax: float = cfg.PSD_FMAX_HZ,
                window_seconds: float = cfg.WELCH_WINDOW_SECONDS,
                overlap: float = cfg.WELCH_OVERLAP,
                window: str = cfg.WELCH_WINDOW
                ) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per epoch and channel, averaged over epochs.

    The 4 s window fixes the frequency resolution at 0.25 Hz, fine enough to
    place peaks but coarse enough that each 5 s epoch still contributes
    several averaged segments.

    The taper is Hamming, which is what the reported analyses used. It is not
    scipy's default, and the difference is not cosmetic: a Hann taper shifts
    the estimated spectrum by about 1% and the recovered exponent by 0.015 --
    the same order as the SOZ effect the study reports. With Hamming this
    function reproduces the reference pipeline's spectra to machine precision.

    Returns ``(freqs, psd)`` with ``psd`` of shape ``(n_channels, n_freqs)``.
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError("expected epochs of shape (n_epochs, n_channels, n_times)")

    nperseg = min(int(round(window_seconds * sfreq)), epochs.shape[-1])
    freqs, psd = welch(epochs, fs=sfreq, nperseg=nperseg,
                       noverlap=int(nperseg * overlap), window=window, axis=-1)

    band = (freqs >= fmin) & (freqs <= fmax)
    return freqs[band], psd[:, :, band].mean(axis=0)


@dataclass(frozen=True)
class AperiodicFit:
    """One channel's aperiodic parameters and how well they fit."""

    exponent: float
    offset: float
    r_squared: float
    error: float

    def passes_qc(self,
                  min_r_squared: float = cfg.MIN_R_SQUARED,
                  min_exponent: float = cfg.MIN_EXPONENT) -> bool:
        """The intracranial criteria of Methods 2.5, applied before any statistic.

        The exponent floor is not a plausibility filter on the biology; in this
        band an exponent below 0.5 means the fit failed.

        Note the asymmetry between the modalities. Contacts are screened this
        way and the analysed set is what survives; scalp channels are not, and
        the whole 19-channel montage is carried through. See the note beside
        the thresholds in :mod:`aperiodic.config`.
        """
        return (np.isfinite(self.exponent)
                and np.isfinite(self.r_squared)
                and self.r_squared >= min_r_squared
                and self.exponent >= min_exponent)


def fit_aperiodic(freqs: np.ndarray, psd: np.ndarray,
                  max_n_peaks: int = cfg.MAX_N_PEAKS_SCALP,
                  fit_range: tuple[float, float] = cfg.FIT_RANGE_HZ
                  ) -> AperiodicFit:
    """Parameterise one channel's spectrum and return its aperiodic component.

    In ``fixed`` aperiodic mode the model is
    ``log10(P) = offset - exponent * log10(f)``: a straight line in log-log
    space, plus a set of Gaussian peaks. A steeper line (larger exponent) is
    the marker interpreted here as a shift towards inhibition.
    """
    from specparam import SpectralModel

    model = SpectralModel(
        aperiodic_mode=cfg.APERIODIC_MODE,
        peak_width_limits=list(cfg.PEAK_WIDTH_LIMITS_HZ),
        max_n_peaks=max_n_peaks,
        min_peak_height=cfg.MIN_PEAK_HEIGHT,
        verbose=False,
    )
    model.fit(np.asarray(freqs, dtype=float),
              np.asarray(psd, dtype=float),
              list(fit_range))

    params = model.results.params.aperiodic.params
    metrics = model.results.metrics.results
    return AperiodicFit(
        exponent=float(params[1]),
        offset=float(params[0]),
        r_squared=float(metrics.get("gof_rsquared", np.nan)),
        error=float(metrics.get("error_mae", np.nan)),
    )


def ols_log_log_slope(freqs: np.ndarray, psd: np.ndarray,
                      fit_range: tuple[float, float] = cfg.FIT_RANGE_HZ
                      ) -> AperiodicFit:
    """Raw spectral slope: no peak model, same frequency range.

    Reported with its sign flipped, so that it is directly comparable to the
    parameterised exponent (both larger = steeper).
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    band = (freqs >= fit_range[0]) & (freqs <= fit_range[1])

    x = np.log10(freqs[band])
    y = np.log10(psd[band])
    slope, intercept = np.polyfit(x, y, 1)

    resid = y - (intercept + slope * x)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    return AperiodicFit(
        exponent=float(-slope),
        offset=float(intercept),
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        error=float(np.mean(np.abs(resid))),
    )


def relative_band_power(freqs: np.ndarray, psd: np.ndarray,
                        band: tuple[float, float] = cfg.BREACH_BAND_HZ,
                        total: tuple[float, float] = (cfg.PSD_FMIN_HZ,
                                                      cfg.PSD_FMAX_HZ)) -> float:
    """Power in ``band`` as a percentage of power over ``total``.

    Used for the breach-rhythm check. A breach rhythm -- the high-frequency
    amplitude increase seen over a skull defect -- would be the obvious
    artefactual explanation for a post-operative spectral change, so it is
    tested for even though the procedure leaves the skull intact.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    in_total = (freqs >= total[0]) & (freqs <= total[1])

    denom = float(np.trapezoid(psd[in_total], freqs[in_total]))
    if denom <= 0:
        return float("nan")
    return 100.0 * float(np.trapezoid(psd[in_band], freqs[in_band])) / denom
