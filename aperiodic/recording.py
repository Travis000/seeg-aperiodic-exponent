"""Optional adapter: recordings on disk -> the arrays the rest of the code takes.

Kept separate, and imported nowhere else in the package, for two reasons.

First, it is the only part that needs MNE and YASA, and the only part that
cannot be exercised without a recording. The computation proper works on plain
arrays and runs anywhere.

Second, this is where a pipeline usually grows site-specific: file layout,
channel naming, which montage a particular amplifier writes. None of that
belongs in a description of a method, so this module states the contract and
leaves the file handling to whoever has the files.

Nothing here reads a fixed path. ``load_recording`` takes the path you give it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as cfg
from .preprocess import (average_reference, bandpass, epoch, reject_adaptive,
                         stage_mask_to_samples)


@dataclass
class Recording:
    """A staged, epoched recording ready for spectral estimation.

    ``data`` is ``(n_channels, n_samples)`` in volts, band-passed and
    average-referenced. ``n2_mask`` marks the samples staged N2.
    """

    channels: list[str]
    data: np.ndarray
    sfreq: float
    n2_mask: np.ndarray

    @property
    def n2_seconds(self) -> float:
        return float(self.n2_mask.sum()) / self.sfreq


#: Preference order for the channel handed to the sleep stager. Automated
#: staging is trained on central and parietal derivations, so a frontal or
#: occipital channel degrades it. In intracranial recordings there is no such
#: channel at all, and the stager is given a fixed contact chosen per recording
#: before any analysis -- that choice is part of the dataset, not of the
#: method, and is therefore supplied by the caller.
STAGING_PREFERENCE = ("C4", "C3", "Cz", "P4", "P3", "Pz", "F4", "F3")


def pick_staging_channel(channels, preference=STAGING_PREFERENCE) -> str:
    """First available channel from the preference order."""
    for candidate in preference:
        if candidate in channels:
            return candidate
    eeg = [ch for ch in channels if ch in cfg.MONTAGE_1020]
    if eeg:
        return eeg[0]
    return list(channels)[0]


def load_recording(path, picks=None, l_freq=cfg.L_FREQ_HZ,
                   h_freq=cfg.H_FREQ_HZ) -> Recording:
    """Read one recording, pick channels, average-reference and band-pass.

    ``picks`` selects channels by name; for scalp data pass
    ``config.MONTAGE_1020``, for intracranial data pass the contacts that
    :func:`aperiodic.geometry.parse_contact_label` accepted.

    The N2 mask is left empty -- call :func:`stage` to fill it.
    """
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    if picks is not None:
        available = [ch for ch in picks if ch in raw.ch_names]
        raw.pick(available)

    data = bandpass(average_reference(raw.get_data()),
                    raw.info["sfreq"], l_freq, h_freq)
    return Recording(channels=list(raw.ch_names), data=data,
                     sfreq=float(raw.info["sfreq"]),
                     n2_mask=np.zeros(data.shape[-1], dtype=bool))


def stage(recording: Recording, staging_channel: str | None = None) -> Recording:
    """Fill in ``n2_mask`` by automated sleep staging.

    Restricting to N2 is not a convenience. The aperiodic exponent varies with
    vigilance state, so comparing a pre-operative recording to a post-operative
    one without fixing the state would measure how much sleep each contained.
    N2 is chosen because it is reliably present in a routine clinical
    recording, unlike slow-wave sleep or REM.
    """
    import mne
    import yasa

    channel = staging_channel or pick_staging_channel(recording.channels)
    info = mne.create_info(recording.channels, recording.sfreq, "eeg")
    raw = mne.io.RawArray(recording.data, info, verbose=False)

    hypnogram = yasa.SleepStaging(raw, eeg_name=channel).predict()
    recording.n2_mask = stage_mask_to_samples(
        list(hypnogram), recording.data.shape[-1], recording.sfreq)
    return recording


def epoch_and_reject(recording: Recording, intracranial: bool = False):
    """N2 mask -> epochs -> adaptive rejection. Returns the rejection record."""
    epochs = epoch(recording.data, recording.sfreq, recording.n2_mask)
    floor = (cfg.PTP_FLOOR_V_INTRACRANIAL if intracranial
             else cfg.PTP_FLOOR_V_SCALP)
    return reject_adaptive(epochs, floor_v=floor)
