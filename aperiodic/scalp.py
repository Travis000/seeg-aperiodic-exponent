"""Scalp summaries: lesion-locked frame, regions, channel classes, change.

Two different groupings of the same 19 channels are used, for two different
questions, and they must not be confused (Methods 2.9):

*Regions* -- ipsilateral / contralateral / midline -- exhaust the montage
(8 + 8 + 3) and are the unit of the topographic comparisons.

*Classes* -- midline / region-mapped / non-corresponding -- are defined by each
patient's own seizure onset zone and are the unit of the cross-modal analyses.
They do not partition the montage cleanly: the region-mapped class can overlap
the midline in patients whose SOZ region maps onto midline leads. The
non-corresponding class is disjoint from both, which is what the "selective"
comparison relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as cfg


def mirror_to_lesion_frame(values: dict[str, float], lesion_side: str
                           ) -> dict[str, float]:
    """Re-express a channel map so that "ipsilateral" means the lesion side.

    Patients differ in which hemisphere is affected, so a plain left/right
    average would cancel any lateralised effect across the group. Left-lesion
    patients are mirror-flipped through the midline; right-lesion patients are
    returned unchanged. Midline channels map to themselves.

    After this, ``RIGHT_CHANNELS`` holds the ipsilateral region in every
    patient. The convention is arbitrary but must be fixed, because the
    topographic contrast is read off it.
    """
    side = (lesion_side or "").strip().lower()
    if side.startswith("r"):
        return dict(values)
    if not side.startswith("l"):
        raise ValueError(f"lesion_side must be left or right, got {lesion_side!r}")
    return {cfg.MIRROR_MAP[ch]: v for ch, v in values.items()
            if ch in cfg.MIRROR_MAP}


def region_channels() -> dict[str, tuple[str, ...]]:
    """The three regions, in the lesion-locked frame."""
    return {
        "ipsilateral": cfg.RIGHT_CHANNELS,
        "contralateral": cfg.LEFT_CHANNELS,
        "midline": cfg.MIDLINE_CHANNELS,
    }


def region_mean(values: dict[str, float], channels) -> float:
    """Mean over the channels of a region, ignoring any that are absent."""
    present = [values[ch] for ch in channels
               if ch in values and np.isfinite(values[ch])]
    return float(np.mean(present)) if present else float("nan")


def whole_scalp(values: dict[str, float]) -> float:
    """Mean over all 19 channels -- the study's macroscopic summary."""
    return region_mean(values, cfg.MONTAGE_1020)


@dataclass(frozen=True)
class ChannelClasses:
    """One patient's cross-modal channel classes.

    ``overlap`` records which region-mapped channels are also midline. It is
    kept explicitly rather than resolved silently, because it is the reason the
    two classes cannot be treated as independent, and it drives the sensitivity
    analysis that drops the shared channels.
    """

    midline: tuple[str, ...]
    region_mapped: tuple[str, ...]
    non_corresponding: tuple[str, ...]
    overlap: tuple[str, ...]

    @property
    def region_mapped_strict(self) -> tuple[str, ...]:
        """Region-mapped channels with any midline overlap removed."""
        return tuple(ch for ch in self.region_mapped if ch not in self.midline)


def channel_classes(soz_region: str, lesion_side: str,
                    available: tuple[str, ...] = cfg.MONTAGE_1020
                    ) -> ChannelClasses:
    """Build the three classes for one patient.

    ``soz_region`` is the patient's modal SOZ region across their SOZ-bearing
    electrodes; ``REGION_TO_LEADS`` maps it to scalp leads by a fixed table.
    Contralateral leads are then dropped, because the class is defined as the
    *ipsilateral* derivations overlying the region -- which is what makes it
    one to three channels per patient rather than a symmetric pair.

    Channel names are assumed to be already in the lesion-locked frame.
    """
    leads = cfg.REGION_TO_LEADS.get(soz_region)
    if leads is None:
        raise KeyError(f"no lead mapping for region {soz_region!r}; "
                       f"known: {sorted(cfg.REGION_TO_LEADS)}")

    contralateral = set(cfg.LEFT_CHANNELS if (lesion_side or "").lower().startswith("r")
                        else cfg.RIGHT_CHANNELS)
    mapped = tuple(ch for ch in leads
                   if ch in available and ch not in contralateral)

    midline = tuple(ch for ch in cfg.MIDLINE_CHANNELS if ch in available)
    non_corresponding = tuple(ch for ch in available
                              if ch not in midline and ch not in mapped)

    return ChannelClasses(
        midline=midline,
        region_mapped=mapped,
        non_corresponding=non_corresponding,
        overlap=tuple(ch for ch in mapped if ch in midline),
    )


def delta(pre: dict[str, float], post: dict[str, float],
          channels) -> float:
    """``post - pre`` for a channel group -- the study's ΔExponent.

    Post minus pre throughout, so a negative value is a fall after surgery.
    """
    return region_mean(post, channels) - region_mean(pre, channels)


def selective_contrast_index(pre: dict[str, float], post: dict[str, float],
                             classes: ChannelClasses) -> float:
    """``SCI = ΔMidline - ΔNon-corresponding``.

    A difference of differences, and that is the point: any change affecting
    the whole head equally -- a drift in recording conditions, a global
    medication or vigilance effect -- moves both terms and cancels. What
    survives is change specific to the midline.

    The two classes are disjoint by construction, so the subtraction does not
    involve a channel twice.
    """
    return (delta(pre, post, classes.midline)
            - delta(pre, post, classes.non_corresponding))


def soz_contrast(soz_exponents, non_soz_exponents) -> float:
    """``mean(SOZ) - mean(non-SOZ)`` for one patient.

    Negative places the SOZ below its surround, which is the study's
    hyper-excitability sign. Measured once, before surgery, and treated as a
    fixed per-patient quantity in every cross-modal analysis.

    In bilaterally implanted patients the non-SOZ pool includes contralateral
    contacts. That is stated rather than corrected: it makes the contrast a
    whole-cohort quantity, and it is why the spatial analyses are specified
    within trajectories instead.
    """
    soz = np.asarray(list(soz_exponents), dtype=float)
    non_soz = np.asarray(list(non_soz_exponents), dtype=float)
    if soz.size == 0 or non_soz.size == 0:
        return float("nan")
    return float(np.nanmean(soz) - np.nanmean(non_soz))
