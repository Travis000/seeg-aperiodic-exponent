"""Where a contact sits, and how far it is from the seizure onset zone.

A depth electrode is a straight shaft carrying evenly spaced contacts, so
distance along it is just contact index times the spacing. That is the whole of
the distance metric used in the paper, and the reason it is defensible without
imaging: it is a property of the hardware, not a reconstruction.

The cost is equally plain -- it measures along the shaft, not through tissue,
so two contacts on different electrodes are not comparable. Every distance
analysis is therefore specified *within* a trajectory (Methods 2.6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import config as cfg

#: Vendor prefixes that EDF headers commonly prepend to a channel label.
_PREFIX = re.compile(r"^(EEG|SEEG|POL|DC|BIP)\s+", re.IGNORECASE)

#: Reference suffixes, likewise.
_SUFFIX = re.compile(r"[-_](Ref|LE|AVG|AV|A1|A2)\-?\d*$", re.IGNORECASE)

#: Physiological and housekeeping channels that are not brain contacts.
_NON_BRAIN = re.compile(
    r"^(ECG|EKG|EMG|EOG|Photic|Pulse|SpO2|IBI|Bursts|Suppr|DC|MK|Event)",
    re.IGNORECASE)

#: An electrode label followed by a contact number, e.g. ``A1``, ``B'12``.
#: The optional apostrophe is the standard SEEG mark distinguishing a shaft
#: from its counterpart; which side it denotes is a per-centre convention and
#: is never inferred here -- laterality comes from the operative record.
_CONTACT = re.compile(r"^([A-Za-z]+['’]?)(\d+)$")


def parse_contact_label(label: str,
                        scalp_labels: frozenset[str] = frozenset(cfg.MONTAGE_1020)
                        ) -> tuple[str, int] | None:
    """``"POL B'12"`` -> ``("B'", 12)``; ``None`` if not a brain contact.

    Scalp labels are rejected explicitly because intracranial recordings
    routinely carry a few simultaneous scalp channels in the same file.
    """
    name = _SUFFIX.sub("", _PREFIX.sub("", label.strip()))
    if _NON_BRAIN.match(name) or name in scalp_labels:
        return None
    match = _CONTACT.match(name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


@dataclass(frozen=True)
class Electrode:
    """One trajectory and the clinical labels attached to its contacts.

    ``soz_contacts`` and ``ablated_contacts`` hold contact *numbers*. Both come
    from the clinical record -- the seizure onset zone as delineated by
    epileptologists from ictal patterns, blind to any aperiodic result, and the
    ablated pairs from the operative record.
    """

    name: str
    soz_contacts: frozenset[int] = frozenset()
    ablated_contacts: frozenset[int] = frozenset()
    target_structure: str = ""

    @property
    def contains_soz(self) -> bool:
        return bool(self.soz_contacts)

    @property
    def soz_center(self) -> float:
        """Mean contact number of the SOZ contacts on this electrode.

        The mean, not the nearest SOZ contact: the SOZ usually spans several
        adjacent contacts, and its centre is the natural origin for a gradient.
        Being a mean it is generally fractional, which is fine -- distance is
        continuous.
        """
        if not self.soz_contacts:
            return float("nan")
        return float(np.mean(sorted(self.soz_contacts)))

    def distance_mm(self, contact: int) -> float:
        """Along-shaft distance from ``contact`` to this electrode's SOZ centre.

        ``nan`` on an electrode with no SOZ contact: there is nothing to
        measure from. This is exactly why the per-electrode slope control of
        Methods 2.6 uses the electrode's own axis instead, so that SOZ-bearing
        and SOZ-free trajectories can be compared at all.
        """
        if not self.soz_contacts:
            return float("nan")
        return abs(contact - self.soz_center) * cfg.CONTACT_SPACING_MM

    def region(self) -> str | None:
        """Anatomical region of this trajectory, or ``None`` if unclassifiable."""
        return classify_region(self.target_structure)


def classify_region(target_structure: str) -> str | None:
    """Apply the ordered keyword rule of Methods 2.8 to a target structure.

    The target structure is the deep end of the trajectory, where contact 1
    sits -- not the entry point. One rule, applied to every electrode, so that
    the regional baselines are not assembled case by case.

    Returns ``None`` for targets that do not fall in any grouped region (white
    matter, heterotopia); those electrodes are excluded from the region-wise
    model rather than forced into a category.
    """
    text = (target_structure or "").lower()
    for keyword, region in cfg.REGION_KEYWORD_RULE:
        if keyword in text:
            return region
    return None


def gradient_set(contacts, electrodes: dict[str, Electrode],
                 max_distance_mm: float = cfg.MAX_DISTANCE_MM):
    """Select the contacts the distance gradient is modelled on.

    Three conditions, all from Methods 2.6, and each with a reason:

    * on a SOZ-containing trajectory -- otherwise there is no origin;
    * within the distance cap -- beyond it the contacts thin out and a handful
      of far contacts on a few long electrodes would carry the slope;
    * passing spectral quality control -- applied before, not after, looking at
      the distance.

    ``contacts`` is any iterable of records exposing ``electrode``, ``contact``
    and ``passes_qc``. Yields ``(record, distance_mm)``.
    """
    for record in contacts:
        electrode = electrodes.get(record.electrode)
        if electrode is None or not electrode.contains_soz:
            continue
        if not record.passes_qc:
            continue
        distance = electrode.distance_mm(record.contact)
        if not np.isfinite(distance) or distance > max_distance_mm:
            continue
        yield record, distance


def electrode_axis_positions(contact_numbers) -> np.ndarray:
    """Position along the shaft in mm, measured from the electrode's own tip.

    Used by the per-electrode slope control, where SOZ-free trajectories have
    no SOZ centre to measure from and both classes must be treated the same
    way for the comparison to mean anything.
    """
    return np.asarray(sorted(contact_numbers), dtype=float) * cfg.CONTACT_SPACING_MM
