"""Every fixed threshold and setting used in the study, in one place.

Each constant carries the Methods section it is stated in, so that a reader can
check the code against the paper without searching. Nothing here is fitted to
the data; all values were fixed before analysis and applied identically to
every recording.
"""

from __future__ import annotations

# ── Filtering and epoching (Methods 2.4) ─────────────────────────────────────

L_FREQ_HZ = 0.5
H_FREQ_HZ = 45.0
#: The upper edge stops below the 50 Hz mains frequency of the recording site.

EPOCH_SECONDS = 5.0
#: Fixed-length, non-overlapping epochs cut from the retained N2 signal.

STAGING_EPOCH_SECONDS = 30.0
#: The sleep stager labels 30 s epochs; N2 labels are expanded to a sample mask.

MIN_N2_SECONDS = 30.0
#: A recording contributing less N2 than this is dropped entirely.


# ── Adaptive artefact rejection (Methods 2.4, Supplementary Methods SM3) ─────

PTP_PERCENTILE = 90.0
#: Per recording, the threshold starts at the 90th percentile of the epoch
#: peak-to-peak amplitudes, i.e. the noisiest tenth is a candidate for removal.

PTP_FLOOR_V_SCALP = 150e-6
PTP_FLOOR_V_INTRACRANIAL = 300e-6
#: ...but the threshold is never allowed below this floor, so that a clean
#: recording does not have 10% of its epochs discarded for nothing. The
#: intracranial floor is higher because intracranial amplitudes are larger.

MIN_CLEAN_EPOCHS = 6
#: Fewer surviving epochs than this and the recording yields no estimate.


# ── Spectral estimation (Methods 2.5) ────────────────────────────────────────

PSD_FMIN_HZ, PSD_FMAX_HZ = 1.0, 45.0
WELCH_WINDOW_SECONDS = 4.0
WELCH_OVERLAP = 0.5
WELCH_WINDOW = "hamming"
#: Welch periodogram, 4 s segments with 50% overlap, averaged over epochs.
#: The taper is stated explicitly because the two obvious defaults disagree:
#: the EEG toolkit used for the reported analyses tapers with Hamming, while
#: scipy's own default is Hann, and the choice moves the recovered exponent by
#: about 0.015 -- the same order as the effect being measured.

FIT_RANGE_HZ = (1.0, 45.0)
APERIODIC_MODE = "fixed"
#: "fixed" = no spectral knee; the aperiodic component is a straight line in
#: log-log space, and its negated slope is the exponent.
PEAK_WIDTH_LIMITS_HZ = (1.0, 8.0)
MIN_PEAK_HEIGHT = 0.05
MAX_N_PEAKS_SCALP = 4
MAX_N_PEAKS_INTRACRANIAL = 6
#: More peaks are allowed intracranially: a depth contact sees narrower,
#: more numerous rhythms than a scalp electrode averaging over centimetres.


# ── Quality control (Methods 2.5) ────────────────────────────────────────────

MIN_R_SQUARED = 0.85
MIN_EXPONENT = 0.5
#: An exponent below 0.5 in this band indicates a failed fit rather than a
#: genuinely flat spectrum, and is discarded with the fit.
#:
#: These are the **intracranial** criteria, and the paper states them as such:
#: quality control selects the analysed contacts out of the full implantation.
#: Scalp channels are not screened this way. The scalp fit quality is recorded
#: but not used to drop channels, because the scalp regions are defined to
#: exhaust a fixed 19-channel montage -- dropping a channel would change which
#: channels a regional mean is taken over, and so what the mean means, from one
#: patient to the next. Pass these thresholds explicitly rather than relying on
#: a default if you screen scalp data.


# ── Electrode geometry (Methods 2.6) ─────────────────────────────────────────

CONTACT_SPACING_MM = 3.5
MAX_DISTANCE_MM = 63.0
MAX_CONTACT_OFFSET = 18
#: 18 x 3.5 mm = 63 mm. The same cap is expressed in millimetres for the
#: distance gradient and in contact counts for the region-wise baseline model,
#: so that the two analyses cover the same span of tissue.

MIN_CONTACTS_FOR_SLOPE = 5
#: An electrode needs at least this many analysed contacts before a per-
#: electrode slope is estimated from it.

MIN_ELECTRODES_PER_REGION = 3
#: A region enters the baseline-corrected model only with at least this many
#: electrodes of each trajectory class.


# ── Interictal discharge detection (Methods 2.7, SM8) ────────────────────────

IED_BAND_HZ = (20.0, 60.0)
#: The lower edge is 20 Hz rather than the 10 Hz of the source method. All
#: analysis is on N2, where 10-16 Hz sleep spindles are dense; a detector with
#: a 10 Hz lower edge fires on them constantly. Discharges retain ample energy
#: above 20 Hz, so raising the edge removes spindles and keeps discharges.

NEO_SMOOTH_MS = 20.0
REFRACTORY_MS = 100.0
#: Supra-threshold samples within one refractory period are one event.

NEO_C_PER_CHANNEL = (3.0, 4.0, 6.0, 8.0)
#: Threshold = C x the median energy *of that channel*.
NEO_C_SHARED = (4.0, 8.0)
#: Threshold = C x a median pooled *across channels*.
ENVELOPE_K = (5.0,)
#: A different detector family entirely, as a cross-method check.

#: The two threshold families trade off against each other, so both are run.
#: A per-channel threshold controls for amplitude differences due to impedance
#: and grey/white matter, but it also normalises away "this contact is
#: intrinsically spikier", which is the very signal under test. A shared
#: threshold preserves between-contact differences but admits amplitude bias.
#: A conclusion that holds under both does not depend on the choice.

MIN_DISCHARGE_FREE_EPOCHS = 30
#: Below this the exponent is not re-estimated on discharge-free epochs.


# ── Statistics (Methods 2.11) ────────────────────────────────────────────────

ALPHA = 0.05
#: All tests two-tailed.

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
#: Correlation confidence intervals are percentile bootstrap over patients,
#: with a fixed seed so that a reported interval is reproducible exactly.
#: A thousand resamples is enough for a percentile interval reported to two
#: decimals; the seed matters more than the count, because at n = 16 the
#: resampling itself is the coarse step.

PERMUTATION_N = 10000
#: Label permutations for the outcome-coefficient null distribution. Larger
#: than the bootstrap count because a permutation p value near 0.001 needs
#: enough draws to resolve it -- that is the resolution the reporting rule in
#: :mod:`aperiodic.reporting` asks for.

FDR_METHOD = "fdr_bh"
#: Benjamini-Hochberg, applied across the three scalp regions and across the
#: six region x outcome cells.


# ── Scalp montage (Methods 2.9) ──────────────────────────────────────────────

MONTAGE_1020 = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz",
)

LEFT_CHANNELS = ("Fp1", "F3", "F7", "C3", "P3", "P7", "T7", "O1")
RIGHT_CHANNELS = ("Fp2", "F4", "F8", "C4", "P4", "P8", "T8", "O2")
MIDLINE_CHANNELS = ("Fz", "Cz", "Pz")
#: The three regions exhaust the 19-channel montage: 8 + 8 + 3.

MIRROR_MAP = {
    "Fp1": "Fp2", "Fp2": "Fp1", "F3": "F4", "F4": "F3",
    "F7": "F8", "F8": "F7", "C3": "C4", "C4": "C3",
    "P3": "P4", "P4": "P3", "P7": "P8", "P8": "P7",
    "T7": "T8", "T8": "T7", "O1": "O2", "O2": "O1",
    "Fz": "Fz", "Cz": "Cz", "Pz": "Pz",
}
#: Applied to left-hemisphere patients so that "ipsilateral" means the same
#: side of the head in every patient (Supplementary Methods SM9).

REGION_TO_LEADS = {
    "MedialTemporal": ("T7", "T8", "F7", "F8"),
    "LateralTemporal": ("T7", "T8", "F7", "F8"),
    "Insula": ("T7", "T8"),
    "Prefrontal": ("Fz", "F3", "F4"),
    "Sensorimotor": ("Cz", "C3", "C4"),
    "Parietal": ("Pz", "P3", "P4"),
    "Occipital": ("O1", "O2"),
    "Cingulate": ("Fz", "Cz", "Pz"),
}
#: The fixed region-to-lead table used to build the region-mapped channel
#: class. It is applied identically to every patient and was not adjusted
#: after seeing any result. Four of the eight entries include a midline lead
#: (Prefrontal, Sensorimotor and Parietal one each, Cingulate all three),
#: which is why the region-mapped and midline classes can overlap in some
#: patients while the non-corresponding class stays disjoint from both.

BREACH_BAND_HZ = (20.0, 45.0)
#: Relative power in this band is compared pre- vs post-operatively to test
#: for a breach rhythm, which is not expected because the skull is left
#: intact (Methods 2.4, Supplementary Methods SM4).


# ── Anatomical grouping (Methods 2.8) ────────────────────────────────────────

REGION_KEYWORD_RULE = (
    ("amygdala", "MedialTemporal"),
    ("hippocamp", "MedialTemporal"),
    ("entorhinal", "MedialTemporal"),
    ("parahippocamp", "MedialTemporal"),
    ("piriform", "MedialTemporal"),
    ("uncus", "MedialTemporal"),
    ("limen insulae", "Insula"),
    ("insul", "Insula"),
    ("cingul", "Cingulate"),
    ("mcc", "Cingulate"),
    ("pcc", "Cingulate"),
    ("fusiform", "LateralTemporal"),
    ("temporal gyrus", "LateralTemporal"),
    ("temporal pole", "LateralTemporal"),
    ("supplementary motor", "Sensorimotor"),
    ("sma", "Sensorimotor"),
    ("paracentral", "Sensorimotor"),
    ("precentral", "Sensorimotor"),
    ("postcentral", "Sensorimotor"),
    ("orbitofrontal", "Prefrontal"),
    ("frontal", "Prefrontal"),
    ("precuneus", "Parietal"),
    ("supramarginal", "Parietal"),
    ("angular", "Parietal"),
    ("parietal", "Parietal"),
    ("operculum", "Parietal"),
    ("lingual", "Occipital"),
    ("occipital", "Occipital"),
)
#: A single ordered keyword rule, applied to the documented target structure of
#: every electrode: the deep end of the trajectory, where contact 1 sits. Order
#: matters -- specific terms precede general ones, so that "orbitofrontal"
#: is not swallowed by "frontal", and "limen insulae" not by "insul".
