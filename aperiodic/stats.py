"""The statistical models named in Methods 2.11, as small explicit functions.

Two conventions run through all of it and are worth stating once.

*The unit of analysis is declared, never implied.* Contact-level quantities go
into mixed models with a random intercept per patient, because contacts within
a patient are not independent observations. Patient-level quantities go into
ordinary tests. Mixing the two -- pooling every contact as though each were a
separate patient -- would shrink every interval by roughly an order of
magnitude, and there are two orders of magnitude more contacts than patients.

*Effect estimates come with intervals.* A p value alone says whether an effect
is distinguishable from zero at this sample size; it does not say how large it
is, which at n = 16 is the more honest question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config as cfg


# ── Effect sizes and intervals ───────────────────────────────────────────────

def cohens_d_paired(pre, post) -> float:
    """Cohen's d for a paired comparison: mean difference over its own SD.

    The SD of the *differences*, not of the raw values -- the pairing is the
    design, and using the raw SD would understate the effect.
    """
    diff = np.asarray(post, dtype=float) - np.asarray(pre, dtype=float)
    diff = diff[np.isfinite(diff)]
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def cohens_d_ci(d: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Large-sample interval for d: ``d ± z * sqrt(1/n + d^2 / 2n)``.

    The analytic form is used for every d in the paper rather than a bootstrap,
    for one reason: it is monotone in the same statistic as the paired t test,
    so an interval excluding zero and a significant t always agree. A bootstrap
    interval can disagree with its own test, which is confusing to read and
    impossible to defend in a figure caption.
    """
    se = np.sqrt(1.0 / n + d * d / (2.0 * n))
    return float(d - z * se), float(d + z * se)


@dataclass(frozen=True)
class Correlation:
    """A rank correlation with a bootstrap interval."""

    rho: float
    p: float
    ci_low: float
    ci_high: float
    n: int


def spearman_bootstrap(x, y, n_boot: int = cfg.BOOTSTRAP_N,
                       seed: int = cfg.BOOTSTRAP_SEED) -> Correlation:
    """Spearman rho with a percentile bootstrap interval over patients.

    Resampling is over *patients*, the independent unit, so the interval
    reflects the sample size that actually limits the study.

    The p value is scipy's asymptotic approximation and is reported as such.
    At n = 16 it is not exact, and where it lands near a reporting boundary the
    paper falls back on an explicit permutation test rather than trusting the
    approximation to place a value on one side of the line.

    The seed is fixed so that a published interval can be reproduced digit for
    digit; it is not tuned, and the interval is stable across seeds to the
    precision reported.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size

    rho, p = st.spearmanr(x, y)
    rng = np.random.default_rng(seed)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # A resample can draw too few distinct values for a rank correlation to
        # be defined; those draws are skipped rather than counted as zero.
        if np.unique(x[idx]).size < 3 or np.unique(y[idx]).size < 3:
            continue
        r = st.spearmanr(x[idx], y[idx])[0]
        if np.isfinite(r):
            boots.append(r)

    low, high = np.percentile(boots, [2.5, 97.5])
    return Correlation(float(rho), float(p), float(low), float(high), n)


def partial_spearman(x, y, covariate) -> Correlation:
    """Spearman correlation of x and y adjusting for one covariate.

    Rank all three, regress out the covariate from each of x and y by ordinary
    least squares, and correlate the residuals. Used to check that a
    cross-modal association is not carried by age or by the pre-operative
    level.
    """
    frame = pd.DataFrame({"x": x, "y": y, "c": covariate}).dropna()
    ranks = frame.rank()

    def residual(col):
        slope, intercept = np.polyfit(ranks["c"], ranks[col], 1)
        return ranks[col] - (intercept + slope * ranks["c"])

    rho, p = st.pearsonr(residual("x"), residual("y"))
    n = len(frame)
    # Fisher z interval on the residual correlation, one degree of freedom
    # spent on the covariate.
    if n > 4:
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(n - 4)
        low, high = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    else:
        low = high = float("nan")
    return Correlation(float(rho), float(p), float(low), float(high), n)


# ── Paired comparisons ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairedResult:
    mean_change: float
    t: float
    p: float
    d: float
    d_ci: tuple[float, float]
    wilcoxon_p: float
    bf01: float
    n: int


def paired_change(pre, post) -> PairedResult:
    """Paired t test, with a signed-rank sensitivity test and a Bayes factor.

    ``BF01`` is the evidence *for* no change. It is reported because several of
    the study's claims are null claims -- that the whole-scalp exponent does
    not move in the poor-outcome group, for instance -- and a non-significant p
    value cannot support one. A Bayes factor can distinguish evidence of
    absence from absence of evidence.
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    ok = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[ok], post[ok]
    n = pre.size

    t, p = st.ttest_rel(post, pre)
    d = cohens_d_paired(pre, post)

    try:
        wilcoxon_p = float(st.wilcoxon(post, pre).pvalue)
    except ValueError:      # all differences zero
        wilcoxon_p = 1.0

    return PairedResult(
        mean_change=float(np.mean(post - pre)),
        t=float(t), p=float(p), d=d, d_ci=cohens_d_ci(d, n),
        wilcoxon_p=wilcoxon_p, bf01=bayes_factor_01(pre, post), n=n,
    )


def bayes_factor_01(pre, post) -> float:
    """Evidence for the null in a paired design, via a default Cauchy prior.

    Uses pingouin when available; returns ``nan`` if it is not installed,
    rather than substituting a different prior silently.
    """
    try:
        import pingouin as pg
    except ImportError:
        return float("nan")
    result = pg.ttest(np.asarray(post, dtype=float),
                      np.asarray(pre, dtype=float), paired=True)
    bf10 = float(result["BF10"].iloc[0])
    return 1.0 / bf10 if bf10 > 0 else float("nan")


def fdr(pvalues, method: str = cfg.FDR_METHOD):
    """Benjamini-Hochberg correction. Returns ``(reject, p_adjusted)``.

    Applied across the three scalp regions, and across the six region x outcome
    cells -- the families are declared in advance, so that the correction is
    not tightened or loosened after seeing which tests survived.
    """
    from statsmodels.stats.multitest import multipletests
    reject, p_adj, _, _ = multipletests(np.asarray(pvalues, dtype=float),
                                        alpha=cfg.ALPHA, method=method)
    return reject, p_adj


# ── Mixed models ─────────────────────────────────────────────────────────────

def soz_contrast_model(contacts: pd.DataFrame, value: str = "Exponent"):
    """``value ~ Is_SOZ`` with a random intercept per patient.

    The study's primary intracranial test. The random intercept is what makes
    it a within-patient comparison: patients differ in overall exponent level
    for reasons that have nothing to do with the SOZ, and without it those
    differences would leak into the SOZ coefficient.

    ``contacts`` needs columns ``Subject``, ``Is_SOZ`` and ``value``.
    """
    import statsmodels.formula.api as smf
    data = contacts.dropna(subset=[value, "Is_SOZ", "Subject"]).copy()
    data["Is_SOZ"] = data["Is_SOZ"].astype(int)
    return smf.mixedlm(f"{value} ~ Is_SOZ", data, groups=data["Subject"]).fit()


def distance_gradient_model(contacts: pd.DataFrame, value: str = "Exponent",
                            covariates: tuple[str, ...] = ()):
    """``value ~ Distance_mm`` with patient and electrode nested random intercepts.

    Nesting matters here in a way it does not for the SOZ contrast. Contacts
    on one electrode share a trajectory, a tissue type and an impedance
    history; treating them as independent within a patient would credit the
    slope with precision it has not earned. Electrode enters as a variance
    component inside patient.

    ``covariates`` adds fixed effects -- the discharge-rate control of
    Methods 2.7 is this function with ``covariates=("Discharge_rate",)``.

    ``contacts`` needs ``Subject``, ``Electrode``, ``Distance_mm`` and ``value``.
    ``Electrode`` must identify a trajectory **globally**, not within a patient:
    electrodes are conventionally lettered per implantation, so a bare "A"
    occurs in every patient, and a variance component keyed on it would pool
    unrelated trajectories into one group. Prefix it with the patient.
    """
    import statsmodels.formula.api as smf
    needed = ["Subject", "Electrode", "Distance_mm", value, *covariates]
    data = contacts.dropna(subset=needed).copy()

    # With one electrode per patient the two random intercepts describe the
    # same grouping and the model is not identifiable. statsmodels reports this
    # as a singular Hessian several layers down, so it is caught here instead.
    per_subject = data.groupby("Subject")["Electrode"].nunique()
    if (per_subject <= 1).all():
        raise ValueError(
            "electrode and patient are the same grouping in this data "
            "(every patient contributes one electrode), so patient and "
            "electrode random intercepts cannot both be estimated; drop the "
            "electrode variance component or supply more electrodes")

    shared = data.groupby("Electrode")["Subject"].nunique()
    if (shared > 1).any():
        offenders = sorted(shared[shared > 1].index)[:5]
        raise ValueError(
            f"electrode labels are reused across patients ({offenders}); the "
            "variance component would treat one label as a single trajectory "
            "in several patients. Make the label globally unique, e.g. "
            "df['Electrode'] = df.Subject + '-' + df.Electrode")

    terms = " + ".join(["Distance_mm", *covariates])
    return smf.mixedlm(f"{value} ~ {terms}", data, groups=data["Subject"],
                       re_formula="1",
                       vc_formula={"Electrode": "0 + C(Electrode)"}).fit()


def per_electrode_slopes(contacts: pd.DataFrame, value: str = "Exponent",
                         min_contacts: int = cfg.MIN_CONTACTS_FOR_SLOPE
                         ) -> pd.DataFrame:
    """One ordinary least-squares slope per electrode, along its own axis.

    This is the control that decides whether the gradient belongs to
    SOZ-bearing trajectories or to depth electrodes in general. SOZ-free
    electrodes have no SOZ centre to measure from, so both classes are
    regressed on position along the electrode's own shaft -- the only metric
    available to both.

    Returns one row per electrode with its slope in exponent units per mm.
    """
    rows = []
    for (subject, electrode), group in contacts.groupby(["Subject", "Electrode"]):
        group = group.dropna(subset=[value, "Contact"])
        if len(group) < min_contacts:
            continue
        position = group["Contact"].to_numpy(float) * cfg.CONTACT_SPACING_MM
        slope, _ = np.polyfit(position, group[value].to_numpy(float), 1)
        rows.append({
            "Subject": subject, "Electrode": electrode,
            "Contains_SOZ": bool(group["Contains_SOZ"].iloc[0]),
            "n_contacts": len(group), "Slope": float(slope),
        })
    return pd.DataFrame(rows)


def compare_trajectory_classes(slopes: pd.DataFrame) -> dict:
    """Average slopes within patient by class, then compare the two classes.

    Averaging within patient first is what keeps the unit of analysis at the
    patient: a patient with eleven electrodes should not outvote one with four.
    Each class is also tested against zero, because "both classes slope" and
    "only SOZ trajectories slope" are different claims and the between-class
    test alone cannot separate them.
    """
    per_patient = (slopes.groupby(["Subject", "Contains_SOZ"])["Slope"]
                   .mean().unstack())
    soz = per_patient.get(True)
    non_soz = per_patient.get(False)

    result = {
        "n_soz_patients": int(soz.notna().sum()),
        "n_non_soz_patients": int(non_soz.notna().sum()),
        "soz_mean": float(soz.mean()),
        "non_soz_mean": float(non_soz.mean()),
        "soz_vs_zero_p": float(st.ttest_1samp(soz.dropna(), 0.0).pvalue),
        "non_soz_vs_zero_p": float(st.ttest_1samp(non_soz.dropna(), 0.0).pvalue),
    }
    paired = per_patient.dropna()
    t, p = st.ttest_rel(paired[True], paired[False])
    result["paired_t"] = float(t)
    result["paired_p"] = float(p)
    result["n_paired"] = int(len(paired))
    return result


# ── Outcome comparison ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class AncovaResult:
    coefficient: float
    p: float
    partial_eta_squared: float
    permutation_p: float
    n: int


def ancova_outcome(frame: pd.DataFrame, dependent: str = "Post",
                   covariate: str = "Pre", group: str = "Outcome",
                   n_permutations: int = cfg.PERMUTATION_N,
                   seed: int = cfg.BOOTSTRAP_SEED) -> AncovaResult:
    """Post-operative level by outcome group, adjusted for the pre-operative level.

    Adjusting for the pre-operative level is not cosmetic. Groups that start
    from different baselines will differ afterwards for that reason alone, and
    a raw between-group test could not tell that apart from a treatment effect.

    The outcome coefficient is then evaluated against a permutation null built
    by shuffling the group labels. With 16 patients split unevenly, the F
    distribution's assumptions are not comfortably met; permuting the labels
    makes no distributional assumption at all, and it is the permutation p that
    the paper reports where the two disagree.
    """
    import statsmodels.formula.api as smf

    data = frame.dropna(subset=[dependent, covariate, group]).copy()
    formula = f"{dependent} ~ {covariate} + C({group})"
    model = smf.ols(formula, data).fit()

    term = next(t for t in model.params.index if t.startswith(f"C({group})"))
    observed = float(model.params[term])

    # Partial eta squared for the group term: its sum of squares over that plus
    # the residual sum of squares.
    reduced = smf.ols(f"{dependent} ~ {covariate}", data).fit()
    ss_group = float(reduced.ssr - model.ssr)
    partial_eta_sq = ss_group / (ss_group + float(model.ssr))

    rng = np.random.default_rng(seed)
    labels = data[group].to_numpy()
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = data.assign(**{group: rng.permutation(labels)})
        refit = smf.ols(formula, shuffled).fit()
        null[i] = float(refit.params[term])

    perm_p = float((np.abs(null) >= abs(observed)).mean())

    return AncovaResult(
        coefficient=observed,
        p=float(model.pvalues[term]),
        partial_eta_squared=float(partial_eta_sq),
        permutation_p=perm_p,
        n=len(data),
    )


def permutation_test(x, y, statistic=None, n_permutations: int = cfg.PERMUTATION_N,
                     seed: int = cfg.BOOTSTRAP_SEED) -> float:
    """Two-tailed permutation p for a two-sample statistic.

    Defaults to the difference in means. Used wherever a small, unbalanced
    comparison makes an asymptotic p value hard to defend.
    """
    if statistic is None:
        def statistic(a, b):
            return np.mean(a) - np.mean(b)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = statistic(x, y)

    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    n_x = x.size

    extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        if abs(statistic(shuffled[:n_x], shuffled[n_x:])) >= abs(observed):
            extreme += 1
    return extreme / n_permutations
