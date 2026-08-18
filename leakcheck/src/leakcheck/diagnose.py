"""Study-design diagnostics.

These run before any model is fitted and cost nothing.  Most of them catch
problems that no amount of careful modelling can repair, because they are
properties of how the recordings were collected:

  * the label is confounded with the group (every animal contributes one class)
  * a handful of groups carry most of the clips
  * there are six recording sites, so the honest score has an error bar the
    width of a barn door and nobody reports it
  * identical feature rows appear on both sides of the split

The last one is rare but fatal, and takes one line to check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Design", "Missing", "design_diagnostics", "cramers_v", "group_bootstrap_ci"]


@dataclass
class Missing:
    """Where the holes are.

    Missingness is rarely random in field data. A microphone that clipped, a
    pitch tracker that gave up on the quiet clips, a lab that ran a different
    version of the extractor -- each leaves a pattern, and if that pattern
    lines up with the label then "which features failed" is itself a feature.
    A model can score well by learning it, and nobody looks at it.
    """
    any_nan_rows: int
    all_nan_rows: int
    nan_cells: float                      # fraction of the matrix
    per_group_rate: dict                  # group -> fraction of rows with any NaN
    per_group_all_nan_rate: dict          # group -> fraction of rows with NO features at all
    per_class_rate: dict
    v_missing_label: float                # association between "row is missing" and the label
    v_missing_group: float
    all_nan_by_class: dict


@dataclass
class Design:
    n_clips: int
    n_classes: int
    n_groups: int
    class_counts: dict
    group_sizes: dict
    smallest_group: int
    largest_group: int
    majority_class_rate: float
    single_class_groups: list[str]
    clips_in_single_class_groups: float      # as a fraction of all clips
    label_group_cramers_v: float
    duplicate_rows: int
    duplicate_rows_across_groups: int
    missing: Missing | None = None
    warnings: list[str] = field(default_factory=list)


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Association between two categorical variables, 0 (independent) to 1
    (one determines the other).  Bias-corrected (Bergsma 2013)."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    if len(ua) < 2 or len(ub) < 2:
        return 0.0
    obs = np.zeros((len(ua), len(ub)))
    np.add.at(obs, (ia, ib), 1)
    n = obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    chi2 = float((((obs - exp) ** 2) / np.maximum(exp, 1e-12)).sum())
    phi2 = chi2 / n
    r, k = obs.shape
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / max(n - 1, 1))
    rc = r - (r - 1) ** 2 / max(n - 1, 1)
    kc = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = max(min(kc - 1, rc - 1), 1e-12)
    return float(np.sqrt(phi2c / denom))


def group_bootstrap_ci(per_group_scores, weights=None, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0):
    """Confidence interval for the honest score, resampling GROUPS not clips.

    The honest score has as many independent observations as you have animals
    or sites, not as many as you have clips.  Six labs is six observations.
    Reporting a clip-level interval on a group-held-out score is the single
    most common way to make a fragile result look settled.
    """
    s = np.asarray(per_group_scores, float)
    if len(s) < 2:
        return (float("nan"), float("nan"))
    w = np.ones(len(s)) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        means[i] = np.average(s[idx], weights=w[idx])
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def design_diagnostics(X, y, groups, min_group: int = 5) -> Design:
    y = np.asarray(y); groups = np.asarray(groups)
    uy, cy = np.unique(y, return_counts=True)
    ug, cg = np.unique(groups, return_counts=True)

    single = [str(g) for g in ug if len(np.unique(y[groups == g])) < 2]
    in_single = float(sum((groups == g).sum() for g in single) / len(y))

    # ---- missingness, before anything is imputed
    nanmask = np.isnan(X)
    any_nan = nanmask.any(1)
    all_nan = nanmask.all(1)
    miss = Missing(
        any_nan_rows=int(any_nan.sum()), all_nan_rows=int(all_nan.sum()),
        nan_cells=float(nanmask.mean()),
        per_group_rate={str(g): float(any_nan[groups == g].mean()) for g in ug},
        per_group_all_nan_rate={str(g): float(all_nan[groups == g].mean()) for g in ug},
        per_class_rate={str(c): float(any_nan[y == c].mean()) for c in uy},
        v_missing_label=cramers_v(any_nan.astype(int), y) if any_nan.any() else 0.0,
        v_missing_group=cramers_v(any_nan.astype(int), groups) if any_nan.any() else 0.0,
        all_nan_by_class={str(c): int((all_nan & (y == c)).sum()) for c in uy},
    ) if nanmask.any() else None

    # for the duplicate check, impute so that "all rows identical after
    # imputation" is caught -- that is the shape the problem usually takes
    Xd = X
    if nanmask.any():
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(X, axis=0)
        Xd = np.where(nanmask, np.where(np.isfinite(mu), mu, 0.0), X)

    # exact duplicate feature rows -- cheap, catches copied files and
    # overlapping windows written twice
    _, first, counts = np.unique(np.ascontiguousarray(Xd).view(
        np.dtype((np.void, Xd.dtype.itemsize * Xd.shape[1]))).ravel(),
        return_index=True, return_counts=True)
    dup = int((counts - 1).sum())
    dup_cross = 0
    if dup:
        keys = np.ascontiguousarray(Xd).view(
            np.dtype((np.void, Xd.dtype.itemsize * Xd.shape[1]))).ravel()
        for k in keys[first[counts > 1]]:
            m = keys == k
            if len(np.unique(groups[m])) > 1:
                dup_cross += int(m.sum() - 1)

    v = cramers_v(y, groups)
    w: list[str] = []

    if single:
        w.append(f"{len(single)} of {len(ug)} groups contain only one class "
                 f"({in_single:.0%} of clips). For those groups the held-out score is a "
                 f"recall, not a balanced accuracy, and the label is partly a group label.")
    if v >= 0.5:
        w.append(f"Label and group are strongly associated (Cramer's V = {v:.2f}). "
                 f"A held-out group brings its own class mix with it, so part of the "
                 f"honest score is base-rate guessing rather than acoustics.")
    if len(ug) < 8:
        w.append(f"Only {len(ug)} groups. The honest score has {len(ug)} independent "
                 f"observations behind it -- read the bootstrap interval, not the point "
                 f"estimate, and do not compare two methods on a difference smaller than it.")
    if cg.min() < min_group:
        w.append(f"{int((cg < min_group).sum())} groups have fewer than {min_group} clips "
                 f"and are dropped from the identity probe (they stay in the task probe).")
    if cg.max() / cg.sum() > 0.4:
        w.append(f"One group holds {cg.max() / cg.sum():.0%} of all clips. Balanced accuracy "
                 f"protects the class axis but nothing protects the group axis.")
    if dup_cross:
        w.append(f"{dup_cross} feature rows are exact duplicates of a row in a DIFFERENT "
                 f"group. That is a copy, a re-used file, or an overlapping window: the "
                 f"same clip is on both sides of the split.")
    elif dup:
        w.append(f"{dup} exact duplicate feature rows (all within a single group). "
                 f"Harmless for the group-held-out score, inflates the shuffled one.")
    if miss is not None:
        if miss.all_nan_rows:
            worst_g = max(miss.per_group_all_nan_rate, key=miss.per_group_all_nan_rate.get)
            by_cls = {k: v for k, v in miss.all_nan_by_class.items() if v}
            w.append(f"{miss.all_nan_rows} rows have NO features at all -- every column is "
                     f"missing. After imputation they are {len(by_cls)} identical constant "
                     f"vector(s) carrying the labels {sorted(by_cls)}. Any score on those rows "
                     f"is the class prior, not acoustics. Half of them or more sit in "
                     f"'{worst_g}', where {miss.per_group_all_nan_rate[worst_g]:.0%} of rows "
                     f"have no features at all.")
        if miss.v_missing_label >= 0.2:
            w.append(f"Whether a row has missing features predicts the label "
                     f"(Cramer's V = {miss.v_missing_label:.2f}). Missingness is a feature here: "
                     f"a model can score by learning which extractions failed. Check that the "
                     f"pattern is not an artefact of how each class was recorded.")
        if miss.v_missing_group >= 0.3:
            w.append(f"Missingness is tied to the group (Cramer's V = "
                     f"{miss.v_missing_group:.2f}), so imputing from column means computed "
                     f"across all groups would move held-out rows toward the training groups. "
                     f"leakcheck imputes inside each fold; anything upstream of it may not.")
        elif miss.any_nan_rows:
            w.append(f"{miss.any_nan_rows} of {len(y)} rows ({miss.any_nan_rows / len(y):.0%}) "
                     f"have at least one missing feature; {miss.nan_cells:.1%} of all cells. "
                     f"Imputed with training-fold column means inside every fold.")

    if len(uy) > 2 and (cy.min() / cy.max()) < 0.1:
        w.append(f"Class sizes span {cy.min()}-{cy.max()}. The rarest class has {cy.min()} "
                 f"clips; a balanced-accuracy point estimate rests heavily on them.")

    return Design(
        n_clips=int(len(y)), n_classes=int(len(uy)), n_groups=int(len(ug)),
        class_counts={str(k): int(v_) for k, v_ in zip(uy, cy)},
        group_sizes={str(k): int(v_) for k, v_ in zip(ug, cg)},
        smallest_group=int(cg.min()), largest_group=int(cg.max()),
        majority_class_rate=float(cy.max() / cy.sum()),
        single_class_groups=single, clips_in_single_class_groups=in_single,
        label_group_cramers_v=v, duplicate_rows=dup,
        duplicate_rows_across_groups=dup_cross, missing=miss, warnings=w)
