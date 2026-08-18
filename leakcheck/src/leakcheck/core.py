"""Core measurements.

Three numbers, one protocol:

    honest    task accuracy with whole GROUPS held out      (what you get in the field)
    shuffled  task accuracy with a random split             (what papers usually report)
    identity  how well the SAME features recover the group  (the explanation)

inflation = shuffled - honest.  If it is large, the shuffled number was partly
the model recognising the individual animal or the recording site, not the
behaviour.  The identity number says whether that shortcut was available.

Estimator conventions
---------------------
Standardisation is fitted on the training fold only, never on the full matrix.
The probe is plain multinomial logistic regression, because a linear probe
measures what is *linearly present in the representation* rather than what a
big head can dig out of it.

Task scores are BALANCED accuracy (chance = 1/n_classes).  Identity is reported
as plain accuracy (the convention in the speaker-ID literature) with balanced
accuracy and the majority-group rate alongside, because group sizes are
usually very unequal.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

__all__ = [
    "ProbeConfig", "GroupScore", "TaskScore", "IdentityScore", "PermutationNull",
    "honest_score", "shuffled_score", "identity_score", "permutation_null",
    "group_splits",
]


# --------------------------------------------------------------------------- config
@dataclass
class ProbeConfig:
    """Everything that can change a number, in one object that gets serialised
    into the report so a result is reproducible from the report alone."""
    C: float = 1.0
    max_iter: int = 3000
    balanced: bool = True          # class_weight='balanced' for the task probe
    folds: int = 5                 # random-split folds, and identity folds
    group_folds: int = 0           # 0 = leave-one-group-out; k = GroupKFold(k)
    loo_limit: int = 30            # above this many groups, auto-switch to GroupKFold
    min_group: int = 5             # groups smaller than this are dropped from identity
    seed: int = 0

    def task_lr(self) -> LogisticRegression:
        return LogisticRegression(max_iter=self.max_iter, C=self.C,
                                  class_weight="balanced" if self.balanced else None)

    def id_lr(self) -> LogisticRegression:
        # no class_weight: identity accuracy is compared against the majority
        # rate, and balancing would make that comparison incoherent
        return LogisticRegression(max_iter=self.max_iter, C=self.C)


# --------------------------------------------------------------------------- results
@dataclass
class GroupScore:
    group: str
    n: int
    score: float
    metric: str        # "balanced_accuracy" or "recall" (single-class group)
    n_classes: int


@dataclass
class TaskScore:
    accuracy: float
    chance: float
    n_scored: int
    n_classes: int
    split: str                                   # "leave-one-group-out" / "GroupKFold(5)" / "StratifiedKFold(5)"
    per_group: list[GroupScore] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class IdentityScore:
    accuracy: float
    balanced_accuracy: float
    chance: float
    majority_rate: float
    n_groups: int
    n_clips: int
    seconds: float = 0.0


@dataclass
class PermutationNull:
    observed: float
    p_value: float
    null_mean: float
    null_sd: float
    null_max: float
    n_permutations: int
    unit: str                                    # "within-group" or "group"
    seconds: float = 0.0
    note: str = ""


# --------------------------------------------------------------------------- splits
def group_splits(groups: np.ndarray, cfg: ProbeConfig):
    """Yield (train_mask, test_mask, name).  Leave-one-group-out while the
    number of groups is manageable, GroupKFold above that so the tool stays
    usable on corpora with hundreds of individuals."""
    uniq = np.unique(groups)
    k = cfg.group_folds or (0 if len(uniq) <= cfg.loo_limit else min(cfg.folds, len(uniq)))
    if k == 0:
        for g in uniq:
            te = groups == g
            yield ~te, te, str(g)
        return
    idx = np.arange(len(groups))
    for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(idx, groups=groups)):
        m = np.zeros(len(groups), bool); m[te] = True
        yield ~m, m, f"fold{i + 1}"


def split_name(groups: np.ndarray, cfg: ProbeConfig) -> str:
    uniq = np.unique(groups)
    k = cfg.group_folds or (0 if len(uniq) <= cfg.loo_limit else min(cfg.folds, len(uniq)))
    return "leave-one-group-out" if k == 0 else f"GroupKFold({k})"


# --------------------------------------------------------------------------- probes
def _impute(Xtr, Xte):
    """Fill missing values with TRAINING-fold column means.

    Imputing from the whole matrix before splitting is a leak in its own right
    -- the test rows help decide what the training rows are filled with -- and
    it is a leak that a tool about leaks has no business committing.
    Columns that are entirely missing in the training fold become zero.
    """
    if not (np.isnan(Xtr).any() or np.isnan(Xte).any()):
        return Xtr, Xte
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN column
        mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return np.where(np.isnan(Xtr), mu, Xtr), np.where(np.isnan(Xte), mu, Xte)


def _fit_predict(Xtr, ytr, Xte, cfg: ProbeConfig, lr=None):
    Xtr, Xte = _impute(Xtr, Xte)
    sc = StandardScaler().fit(Xtr)
    lr = lr if lr is not None else cfg.task_lr()
    return lr.fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def honest_score(X, y, groups, cfg: ProbeConfig = ProbeConfig(),
                 per_group: bool = True) -> TaskScore:
    """The number that survives deployment: entire groups held out."""
    t0 = time.time()
    yp = np.full(len(y), None, dtype=object)
    for tr, te, _name in group_splits(groups, cfg):
        if len(np.unique(y[tr])) < 2:
            continue                                   # nothing to learn from
        yp[te] = _fit_predict(X[tr], y[tr], X[te], cfg)
    scored = np.array([v is not None for v in yp])
    if scored.sum() == 0:
        raise ValueError(
            "No fold could be scored: every training split had a single class. "
            "The label is confounded with the group -- see the design diagnostics.")
    yt = y[scored]
    yh = np.asarray(list(yp[scored]), dtype=y.dtype)
    acc = float(balanced_accuracy_score(yt, yh))

    rows: list[GroupScore] = []
    if per_group:
        for g in np.unique(groups[scored]):
            m = (groups == g) & scored
            gy, gh = y[m], np.asarray(list(yp[m]), dtype=y.dtype)
            nc = len(np.unique(gy))
            if nc >= 2:
                with warnings.catch_warnings():
                    # a held-out group need not contain every class; balanced
                    # accuracy over the classes it does contain is the right
                    # number and sklearn's warning about it is noise here
                    warnings.simplefilter("ignore", UserWarning)
                    sc_g = float(balanced_accuracy_score(gy, gh))
                rows.append(GroupScore(str(g), int(m.sum()), sc_g, "balanced_accuracy", nc))
            else:
                rows.append(GroupScore(str(g), int(m.sum()), float((gy == gh).mean()),
                                       "recall", 1))
        rows.sort(key=lambda r: r.score)

    return TaskScore(acc, 1.0 / len(np.unique(y)), int(scored.sum()),
                     int(len(np.unique(y))), split_name(groups, cfg), rows,
                     round(time.time() - t0, 2))


def shuffled_score(X, y, cfg: ProbeConfig = ProbeConfig()) -> TaskScore:
    """The number the field usually reports: clips shuffled, groups ignored."""
    t0 = time.time()
    yp = np.empty_like(y)
    skf = StratifiedKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    for tr, te in skf.split(X, y):
        yp[te] = _fit_predict(X[tr], y[tr], X[te], cfg)
    return TaskScore(float(balanced_accuracy_score(y, yp)), 1.0 / len(np.unique(y)),
                     len(y), int(len(np.unique(y))), f"StratifiedKFold({cfg.folds})",
                     [], round(time.time() - t0, 2))


def identity_score(X, groups, cfg: ProbeConfig = ProbeConfig()) -> IdentityScore:
    """How much of the group is sitting in the features.  Shuffled on purpose:
    we WANT the most favourable estimate of the shortcut's availability."""
    t0 = time.time()
    keep = np.isin(groups, [g for g in np.unique(groups)
                            if (groups == g).sum() >= cfg.min_group])
    Xi, gi = X[keep], groups[keep]
    uniq = np.unique(gi)
    if len(uniq) < 2:
        return IdentityScore(float("nan"), float("nan"), float("nan"), float("nan"),
                             len(uniq), int(keep.sum()), round(time.time() - t0, 2))
    k = min(cfg.folds, int(min(np.bincount(np.unique(gi, return_inverse=True)[1]))))
    k = max(k, 2)
    gp = np.empty(len(gi), dtype=object)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=cfg.seed).split(Xi, gi):
        gp[te] = _fit_predict(Xi[tr], gi[tr], Xi[te], cfg, lr=cfg.id_lr())
    gh = np.asarray(list(gp), dtype=gi.dtype)
    counts = np.array([(gi == g).sum() for g in uniq], float)
    return IdentityScore(float((gh == gi).mean()),
                         float(balanced_accuracy_score(gi, gh)),
                         1.0 / len(uniq),
                         float(counts.max() / counts.sum()),
                         int(len(uniq)), int(keep.sum()), round(time.time() - t0, 2))


# --------------------------------------------------------------------------- null
def _label_varies_within_groups(y, groups) -> bool:
    return any(len(np.unique(y[groups == g])) > 1 for g in np.unique(groups))


def permutation_null(X, y, groups, cfg: ProbeConfig = ProbeConfig(),
                     n: int = 100, budget_s: float | None = None) -> PermutationNull:
    """Is the honest score above chance for a reason?

    The unit of permutation is chosen by the data, because getting it wrong is
    the classic way to manufacture significance:

      * label varies inside groups -> shuffle labels WITHIN each group.  The
        group-level class mix is preserved, so a model that scores by guessing
        each group's base rate keeps its advantage and cannot look significant.
      * label is constant inside a group (one label per animal / per site) ->
        shuffle the group->label assignment.  Shuffling clips would break the
        block structure and produce a hopelessly optimistic null.
    """
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    within = _label_varies_within_groups(y, groups)
    obs = honest_score(X, y, groups, cfg, per_group=False)

    note = ""
    if budget_s is not None and obs.seconds > 0:
        affordable = max(1, int(budget_s / obs.seconds))
        if affordable < n:
            note = (f"reduced from {n} to {affordable} permutations to stay inside "
                    f"the {budget_s:.0f}s budget (one fit takes {obs.seconds:.1f}s); "
                    f"pass a larger budget for a finer p-value")
            n = affordable

    null = np.empty(n)
    uniq = np.unique(groups)
    for i in range(n):
        yp = y.copy()
        if within:
            for g in uniq:
                m = groups == g
                yp[m] = rng.permutation(y[m])
        else:
            lab = {g: y[groups == g][0] for g in uniq}
            shuffled = rng.permutation([lab[g] for g in uniq])
            for g, v in zip(uniq, shuffled):
                yp[groups == g] = v
        try:
            null[i] = honest_score(X, yp, groups, cfg, per_group=False).accuracy
        except ValueError:
            null[i] = np.nan

    null = null[~np.isnan(null)]
    p = float((1 + (null >= obs.accuracy).sum()) / (1 + len(null))) if len(null) else float("nan")
    return PermutationNull(obs.accuracy, p,
                           float(null.mean()) if len(null) else float("nan"),
                           float(null.std()) if len(null) else float("nan"),
                           float(null.max()) if len(null) else float("nan"),
                           int(len(null)), "within-group" if within else "group",
                           round(time.time() - t0, 2), note)
