"""Can the shortcut be removed?

Three transforms, all fitted on training groups only, all applied per fold:

  LEACE  least-squares concept erasure (Belrose et al. 2023), closed form.
         Moves every group-conditional mean onto the global mean.  The
         condition a linear probe actually needs in order to fail.
  INLP   iterative nullspace projection (Ravfogel et al. 2020).  Repeatedly
         deletes the most group-predictive direction.  Included because it is
         the better-known method and because watching it plateau is
         instructive: deleting the most useful direction does not move the
         means together, so a fresh probe finds another angle.
  random project out the same number of random directions.  The control that
         separates invariance from vandalism.

Two traps this module exists to avoid
-------------------------------------
1. Fitting the eraser on all the data and then probing out of fold.  The probe
   then sees a transform built with the test group's statistics in it and can
   score BELOW chance.  Every eraser here is fitted inside the fold.
2. Declaring victory on a linear probe.  Erasure equalises means; it does not
   equalise covariances.  `nonlinear=True` re-runs the group probe with an RBF
   kernel afterwards.  If the group comes back, the leak was in the shape of
   the cloud, not its position, and no mean-matching method will touch it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .core import ProbeConfig, _impute, group_splits, identity_score, split_name

__all__ = ["EraseResult", "leace_fit", "inlp_fit", "random_fit", "erase_audit"]


# --------------------------------------------------------------------- algebra
def _sym_pow(S, p, rtol=1e-10):
    w, U = np.linalg.eigh((S + S.T) / 2.0)
    w = np.maximum(w, 0.0)
    keep = w > rtol * max(w.max(), 1e-30)
    ws = np.zeros_like(w)
    ws[keep] = w[keep] ** p
    return (U * ws) @ U.T


def _onehot(g):
    labs = np.unique(g)
    return (g[:, None] == labs[None, :]).astype(np.float64)


class _Affine:
    """x -> x - P (x - mu).  Fitted once, applied to anything."""

    def __init__(self, mu, P, rank, name):
        self.mu, self.P, self.rank, self.name = mu, P, rank, name

    def __call__(self, X):
        X = np.asarray(X, np.float64)
        return X - (X - self.mu) @ self.P.T


def leace_fit(X, groups, y=None, conditional: bool = False) -> _Affine:
    """Closed-form LEACE.  conditional=True partials the LABEL out first, so
    the direction that carries the task is protected from erasure."""
    X = np.asarray(X, np.float64)
    n, d = X.shape
    Z = _onehot(groups)
    mu = X.mean(0)
    if not conditional or y is None:
        Xc, Zc = X - mu, Z - Z.mean(0)
        Sxx, Sxz = Xc.T @ Xc / n, Xc.T @ Zc / n
    else:
        Sxx = np.zeros((d, d)); Sxz = np.zeros((d, Z.shape[1]))
        for c in np.unique(y):
            m = y == c
            Xc, Zc = X[m] - X[m].mean(0), Z[m] - Z[m].mean(0)
            Sxx += Xc.T @ Xc / n; Sxz += Xc.T @ Zc / n
    W, Wi = _sym_pow(Sxx, -0.5), _sym_pow(Sxx, 0.5)
    U, s, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    r = int((s > 1e-8 * max(s.max(), 1e-30)).sum())
    Q = U[:, :r]
    return _Affine(mu, Wi @ Q @ Q.T @ W, r, "LEACE" + ("-conditional" if conditional else ""))


def inlp_fit(X, groups, y=None, k: int = 32, seed: int = 0) -> _Affine:
    X = np.asarray(X, np.float64)
    Xw, dirs = X.copy(), []
    while len(dirs) < k:
        s = np.maximum(Xw.std(0), 1e-3 * Xw.std(0).mean() + 1e-12)
        clf = LogisticRegression(max_iter=1500, C=1.0, random_state=seed).fit(
            (Xw - Xw.mean(0)) / s, groups)
        grew = False
        for w in np.atleast_2d(clf.coef_):
            w = w / s                              # back to raw space
            for dvec in dirs:
                w = w - np.dot(w, dvec) * dvec
            nrm = np.linalg.norm(w)
            if nrm < 1e-8:
                continue
            dvec = w / nrm
            dirs.append(dvec)
            Xw = Xw - np.outer(Xw @ dvec, dvec)
            grew = True
            if len(dirs) >= k:
                break
        if not grew:
            break
    Q = np.array(dirs).T if dirs else np.zeros((X.shape[1], 0))
    return _Affine(np.zeros(X.shape[1]), Q @ Q.T, Q.shape[1], f"INLP-{Q.shape[1]}")


def random_fit(X, groups, y=None, k: int = 32, seed: int = 0) -> _Affine:
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((X.shape[1], k)))[0]
    return _Affine(np.zeros(X.shape[1]), Q @ Q.T, k, f"random-{k}")


# --------------------------------------------------------------------- probes
def _nonlinear_group_acc(X, groups, cfg: ProbeConfig, cap: int = 3000) -> float:
    """Same question, curved decision boundary.  Answers 'is the group still
    there in the second-order structure after the means were matched?'"""
    from sklearn.svm import SVC
    rng = np.random.default_rng(cfg.seed)
    keep = np.isin(groups, [g for g in np.unique(groups)
                            if (groups == g).sum() >= cfg.min_group])
    Xi, gi = X[keep], groups[keep]
    if len(np.unique(gi)) < 2:
        return float("nan")
    if len(Xi) > cap:
        sel = rng.choice(len(Xi), cap, replace=False)
        Xi, gi = Xi[sel], gi[sel]
    k = max(2, min(cfg.folds, int(np.bincount(np.unique(gi, return_inverse=True)[1]).min())))
    gp = np.empty(len(gi), dtype=object)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=cfg.seed).split(Xi, gi):
        sc = StandardScaler().fit(Xi[tr])
        gp[te] = SVC(kernel="rbf", C=1.0, gamma="scale").fit(
            sc.transform(Xi[tr]), gi[tr]).predict(sc.transform(Xi[te]))
    return float((np.asarray(list(gp), dtype=gi.dtype) == gi).mean())


def _honest_with(X, y, groups, cfg: ProbeConfig, fitter, **kw) -> float:
    """Honest task score with the eraser fitted INSIDE each fold."""
    yp = np.full(len(y), None, dtype=object)
    for tr, te, _ in group_splits(groups, cfg):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, Xte = _impute(X[tr], X[te])          # training-fold means, then erase
        f = fitter(Xtr, groups[tr], y[tr], **kw) if fitter else None
        A, B = (f(Xtr), f(Xte)) if f else (Xtr, Xte)
        sc = StandardScaler().fit(A)
        yp[te] = cfg.task_lr().fit(sc.transform(A), y[tr]).predict(sc.transform(B))
    m = np.array([v is not None for v in yp])
    return float(balanced_accuracy_score(y[m], np.asarray(list(yp[m]), dtype=y.dtype)))


# --------------------------------------------------------------------- result
@dataclass
class EraseResult:
    method: str
    rank: int
    task_before: float
    task_after: float
    identity_before: float
    identity_after: float                 # eraser saw these groups: best case for it
    identity_after_unseen_groups: float   # eraser fitted on a disjoint half of groups
    identity_after_nonlinear: float
    chance_task: float
    chance_identity: float
    majority_identity: float = float("nan")
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def floor_identity(self) -> float:
        """Group sizes are unequal, so plain accuracy cannot fall below the
        majority-group rate no matter how complete the erasure."""
        m = self.majority_identity
        return max(self.chance_identity, m if np.isfinite(m) else 0.0)

    @property
    def verdict(self) -> str:
        task_held = self.task_after >= self.task_before - 0.02
        id_gone = self.identity_after <= self.floor_identity + 0.05
        if id_gone and task_held:
            nl_back = (np.isfinite(self.identity_after_nonlinear)
                       and self.identity_after_nonlinear > self.identity_after + 0.10)
            no_transfer = (np.isfinite(self.identity_after_unseen_groups)
                           and self.identity_after_unseen_groups > self.identity_after + 0.10)
            if nl_back and no_transfer:
                return ("LINEAR AND IN-SAMPLE ONLY: a straight probe fails, but a curved one "
                        "still finds the group and the transform does not survive new groups. "
                        "Do not ship this as invariance.")
            if nl_back:
                return ("LINEAR ONLY: the group is gone from a straight probe and still there "
                        "for a curved one. The means were matched, the covariances were not.")
            if no_transfer:
                return ("IN-SAMPLE ONLY: erased on the groups it was fitted on, intact on new "
                        "ones. It learned these groups' means, not a group axis.")
            return "INVARIANCE: the group is gone from a linear probe and the task survived."
        if id_gone and not task_held:
            return "COMPRESSION: the group is gone but so is the task. You deleted signal, not nuisance."
        if not id_gone and task_held:
            return "NO EFFECT: the group is still linearly decodable. This method did not erase it."
        return "FAILURE: the group survived and the task did not."


def erase_audit(X, y, groups, method: str = "leace", *, k: int = 32,
                cfg: ProbeConfig | None = None, nonlinear: bool = True,
                conditional: bool = False, progress=None) -> EraseResult:
    cfg = cfg or ProbeConfig()
    X = np.asarray(X, np.float64); y = np.asarray(y); groups = np.asarray(groups)
    say = progress or (lambda *_: None)
    t0 = time.time()

    fitters = {
        "leace": lambda A, g, yy: leace_fit(A, g, yy, conditional=conditional),
        "inlp": lambda A, g, yy: inlp_fit(A, g, yy, k=k, seed=cfg.seed),
        "random": lambda A, g, yy: random_fit(A, g, yy, k=k, seed=cfg.seed),
    }
    if method not in fitters:
        raise ValueError(f"method must be one of {sorted(fitters)}")
    fit = fitters[method]

    say("task score before erasure")
    t_before = _honest_with(X, y, groups, cfg, None)
    say("task score after erasure (eraser refitted inside every fold)")
    t_after = _honest_with(X, y, groups, cfg, fit)

    say("identity before")
    id_b = identity_score(X, groups, cfg)
    say("identity after, in-sample (most favourable case for the eraser)")
    Xi, _ = _impute(X, X[:1])                     # in-sample throughout this block
    f_all = fit(Xi, groups, y)
    id_a = identity_score(f_all(Xi), groups, cfg).accuracy

    # does the erasure transfer to groups the eraser never saw?
    say("identity after, on a disjoint half of the groups")
    ug = np.unique(groups)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(ug)
    half_a, half_b = perm[: len(ug) // 2], perm[len(ug) // 2:]
    id_unseen = float("nan")
    notes: list[str] = []
    if len(half_a) >= 2 and len(half_b) >= 2:
        ma, mb = np.isin(groups, half_a), np.isin(groups, half_b)
        Xa, Xb = _impute(X[ma], X[mb])
        f_half = fit(Xa, groups[ma], y[ma])
        id_unseen = identity_score(f_half(Xb), groups[mb], cfg).accuracy
    else:
        notes.append("too few groups to test whether the erasure transfers to unseen groups")

    id_nl = float("nan")
    if nonlinear:
        say("nonlinear group probe after erasure (RBF)")
        id_nl = _nonlinear_group_acc(f_all(Xi), groups, cfg)
        if np.isfinite(id_nl) and np.isfinite(id_a) and id_nl > id_a + 0.10:
            notes.append(
                f"A curved boundary recovers the group at {id_nl:.3f} where a straight one "
                f"gets {id_a:.3f}. The means were matched; the covariances were not. "
                f"Mean-matching methods -- LEACE, INLP, NAP, CORAL -- cannot fix this, and "
                f"neither did adversarial training in our hands.")

    if np.isnan(X).any():
        notes.append("Missing values were imputed with training-fold column means before "
                     "each eraser was fitted; the in-sample rows use their own column means.")
    r = EraseResult(f_all.name, int(f_all.rank), t_before, t_after,
                    id_b.accuracy, id_a, id_unseen, id_nl,
                    1.0 / len(np.unique(y)), id_b.chance, id_b.majority_rate,
                    round(time.time() - t0, 2), notes)
    if np.isfinite(id_unseen) and id_unseen > id_a + 0.10:
        r.notes.append(
            f"The erasure does not transfer: on groups the eraser never saw, identity is "
            f"{id_unseen:.3f} against {id_a:.3f} in-sample. It memorised these groups' "
            f"means rather than removing a group axis.")
    return r
