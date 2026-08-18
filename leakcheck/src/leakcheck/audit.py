"""Orchestration: run every measurement, derive the headline, write the verdict."""
from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from .core import (IdentityScore, PermutationNull, ProbeConfig, TaskScore,
                   honest_score, identity_score, permutation_null, shuffled_score)
from .diagnose import Design, design_diagnostics, group_bootstrap_ci

__all__ = ["Report", "audit"]

SEVERE, MODERATE, LOW = "SEVERE", "MODERATE", "LOW"


@dataclass
class Report:
    dataset: str
    label_name: str
    group_name: str
    design: Design
    honest: TaskScore
    shuffled: TaskScore
    identity: IdentityScore
    null: PermutationNull | None
    config: ProbeConfig
    honest_ci: tuple = (float("nan"), float("nan"))
    created: str = ""
    version: str = ""
    environment: str = ""
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- derived
    @property
    def inflation(self) -> float:
        return self.shuffled.accuracy - self.honest.accuracy

    @property
    def leak_fraction(self) -> float:
        """Of the skill the shuffled split appeared to show, how much vanishes
        when whole groups are held out.  1.0 = all of it."""
        apparent = self.shuffled.accuracy - self.shuffled.chance
        if apparent <= 1e-9:
            return float("nan")
        return float(1.0 - (self.honest.accuracy - self.honest.chance) / apparent)

    @property
    def identity_ratio(self) -> float:
        if not np.isfinite(self.identity.chance) or self.identity.chance <= 0:
            return float("nan")
        return float(self.identity.accuracy / self.identity.chance)

    @property
    def level(self) -> str:
        """Severity on two axes, because they disagree in opposite regimes.

        Raw inflation catches the big absolute gaps.  The fraction catches the
        case where a task is hard, the shuffled score is 0.60 against a 0.50
        chance rate, and two thirds of that thin margin is the animal -- small
        in points, fatal to the claim.  The fraction is only consulted when
        there is enough apparent skill for the ratio to mean anything.
        """
        i, f = self.inflation, self.leak_fraction
        apparent = self.shuffled.accuracy - self.shuffled.chance
        usable = np.isfinite(f) and apparent >= 0.05
        i = round(i, 3)          # compare at the precision we print
        if i >= 0.10 or (usable and f >= 0.50 and i >= 0.02):
            return SEVERE
        if i >= 0.05 or (usable and f >= 0.25 and i >= 0.01):
            return MODERATE
        return LOW

    @property
    def worst_group(self):
        return self.honest.per_group[0] if self.honest.per_group else None

    # ---------------------------------------------------------------- verdict
    def verdict(self) -> list[str]:
        out: list[str] = []
        i, lf = self.inflation, self.leak_fraction
        frac = "" if not np.isfinite(lf) else f" That is {max(0.0, min(1.0, lf)):.0%} of the apparent skill."

        if self.level == SEVERE:
            out.append(f"Shuffling the clips adds {i:+.3f} to the reported score.{frac} "
                       f"A number produced by a random split of this dataset is not a "
                       f"number about behaviour."
                       + ("" if i >= 0.10 else
                          " The gap is small in points only because the task is hard; "
                          "as a share of the margin over chance it is most of it."))
        elif self.level == MODERATE:
            out.append(f"Shuffling the clips adds {i:+.3f} to the reported score.{frac} "
                       f"Small enough to argue about, large enough to change a ranking "
                       f"between two methods.")
        else:
            out.append(f"Shuffling the clips changes the score by {i:+.3f}. On this dataset "
                       f"the random split is not badly misleading -- which is worth stating "
                       f"explicitly, because it is not the usual case.")

        if np.isfinite(self.identity_ratio):
            if self.identity.accuracy > max(3 * self.identity.chance, self.identity.majority_rate):
                out.append(f"The shortcut was available: the same features identify the "
                           f"{self.group_name} at {self.identity.accuracy:.3f} against a "
                           f"{self.identity.chance:.3f} chance rate "
                           f"({self.identity_ratio:.0f}x) and a {self.identity.majority_rate:.3f} "
                           f"majority rate.")
            else:
                out.append(f"{self.group_name.capitalize()} identity is weak in these features "
                           f"({self.identity.accuracy:.3f} vs {self.identity.chance:.3f} chance), "
                           f"so whatever the shuffled split is buying, it is probably not "
                           f"individual recognition.")

        lo, hi = self.honest_ci
        if np.isfinite(lo):
            if lo <= self.honest.chance:
                out.append(f"The honest score is {self.honest.accuracy:.3f} with a "
                           f"{self.design.n_groups}-group bootstrap interval of [{lo:.3f}, {hi:.3f}], "
                           f"which includes chance ({self.honest.chance:.3f}). Held out properly, "
                           f"this result is not established.")
            else:
                out.append(f"The honest score is {self.honest.accuracy:.3f}, interval "
                           f"[{lo:.3f}, {hi:.3f}] over {self.design.n_groups} groups, above the "
                           f"{self.honest.chance:.3f} chance rate.")

        if self.null is not None and np.isfinite(self.null.p_value):
            unit = ("labels shuffled inside each group" if self.null.unit == "within-group"
                    else f"the {self.group_name}->label assignment shuffled")
            verdict = "survives" if self.null.p_value <= 0.05 else "does NOT survive"
            out.append(f"Permutation null ({unit}, n={self.null.n_permutations}): "
                       f"p = {self.null.p_value:.3f}, null mean {self.null.null_mean:.3f}. "
                       f"The honest score {verdict} its own null.")

        w = self.worst_group
        if w is not None and len(self.honest.per_group) >= 3:
            out.append(f"Worst {self.group_name}: {w.group} at {w.score:.3f} over {w.n} clips "
                       f"(best {self.honest.per_group[-1].score:.3f}). An average over groups "
                       f"is not a promise to any one of them.")
        return out

    # ---------------------------------------------------------------- export
    def to_dict(self) -> dict:
        return {
            "leakcheck": self.version, "created": self.created,
            "environment": self.environment,
            "dataset": self.dataset, "label": self.label_name, "group": self.group_name,
            "headline": {
                "honest": self.honest.accuracy, "shuffled": self.shuffled.accuracy,
                "identity": self.identity.accuracy, "inflation": self.inflation,
                "leak_fraction": self.leak_fraction, "level": self.level,
                "chance_task": self.honest.chance, "chance_identity": self.identity.chance,
                "honest_ci95": list(self.honest_ci),
            },
            "honest": asdict(self.honest), "shuffled": asdict(self.shuffled),
            "identity": asdict(self.identity),
            "null": asdict(self.null) if self.null else None,
            "design": asdict(self.design), "config": asdict(self.config),
            "verdict": self.verdict(), "notes": self.notes,
        }


# --------------------------------------------------------------------------- run
def audit(X, y, groups, *, dataset: str = "dataset", label_name: str = "label",
          group_name: str = "group", config: ProbeConfig | None = None,
          permutations: int = 0, permutation_budget_s: float | None = 120.0,
          progress=None) -> Report:
    """Run the full audit.

    Parameters
    ----------
    X : (n_clips, n_features) features -- embeddings, descriptors, anything.
    y : (n_clips,) the label you are claiming to predict.
    groups : (n_clips,) the unit that will be different at deployment time:
        the individual animal, the recording site, the session, the deployment.
        Getting this column right is the whole exercise.
    permutations : 0 disables the null; 100-1000 is a useful range.
    """
    from . import __version__
    cfg = config or ProbeConfig()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(f"length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}")
    if len(np.unique(groups)) < 2:
        raise ValueError("need at least 2 groups -- with one group there is nothing to hold out")
    if len(np.unique(y)) < 2:
        raise ValueError("need at least 2 classes in the label")
    if np.isinf(X).any():
        raise ValueError(f"X contains infinities in "
                         f"{int(np.isinf(X).any(1).sum())} rows -- clean them before auditing")
    # NaN is not rejected: missingness is data, and where it sits is often the
    # most informative thing in the file. It is imputed inside each fold and
    # reported by the design diagnostics.

    say = progress or (lambda *_: None)

    say("design diagnostics")
    design = design_diagnostics(X, y, groups, cfg.min_group)

    say("honest score (groups held out)")
    hon = honest_score(X, y, groups, cfg)
    say("shuffled score (random split)")
    shu = shuffled_score(X, y, cfg)
    say("identity probe")
    ide = identity_score(X, groups, cfg)

    ci = group_bootstrap_ci([g.score for g in hon.per_group],
                            [g.n for g in hon.per_group], seed=cfg.seed) \
        if len(hon.per_group) >= 2 else (float("nan"), float("nan"))

    nul = None
    if permutations > 0:
        say(f"permutation null ({permutations} refits)")
        nul = permutation_null(X, y, groups, cfg, permutations, permutation_budget_s)

    return Report(dataset=dataset, label_name=label_name, group_name=group_name,
                  design=design, honest=hon, shuffled=shu, identity=ide, null=nul,
                  config=cfg, honest_ci=ci,
                  created=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                  version=__version__,
                  environment=f"python {platform.python_version()} on {platform.system()}")
