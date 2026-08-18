"""Tests for leakcheck.

The interesting ones are the cases where a wrong implementation still looks
plausible: a leak that only a random split can see, an imputation that quietly
uses the test fold, a permutation null taken at the wrong unit.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from leakcheck import (ProbeConfig, audit, cramers_v, design_diagnostics,
                       erase_audit, group_bootstrap_ci, make_demo, to_html,
                       to_markdown, to_text)
from leakcheck.core import _impute, honest_score, identity_score, permutation_null, shuffled_score

CFG = ProbeConfig(folds=3)


# --------------------------------------------------------------- the headline
def test_leak_appears_only_under_a_random_split():
    X, y, g, _ = make_demo(leak=0.95, seed=1)
    r = audit(X, y, g, config=CFG)
    assert r.inflation > 0.05, "a planted per-animal class mix must inflate the random split"
    assert r.identity.accuracy > 5 * r.identity.chance
    assert r.level in ("SEVERE", "MODERATE")


def test_no_leak_when_every_animal_has_the_same_class_mix():
    X, y, g, _ = make_demo(leak=0.0, seed=1)
    r = audit(X, y, g, config=CFG)
    assert abs(r.inflation) < 0.05, f"nothing to memorise, yet inflation was {r.inflation:+.3f}"
    assert r.level == "LOW"


def test_leak_is_monotone_in_the_knob():
    infl = []
    for leak in (0.0, 0.5, 0.95):
        X, y, g, _ = make_demo(leak=leak, seed=3)
        infl.append(audit(X, y, g, config=CFG).inflation)
    assert infl[0] < infl[2], f"inflation should grow with leak, got {infl}"


def test_pure_noise_scores_chance_both_ways():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((360, 24))
    y = rng.choice(["a", "b"], 360)
    g = np.repeat([f"g{i}" for i in range(9)], 40)
    r = audit(X, y, g, config=CFG)
    assert abs(r.honest.accuracy - 0.5) < 0.09
    assert abs(r.shuffled.accuracy - 0.5) < 0.09
    assert r.identity.accuracy < 4 * r.identity.chance


# --------------------------------------------------------------- correctness
def test_imputation_uses_only_the_training_fold():
    """Global mean imputation would give the test rows a value computed partly
    from themselves. Fold-local imputation must not."""
    Xtr = np.array([[1.0], [3.0]])
    Xte = np.array([[np.nan]])
    a, b = _impute(Xtr, Xte)
    assert b[0, 0] == pytest.approx(2.0)          # mean of TRAIN only
    assert not np.isnan(a).any()


def test_column_missing_entirely_in_training_becomes_zero():
    a, b = _impute(np.array([[np.nan], [np.nan]]), np.array([[np.nan]]))
    assert a[0, 0] == 0.0 and b[0, 0] == 0.0


def test_all_nan_rows_are_reported_not_silently_imputed():
    X, y, g, _ = make_demo(n_groups=6, n_per_group=30, seed=0)
    X = X.astype(float)
    X[:20] = np.nan
    d = design_diagnostics(X, y, g)
    assert d.missing is not None and d.missing.all_nan_rows == 20
    assert any("NO features at all" in w for w in d.warnings)


def test_missingness_that_predicts_the_label_is_flagged():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8))
    y = np.array(["a"] * 100 + ["b"] * 100)
    g = np.repeat([f"g{i}" for i in range(10)], 20)
    X[y == "a", 0] = np.nan                      # missingness == the label
    d = design_diagnostics(X, y, g)
    assert d.missing.v_missing_label > 0.9
    assert any("predicts the label" in w for w in d.warnings)


def test_duplicate_rows_spanning_groups_are_flagged():
    X, y, g, _ = make_demo(n_groups=6, n_per_group=30, seed=0)
    X = X.astype(float)
    X[0] = X[-1]                                  # same clip in two animals
    d = design_diagnostics(X, y, g)
    assert d.duplicate_rows_across_groups >= 1
    assert any("DIFFERENT" in w for w in d.warnings)


def test_single_class_groups_are_flagged_and_scored_as_recall():
    X, y, g, _ = make_demo(n_groups=8, n_per_group=40, n_classes=2, seed=0)
    y = y.copy()
    y[g == "animal_00"] = "isolation"             # one animal, one class
    d = design_diagnostics(X, y, g)
    assert "animal_00" in d.single_class_groups
    h = honest_score(X, y, g, CFG)
    assert [r.metric for r in h.per_group if r.group == "animal_00"] == ["recall"]


def test_cramers_v_endpoints():
    a = np.array([0, 0, 1, 1])
    assert cramers_v(a, a) == pytest.approx(1.0, abs=0.05)
    rng = np.random.default_rng(0)
    x, z = rng.integers(0, 3, 4000), rng.integers(0, 3, 4000)
    assert cramers_v(x, z) < 0.1


def test_bootstrap_interval_brackets_the_mean():
    lo, hi = group_bootstrap_ci([0.4, 0.5, 0.6, 0.7, 0.8], seed=0)
    assert lo < 0.6 < hi


# --------------------------------------------------------------- the null
def test_permutation_null_is_not_significant_on_noise():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((240, 12))
    y = rng.choice(["a", "b"], 240)
    g = np.repeat([f"g{i}" for i in range(8)], 30)
    n = permutation_null(X, y, g, CFG, n=30)
    assert n.p_value > 0.05
    assert n.unit == "within-group"


def test_permutation_unit_switches_when_the_label_is_a_group_property():
    X, _, g, _ = make_demo(n_groups=8, n_per_group=30, seed=0)
    lab = {f"animal_{i:02d}": ("a" if i % 2 else "b") for i in range(8)}
    y = np.array([lab[v] for v in g])
    n = permutation_null(X, y, g, CFG, n=5)
    assert n.unit == "group", "one label per animal must be permuted at the animal level"


# --------------------------------------------------------------- erasure
def test_random_projection_control_does_not_remove_identity():
    X, y, g, _ = make_demo(n_groups=8, n_per_group=40, seed=0)
    e = erase_audit(X, y, g, "random", k=8, cfg=CFG, nonlinear=False)
    assert e.identity_after > 0.5 * e.identity_before, \
        "projecting out random directions must not count as erasure"


def test_leace_removes_a_pure_mean_offset():
    rng = np.random.default_rng(0)
    d, n = 12, 60
    off = rng.standard_normal((4, d)) * 4.0
    X = np.vstack([off[i] + rng.standard_normal((n, d)) for i in range(4)])
    g = np.repeat([f"g{i}" for i in range(4)], n)
    y = np.array(list("ab") * (2 * n))
    e = erase_audit(X, y, g, "leace", cfg=CFG, nonlinear=False)
    assert e.identity_before > 0.8
    assert e.identity_after < 0.45, f"means were the only signal, yet identity survived at {e.identity_after}"


def test_second_order_leak_survives_leace_and_is_reported():
    """Same centre, different spread: mean-matching cannot touch it."""
    rng = np.random.default_rng(0)
    d, n = 8, 200
    A = rng.standard_normal((n, d)) * np.array([3.0] + [0.4] * (d - 1))
    B = rng.standard_normal((n, d)) * np.array([0.4] + [3.0] * (d - 1))
    X = np.vstack([A, B])
    g = np.array(["a"] * n + ["b"] * n)
    y = np.array(list("xy") * n)
    e = erase_audit(X, y, g, "leace", cfg=CFG, nonlinear=True)
    assert e.identity_after < 0.65, "LEACE should defeat the LINEAR probe here"
    assert e.identity_after_nonlinear > 0.85, "a curved boundary should still find the group"
    assert any("covariances" in n_ for n_ in e.notes)


# --------------------------------------------------------------- plumbing
def test_report_serialises_and_renders():
    X, y, g, _ = make_demo(n_groups=6, n_per_group=30, seed=0)
    r = audit(X, y, g, config=CFG)
    d = json.loads(json.dumps(r.to_dict(), default=str))
    assert d["headline"]["inflation"] == pytest.approx(r.inflation)
    txt, md, htm = to_text(r, color=False), to_markdown(r), to_html(r)
    assert "inflation" in txt and "| **honest** |" in md
    assert htm.count("<svg") == htm.count("</svg>") and "<figcaption>" in htm
    stripped = htm.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in stripped and "https://" not in stripped, "no external refs"


def test_rejects_impossible_inputs():
    X, y, g, _ = make_demo(n_groups=4, n_per_group=20, seed=0)
    with pytest.raises(ValueError, match="at least 2 groups"):
        audit(X, y, np.array(["only"] * len(y)))
    with pytest.raises(ValueError, match="length mismatch"):
        audit(X, y[:-1], g)
    with pytest.raises(ValueError, match="infinities"):
        Xi = X.astype(float).copy(); Xi[0, 0] = np.inf
        audit(Xi, y, g)


def test_group_kfold_kicks_in_for_many_groups():
    X, y, g, _ = make_demo(n_groups=40, n_per_group=12, seed=0)
    h = honest_score(X, y, g, ProbeConfig(folds=3, loo_limit=30))
    assert h.split.startswith("GroupKFold")


def test_cli_demo_and_fail_over(tmp_path):
    out = tmp_path / "r.html"
    p = subprocess.run([sys.executable, "-m", "leakcheck.cli", "demo", "--no-color", "-q",
                        "--groups", "6", "--per-group", "30", "--html", str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "inflation" in p.stdout and out.exists()

    npy, meta = tmp_path / "x.npy", tmp_path / "m.csv"
    X, y, g, _ = make_demo(n_groups=8, n_per_group=40, leak=0.95, seed=0)
    np.save(npy, X)
    meta.write_text("ctx,animal\n" + "\n".join(f"{a},{b}" for a, b in zip(y, g)) + "\n")
    p = subprocess.run([sys.executable, "-m", "leakcheck.cli", "audit", str(npy),
                        "--meta", str(meta), "--label", "ctx", "--group", "animal",
                        "--no-color", "-q", "--fail-over", "0.01"],
                       capture_output=True, text=True)
    assert p.returncode == 1, "an inflated dataset must fail the CI gate"
    assert "exceeds --fail-over" in p.stderr
