"""Command line.

    leakcheck demo
    leakcheck audit emb.npy --meta meta.csv --label context --group animal_id
    leakcheck audit feats.csv --feature-cols "mfcc*" --label context --group animal_id
    leakcheck erase emb.npy --meta meta.csv --label context --group site

`audit` exits 1 when inflation exceeds --fail-over, so it can sit in CI next to
the tests and fail a pull request that reintroduces a shuffled split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _common(p):
    p.add_argument("--label", required=True, help="metadata column holding the label you predict")
    p.add_argument("--group", required=True,
                   help="metadata column holding the unit that changes at deployment: "
                        "individual, site, session, deployment")
    p.add_argument("--meta", help="csv/tsv/json metadata, one row per clip, matched by position")
    p.add_argument("--feature-cols", help="for a csv of features: 'a,b,c', 'prefix*', or 'rest'")
    p.add_argument("--key", help="array name inside a .npz")
    p.add_argument("--layer", type=int, help="layer index for a 3-D (layers, clips, dim) array")
    p.add_argument("--name", help="dataset name for the report")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--group-folds", type=int, default=0,
                   help="0 = leave-one-group-out (auto-switches to GroupKFold above 30 groups)")
    p.add_argument("--C", type=float, default=1.0, dest="C")
    p.add_argument("--no-balance", action="store_true", help="drop class_weight='balanced'")
    p.add_argument("--min-group", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="leakcheck",
        description="Report how well your features identify the ANIMAL next to how well "
                    "they identify the BEHAVIOUR. If the first number is high, the second "
                    "one is borrowing from it.")
    ap.add_argument("--version", action="version", version=f"leakcheck {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="the three numbers, the null, and the verdict")
    a.add_argument("features", help=".npy / .npz / .csv")
    _common(a)
    a.add_argument("--permutations", type=int, default=0,
                   help="permutation null for the honest score (100-1000 is useful)")
    a.add_argument("--permutation-budget", type=float, default=120.0,
                   help="seconds; the null is trimmed to fit rather than hanging")
    a.add_argument("--fail-over", type=float, default=None,
                   help="exit 1 if inflation exceeds this (for CI)")
    a.add_argument("--json"); a.add_argument("--html"); a.add_argument("--md")

    d = sub.add_parser("demo", help="synthetic corpus with a planted leak; no downloads")
    d.add_argument("--leak", type=float, default=0.95,
                   help="0 = every animal has the same class mix, 1 = each animal's mix is lopsided")
    d.add_argument("--groups", type=int, default=12)
    d.add_argument("--per-group", type=int, default=60)
    d.add_argument("--dim", type=int, default=64)
    d.add_argument("--classes", type=int, default=3)
    d.add_argument("--permutations", type=int, default=0)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-color", action="store_true"); d.add_argument("--quiet", "-q", action="store_true")
    d.add_argument("--json"); d.add_argument("--html"); d.add_argument("--md")

    e = sub.add_parser("erase", help="try to remove the group, and check you did not cheat")
    e.add_argument("features")
    _common(e)
    e.add_argument("--method", default="leace", choices=["leace", "inlp", "random"])
    e.add_argument("-k", type=int, default=32, help="rank for inlp/random (leace picks its own)")
    e.add_argument("--conditional", action="store_true",
                   help="LEACE with the label partialled out first, protecting the task direction")
    e.add_argument("--no-nonlinear", action="store_true", help="skip the RBF second-order check")
    e.add_argument("--json")
    return ap


def _cfg(args):
    from .core import ProbeConfig
    return ProbeConfig(C=args.C, balanced=not args.no_balance, folds=args.folds,
                       group_folds=args.group_folds, min_group=args.min_group, seed=args.seed)


def _write(args, report):
    from .render import to_html, to_markdown
    for attr, fn, what in (("json", lambda r: json.dumps(r.to_dict(), indent=2, default=str), "json"),
                           ("html", to_html, "html"), ("md", to_markdown, "markdown")):
        path = getattr(args, attr, None)
        if path:
            with open(path, "w") as f:
                f.write(fn(report))
            if not args.quiet:
                print(f"  wrote {what}: {path}", file=sys.stderr)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    color = not args.no_color and sys.stdout.isatty()
    say = (lambda m: print(f"  … {m}", file=sys.stderr)) if not args.quiet else (lambda m: None)

    from .audit import audit
    from .render import to_text

    if args.cmd == "demo":
        from .demo import make_demo
        X, y, g, spec = make_demo(n_groups=args.groups, n_per_group=args.per_group,
                                  dim=args.dim, n_classes=args.classes,
                                  leak=args.leak, seed=args.seed)
        r = audit(X, y, g, dataset=f"demo (leak={args.leak})", label_name="context",
                  group_name="animal", permutations=args.permutations, progress=say)
        r.notes.append("Synthetic data. " + spec["planted"])
        print(to_text(r, color))
        _write(args, r)
        return 0

    from .io import resolve
    try:
        X, y, g = resolve(args.features, args.label, args.group, args.meta,
                          args.feature_cols, args.key, args.layer)
    except Exception as exc:                                  # noqa: BLE001
        print(f"leakcheck: {exc}", file=sys.stderr)
        return 2

    name = args.name or os.path.basename(args.features)

    if args.cmd == "erase":
        from .erase import erase_audit
        from .render import erase_text
        e = erase_audit(X, y, g, args.method, k=args.k, cfg=_cfg(args),
                        nonlinear=not args.no_nonlinear, conditional=args.conditional,
                        progress=say)
        print(erase_text(e, args.group, color))
        if args.json:
            from dataclasses import asdict
            d = asdict(e); d["verdict"] = e.verdict
            with open(args.json, "w") as f:
                json.dump(d, f, indent=2)
        return 0

    try:
        r = audit(X, y, g, dataset=name, label_name=args.label, group_name=args.group,
                  config=_cfg(args), permutations=args.permutations,
                  permutation_budget_s=args.permutation_budget, progress=say)
    except ValueError as exc:
        print(f"leakcheck: {exc}", file=sys.stderr)
        return 2

    print(to_text(r, color))
    _write(args, r)

    if args.fail_over is not None and r.inflation > args.fail_over:
        print(f"\nleakcheck: inflation {r.inflation:+.3f} exceeds --fail-over "
              f"{args.fail_over:+.3f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
