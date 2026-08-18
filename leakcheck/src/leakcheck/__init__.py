"""leakcheck -- is your model reading the behaviour, or recognising the animal?

    from leakcheck import audit
    r = audit(X, y, groups)      # groups = the animal, the site, the session
    print(r.honest, r.shuffled, r.identity, r.inflation)

One rule underneath all of it: the thing that will be different at deployment
time must be entirely absent from training. In bioacoustics that thing is
almost never the clip. It is the animal, the microphone, the room, the season.
"""
from .audit import Report, audit
from .core import (IdentityScore, PermutationNull, ProbeConfig, TaskScore,
                   honest_score, identity_score, permutation_null, shuffled_score)
from .demo import make_demo
from .diagnose import Design, cramers_v, design_diagnostics, group_bootstrap_ci
from .erase import EraseResult, erase_audit, inlp_fit, leace_fit
from .render import to_html, to_markdown, to_text

__version__ = "0.1.0"

__all__ = [
    "audit", "Report", "ProbeConfig", "TaskScore", "IdentityScore", "PermutationNull",
    "honest_score", "shuffled_score", "identity_score", "permutation_null",
    "design_diagnostics", "Design", "cramers_v", "group_bootstrap_ci",
    "erase_audit", "EraseResult", "leace_fit", "inlp_fit",
    "make_demo", "to_text", "to_markdown", "to_html", "__version__",
]
