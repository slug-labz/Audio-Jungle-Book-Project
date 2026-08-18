"""A synthetic corpus with a leak you can dial.

The generative story is the one the real corpora actually have, which is not
the one you would guess.  A linear probe cannot learn a private rule per
animal -- it has one weight vector for everybody.  What it *can* learn is
where each animal sits in feature space, and what that animal usually happens
to be doing:

    animal 7 is loud and nasal, and 80% of animal 7's clips are labelled
    "aroused", so predict "aroused" whenever the clip smells of animal 7.

Under a random split that rule is available and pays.  Hold animal 7 out and
it is worthless, because the model has never met animal 7 and its base rate.
Every real corpus has this: individuals are not recorded in equal proportions
across contexts, because animals do not take turns.

Two knobs:

    leak     spread of the per-animal class base rates. 0 = every animal is
             50/50 and there is nothing to memorise; 1 = base rates run the
             whole range and the shuffled split is mostly reading the animal.
    shared   strength of the one context direction that every animal really
             does share. This is the signal that survives an honest split.

Note what the per-group table does to this: an animal-level score of 0.50
everywhere can still pool into 0.80 overall, because the pooled sensitivity
comes from the high-base-rate animals and the pooled specificity from the low
ones. That is not a bug in balanced accuracy. It is why one average is never
enough.
"""
from __future__ import annotations

import numpy as np

__all__ = ["make_demo"]


def make_demo(n_groups: int = 12, n_per_group: int = 60, dim: int = 64,
              n_classes: int = 3, leak: float = 0.95, shared: float = 0.55,
              identity: float = 0.55, private: float = 0.2, noise: float = 1.0,
              seed: int = 0):
    """Return (X, y, groups, spec)."""
    rng = np.random.default_rng(seed)
    names = ["isolation", "brushing", "waiting", "feeding", "play"][:n_classes]

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    # one direction per class that every animal shares -- the honest signal
    U = unit(rng.standard_normal((n_classes, dim)))
    offsets = rng.standard_normal((n_groups, dim)) * identity
    priv = unit(rng.standard_normal((n_groups, n_classes, dim)))

    # per-animal class mix. leak -> 0 gives every animal the same flat mix;
    # leak -> 1 gives each animal a lopsided one, which is what real corpora
    # look like because animals are not recorded doing each thing equally often
    alpha = np.full(n_classes, max(1e-3, (1.0 - leak) * 40.0 + 0.25))
    mix = rng.dirichlet(alpha, n_groups)

    X, y, g = [], [], []
    for i in range(n_groups):
        lab = rng.choice(n_classes, n_per_group, p=mix[i])
        sig = shared * U[lab] + private * priv[i][lab]
        X.append(offsets[i] + sig + noise * rng.standard_normal((n_per_group, dim)))
        y.append(np.array([names[c] for c in lab]))
        g.append(np.array([f"animal_{i:02d}"] * n_per_group))

    return (np.vstack(X).astype(np.float32), np.concatenate(y), np.concatenate(g),
            {"n_groups": n_groups, "n_per_group": n_per_group, "dim": dim,
             "n_classes": n_classes, "leak": leak, "shared": shared,
             "identity": identity, "private": private, "noise": noise, "seed": seed,
             "class_mix_per_animal": [[round(float(v), 2) for v in row] for row in mix],
             "planted": (f"one direction per context that every animal shares (strength "
                         f"{shared}), plus a lopsided per-animal class mix that only a "
                         f"random split can exploit")})
