# leakcheck

**Report how well your features identify the *animal*, next to how well they identify the
*behaviour*. If the first number is high, the second one is borrowing from it.**

```bash
pip install "git+https://github.com/slug-labz/Audio-Jungle-Book-Project.git#subdirectory=leakcheck"
leakcheck demo
```

No downloads, no GPU, no model. It runs on features you already have.

---

## The problem, in one paragraph

Animal recordings come from a handful of individuals in a handful of places. Shuffle the clips and
the same cat, recorded minutes apart in the same room, lands on both sides of the split. A model can
then score well by recognising *the cat* rather than understanding *the situation*, and accuracy
cannot tell the difference. Across four corpora we measured the cost at **10 to 24 points**, and the
same thing happens to hand-crafted features, so it is not a neural-network problem — it is an
evaluation problem.

`leakcheck` measures it. Three numbers, one command:

| | what it is |
|---|---|
| **honest** | your task, with whole animals or sites held out — what you get in the field |
| **shuffled** | your task, random split — what papers usually report |
| **identity** | how well the *same features* recover the animal or site — the explanation |

---

## Use it

```bash
# embeddings in a .npy, one metadata row per clip
leakcheck audit embeddings.npy --meta clips.csv --label context --group animal_id

# a plain csv of features
leakcheck audit features.csv --feature-cols "mfcc*" --label context --group site

# a 3-D (layers, clips, dim) array from a transformer
leakcheck audit wavlm.npy --layer 9 --meta clips.csv --label valence --group lab

# save the report next to your results
leakcheck audit emb.npy --meta m.csv --label ctx --group animal --html report.html --json report.json
```

```python
from leakcheck import audit
r = audit(X, y, groups, group_name="cat")
print(r.honest.accuracy, r.shuffled.accuracy, r.identity.accuracy, r.inflation)
```

**In CI.** `--fail-over` exits 1 when inflation crosses a threshold, so a pull request that
reintroduces a shuffled split fails the build:

```bash
leakcheck audit emb.npy --meta m.csv --label ctx --group animal --fail-over 0.05 -q
```

### Choosing `--group`

This is the only decision that matters. **The group is whatever will be different at deployment
time.** Individual animal, recording site, deployment, season, microphone, annotator. If you pick
the clip, you have measured nothing. If your recordings have more than one such axis, run it once
per axis — the answers are usually not the same.

---

## What it checks that you did not ask for

The design diagnostics run before any model is fitted, and they cost nothing:

- **Missing features.** Where the holes are, whether they cluster by group, and — the one people
  miss — whether *having* a missing value predicts the label. If it does, "which extraction failed"
  is a feature and your model can score on it.
- **Rows with no features at all.** After imputation these become one constant vector. Any accuracy
  on them is the class prior.
- **Duplicate rows that span groups.** The same clip on both sides of the split.
- **Label–group confounding** (Cramér's V). If every animal contributes one class, a held-out animal
  brings its own base rate with it and part of your honest score is guessing.
- **How many groups you actually have.** The honest score has as many independent observations as
  you have animals, not as you have clips. Six labs is six observations, and the reported bootstrap
  interval is over groups for that reason.
- **Per-group scores.** An average over animals is not a promise to any one of them.

It also runs a **permutation null at the right unit** — labels shuffled inside each group when the
label varies within a group, group→label assignment shuffled when it does not. Permuting clips when
the label is a property of the animal manufactures significance, and it is the standard way to do it
by accident.

---

## Trying to remove the shortcut

```bash
leakcheck erase emb.npy --meta m.csv --label valence --group lab --method leace
```

Fits LEACE (or INLP, or a random-projection control) **inside every fold**, then reports four
numbers instead of the usual one:

```
LEACE  rank 5
  task      0.436 → 0.456   (chance 0.500) must hold
  lab       0.541 → 0.242   (floor 0.291) must fall

  identity on groups the eraser never saw   0.773
  identity with a curved boundary (RBF)     0.661

  LINEAR AND IN-SAMPLE ONLY: a straight probe fails, but a curved one still finds
  the group and the transform does not survive new groups. Do not ship this as invariance.
```

Those last two lines are the point. Erasure equalises group *means*; it does not equalise group
*covariances*, and it can memorise the means of the groups it was fitted on. Report a linear probe
alone and both failures are invisible — which is how a method that does nothing useful gets
published as invariance.

---

## Design decisions worth knowing

- **Linear probes throughout.** A linear probe measures what is *present in the representation*.
  A deep head measures what a deep head can dig out, which is a different question and a worse one
  for this purpose.
- **Balanced accuracy** for the task (chance = 1/classes); **plain accuracy** for identity, compared
  against both `1/groups` and the majority-group rate, because group sizes are never equal.
- **The scaler is fitted inside the fold.** So is imputation. Filling missing values from column
  means computed over the whole matrix lets the test rows help decide what the training rows become
  — a leak, in a tool about leaks.
- **Leave-one-group-out** while the group count is manageable, `GroupKFold` above 30 groups, so this
  stays usable on a corpus with 173 individuals.
- Two dependencies: `numpy`, `scikit-learn`.

---

## Where the numbers in this README come from

The pig example is real: the 18 acoustic features published with the Soundwel corpus, 5,031 clips,
6 recording labs, binary valence. `leakcheck` reports honest **0.436** against shuffled **0.575**,
and flags that **700 of the 1,381 NMBU clips have no features at all** — every column missing, all
of them labelled negative. Full write-up in [`../FINDINGS.md`](../FINDINGS.md).

MIT licensed. Issues and corrections: **vishrutmalhotra4@gmail.com**
