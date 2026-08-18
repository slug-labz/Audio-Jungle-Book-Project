# Overnight run — findings, 17 Aug 2026

Everything below was run on the laptop (Apple M2, CPU only) between roughly 04:30 and 07:00.
No GPU, no cluster. Raw outputs, code and figures:
`experiments/context_probe/` (code), `results/` (JSON), `figures/` (plots).

---

## 0. One-paragraph summary

Frozen **human-speech** encoders classify a cat's behavioural context better than hand-crafted
features (0.571 vs 0.399, chance 0.333) and better than ESP's own bioacoustic encoder AVES (0.540).
But most of that within-species signal is **not the animal** — the quiet background frames alone
reach 0.725 on the binary task, because CatMeows induced isolation by moving the cat to a different
room. The result that *does* survive every control is cross-species: an affect direction learned on
**dogs** (aggression vs play) predicts **cat** contexts (isolation vs brushing) at 0.583 mean over
all layers, p=0.0033, replicated in a second encoder, rising monotonically with depth, and **not**
reproducible from the background or from simple acoustics. The reverse direction (cat→dog) does not
survive. Most striking: duration is the only simple feature that points the same way in both species
— energy, spectral centroid and zero-crossing rate all point in **opposite** directions — yet the
deep embedding still finds a transferable direction. So whatever transfers is *not* "loud, long and
high-pitched means distress."

---

## 1. Experiment 1 — within-species context (CatMeows)

440 meows · 21 cats · 3 contexts (brushing / waiting-for-food / isolation) · leave-one-cat-out ·
chance 0.333.

| encoder | best layer | context (held-out cat) | context (random split) | cat identity (21-way) |
|---|---|---|---|---|
| hubert-base-ls960 | 12 | **0.571** | 0.671 | 0.700 |
| wavlm-base-plus | 3 | 0.559 | 0.725 | 0.793 |
| wav2vec2-base | 2 | 0.554 | 0.737 | 0.755 |
| **AVES-bio (ESP)** | 5 | 0.540 | 0.736 | 0.793 |
| MFCC baseline (85-d) | – | 0.399 | 0.602 | 0.768 |

**(a) Speech encoders beat hand-crafted features.** +0.17 over MFCC. Solid.

**(b) Speech encoders beat ESP's bioacoustic AVES** (0.571 vs 0.540). Consistent with *Crossing the
Species Divide* (DCASE 2025), which found frozen speech models rival fine-tuned bioacoustic models on
BEANS. Here it holds on a task none of them were built for. Caveat: one dataset, one species, and
AVES-bio is the 2023 model — the newer AVEX `esp_aves2_*` checkpoints were not tested.

**(c) Random splits inflate massively.** Same features, same classifier: 0.399 → 0.602 (MFCC),
0.559 → 0.725 (WavLM). The reason is in the last column — these features identify *which of 21 cats*
is meowing at 0.70–0.79 accuracy (chance 0.048). Any split that lets a cat appear in both train and
test is measuring voice recognition.

**(d) Identity fades with depth; context does not.** WavLM identity falls 0.820 (L1) → 0.655 (L12)
while context stays ~0.51–0.56. The context/identity ratio is best at the deepest layers. If you want
an encoder that hears the *situation* rather than the *animal*, go deep.

**Confusion matrix (MFCC, held-out cat):** brushing 0.44, isolation 0.44, **waiting-for-food 0.29 —
below chance**, mostly misfiled as isolation. The only axis with real signal is calm-at-home versus
alone-in-a-strange-room. That is an **arousal** axis, not a lexicon.

---

## 2. THE BIG CAVEAT — the room, not the cat

CatMeows induces isolation by **moving the cat to an unfamiliar room**. So the negative class differs
by recording environment *by construction*. Test: discard the vocalisation, keep only the quietest
30% of frames, rerun.

| | background-only | full embedding |
|---|---|---|
| within-species, cat (binary) | **0.725** | 0.69–0.79 |
| within-species, dog (binary) | 0.614 | 0.74–0.89 |
| transfer cat→dog | 0.511 (p=0.460) | 0.527 |
| transfer dog→cat | 0.553 (p=0.113) | **0.583 (p=0.0033)** |

**The background alone does about as well as the full embedding within cats.** A large share of
"context classification" on this dataset is room/channel, not voice. Reverb proxy confirms it: the
energy-decay slope separates classes at +0.67/−0.32 SD in dogs and +0.10/−0.18 in cats.

This is a caveat for Experiment 1 and, separately, **a finding in its own right** — it is the
leakage-audit idea landing on real data. Published accuracies on CatMeows are much higher than ours;
before believing any of them, check both the split *and* whether the room is doing the work.

Crucially, **the background does not transfer across species** (p=0.113), so §3 survives this.

---

## 3. Experiment 3 — cross-species affect transfer

Added dog barks (Molnár 2008, via ESP's BEANS `dogs`): 693 barks, 10 named dogs.
Binary affect axis, chosen to be defensible in both species:

- **negative / agonistic-distress** = cat isolation (221) + dog aggression (99)
- **positive / affiliative** = cat brushing (127) + dog play (209)
- excluded as ambiguous: cat waiting-for-food, dog contact

656 clips, 20 cats + 10 dogs. Z-scored **within species** (removes the species offset — without this
you are partly just detecting cat-vs-dog). Train on all of species A, test on all of species B.
300-permutation null. Chance 0.500.

**Selection-free statistic — mean over all 13 layers, nothing chosen after seeing the answer:**

| encoder | dog→cat | null | p | cat→dog | p |
|---|---|---|---|---|---|
| wavlm | **0.583** | 0.490±0.043 | **0.0033** | 0.527 | 0.283 |
| hubert | **0.562** | 0.490 | **0.0367** | 0.522 | 0.293 |
| wav2vec2 | 0.530 | 0.495 | 0.210 | 0.551 | 0.090 |

Per-layer dog→cat (WavLM) is significant at L7 (0.624, p=0.013), L9 (0.656, p=0.000),
L10 (0.646, p=0.007), L11 (0.632, p=0.030), L12 (0.623, p=0.017) — **a monotonic rise with depth, not
an isolated spike.** cat→dog reaches p<0.05 only at L3 and is noise elsewhere.

Bootstrap over target individuals (resampling whole animals): hubert best layer 0.631, 95% CI
[0.582, 0.676] — excludes 0.5. Per-cat consistency 12/19 above chance (median 0.579).

**Verdict: dog→cat is probably real. cat→dog is not established.**

---

## 4. The most interesting result — simple acoustics point the *opposite* way

Control: 4 interpretable features (log duration, log energy, spectral centroid, zero-crossing rate).

Within species they work (cat 0.638, dog 0.643). **Across species they fail completely**:
cat→dog 0.459 (p=0.607), dog→cat 0.448 (p=0.683).

Per-feature separation, negative minus positive, in SD:

| feature | cat (isolation − brushing) | dog (aggression − play) | same direction? |
|---|---|---|---|
| log duration | +0.28 | +0.31 | **yes** |
| log energy | +0.69 | −0.19 | no |
| spectral centroid | +0.12 | −0.47 | no |
| zero-crossing rate | +0.14 | −0.48 | no |

Only duration agrees. The textbook arousal cues (louder, higher, harsher) **invert between these two
species on these two contrasts** — dog play is louder and harsher than dog aggression, while cat
isolation is louder and harsher than cat brushing. And yet the deep embedding still transfers.

So the transferable structure is **not** the classical acoustic-universals story. That is the single
most publishable sentence of the night, and also the one most in need of a third species.

---

## 5. What would still kill it

Not yet controlled, in rough order of danger:

1. **Different corpora, different labs, different microphones.** Cats (Italy, 2021) and dogs
   (Hungary, 2008) were recorded by different teams. Within-species z-scoring removes the mean
   offset but not a class-correlated channel difference that happens to align across corpora.
   *Settles it:* a third species from a fourth lab; or matched-channel augmentation (convolve both
   corpora with the same set of impulse responses and re-run).
2. **Only two species, one contrast each.** The "affect axis" is really "isolation-vs-brushing"
   mapped onto "aggression-vs-play". Those may share something specific rather than general.
   *Settles it:* bats, pigs, zebra finches — datasets that carry several contexts each.
3. **Ten dogs, twenty cats.** Individual-level bootstrap CIs are wide.
   *Settles it:* more individuals, or a mixed-effects model with animal as a random effect.
4. **Class imbalance interacting with `class_weight='balanced'`.** Cat is 221/127 negative-heavy,
   dog is 99/209 negative-light — the imbalance is *inverted* between species, which is exactly the
   configuration where a badly-calibrated probe can look like it transfers.
   *Settles it:* subsample to equal class sizes and rerun; report both.
5. **Transductive z-scoring.** The target-species mean/SD uses test data. Standard in domain
   adaptation, but worth reporting a strict version that z-scores the target using held-out-animal
   statistics only.
6. **Excluded categories.** Dropping cat-food and dog-contact was a judgement call made before
   seeing results, but it is still a researcher degree of freedom. *Settles it:* report the 3-class
   and 4-class variants too.

---

## 6. Housekeeping / state

- Environment: python 3.12, torch 2.13, transformers 5.15, CPU only.
- Datasets: CatMeows (8.9 MB), dog barks (901 MB); later pigs (326 MB) and bats (5.2 GB, subsampled). See `data/README.md`.
- Total compute cost: **$0**.

## 7. Files

```
ctx/
  context_probe.py      experiment 1, MFCC baseline           (stock anaconda python)
  encoders.py           experiment 1b, 5 frozen encoders      (venv)
  cross_species.py      experiment 3, cat<->dog transfer
  validate_transfer.py  permutation tests, nested selection, controls
  replicate.py          3-encoder replication + bootstrap
  room_control.py       the background/room confound test
  figures.py            all six figures
  out_enc/ out_cross/ out_fig/   results json, cached embeddings, PNGs
```

---

# ADDENDUM — adversarial review (Fable 5 synthesis) and what it changed

A senior-reviewer pass over the numbers above found three things I had wrong or overstated.
Recording them here because they matter more than the results did.

## R1. The headline was the wrong result

I led with cross-species transfer. The reviewer's judgement — and I now agree — is that the
**identity-leakage finding is the solid one** and the transfer is "suggestive, not established."
Leakage is replicated across all five feature sets, is mechanistically coherent, needs no new data,
and is already chapter 4 of the plan ("random splits leak recorder/site/time; nobody measured").
We have now measured a fourth leak axis: **individual**.

## R2. "Embeddings transfer where acoustics cannot" was overstated

**log-duration is SAME-SIGN in both species** (cat +0.28 SD, dog +0.31 SD). A same-sign feature of
d≈0.3 predicts transfer around Φ(d/2) ≈ 0.556 — which sits exactly inside the observed band
(0.530–0.583). Mean-pooled SSL states encode clip duration well. So **"the probe found duration"**
is a live and sufficient null, and I had not tested it.

The 4-feature control failed to transfer not because "handcrafted features can't", but because,
trained on cats, it loads on **energy** (d=0.69 in cats) which **anti-transfers** (−0.19 in dogs) and
drags the probe below chance. That is a *weighting* artifact, not evidence about handcrafted features
in general. The correct claim is narrower: "embeddings transfer where *this particular 4-feature
probe* does not."

Also worth noting: the sign flips are *predicted* by Morton's motivation-structural rules — the two
"negatives" are different emotions (dog hostility lowers pitch and adds roughness; cat distress
raises pitch). That makes any genuine transfer more interesting, not less.

## R3. The statistics were anti-conservative

- **Permutation unit was wrong.** 300 shuffles of *clip* labels treat 348 correlated meows as
  independent, when identity is 79% decodable. The exchangeable unit is the **animal**. At animal
  level, 12/19 cats above 0.5 is a sign test at p≈0.18 — *not* significant.
- **p-floor.** With 300 permutations the smallest reportable p is 1/301 = 0.0033. WavLM hit the
  floor; "p=0.0033" should have read "p < 0.0033". Rerunning with 2,000.
- **The asymmetry may not exist.** dog→cat 0.583 vs cat→dog 0.527, with null SDs ≈0.04, gives
  z≈0.95, p≈0.34. All three cat→dog point estimates are *above* 0.5. The honest sentence is
  "cat→dog is underpowered", never "cat→dog fails". Explaining an asymmetry before establishing it
  is the classic Gelman error.
- **b was overstated too.** HuBERT 0.571 vs AVES 0.540 is ~14 clips in a 21-cluster design where
  per-cat accuracy spans 0.00–0.86. The defensible version is the *negative*: "bioacoustic
  pretraining confers no measurable advantage over speech SSL on domestic-animal vocalisations."

## R4. And one correction to the reviewer

It proposed that identity accuracy *predicts* each encoder's inflation gap — an elegant internal
coherence. Computed: Pearson r = 0.813 but **p = 0.094**; Spearman r = 0.410, p = 0.493. With n=5
encoders this is **suggestive, not significant**. Worth a figure, not a claim.

## The two papers

**Paper A — exists now.** "Individual identity, not context: an evaluation-leakage audit of frozen
encoders for animal vocalisations." Needs: eGeMAPS baseline (the honest paralinguistics baseline,
not MFCC-85), one more individually-labelled dataset (Prat bats, 15k calls), animal-level statistics
throughout. Venue: ICBINB @ NeurIPS (negative/surprising results) — **check the deadline first**.

**Paper B — contingent on the kill-test.** Cross-species transfer. Write nothing until duration is
partialled out and animal-level permutation is run. If it dies, the autopsy ("apparent cross-species
affect transfer reduces to call duration") is itself a good ICBINB submission. Win either way.

## Ranked next experiments

1. **E1 kill-test battery** — duration-only transfer, duration-partialled embeddings, animal-level
   permutation (2,000 draws), single-feature ablations, asymmetry test, individual×class tables.
   *Running now.* Decides whether Paper B exists.
2. **E2 third species: pigs** (Briefer 2022, 7,414 calls, explicitly valence-labelled with arousal
   covariates). Turns a pair into a 3×3 matrix and separates the arousal vs valence hypothesis.
3. **E3 scale replication: Egyptian fruit bats** (Prat 2016, ~15k calls, emitter ID). Does
   identity-dominance hold on a wild, non-human-directed species at 30× the sample size?
4. **E4 placebo axis.** Train a probe on an arbitrary non-affect binary, test on cat affect. Must be
   ≈0.5, or generic corpus alignment is driving everything.
5. **E5 channel stress test.** Re-embed under low-pass / reverb / gain; do context decisions survive
   where identity decisions do not? Doubles as the thesis' trustworthiness figure.

## Two bookkeeping items to fix before anything ships

- Cat count is inconsistent across sections: 21 (Exp 1), 20 (Exp 3 header), 19 (per-cat consistency).
  Resolve which animals were dropped at each stage and why.
- State openly that within-species z-scoring is **transductive** (uses test-corpus statistics). It is
  defensible for deployment and the permutation null shares it, but unstated it reads as leakage.
  Report AUC alongside balanced accuracy — AUC is immune to intercept transfer.

---

# KILL-TEST VERDICT (E1 battery, complete)

| test | dog→cat | note |
|---|---|---|
| **duration alone** (1 feature) | **0.557** | almost exactly the predicted Φ(0.3/2)≈0.556 |
| embedding, raw (WavLM, mean over 13 layers) | 0.583 | |
| embedding, **log-duration regressed out** | **0.552** | drop of 0.031 |
| animal-level permutation, 2,000 draws (raw) | **p = 0.0045** | correct exchangeable unit; survives |
| asymmetry dog→cat − cat→dog = +0.056 | **p = 0.44** | **not significant** — never say "cat→dog fails" |
| single features: energy / centroid / zcr | 0.354 / 0.504 / 0.507 | energy anti-transfers, as predicted |

**Reading.** The transfer is real at the animal level, but roughly **half of it is duration** — the one
cue that points the same way in both species. What remains after removing duration (0.552 vs null
0.490±0.036, z≈1.7) is small and borderline; it needs its own permutation p-value and, more than
that, a third species. Combined with the earlier controls (equal class sizes 0.618 p<0.0001; strict
scaling 0.591 p=0.005), the honest one-liner is:

> *An affect axis transfers dog→cat in frozen speech embeddings and survives animal-level, balanced
> and strict-scaling tests; about half the effect is call duration, the residual is small, and the
> apparent directional asymmetry is not statistically supported.*

That is a clean, modest, ICBINB-shaped result — not the headline. The headline remains identity
leakage + the room confound (Paper A).

**Next:** third species (pigs, Briefer 2022, valence-labelled) — the only thing that can break the
two-corpus channel confound and tell arousal from valence.

---

# ITERATION 2 (afternoon 17 Aug) — specificity, residual, third species

## Placebo axes: the transfer is SPECIFIC to affect

If *any* dog binary predicted *any* cat binary, "transfer" would just be generic corpus alignment.
Mean over 13 WavLM layers, chance 0.500:

| pair | acc | verdict |
|---|---|---|
| dog AFFECT → cat AFFECT (the claim) | **0.583** | |
| dog SEX → cat AFFECT (placebo) | 0.513 | chance |
| dog AFFECT → cat SEX (placebo) | 0.498 | chance |
| cat SEX → dog AFFECT (placebo) | 0.496 | chance |
| dog SEX → cat SEX (matched, non-affect) | 0.504 | chance |

Every placebo sits at chance. Whatever transfers, it is the affect axis and nothing else. This is
the single most convincing control so far, and it costs nothing.

## The residual after removing duration IS significant (barely)

Regress log-duration out of every embedding dimension, then animal-level permutation (500 draws):
**0.552, null 0.489 ± 0.037, p = 0.048.** Intercept-free AUC: raw mean 0.673, residual mean 0.627,
deep layers 0.70–0.75. So there is real signal beyond duration — small, and it lives in the deep
layers, same place the identity signal fades.

Updated one-liner:
> *An affect axis learned on dog barks transfers to cat meows (0.583, animal-level p = 0.0045),
> survives balanced classes, strict scaling, and every placebo axis; roughly half is call duration,
> and the residual is small but significant (0.552, p = 0.048; AUC 0.63).*

## Bats: BEANS packaging has NO context labels

The BEANS `egyptian_fruit_bats` zip (5.2 GB) is 10 emitters × 1,000 calls with **Emitter only** —
the file IDs were renumbered, so the original Prat 2017 `Annotations.csv` (91,080 rows, contexts,
addressees; figshare article 4555903, downloaded) **cannot be joined** (emitter agreement 0.1%).
The full annotated audio is on figshare in ~3.6 GB chunks (article "files 207–224"). Parked.
Instead: BEANS bats is a perfectly balanced **identity** set → running the identity-decodability
replication on a wild species (2,000 calls, native-16k vs 4× time-stretch to fold 0–32 kHz into the
encoder's band — ESP's bandwidth problem #6, tested for free).

## Pigs: Soundwel is public — the real third species

Briefer et al. 2022 → **Soundwel database, Zenodo 8252482, CC-BY-4.0, 326 MB, 6,888 calls, 17
contexts with positive/negative valence.** Downloading. This is the dataset that turns one species
pair into a matrix and can separate the arousal-vs-valence hypotheses.

---

# ITERATION 3 (17 Aug, afternoon) — third species, wild species, and a rejected hypothesis

## Pigs (Soundwel, 5,031 calls after capping 700/context, 6 labs, valence Neg/Pos)

The honest unit is the **lab** (leave-one-team-out): valence is confounded with recording team.

| | held-out lab | random split | inflation |
|---|---|---|---|
| WavLM, mean over layers | **0.604** | 0.813 | **+0.21** |
| WavLM, best layer (L0) | 0.660 | | |
| **paper's own 18 acoustic features** | **0.386** (below chance) | 0.575 | |
| duration alone | 0.462 | | |

Which-lab decodability from the embedding: **0.824** (chance 0.167). Per-lab held-out valence:
IASPA 0.75, ETHZ 0.68, FBN 0.53, NMBU 0.51 — two of four labs at chance. And the published feature set
scores *below* chance when a lab is held out, meaning the feature→valence mapping partly *inverts*
between labs. That is the leakage story on the very features the paper used. (Caveat: my LOTO uses
the features as-is with a linear probe; the paper used other classifiers and did not claim cross-lab
generalisation. This is a reframing, not a refutation.)

## Bats (BEANS Egyptian fruit bats, 10 emitters × 200 calls, wild, 250 kHz)

10-way identity, chance 0.10:

| | native 16 kHz | 4× time-stretch (0–32 kHz → 0–8 kHz) |
|---|---|---|
| WavLM | 0.631 (mean 0.561) | **0.700** (mean 0.615) |
| AVES-bio | 0.651 (mean 0.629) | 0.681 (mean 0.670) |

Identity dominance is **not a domestic-animal quirk**. And a 4× time-stretch, which folds ultrasonic
content into the encoder's band, improves identity by +7 pts (WavLM).
**Correction (17 Aug):** this is the *time-expansion baseline* that ESP's own multiband paper (Sarkar
et al. 2026, arXiv 2604.27936, `earthspecies/multiband-audio`) is designed to beat — a known technique,
not a new result. Honest framing: we reproduced the time-expansion effect on a wild species with a
speech encoder. The natural next step is to run their heterodyne-and-fuse method on the same 2,000
calls and see if it beats 0.700. No context labels in the BEANS packaging (see iteration 2).

## The 3×3 transfer matrix (WavLM, mean over layers, group-level permutation)

| trained → tested | cat | dog | pig |
|---|---|---|---|
| **cat** | 0.744 within | 0.527 (p=.59) | 0.524 (p=.10) |
| **dog** | **0.583 (p=.015)** | 0.811 within | 0.495 (p=.56) |
| **pig** | **0.591 (p=.005)** | 0.523 (p=.25) | 0.60 within* |

\* pig within-species shown after within-lab z-scoring is 0.53; that strips valence signal from the two
single-valence labs, so the honest number is the plain held-out-lab 0.604 above.

**Two of six off-diagonal cells are significant, and both point at cats.** Nothing transfers *into*
dogs or pigs. Placebo pig SEX → cat AFFECT: 0.510 (chance). Cats — isolation-in-a-strange-room vs
brushed-at-home — are the most predictable target for an affect axis learned on another species.

## Hypothesis tested and REJECTED: "it's isolation calls"

Pig negatives are dominated by piglet isolation; cat negatives *are* isolation. If the shared axis were
separation-distress calls, removing pig isolation from the source should collapse pig→cat.

| pig source | pig→cat | p |
|---|---|---|
| all contexts | 0.591 | .020 |
| **without isolation** | **0.568** | **.030** |
| isolation vs positive only | 0.552 | .020 |

It barely moves. Not isolation-specific. Whatever transfers is more general than that.

## Construct validity — the excluded category lands in the middle

Score every cat context with the **pig-trained** probe (fraction called negative):
brushing 0.443 → **waiting-for-food 0.495** → isolation 0.621. The category we excluded *before*
seeing results, as ambiguous, is rated as exactly ambiguous. That is the check the reviewer asked for
(E4), and it passes.

## Where this leaves the two papers

**Paper A (leakage) is now three species strong:** cats (individual, +0.10–0.20), pigs (lab, +0.21,
published features below chance across labs), bats (identity 0.63–0.70 in a wild species). Add the
bandwidth aside. This is a real, multi-dataset methods paper.

**Paper B (transfer):** two significant off-diagonal cells (dog→cat, pig→cat), placebo-clean,
survives duration partialling (borderline), balanced classes, strict scaling, group-level permutation;
excluded category intermediate; isolation hypothesis rejected. Still: only *into* cats, and only with
one encoder family. Honest framing: *a cat affect axis is predictable from probes trained on two other
species; the reverse is not; the shared component is small, partly duration, and not isolation-specific.*

## eGeMAPS — the honest handcrafted baseline (added)

openSMILE eGeMAPSv02, 88 functionals (F0, jitter, shimmer, HNR, formants…), CatMeows, leave-one-cat-out:
**0.486** (MFCC-85 was 0.399; best encoder 0.571). Random split 0.641; cat identity 0.727.
→ Claim (a) shrinks from "+0.17 over handcrafted" to **"+0.07–0.09 over eGeMAPS"** and holds. Same
leakage pattern (inflation +0.155). This is the number to quote.

---

# ITERATION 4 (17 Aug, 13:30–) — encoder generality

## HuBERT replication of the 3×3 — pig→cat does NOT replicate

Same 5,031 pig calls, same cats and dogs, HuBERT-base instead of WavLM (mean over layers, group perm):

| cell | WavLM | HuBERT |
|---|---|---|
| dog→cat | 0.583 (p=.015) | **0.561 (p=.060)** — holds, borderline |
| pig→cat | 0.591 (p=.005) | **0.530 (p=.229)** — does not replicate |
| cat→pig | 0.524 (p=.10) | 0.523 (p=.14) |
| pig→dog | 0.523 (p=.25) | 0.493 (p=.55) |
| pig within, held-out lab (L0) | 0.660 | 0.628 |

So: **dog→cat is the only cell that holds across two encoders** (WavLM p=.015/.0045 animal-level; HuBERT
p=.06/.037 selection-free earlier). pig→cat is a WavLM-only result and must be reported as such.
The transfer story is now: *one robust cell, one encoder-dependent cell, everything else at chance.*

Running: AVES-bio (a **bioacoustic**, non-speech encoder) on the full 3×3 — the third encoder family.
If dog→cat holds there too, it is a property of frozen encoders generally; if not, it is a speech-model
property (which would itself be interesting — speech pretraining as the carrier).

## eGeMAPS added to Paper A (see above): honest handcrafted baseline 0.486; encoders +0.07–0.09 over it.

## AVES-bio (third encoder family — bioacoustic, not speech) — the 3×3

| cell | WavLM | HuBERT | AVES-bio |
|---|---|---|---|
| dog→cat | **0.583 (p=.015)** | 0.561 (p=.060) | 0.537 (p=.23) |
| pig→cat | **0.591 (p=.005)** | 0.530 (p=.23) | **0.608 (p=.040)** |
| cat→pig | 0.524 (p=.10) | 0.523 (p=.14) | **0.559 (p=.040)** |
| pig→dog | 0.523 (p=.25) | 0.493 (p=.55) | 0.565 (p=.085) |
| cat→dog | 0.527 (p=.59) | — | 0.456 (p=.78) |
| dog→pig | 0.495 (p=.56) | — | 0.469 (p=.82) |
| placebo (sex→affect) | 0.51 | — | 0.487 |
| pig within, held-out lab | 0.660 | 0.628 | **0.679** |

**Reading.** No cell is significant in all three encoders. **pig→cat is significant in two of three**
(WavLM, AVES); dog→cat in one plus a borderline. Every encoder finds at least one species pair that
transfers *into cats* above chance, but *which* pair differs. AVES — a bioacoustic model — is the only
one where anything transfers *out* of cats (cat→pig 0.559). Placebos stay at chance in all.

Honest Paper-B sentence, final form:
> *Across three frozen encoder families, an affect axis learned on one species predicts cat contexts
> above chance in every encoder, but the specific source species that transfers is encoder-dependent
> and no cell is unanimous; the effect is small, placebo-clean, roughly half explained by call duration,
> and should be read as "encoders share some coarse arousal structure across mammals" rather than as
> a stable cross-species code.*

Figure: `figures/encoders3.png`. Data: `results/out_pigs/aves_matrix.json`, `encoder_consistency.json`.

---

# ITERATION 5 — channel stress test (E5): which axis survives deployment?

Take the same 656 cat/dog clips, perturb the audio four ways, re-embed with WavLM, and ask probes
trained on **clean** audio how often their decisions flip. This is the thesis question in miniature:
deployment never matches the recording session.

Layer 9, pooled cats+dogs, fraction of decisions that flip:

| perturbation | context (binary) | identity, matched binary | identity, full 10–20-way | ratio (matched) |
|---|---|---|---|---|
| low-pass 4 kHz | **0.011** | 0.047 | 0.070 | **4.2×** |
| reverb 0.4 s | 0.116 | 0.221 | 0.329 | 1.9× |
| noise +10 dB SNR | 0.191 | 0.267 | 0.524 | 1.4× |
| gain −12 dB | 0.000 | — | 0.000 | *(not informative — we z-normalise every waveform before the encoder, so gain is removed by construction)* |

**The control mattered.** Raw identity is a 10–20-way task and context is binary, so identity has more
boundaries to cross. Matching difficulty (predict which random half of the animals a clip came from,
20 splits averaged) shrinks the effect from 2.7–6.1× to **1.4–4.2×** — but it does not remove it.

**Reading.** *Context is the channel-robust axis; identity is channel-entangled* — clearly under
bandwidth loss (context flips 1.1% when everything above 4 kHz is deleted), moderately under reverb,
weakly and inconsistently under noise (ratio ≈1.0 at layers 3/6/12). Physically sensible: individual
voice identity lives in fine spectral detail, affect in coarse envelope and duration.

**It also pushes back on the room worry.** If the cat "context" signal were purely room acoustics,
adding 0.4 s of reverb should scramble it. Context flips 11.6%; matched identity flips 22.1%. So the
context axis is not simply an acoustic-environment readout — though the background-only result stands
as a separate warning about *that particular dataset*.

Caveat: probes are fit in-sample on clean audio, so these are decision-**stability** measures, not
generalisation measures. That is the right design for counting flips; it is not an accuracy claim.

Figure `figures/stress.png`; data `results/out_stress/`.

---

# ITERATION 6 — what actually carries the transfer (and what doesn't)

Analysis on cached WavLM embeddings only (no audio): affect directions, 1-D projections, and sparse
dictionary features across cat / dog / pig. Full detail `results/out_sae/`.

## The naive "shared affect direction" hypothesis is FALSE

Take each species' affect direction w_s = normalize(mean_negative − mean_positive) at layer 9 and
compare them:

| pair | cosine | null mean±SD (200 perms) | p |
|---|---|---|---|
| cat–dog | +0.188 | −0.142±0.194 | 0.065 |
| cat–pig | +0.012 | −0.001±0.110 | 0.448 |
| dog–pig | +0.004 | −0.005±0.111 | 0.488 |

Nothing survives at layers 12 or 3 either (one p=0.045 out of 18 comparisons — that is what you expect
by chance). **The species' affect directions are essentially orthogonal.**

Crucially this is *not* a power problem for pigs: split-half reliability of w_s is cat 0.61, dog 0.41,
**pig 0.94**. The pig direction is measured extremely precisely and it still points nowhere near the
others. Cat–dog is attenuated by unreliability (correcting gives +0.374) but still sits inside the null.

## So why does transfer work at all? Whitening.

Transfer as a 1-D projection (AUC, layer 9): **dog→cat 0.790**, pig→cat 0.681, cat→dog 0.632,
pig→dog 0.590, dog→pig 0.534, cat→pig 0.516. Same asymmetry as before — everything transfers *into*
cats, nothing into pigs.

The reconciliation: the **mean-difference** axis transfers much worse (dog→cat 0.616) than the
**logistic probe** (0.790). The probe finds a direction in the covariance-whitened space, not the raw
class-mean difference. So what is shared across species is not a direction in embedding space — it is
a direction *relative to each species' own covariance structure*. That also explains the
encoder-dependence we saw: whitened directions depend on the covariance, which differs per encoder.

## Correction to the earlier "half of it is duration"

Partialling log-duration out of the projection barely moves it (dog→cat 0.790 → 0.777). Where the
transferred axis tracks a simple acoustic quantity, it is **energy**, not duration (pig→cat ρ=+0.544,
dog→cat +0.344, cat→dog +0.337).

This does not contradict the kill-test (duration alone transfers at 0.557 balanced accuracy) — both
are true. Duration is *independently* predictive across species, but the embedding's transfer does not
run *through* duration. Revised sentence: **duration is a parallel cross-species cue, not the mechanism.**

One genuine oddity: the **pig affect direction essentially IS duration** — cos(w_pig, duration
direction) = **+0.897**, and the pig affect score correlates with log-duration at ρ=+0.745. Yet pig
duration does not align with cat/dog duration (+0.008 / −0.275), which is why pigs transfer poorly.

## Sparse features: one shared feature in two species, none in three

MiniBatchDictionaryLearning, 192 atoms, 13 s on CPU, 68.6% variance explained.

- **0 of 192** atoms are affect-selective in all three species.
- **1 of 192** in ≥2 species with the same sign (null 0.00±0.00 over 200 perms, p=0.005).
- Per-species selective atoms: cat 9 (null 3.3±1.4), dog 10 (null 0.05±0.24), **pig 0**. The best
  single pig atom reaches AUC 0.580 across 5,031 calls — pig valence is genuinely distributed, not
  carried by any sparse feature.
- **Component 33** (cat 0.299, dog 0.334, pig 0.522): *not* duration (ρ = −0.166/+0.022/+0.090),
  survives the duration control and a full duration+energy+centroid control (0.392/0.408), and holds
  when computed *within* each animal (all 10 dogs on the same side).
- Robustness: FastICA(64) also finds 1-in-≥2 and 0-in-3 — but a *different, uncorrelated* feature
  (ρ=−0.001). A species-balanced dictionary finds 0 passing.

**Verdict:** there is no single sparse feature that means "distressed" across mammals. There is a
weak, cat–dog-only shared structure, and it is not explained by the obvious acoustics.

## What this does to Paper B

It makes it a better and more honest paper. The claim is no longer "animals share an affect direction."
It is: *frozen encoders support above-chance cross-species affect transfer into some species, but the
shared structure is covariance-relative rather than a common direction, is absent for pigs despite a
precisely-measured pig affect axis, and is not reducible to duration, energy or any single sparse
feature.* That is a cautionary methods result about probing, which is exactly the ICBINB genre.

---

# ITERATION 7 — the leakage audit, done properly (Paper A's core)

Four datasets × six feature sets × all layers, one protocol: held-out-group context accuracy vs
random-split accuracy vs group-identity decodability. Full detail `results/out_audit/`.

| dataset | features | held-out group | random | inflation | group identity | chance (task / id) |
|---|---|---|---|---|---|---|
| CatMeows | WavLM (L3) | 0.559 | 0.725 | +0.167 | 0.793 | .333 / .048 |
| CatMeows | HuBERT (L12) | 0.571 | 0.671 | +0.099 | 0.700 | .333 / .048 |
| CatMeows | wav2vec2 (L2) | 0.554 | 0.737 | +0.183 | 0.755 | .333 / .048 |
| CatMeows | AVES-bio (L5) | 0.540 | 0.736 | +0.196 | 0.793 | .333 / .048 |
| CatMeows | eGeMAPS-88 | 0.492 | 0.628 | +0.136 | 0.727 | .333 / .048 |
| CatMeows | MFCC-85 | 0.399 | 0.602 | +0.203 | 0.768 | .333 / .048 |
| Dog barks | WavLM (L6) | **0.728** | 0.878 | +0.150 | 0.817 | .333 / .100 |
| Dog barks | eGeMAPS-88 | 0.619 | 0.733 | +0.114 | 0.792 | .333 / .100 |
| Dog barks | MFCC-85 | 0.551 | 0.793 | +0.242 | 0.892 | .333 / .100 |
| Dog barks | log-duration only | 0.341 | 0.360 | +0.018 | 0.222 | .333 / .100 |
| Soundwel pigs | WavLM (L0) | 0.660 | 0.883 | +0.223 | **0.938** | .500 / .167 |
| Soundwel pigs | eGeMAPS-88 | **0.669** | 0.823 | +0.153 | 0.865 | .500 / .167 |
| Soundwel pigs | paper's 18 features | 0.386 | 0.575 | +0.189 | 0.513 | .500 / .167 |
| BEANS bats (excluded) | WavLM (L11) | 0.301 | 0.303 | +0.003 | 0.394 | .250 / .100 |

## The inflation law — real *within* a corpus, NOT across corpora

- Pooled (n=83): Pearson **r=0.611, p=8.7e-10**; Spearman 0.610
- Within-dataset partial: **r=0.646, df=79, p=7.1e-11**
- Layer-only variation inside a single encoder (n=77): r=0.636, p=2.0e-9; **all six encoder clusters
  positive** (sign test p=0.031)
- Neural-only, 768-d (n=77): r=0.694 — not a dimensionality artifact
- Leverage-safe: dropping the extreme point *raises* r to 0.670; jackknife keeps r ∈ [0.480, 0.689]
- Identity tracks the **leaky** number (r=0.754) more than the honest one (r=0.509) — exactly what the
  leakage account predicts
- ⚠️ **Cluster level (12 independent dataset×feature-set units): r=0.221, p=0.489 — NULL**

**Honest claim:** *among feature sets and layers for a given dataset, the one that leaks less identity
inflates less.* **Not:** "datasets with more identity leakage inflate more." Earlier I reported the
cats-only n=5 version (r=0.81, p=0.094); with n=6 it is r=0.866, p=0.026 — but the cross-corpus
generalisation is not established and must not be claimed.

## Three findings that change the story

1. **eGeMAPS BEATS WavLM on pigs under held-out lab** (0.669 vs 0.660) while leaking less identity
   (0.865 vs 0.938). The encoder's apparent advantage on pigs lives *entirely in the leaky number.*
   That is the sharpest single sentence Paper A has.
2. **Dogs generalise far better than cats or pigs**: held-out-dog context 0.728 (chance 0.333), with a
   duration-only floor at 0.341 — so it is not a bout-length artifact. Caveat: only 10 dogs, and dog
   context is confounded with recording session, so this is not a leave-one-session-out number.
3. **Recovery curves differ in kind.** ~11 labelled clips from the target cat lift WavLM 0.576 → 0.727,
   which is the random-split ceiling (0.729) — cat "shift" is individual idiosyncrasy a handful of
   labels fully absorbs. For pigs, 100 labelled clips from the target lab close only **50%** of the gap
   and are still climbing. eGeMAPS recovers far less in both: the encoder is more *adaptable*, not more
   *transferable*.

## Honesty notes
- **Bats excluded from the fit**: no bat context label is decodable at all (0.301 vs 0.250 chance), so
  their near-zero inflation is a degenerate zero. Including them would pump pooled r to 0.874 purely as
  leverage; within-bats the correlation is *negative*.
- Pig WavLM L0 was recomputed from scratch as a reproduction check and matched to within 0.002.

---

# ITERATION 8 — ESP's multiband package vs our time-expansion baseline (bats)

Ran ESP's own `multiband-audio` v0.1.0 (Sarkar et al. 2026, arXiv 2604.27936) on the 10-way bat
identity task, 1,000 calls (100 × 10 emitters), chance 0.100, best layer:

| encoder | baseband | time-expansion 4× | multiband mean | multiband concat |
|---|---|---|---|---|
| WavLM-base-plus | 0.587 (L2) | **0.666** (L1) | 0.604 (L2) | 0.637 (L4) |
| AVES-bio | 0.608 (L9) | **0.641** (L4) | 0.561 (L2) | 0.616 (L8) |

Paired tests (same folds): time-expansion − multiband-concat = +0.029 (p=0.091) and +0.025 (p=0.13);
time-expansion − multiband-mean = +0.062 (p=3e-4) and +0.080 (p=5e-6).

**Answer: multiband did not beat time-expansion here — but this does NOT refute ESP's method.**
We tested only the *parameter-free* fusions (mean, concat). The paper's headline is **adaptive** fusion
(`gp` gated-pool is their default, plus MoE / hybrid / self-attention), which is learned end-to-end.
A learned gate could down-weight the weak bands and plausibly close a 0.03 gap. `MultibandWrapper`
requires a backbone returning one (N,D) vector plus a trained head, which is incompatible with our
frozen-encoder + per-layer logistic probe, so we used the package's band waveforms and fused ourselves.

## Two verifiable observations about the v0.1.0 package (worth reporting upstream)

1. **The shipped heterodyne is a single-phase mixer, so each non-baseband band folds 2:1.** Verified
   with tones: 9 kHz and 15 kHz both land at 3 kHz in band 1; a sine exactly at the 12 kHz band centre
   is nulled (rms 0.0006) while a cosine passes (0.500). There is no quadrature path.
2. **Bands overlap substantially.** The band-pass is a single 2nd-order biquad, not a brick wall — a
   15 kHz tone shows rms 0.287 in band 1 *and* 0.176 in band 2 (≈ −4 dB leak). The README's
   "non-overlapping bands" is aspirational at v0.1.0.

Neither is a criticism of the paper's idea; both are concrete, checkable notes about the released code.

## Two findings worth keeping

- **Energy does not predict usefulness.** 76.2% of bat call energy sits in band 1 (8–16 kHz), yet
  band 0 alone (0–8 kHz) out-probes band 1 alone by ~9 points (0.600 vs 0.516 WavLM).
- **Mean-fusion is actively harmful** for AVES: −0.049 against its own baseband band (p=0.004).
  Averaging dilutes the one informative band.

## Methodological hygiene from this run
- **Built-in noise ruler:** `baseband` and `band0_only` are the same 0–8 kHz audio down two different
  resamplers; they differ by 0.013 / 0.002 (n.s.). So ~1–3 points is the noise floor on this task.
- n=1,000 (half of the `out_bats` run), which is why everything sits ~4 points below it
  (0.587 vs 0.631; 0.666 vs 0.700) — the *ordering* replicates, which is the part that matters.
- Best-layer selection is not cross-validated and `C=0.5` was not swept across 768-d vs 3072-d.

## Why this matters beyond the number
This is the strongest cold-email content we have for ESP: a reproduction of their baseline comparison
on a dataset they did not use, plus two tone-verified observations about their released code — and
Gagan Narula, our #1 contact, is a co-author on that paper.

---

# ITERATION 9 — chapters 1 and 4 meet: conformal coverage is per-group broken

Our conformal code had only ever been validated on synthetic data. The audit said per-animal accuracy
varies wildly (0.00–0.86). Prediction: **marginal coverage will look perfect while per-group coverage
is catastrophic.** Run on real out-of-fold scores, α=0.10 (nominal 0.90), WavLM L9, calibration/test
split by group, 40 repeats.

| dataset | marginal | per-group **min** | IQR | groups <0.80 | mean set size |
|---|---|---|---|---|---|
| cats (20 individuals) | 0.904 | **0.412** | 0.186 | **5/20** | 1.32 |
| dogs (10 individuals) | 0.892 | 0.624 | 0.059 | 2/10 | 1.10 |
| pigs (6 labs) | 0.891 | 0.735 | 0.079 | 1/6 | 1.67 |

Worst offenders: `cat_TIG01` 0.412, `dog_Freid` 0.624, pig lab `IASPB` 0.735. **The headline number is
exactly on target and one cat in five is getting a guarantee that is 50 points short.**

**It is not a bug in our code.** Synthetic exchangeable control: 0.9048 over 2,000 trials. Real scores
under an *exchangeable within-group* split: cats 0.907, dogs 0.903, pigs 0.901. And the per-group
spread is just as wide under the exchangeable split — so this is the **conditional-coverage gap**, a
known theoretical limitation of conformal prediction, demonstrated on real bioacoustic data.

## Does the textbook fix work? Two important negatives and one positive

- **Mondrian by predicted class — no.** Worst cat 0.412 → 0.425; worst dog unchanged at 0.624; for pigs
  marginal coverage drops *below* nominal (0.873) and the IQR widens.
- **Mondrian by group, in deployment — structurally vacuous.** Under a group-disjoint split, no test
  group has calibration data, so every group falls back to the pooled threshold. Verified
  programmatically: it is bit-for-bit identical to pooled (0/20, 0/10, 0/6 groups get their own
  threshold). *The obvious fix does not exist precisely when you need it.*
- **Mondrian by group, given some of that group's own labels — complete repair, where data allows.**
  Dogs worst 0.696 → **0.905**, IQR 0.068 → 0.039, 0/10 below 0.80. Pigs worst 0.775 → **0.898**,
  IQR 0.065 → **0.002**. But **cats: no repair at all** (0.467 → 0.467, still 6/20 below 0.80) —
  at α=0.10 you need **≥9 calibration clips from that individual**, and only 6 of 20 cats have enough.

**Practitioner rule that falls out:** *α=0.10 requires ≥9 labelled clips from the animal or site you
are deploying on. Below that, no per-group guarantee is available at any price.* That dovetails exactly
with the audit's recovery curve (~11 target-cat clips saturate accuracy).

The repair is also **paid for in set size, unevenly**: `dog_Luke` 1.91 (abstains on almost everything)
vs `dog_Rudy` 0.91 (emits empty sets). Coverage repaired, usefulness redistributed.

## Shift case: species change breaks even the marginal guarantee

Calibrate on dogs, test on cats: marginal coverage **0.747** at L9 and **0.556** at L6, against a 0.920
same-species reference; worst cat 0.330, 12/20 below 0.80. Cats→dogs 0.746.
**Swapping individuals within a species costs nothing marginally; swapping species costs 15–35 points.**
That is the shift ladder the thesis frame predicts, measured.

## Caveats (in the agent's own README)
- The extreme cat minima come partly from tiny groups. Restricted to the 15 cats with n≥10, the worst
  is 0.748, IQR 0.113, 3/15 below 0.80 — the effect survives, less dramatically.
- Per-group coverage is partly confounded with per-group class prior (pig labs IASPB/IASPC are
  single-valence by construction, as are six cats).
- OOF scores are cross-conformal-flavoured rather than a single frozen scorer.
- Pig sets are barely informative anyway (1.67 mean size at 0.63 probe accuracy).

## What this does to Paper A
It supplies the "so what". The paper no longer just says *your accuracy is inflated*; it says
**your uncertainty guarantee is also per-group broken, the standard remedy is unavailable in
deployment, and here is the labelling threshold at which it becomes available.** That is a concrete,
actionable failure mode — exactly the ICBINB-BIO genre.

---

# ITERATION 12 — Phase 1 of the encoder plan: identity is removable, site identity is not

Two cheap experiments on cached embeddings (numpy/sklearn, no GPU, no retraining), testing the core
premise of `paper/ENCODER_PLAN.md`: *the information is there, it is just dominated by nuisance.*

## 1a. Fusion — eGeMAPS + WavLM beats either, slightly

Pigs (held-out lab): WavLM 0.654, eGeMAPS 0.641, **fused 0.668 (+0.014 over the better one)**.
So the two feature families do make partly different errors, but the gain is small — consistent with
them encoding largely the same thing. Cats/dogs skipped: the audit's eGeMAPS arrays cover all 440
meows while the cross-species subset is the 348 brushing/isolation clips. *(To fix: re-index rather
than recompute.)*

## 1b. Identity-subspace removal (INLP) — the headline

Estimate identity directions with iterative nullspace projection on **training groups only**, project
them out, re-run the honest context probe. Report both numbers plus a random-direction control of
equal rank.

| | identity before | identity after (rank 32) | Δ identity | context before | context after | Δ context |
|---|---|---|---|---|---|---|
| **cats** (20 individuals) | 0.721 | **0.256** | **−0.465** | 0.782 | 0.780 | −0.002 |
| **dogs** (10 individuals) | 0.753 | **0.208** | **−0.545** | 0.846 | 0.849 | +0.003 |
| **pigs** (6 **labs**) | 0.941 | 0.828 | −0.113 | 0.654 | 0.662 | +0.008 |

At the context-optimal rank: cats rank 8 → identity −0.244 with context **+0.013**; dogs rank 2 →
identity −0.114 with context +0.010.

**The random-direction control is the important one.** Projecting out the same number of *random*
directions leaves identity untouched (cats 0.721 → 0.724; dogs 0.753 → 0.747; pigs 0.941 → 0.940).
So INLP is removing something specific, not merely shrinking the representation. This is the control
that separates invariance from compression, and it passes.

## The three-way contrast IS the finding

- **Individual identity is a low-rank, removable subspace.** Eight directions out of 768 carry a
  quarter of cat identity; thirty-two carry more than half of dog identity — and context is
  completely unharmed, sometimes marginally better.
- **Recording-site identity is not.** Thirty-two directions remove only 11 points of pig *lab*
  identity, which starts at 0.941 and stays at 0.828. Lab identity is distributed across the
  representation, not concentrated in a subspace.

That distinction is mechanistically sensible — individual voice is a property of one animal's vocal
apparatus, while a lab is channel plus room plus population plus equipment, which touches everything.

**And it independently reproduces the recovery-curve result.** In the audit, ~11 labelled clips from
a target cat closed the individual gap completely, while 100 clips from a target pig lab closed only
half. Two unrelated methods — labelling curves and subspace geometry — arrive at the same structural
claim: *individual shift is simple and cheap to fix; site shift is deep and neither labels nor linear
projection dispatch it.*

## What this licenses, and what it does not

It licenses a practical tool: an identity-invariant projection for individual-level nuisance, free,
post-hoc, on any frozen encoder, with the honest evaluation protocol attached. It does **not** license
a claim about site invariance — linear projection fails there, and the next thing to try is a
nonlinear or adversarial method, which needs training and therefore a GPU.

Figure `figures/invariance.png`; data `results/out_invariance/results.json`; code
`experiments/encoder/phase1_invariance.py`.

---

# ITERATION 13 — site invariance: two methods fail, and the failure is well characterised

Following iteration 12 (individual identity is a removable low-rank subspace; pig LAB identity is not),
two follow-ups asked whether we simply stopped too early or used too weak a method.

## Deeper linear projection — it plateaus, definitively

INLP on pig lab identity, all the way to rank 384 of 768 dimensions:

| rank | valence (held-out lab) | lab identity | lab identity, random dirs |
|---|---|---|---|
| 0 | 0.654 | 0.941 | 0.941 |
| 32 | 0.662 | 0.828 | 0.940 |
| 64 | 0.665 | **0.819** | 0.941 |
| 128 | 0.665 | 0.820 | 0.940 |
| 256 | 0.665 | 0.820 | 0.935 |
| 384 | 0.665 | 0.819 | 0.937 |

**Flat from rank 64 onward.** Deleting half the dimensions removes nothing further. This is not
"we didn't look hard enough" — linear projection provably saturates at 0.82, still far above the
0.167 chance level. Valence is untouched throughout (+0.011), so nothing is being destroyed either.

## Domain-adversarial training — actively counterproductive

A gradient-reversal MLP (768→256→128, context head + lab head behind a GRL), λ swept 0→3:

| λ | pigs ctx / lab | cats ctx / id | dogs ctx / id |
|---|---|---|---|
| 0 (plain MLP, no adversary) | 0.617 / 0.878 | 0.740 / 0.581 | 0.834 / 0.562 |
| 0.3 | 0.655 / 0.904 | 0.766 / **0.680** | 0.827 / 0.620 |
| 1.0 | 0.648 / 0.905 | 0.737 / 0.648 | 0.856 / **0.662** |
| 3.0 | 0.662 / 0.894 | 0.742 / 0.608 | 0.840 / 0.630 |

**Turning the adversary up made group identity MORE decodable, not less.** Whatever removal the MLP
achieves comes from its 768→128 bottleneck, not from the adversarial objective.

This is the textbook DANN failure mode and worth stating precisely: the adversary only defeats *its
own* discriminator head. We measure afterwards with a **fresh** logistic probe, which finds the
information sitting where it always was. Beating the adversary during training is not evidence of
invariance — a point the invariance literature makes and that this reproduces cleanly.

## Head-to-head

| | raw ctx / id | linear INLP | adversarial DANN |
|---|---|---|---|
| cats | 0.782 / 0.721 | **0.780 / 0.256** | 0.740 / 0.581 |
| dogs | 0.846 / 0.753 | **0.849 / 0.208** | 0.834 / 0.562 |
| pigs | 0.654 / 0.941 | 0.665 / **0.819** | 0.617 / 0.878 |

**The 30-line linear projection beats the trained neural network on every dataset** — 2–3× more
identity removed, and it preserves context better (the MLP also loses accuracy).

That is now the fourth time on this project the simpler method wins under honest evaluation:
eGeMAPS matched WavLM; time-expansion beat multiband fusion; hand-crafted features beat the encoder
on pigs under held-out lab; and now linear erasure beats adversarial training.

## Status of the site-invariance problem
Individual nuisance: **solved**, free, post-hoc. Site nuisance: **open**, and now characterised —
not low-rank, not reachable by gradient reversal. Remaining candidates (LEACE with its formal
guarantee, WCCN from speaker verification, CORAL, nonlinear probing to test whether the information
is even linearly held) are under test.

Code `experiments/encoder/{phase1_invariance,phase1b_adversarial,deep_rank}.py`;
data `results/out_invariance/`, `results/out_adversarial/`.

---

# ITERATION 14 — allometric pitch normalisation: tested, and it does not work

The idea in `paper/ENCODER_PLAN.md` §2, described there as "the idea I think is genuinely ours" and
"the first thing I would test": body mass predicts fundamental frequency across mammals, so raw pitch
is not comparable between a 3 kg cat and a 300 kg pig — but *pitch relative to what the body predicts*
might be. An animal calling higher than its size predicts is straining, in any species.

**It does not work. Not over raw pitch, not over within-species z-scoring, and not over having no
pitch feature at all.**

## The transfer test (mean balanced accuracy, 6 directed species pairs)

| no pitch at all | raw F0 | allometric residual | allometric, law fitted excluding test species | within-species z-scored F0 |
|---|---|---|---|---|
| **0.577** | 0.572 | **0.578** | 0.580 | 0.569 |

Everything sits within 0.022. Allometric beats raw by +0.009 (sign test p=0.22), beats within-species
z by +0.011 (p=0.22), and beats **dropping pitch entirely** by +0.003 (p=0.69). Only 1 of 6 pairs
clears its permutation null — and the no-pitch baseline clears that one too. Raw F0 clears none.

## Why it fails, and why I should have seen it

Probe weights on the pitch feature (positive = predicts negative affect): **cat −0.20, dog −1.24,
pig +0.21.** The pigs flip sign.

**No location or scale correction can repair a sign flip.** Allometric normalisation shifts and
rescales the pitch axis; it cannot reverse it. I proposed this after noticing that spectral cues point
in opposite directions across species — and then proposed a fix that is mathematically incapable of
addressing opposite directions. That was a conceptual error, not a data problem.

Only **duration** agrees across all three species (+0.27 / +1.43 / +1.25), and duration+energy alone
*is* the 0.577 baseline that every fancier variant merely matches.

## The genuine positive, orthogonal to the hypothesis

Within the 10 dogs — the only animals with **real measured body masses** rather than imputed ones —
log F0 scales with mass at **b = −0.334, R² = 0.48, p = 0.026**. That is an independent replication of
the isometric / carnivore scaling exponent (theory −0.333; Bowling et al. carnivores −0.335) on a
corpus collected in 2008 for an entirely different purpose. Small, but real.

The cross-species fit is much weaker: b = −0.389, R² = 0.205, **p = 0.068, n.s.**, and dropping one
species swings it from −0.089 to −0.907. Three species cannot establish an allometric law.

## Methodological findings worth keeping

- **For cats and pigs, "allometry" was species-mean centering in disguise** (r = 0.956 and 0.963),
  because imputed mass takes only 4 and 3 distinct values. The feature was not doing what its name said.
- **Permutation nulls are not at 0.5** — they run 0.46–0.67 by pair, because several cats and 2 of 6
  pig labs are single-class. Reporting these against naive chance would have overstated everything.
- **F0 extraction failure is not missing-at-random**: 100% success on negative cat clips vs 88% on
  positive; 49.6% vs 40.2% in pigs. Discarded honestly rather than imputed.
- **Our pitch tracker flattered the idea.** Validated against Soundwel's own published F0: log-log
  r = 0.78, with 44% of clips off by more than an octave. Re-running with the corpus's expert F0
  values puts *every* pitch variant **below** the no-pitch baseline.
- **Formant dispersion is not measurable on this data** — LPC returns Df ≈ 4000/model-order for all
  three species and ranks pigs backwards. Cat formant spacing (~2460 Hz) cannot be resolved from a
  600 Hz harmonic comb inside a 4 kHz band. A physical limit, not a bug.
- **Segmentation matters**: dog files are 12 s bouts, pig calls 0.3 s. Analysing the focal call rather
  than the file (validated r = 0.976 against published durations) roughly doubles duration's
  cross-species affect separation.

## Where this leaves the encoder plan

Layer 4 (metadata) loses its flagship idea. What survives is the finding underneath it: **the only
cross-species affect cue we can find that has a consistent sign is call duration**, and adding pitch
in any form adds nothing. That is a cleaner and more surprising sentence than the one I set out to
write, and it strengthens rather than weakens the transfer chapter — it explains *why* the transfer is
small and covariance-relative rather than a shared direction.

Fifth instance of the pattern: the no-pitch baseline beats every pitch-based elaboration.

Data `results/out_allometry/`; code `experiments/allometry/`.

---

# ITERATION 15 — CORRECTION: site identity *is* low-rank. I was wrong.

Iterations 12–13 claimed **"individual identity is a low-rank removable subspace; recording-site
identity is not."** The second half is **false**, and it was published here, in two commit messages,
and in both repositories. Correcting it in full.

## What is actually true

**LEACE** (Belrose et al., least-squares concept erasure) drives 6-way lab decodability from **0.941
to 0.291 — the majority-class floor — at rank 5**, in closed form. Five is exactly *#labs − 1*. Site
identity is not merely low-rank; it sits in the smallest subspace it possibly could.

INLP stalled at 0.82 for a conceptual reason, not a capacity one. **INLP removes the max-likelihood
*discriminative* direction; the condition for a linear probe to fail is *equal class means*, and only
LEACE targets that.** The agent also checked the obvious bug hypothesis — our `inlp_directions`
projects a z-space coefficient out of un-scaled space, so the correct direction is `w/σ` — and fixing
it changes nothing (0.934 vs 0.925 at rank 5). The method was wrong, not the implementation.

| method | rank | valence (held-out lab) | lab, linear probe | lab, MLP probe |
|---|---|---|---|---|
| raw | – | 0.654 | 0.941 | 0.949 |
| INLP | 32 | 0.665 | 0.903 | – |
| **LEACE** | **5** | 0.657 | **0.291** (floor) | **0.918** |
| CORAL | – | 0.651 | 0.291 | **0.694** |
| per-lab z-score *(transductive)* | – | **0.506** | 0.291 | 0.970 |

## The finding underneath the correction

**The leak is second-order.** After LEACE drives *linear* lab decodability to the floor, an **MLP
still reads the lab at 0.918**. Lab covariances differ from pooled by 59–115% in Frobenius norm.
INLP, LEACE, NAP and WCCN are all first-order methods and are blind to this *by construction*.

That also **explains the DANN failure from iteration 13 without appealing to optimisation pathology**
— the information genuinely remains, in second-order structure no first-order surgery touches.

The one second-order method that does reach it, CORAL, takes the task with it: MLP lab 0.949 → 0.694,
but valence collapses to 0.516. Compression, not invariance. Erasing in a 1024-dim random-Fourier
space reproduces the same pattern one level up.

**No method improved valence** under a paired cluster bootstrap. The only significant effect in the
whole study is that per-lab z-scoring is significantly *harmful* (−0.123, CI [−0.199, −0.056]).

## A caveat that lands on Paper A

**On Soundwel, the lab name alone predicts valence at 0.766 — higher than the 0.654 the embedding
earns honestly.** IASPB is 100% positive and IASPC 100% negative by construction. And
leave-one-*context*-out scores 0.569, *worse* than leave-one-lab-out's 0.654, with zero contexts
shared across all six labs.

So **leave-one-lab-out on this corpus conflates site shift with label-distribution shift and context
novelty.** Our pig numbers stand as measured, but the interpretation must be hedged: they are not a
clean site-shift measurement, and the eGeMAPS-vs-WavLM comparison inherits that. The evaluation is
also underpowered — only 4 of 6 labs are scorable, baseline 95% CI [0.526, 0.725].

## Two protocol traps, one of which infected our earlier numbers

1. **Fitting the eraser on all data and then probing out-of-fold gives LEACE 0.10 — *below* the 0.167
   chance level.** That is a fold-anticorrelation artifact, and it is the protocol our earlier INLP
   numbers used. Erasers must be re-fit inside each fold.
2. NAP-2's apparent +0.030 win is an artifact of pooling the two single-class labs; on the 4 scorable
   labs it is **−0.015**.

Also recorded: projecting *onto* the 5-d site subspace scored 0.718 and looked exciting, but a random
5-d subspace reaches 0.714. Not a finding.

## Revised position

Individual nuisance: removable, cheaply, first-order. Site nuisance: **linearly removable at rank
#labs−1, but the information survives in second-order structure**, and every method that reaches it
destroys the task. The honest conclusion is that this is a **study-design and encoder-training
problem** — record shared contexts across sites, or use a second-order-aware training objective — not
something representation surgery fixes after the fact.

Data `results/out_site/`; code `experiments/site/`.

---

# ITERATION 16 — we built the tool, and it immediately corrected the paper

*18 Aug 2026. Two things happened: `leakcheck` exists, and the first real dataset we pointed it at
turned up a defect in a number that is currently in Paper A's abstract.*

## The tool

`leakcheck/` — pip-installable, two dependencies (numpy, scikit-learn), no GPU, no model. It takes
`(X, y, groups)` and reports the three numbers this project has been reporting by hand since
iteration 7:

```
leakcheck audit embeddings.npy --meta clips.csv --label context --group animal_id
```

| | |
|---|---|
| honest | task accuracy, whole groups held out |
| shuffled | task accuracy, random split |
| identity | how well the same features recover the group |

Plus, because we kept getting caught by them ourselves: a permutation null **at the automatically
chosen unit** (within-group when the label varies inside a group, group-level when it does not); a
bootstrap interval over **groups, not clips**; per-group scores; and design diagnostics that run
before any model is fitted.

`leakcheck erase` runs LEACE / INLP / a random-projection control with the eraser **re-fitted inside
every fold** — the protocol trap from iteration 15 is now impossible to fall into by accident — and
reports two numbers a linear probe alone would hide:

- identity on **groups the eraser never saw** (does the transform transfer, or did it memorise?)
- identity under an **RBF probe** (is the leak second-order, i.e. iteration 15's finding?)

Run against our own pig data it independently reproduces iteration 15: LEACE at **rank 5**, lab
identity 0.541 → 0.242 in-sample, task 0.436 → 0.456. And it adds a result we had not measured:
**on labs the eraser never saw, identity is 0.773.** The transform memorises the training labs'
means; it does not remove a lab axis. Combined with the RBF probe recovering 0.661, the verdict line
reads *"LINEAR AND IN-SAMPLE ONLY — do not ship this as invariance,"* which is the correct thing for
a tool to say about our own best method.

21 tests, all passing, including one that fails if imputation ever stops being fold-local.

## What it found in ninety seconds: the pig feature table has a hole

The design diagnostics flagged **700 rows with no features at all** — every one of the 18 published
measurements absent. They are:

- all from **NMBU** (51% of that lab's 1,381 clips)
- all labelled **negative valence**, with no positives among them
- after mean imputation, **one identical constant vector**, 700 times

The paper says the corpus's own feature set "scores below chance (0.386) once a lab is held out."
That number reproduces exactly. But it is produced by imputing 700 clips the published feature table
simply does not cover, and all of them carry one label.

Recomputed with `experiments/audit/audit.py` itself, same estimator, same protocol:

| | held-out lab | random | inflation | identity | n |
|---|---|---|---|---|---|
| as published | **0.386** | 0.575 | +0.189 | 0.513 | 5,031 |
| 700 empty rows dropped | **0.537** | 0.745 | **+0.208** | 0.650 | 4,331 |

**The below-chance claim does not survive.** 0.537 is at chance, not below it, so "the mapping
inverts between labs" is not supported — the supportable claim is that this feature set retains *no*
cross-lab valence signal.

**The leakage claim gets stronger.** Inflation rises from +0.189 to +0.208, and the random-split
score rises from 0.575 to 0.745. Removing the degenerate block makes the paper's actual thesis
cleaner, not weaker.

Paper A updated: abstract, contribution (3), the §4.2 paragraph, and a second table row marked
`†`. Provenance in `results/out_audit/pig_missing_features.json`.

## Why this is the third-most-useful thing that happened this week

The check that caught it is four lines of numpy and runs before any model is fitted. We have been
staring at this dataset since 15 August, across fifteen iterations, and did not look. That is the
argument for the tool existing: not that the analysis is clever, but that the cheap checks only get
run when something runs them for you.

It also generalises. `leakcheck` now warns when **missingness itself predicts the label** — Cramér's
V of 0.21 on this corpus. If a model can tell which extractions failed, and failure correlates with
the label, it can score without hearing anything.

## Related: what ESP shipped in May

Earth Species published *"From What to Who?"* (5 May 2026): BirdAVES adapted into a 173-way
individual classifier for zebra finches, producing "who-sang-when" timelines. The post does not state
the split protocol. This is the same quantity we measure as `identity` — for them a capability, for
us the contaminant. Both readings are correct and they are the same number, which is a better framing
for the outreach email than anything we had.

Code `leakcheck/`; tests `leakcheck/tests/`; report cards write to HTML, Markdown or JSON.
