# Audio Jungle Book Project

**Can a machine tell what an animal was feeling from the sound it made?**

Not what it *said* — nobody can do that, and anyone claiming otherwise is selling something. But a
narrower, answerable question: given a recording, can we tell whether the animal was distressed or
content, alone or with company, playing or fighting?

This project started with a simpler question — *can two dogs understand each other, and could a dog
and a cat?* — and turned into something more useful: an audit of whether the AI models the field
already relies on are measuring what everyone thinks they are.

They mostly aren't. That's the finding.

---

## The story in one minute

Conservation biologists now leave microphones in forests for months and run the audio through large
pretrained AI models. Those models decide which species were present, how many, and increasingly what
the animals were doing. Real decisions follow: which habitat gets protected, which population is
declining, whether a farm's animals are in distress.

Everyone evaluates these models the same way: shuffle the recordings, train on some, test on the rest.

That test is broken, and this repository measures how badly.

Animal recordings come from a handful of individuals in a handful of places. Shuffle them and the same
cat, recorded minutes apart in the same room, lands on both sides of the split. So the model can score
well by recognising **the cat** rather than understanding **the situation** — and the metric can't
tell the difference.

We measured it across four species. Individual identity is decodable at **63–94%** (chance: 5–17%),
while the behavioural context everyone actually cares about sits at 55–73%. Shuffled evaluation
inflates reported accuracy by **10 to 24 points**.

**The models are answering "who is this?" and being graded as though they answered "what is happening?"**

---

## What we found

| | |
|---|---|
| **Identity dominates** | Across cats, dogs, pigs and wild bats, frozen encoders encode *who* and *where* far more strongly than *what situation*. |
| **Shuffled splits inflate by 10–24 points** | Present in every dataset and every feature set, including hand-crafted ones. This is an evaluation problem, not a neural-network problem. |
| **A 2015 baseline matches a modern encoder** | On pig emotional valence, 88 classic acoustic descriptors match a 95M-parameter self-supervised model once you hold out a recording lab — while leaking less identity. The encoder's lead exists only in the broken metric. |
| **Published features can invert** | One corpus's own feature set scores *below chance* when a new lab is held out. |
| **The uncertainty guarantees break too** | Conformal prediction reports a perfect 90% coverage on average while the worst individual gets 41%. |
| **"Who" is erasable, "where" is not** | Deleting 32 directions from the embedding removes half of individual identity with no cost to context — but barely dents recording-site identity. |

Full detail, including everything that *didn't* work: **[FINDINGS.md](FINDINGS.md)**.

---

## Why it matters

**For conservation.** A model that reads the recording site rather than the animal will fail the
moment you move the microphone — which is the entire point of passive acoustic monitoring. Our
recovery curves show individual differences are cheap to fix (about 11 labelled clips) while site
differences are not (100 clips close only half the gap).

**For animal welfare.** Automated distress detection is being deployed on farms now. If the model has
learned "this barn" instead of "this pig is in pain," it will be confidently wrong somewhere new.

**For the field.** The fix costs one extra line in your evaluation: report how well your features
identify the *animal*, next to how well they identify the *behaviour*. If the first number is high,
the second one is borrowing from it.

That line is now a tool. **[`leakcheck`](leakcheck/)** takes your features, your labels and one
column saying which animal or site each clip came from, and reports all three numbers plus the
things people forget to check:

```bash
pip install "git+https://github.com/slug-labz/Audio-Jungle-Book-Project.git#subdirectory=leakcheck"
leakcheck demo
leakcheck audit embeddings.npy --meta clips.csv --label context --group animal_id
```

Two dependencies, no GPU, no model downloads. `--fail-over 0.05` exits non-zero, so it can sit in CI
and fail a pull request that reintroduces a shuffled split. The first real dataset we pointed it at
was our own, and it found a defect in a number that was in our paper's abstract — see
[FINDINGS.md](FINDINGS.md) iteration 16.

---

## How we did it

One protocol, applied everywhere. For every dataset and feature set we report three numbers:

- **honest** — accuracy with an entire animal or recording site held out
- **shuffled** — accuracy with a random split (what the field usually reports)
- **identity** — how well the same features identify the animal or site

The third explains the gap between the first two.

Four public corpora: cat meows (21 individuals), dog barks (10), pig calls (6 recording labs), wild
Egyptian fruit bats (10 emitters). Six feature sets: WavLM, HuBERT, wav2vec2, AVES-bio, eGeMAPS, MFCC —
all frozen, none retrained. Probes are plain logistic regression, because that measures what is
actually *in* the representation.

**Everything here ran on a laptop with no GPU. Total compute cost: $0.**

---

## What's in here

```
FINDINGS.md              every result, every control, and what got corrected
leakcheck/               the tool: pip-installable leakage audit for any (features, labels, groups)
paper/                   the write-up in progress + LaTeX
experiments/
  context_probe/         the audit: encoders, cross-species, controls, kill-tests
  audit/                 the four-corpus leakage study
  conformal_eval/        per-group coverage
  encoder/               identity-subspace removal
  mechanism/             what actually carries the cross-species signal
  multiband/             reproducing a published ultrasonic method on bats
conformal_pam/           uncertainty tooling for detection pipelines
results/                 JSON for every number quoted anywhere
figures/                 19 plots
artifacts/               three self-contained HTML pages
data/README.md           how to fetch each dataset (nothing large is committed)
```

## Running it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# fetch datasets per data/README.md, then from experiments/context_probe/:
python encoders.py          # five frozen encoders, leave-one-animal-out
python validate_transfer.py # permutation nulls and controls
python killtest.py          # the adversarial checks
```

Every script explains at the top what it tests and why. Splits are **always** by individual or site,
never random — that rule is the whole point.

---

## Where this is going

1. **A workshop paper** on the audit, targeting a venue about failure modes in scientific AI.
2. **Site invariance** — the open problem. Individual identity comes out with a linear projection;
   recording-site identity resists both that and an adversarial network. Nobody has solved this, and
   it is the one that matters for deployment.
3. **Cross-species affect** — a small, real, fragile signal that transfers between species. We can
   show it exists and that it isn't explained by loudness or pitch. We can't yet say what it is.
4. **Features nobody uses** — body-size-normalised pitch, nonlinear vocal phenomena, species-calibrated
   filterbanks instead of ones tuned to human hearing.

---

## Honest limitations

One encoder family dominates the comparison. Bats have no usable behavioural labels in the packaging
we could obtain, so they contribute identity results only. Dog contexts are confounded with recording
session. The relationship between identity leakage and inflation holds *within* a corpus and is not
established *across* corpora. Probes are linear throughout.

Where a result is fragile, [FINDINGS.md](FINDINGS.md) says so — including the three claims an
adversarial review made us walk back.

---

## Contact

**vishrutmalhotra4@gmail.com**

Corrections and issues welcome, especially from people who work with these animals. The datasets are
public and every number traces to a script in this repository — if something here is wrong, it should
be straightforward to show it.

Built on open work from the [Earth Species Project](https://earthspecies.org), Google's Perch, and the
teams who released the corpora: CatMeows (Ludovico et al. 2021), dog barks (Molnár et al. 2008),
Soundwel (Briefer et al. 2022), and Egyptian fruit bats (Prat et al. 2017). Code is MIT; datasets and
model weights carry their own licences, several non-commercial — see [data/README.md](data/README.md).
