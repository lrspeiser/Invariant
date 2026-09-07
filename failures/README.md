# Failures

Everything this programme got wrong, and everything it looked for and did not find.

Kept because the successes in this repository are three data-free theorems and a
working audit method, while the failures are ten worked examples of how a
careful analysis manufactures a signal. The second set is more useful to anyone
else, and it is the set nobody publishes.

## What is here

| folder | what it holds | count |
|---|---|---|
| [`artefacts/`](artefacts/) | confident, plausible, wrong results — with the mechanism and the control that killed each | 10 |
| [`archive-traps/`](archive-traps/) | public archives returning something other than what was asked, under a success code | 5 |
| [`eliminated/`](eliminated/) | candidate laws and whole families that were refuted, blocked, or came back null | 11 |
| [`index.json`](index.json) | the same catalogue, machine-readable | — |

Everything is generated from one definition in [`build.py`](build.py), so the
index and the pages cannot drift apart. Edit there, re-run, commit.

## How to read an artefact page

Each one is the same three questions:

1. **What was claimed** — and the number that made it look real.
2. **Why it happened** — the mechanism, not the blame.
3. **What caught it** — the specific control, and what it cost to run.

The cost line matters. Every control in this catalogue was cheaper than the
analysis it invalidated. Several were free.

## The pattern, if you only read one thing

None of the nine was a coding error. The arithmetic was correct every time.
What was wrong sat one layer out:

- the archive returned different data than was asked for, under HTTP 200
- the instrument wrote its own signature into the measurement
- two axes silently shared a term from their construction
- the null distribution was too small to show its own width

All four produce the same symptom: **a confident number of plausible size, with
no exception raised anywhere.** And because a striking result is what one was
hoping for, it is the least likely thing to be interrogated.

The ninth was found while writing up the other eight: a permutation test read
2.7σ at 40 draws and 1.6σ at 200, on identical data. An undersized null is not a
conservative control — it manufactures detections, quietly.

The tenth was found the day after that, in a factor-of-two discrepancy this
programme had just reported. The analysis had filtered out the negative lensing
measurements before taking a ratio. A negative lensing signal is not an
unphysical value to clean away — it is a measurement of a small positive
quantity by a noisy instrument, and removing it deletes only the downward
fluctuations. The factor of two went to 0.914 ± 0.142 when the filter came out.

## On the count

Ten is not a boast. It is the number of times a control was run and returned a
verdict the analysis did not want, in a programme that ran roughly five control
modules for every search module.

A programme with a lower ratio would not have a smaller number. It would have a
smaller *known* number.

## Where the full records are

Each entry names its run. The complete narrative, with every number rendered
from JSON, is in `gravity-discovery-program.md` at the repository root, and each
lane keeps its own `REPORT.md`, self-tests and results under
`work/wellnet-2026-09/`.
