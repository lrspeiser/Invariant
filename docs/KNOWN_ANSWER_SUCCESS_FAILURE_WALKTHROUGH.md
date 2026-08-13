# A Formula Walks Into a Falsification Engine

*How Invariant records one mathematical success, one useful failure, and the line between them*

Suppose we give a formula-discovery system an anonymous sequence defined only by

\[
S(0)=0, \qquad S(n+1)=S(n)+(n+1).
\]

We want more than a formula that agrees with a few examples. We want to see the whole research
record: how a candidate was found, how a tempting alternative failed, why the survivor is a theorem
rather than a curve fit, and which claims the experiment still does **not** justify.

Invariant now produces that record from immutable machine receipts. This article follows two real
candidates through the system:

\[
C(n)=\frac{n(n+1)}2
\]

and the plausible but false shortcut

\[
W(n)=n^2.
\]

The outcome is deliberately asymmetric. The false formula is stopped as soon as a counterexample
is found. The correct formula is allowed to proceed to exact proof, prior-art comparison, and only
then a soft Pareto ranking. No failed candidate is quietly discarded, and no simplicity score is
allowed to rescue a mathematical failure.

## Before the answer was visible

The companion blind benchmark did not begin with the natural-sum formula. It searched the declared
quadratic grammar

\[
a n^2+b n+c
\]

using exact rational coefficients. Within that finite grammar it performed:

- 46,656 raw coefficient trials;
- reduction to 12,167 canonical coefficient classes;
- comparison with six public examples;
- 59 additional counterexample checks for each public-example survivor; and
- an induction proof using only the anonymous base case and successor rule.

Exactly one candidate survived the public examples:

\[
a=\frac12,\qquad b=\frac12,\qquad c=0.
\]

The winning expression and proof were sealed before the withheld theorem fixture could be read.
Only after sealing did the verifier disclose that the result matched the familiar natural-sum
formula. That experiment is a bounded chronological rediscovery control—not a claim that the system
searched every possible formula.

## The successful derivation, written as a mathematician might write it

Let

\[
C(n)=\frac{n^2+n}{2}.
\]

First check the base case:

\[
C(0)=0=S(0).
\]

Now compute the successor difference:

\[
\begin{aligned}
C(n+1)-C(n)
&=\frac{(n+1)^2+(n+1)}2-\frac{n^2+n}2\\
&=\frac{n^2+3n+2-n^2-n}{2}\\
&=n+1.
\end{aligned}
\]

This is exactly the defining recurrence for \(S\). Therefore, if \(C(n)=S(n)\), then

\[
C(n+1)=C(n)+(n+1)=S(n)+(n+1)=S(n+1).
\]

By induction,

\[
\boxed{S(n)=\frac{n(n+1)}2}
\]

for every nonnegative integer \(n\).

The important distinction is that the finite examples suggested and filtered the candidate; they
did not prove the universal statement. The recurrence certificate supplied that proof.

## The failure is part of the result

The shortcut \(W(n)=n^2\) looks respectable at the first two values:

\[
W(0)=0,\qquad W(1)=1.
\]

But the next informative point is enough to falsify it. From the recurrence,

\[
S(2)=S(1)+2=3,
\]

whereas

\[
W(2)=4.
\]

Invariant records \(n=2\) as the first counterexample in the blind benchmark. In the end-to-end
evaluation control, the candidate receives a typed `reject` outcome at
`counterexample_screened`. The later `exactly_verified` and `prior_art_checked` stages are recorded as skipped.
It receives no metric receipt and no Pareto front.

That behavior matters. A formula cannot compensate for being false by being simple, elegant, or a
good numerical fit.

## What the machine actually did

The known-answer control sends both candidates through the same fail-closed ladder:

| Stage | Purpose | Correct identity | Wrong square formula |
|---|---|---:|---:|
| `typed` | Validate the formula and domain schema | Pass | Pass |
| `canonicalized` | Establish the exact canonical expression boundary | Pass | Pass |
| `counterexample_screened` | Search the declared exact/adversarial domain | Pass | **Reject** |
| `exactly_verified` | Require an exact proof certificate | Pass | Skipped |
| `prior_art_checked` | Compare only after proof admission | Pass | Skipped |
| soft metric / Pareto | Explain admitted survivors, never establish truth | Front 1 | Not eligible |

The checked run contains two candidates, one full hard-gate pass, one counterexample rejection, one
formal proof, one post-proof prior-art check, one metric receipt, and one ranked candidate. It
authorizes zero promotions.

The soft metric in this control is the exact byte length of the canonical Math IR formula. It is
intentionally unimportant mathematically. Its purpose is to demonstrate ordering: the metric is
admitted only after all hard gates pass. With one eligible candidate, “Pareto front 1” means only
that the survivor is nondominated in this tiny admitted set. It does not mean “true because ranked
first.” Truth was already a prerequisite.

## Receipts instead of screenshots

The end-to-end control is stored at
[`runs/math-language/math-known-identity-pipeline-control/campaign.json`](../runs/math-language/math-known-identity-pipeline-control/campaign.json).
Its file SHA-256 is
`2bd02f439bd04523a5eeec00291d82e7e9c755e09a601a64f3e271a8e39988f8`, and its canonical content
seal is `c658cfe6e0e6f2146f7c109c7ae262561933f6fbd10e3780a7bd9a7549f639fd`.

The successful candidate is `sig-522092801d108ee92d093f22`; the rejected candidate is
`sig-06099d25141d45f0714a4205`. Their stage and gate outcomes, skipped stages, metric admission,
Pareto explanation, source bindings, and claim boundary are all content-hash bound.

The earlier blind benchmark is stored at
[`runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json`](../runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json).
Its winner is `f48406b1add8e3e88d5b86c0`, and its checked file SHA-256 is
`3158d6031ad0dbf1c3cb955c956af319f39c58d50982d2193a07b0f46c83e685`.

For a notebook-style proof narrative, see the generated
[`natural-sum-rediscovery.md`](notebooks/generated/natural-sum-rediscovery.md) record or open the
adjacent `.ipynb` file in Jupyter.

## Reproduce the success and failure

From the repository root:

```powershell
python -m pytest tests/test_math_known_identity_pipeline_control.py -q
python -c "import json; from pathlib import Path; from sigma_theory_compiler.math_known_identity_pipeline_control import OUTPUT_PATH, validate_result; root=Path('.').resolve(); validate_result(json.loads((root/OUTPUT_PATH).read_text(encoding='utf-8')), root); print('immutable replay: PASS')"
```

The focused test suite independently checks the immutable replay, successful proof path, failed
counterexample path, skipped formal work, ranking exclusion, closed scientific claims, and several
resealed semantic tamper attempts.

## The same presentation pattern on an open research problem

The natural-sum control is intentionally familiar, so its right answer is easy to recognize. The
same notebook machinery is also used where the answer is not known in advance. Two adjacent
quartic-field notebooks show what that looks like.

The
[`quartic-ordered-mixed-d2-differentiability.md`](notebooks/generated/quartic-ordered-mixed-d2-differentiability.md)
notebook follows 20 registered first-derivative roots per candidate through exact sum, product, and
quotient rules. It reaches 31,680 candidate-bound leaf-derivative obligations—but the needed leaf
jets are absent from the registered schema. The notebook therefore records a block. It does not
replace an unknown derivative by zero, reject the candidate, or call the missing schema a physical
no-go.

The
[`quartic-flat-factorized-leaf-jet-d2.md`](notebooks/generated/quartic-flat-factorized-leaf-jet-d2.md)
notebook then adds a legitimate flat-reference coordinate map. Under that declared specialization,
the same obligations factor into 20 formulas per candidate and all 264 requested D2 values are
materialized exactly: 192 are zero and 72 are nonzero. This is a real success, but only at the flat
reference. The arbitrary-background theorem remains open.

Placed side by side, these notebooks expose three states that a useful research system must keep
distinct:

| Receipt state | Plain-language meaning | What Invariant may say |
|---|---|---|
| `reject` | A concrete check disproved the candidate | Show the counterexample and stop downstream work |
| `block` | Required evidence or a registered operation is missing | Name the missing premise and preserve the candidate |
| `pass` at a declared scope | Every obligation in that scope closed | Show the derivation and state exactly where it stops |

That distinction is the core product behavior. A block is not softened into a pass, and a bounded
pass is not inflated into a global theorem.

## What this demonstration establishes—and what it does not

It establishes that, for this registered Math Pack and these bound inputs, Invariant can:

1. retain a candidate that survives exact checks;
2. stop a false candidate at a concrete counterexample;
3. prevent proof, prior-art, and ranking stages from opening after failure;
4. certify a survivor using a universal induction argument rather than finite agreement;
5. compare against a withheld known result only after the discovery/proof seal; and
6. preserve the complete path as replayable, tamper-evident evidence.

It does **not** establish that every future domain pack is correct, that every true formula lies in a
declared grammar, that bounded counterexample exhaustion is proof, that the result was historically
novel, or that a Pareto rank authorizes publication or promotion. Those claims remain explicitly
false in the receipt.

That is the product behavior we want others to inspect: not a machine that merely announces an
answer, but one that shows where an idea came from, where a rival failed, what converted a survivor
into a theorem, and exactly where confidence must stop.
