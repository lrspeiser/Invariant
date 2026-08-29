# Item 41: stochastic-gravity result

## Bottom line

Item 41 completed a real search for gravity formulas that predict variance as well as mean behavior.
It generated 262,144 stochastic laws in four equally sized mechanism niches, admitted 60,905 after
frozen physical checks, and evaluated 25,945,530 candidate-point-fold combinations on an RTX 5090.
The selected law was then transferred unchanged to 20 CLASH galaxy clusters.

The result is useful but negative. A nearly constant white-noise law was selected, but it did not
beat a simple constant-noise control on the paired galaxy data and did not beat an ordinary
out-of-cluster variance model on the cluster data. The exact law is not promoted. It is not killed,
and the stochastic-gravity family remains open.

## What was tested

The search treated the approaching and receding sides of each galaxy as two imperfect realizations
at the same radius. Their average tests the predicted mean drift; their difference tests the
predicted variance. Four mechanism niches received equal search space:

- an Einstein-Langevin-inspired white stochastic field;
- a radially colored Ornstein-Uhlenbeck field;
- multiplicative noise with noise-induced drift;
- a two-state or telegraph-like vacuum field.

The formula generator saw no response values. It used only baryonic acceleration and normalized
radius, then assigned each candidate both a natural-log acceleration drift `m` and variance `S`.
The selected full retrospective candidate was:

`u = gbar/a0`

`m = 0`

`S = 0.56^2 / [1 + (u/10000)^3]`

`Delta ln(g) = sqrt(S) xi`, where `E[xi]=0` and `Var[xi]=1`.

Across the tested weak-acceleration range this is effectively a constant variance of about 0.3136,
not a strongly scale-dependent stochastic effect.

## Retrospective paired-side galaxy result

The test used 15 previously exposed GHASP galaxies and 71 paired radial points. It opened none of
Item 28's ten sealed confirmation galaxies, so this is retrospective development rather than fresh
confirmation.

Equal-galaxy held-out joint mean-and-variance losses were:

| Model | Loss | Reading |
|---|---:|---|
| Homoskedastic control | 1.3003 | best tested control |
| Item 41 stochastic candidate | 1.7493 | about 34.5% worse |
| Ordinary heteroskedastic ridge | 3.0435 | worse than the candidate |

The candidate's predicted variance had essentially no rank relationship with the measured squared
side difference (`rho=-0.0184`, `p=0.8787`). Five galaxies were raw counterexamples and three
remained uncertainty-resolved. The aggregate conclusion reverses when the most influential galaxy,
UGC5786, is removed. That makes this a data-sensitive negative result, not a terminal rejection.

## Unchanged CLASH cluster diagnostic

After the galaxy-selected formula was committed, the identical drift and variance were applied
without retuning to 20 CLASH clusters and 84 published radial points. The primary diagnostic used a
fixed MOND/RAR mean so the stochastic component was tested specifically as a residual drift and
variance law. A separate score placed the same law on a baryonic-Newton mean to test whether it
could act as a direct no-dark-matter replacement.

| Model | Equal-cluster natural-log NLL |
|---|---:|
| MOND + ordinary out-of-cluster heteroskedastic variance | 0.6444 |
| MOND + out-of-cluster constant variance | 0.6616 |
| MOND + Item 41 unchanged stochastic variance | 1.4046 |
| Baryonic Newton + Item 41 stochastic variance | 5.2649 |
| MOND with measurement errors only | 19.0396 |
| Baryonic Newton with measurement errors only | 65.6300 |

The stochastic variance usefully acknowledges that measurement errors alone understate the scatter,
but its fixed strength is far too large and it does not repair the baryonic mean. It lost to the
strongest honest variance control in 18 of 20 clusters; 17 remained uncertainty-resolved. The
result was stable to removing the most influential cluster.

CLASH is an already exposed, model-dependent acceleration proxy rather than a fresh direct
image-likelihood test. It therefore supplies scoped negative evidence but is not counted as an
independent unchanged replication that can retire the formula.

## Why the formula and family are retained

Astronomical data carry measurement, geometry, distance, inclination, stellar-mass, substructure,
and modelling uncertainty. The executable project policy therefore forbids using one empirical
counterexample—or a count by itself—as a kill switch. Item 41 preserves every mismatch, separates
raw from uncertainty-resolved cases, checks sensitivity to individual objects, and requires genuinely
independent unchanged replication before terminal rejection even within a tested scope. A finite
sample cannot prune the entire stochastic-gravity family.

Retention is not success. It means keeping the exact law and its failure map for replication or a
principled future hybrid while refusing to tune it to these exposed responses. The next mechanism
test starts fresh rather than rescuing this result after seeing its failures.

## Claim limits

- No stochastic gravity process, correlation time, or noise kernel was established.
- No dark-matter explanation was excluded.
- No modification of gravity was established.
- No covariant theory was derived.
- Historical novelty was not established; all four niches combine known stochastic ideas.
- Item 28's confirmation galaxies remain sealed.
- No paid model calls were made.
