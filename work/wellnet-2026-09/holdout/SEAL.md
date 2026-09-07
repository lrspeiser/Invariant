# The gold-cluster holdout

Sealed 2026-09-07, before any shear was read from any cluster in the pool.

    pool     605 clusters with a public raw weak-lensing channel (Run BO)
    SEALED   304        digest a58e22431181b23c...
    OPEN     301
    rule     holdout_seal v2
    salt     wellnet-goldcluster-2026-09-07   (declared before the split)

## Why this exists

The programme has had no sealed confirmation set since KiDS and the wide
binaries were scored in round 1. Everything since has been validation dressed as
confirmation: a pattern found in a dataset cannot be confirmed by that dataset.

Run BO created the first opportunity to fix that. It probed 1,863 eRASS1
clusters for weak-lensing COVERAGE and read no shape, no response and no
per-source redshift from any of them — enforced by `goldcluster/guard.py`, not
promised. So at the moment of this split, **no outcome had been observed for any
of the 605**. That is the only condition under which a confirmation set can be
made, and it does not come round often.

## How the split was made

Deterministic, stratified, and committed.

- **Deterministic.** Assignment is a keyed SHA-256 rank of the cluster's own
  name within its stratum. There is no random seed and no shuffle, so there is
  no state to re-roll and re-running reproduces the split exactly.
- **Stratified** into 27 cells — terciles of redshift × X-ray counts × shear
  depth — so neither half is systematically nearer, brighter or better covered.
  Measured balance: medians agree to 0.2% in redshift, 4% in X-ray counts and
  0.8% in shear depth; 182 vs 181 have spectroscopic redshifts, 160 vs 172 have
  a measured temperature.
- **Committed.** `holdout_split.json` carries the digests of the pool, the
  sealed set and the open set. `loader.verify()` recomputes all three on every
  load and raises `SealBroken` on any drift. Moving a single cluster between
  halves is caught (test T5). A split you can quietly re-roll is not a seal.

## What is sealed, and what is not

**Sealed: the outcome.** Anything derived from a sealed cluster's shear — a
shape, a response, a tangential profile, a mass. Reaching it requires a
one-shot token, and taking one is a deliberate, recorded, irreversible act.

**Not sealed: public catalogue metadata.** Name, position, redshift, X-ray
counts and temperature stay readable for sealed clusters, because a lane has to
know which clusters to stay away from, and because reading a published X-ray
count scores nothing about gravity.

One honest wrinkle: `goldcluster/ranking.json` is committed and carries
`n_shear_sources` for all 605, sealed ones included. That number is how many
background galaxies DECam happens to have imaged behind that position — it is
set by exposure depth, not by the cluster's mass, and it encodes nothing about
the lensing signal. It is nonetheless excluded from `open_fields`, so
`loader.sealed_metadata()` will not hand it to you.

## How to use it

    from holdout import loader
    loader.verify()                        # digests intact
    targets = loader.open_pool()           # the 301 you may search on
    loader.assert_not_sealed(names)        # before any archive query

Search, fit and eliminate on the open half as freely as you like. The sealed
half stays blind.

## How to spend it — once

    tok = loader.request_token(
        "Run BX: test law L on the sealed half, pre-registered in <path>",
        requester="BX")
    sealed = loader.open_sealed(tok)

A token needs a written reason of at least 40 characters naming the run and the
pre-registered test, because the answer can only be obtained once. **Write the
pre-registration first** — the exact statistic, the exact threshold, and what
outcome would count as a refutation — and commit it before minting the token.

Opening the seal ends the holdout. After that the programme has no confirmation
set again until one is built from data that has never been searched.

## The CI gate

`test_seal.py` T2 asserts `tokens_spent == 0`. While the holdout is intact, CI
is green. The moment anyone opens it, CI goes red — deliberately — until the
pre-registration is committed alongside it and T2 is updated to reflect that the
set is spent. That is the alarm, and it should be loud.

Run it as `python test_seal.py`, not under pytest: the lane provenance guards
raise on pytest's own `.pytest_cache` write.
