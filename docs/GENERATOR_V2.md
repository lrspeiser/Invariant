# Generator v2: billion-scale sparse-action enumeration

Generator v2 is the compiled front end beneath the trusted Python/SymPy verifier. It does not construct a SymPy expression or write a JSON row for every candidate.

## Exact search claim

The frozen grammar contains 50 distinct dimensionless basis terms constructed from

```text
x = D^2/a_sigma^2
q = L_sigma^2 (partial D)^2/a_sigma^2
z = Z_b^2/Z_0^2
```

using monomials and the transforms `u`, `sqrt(1+u)-1`, and `u/(1+u)`. An action contains one through six distinct sorted terms. Every term receives exactly one coefficient from `{-epsilon,+epsilon}`, where `epsilon` is one shared universal constant.

Consequently, the declared number of actions is

```text
sum(C(50,k) * 2^k, k=1..6) = 1,088,651,720.
```

No arbitrary real coefficients are included. No claim is made about all mathematical actions or all gravity theories.

## Canonical identity and random access

Each candidate is represented by:

- a sorted tuple of basis-term IDs;
- a finite sign bit for each term;
- a unique integer ordinal;
- a SHA-256 content identifier incorporating the frozen protocol version.

The ordinal can be decoded directly through combinatorial unranking. Shard 7 does not have to enumerate shards 0–6 first.

## Gates executed in the compiled traversal

Every action receives exactly one first-failing result:

1. reject flux-only actions with no measured spatial-state dependence;
2. reject actions whose signed aggregate correction fails the declared high-field Newtonian scaling ray, after exact cancellation of equal leading powers;
3. reject actions with no gradient sector under the strict static Hessian contract;
4. reject the exact negative-elasticity control;
5. reject a non-positive radial `(D, partial_r D)` Hessian anywhere on the frozen 27-point grid;
6. survive the sampled-static tier.

Second derivatives of all 50 basis terms are precomputed with a compiled second-order jet. Candidate Hessians are sparse signed sums of those rows. Python/SymPy independently repeats the same test on recorded survivors.

A sampled-static survivor is not a covariantly healthy theory. Global tensor convexity, action variation, constraint algebra, physical degrees of freedom, characteristic cones, and GR/Solar recovery remain later gates.

## Reproducibility and storage

Candidates are processed in fixed absolute blocks. Each block commits to the ordered candidate hashes and gate codes with SHA-256. The manifest root commits to the ordered block records. The complete billion-action manifest is approximately 4 MB rather than terabytes.

Independent exact rechecks use the 32 globally lowest SHA-256 survivor hashes. This gives a deterministic pseudorandom sample distributed across the ordinal space rather than privileging early candidates. The manifest also records the lowest-hash reproducible witness for every rejection family.

`--checkpoint-dir` atomically stores completed block results. Repeating the same command reuses those blocks, reproduces the same root and gate counts, and computes only missing blocks. `--start`, `--limit`, `--shard-count`, and `--shard-index` provide bounded resume and distributed ranges.

## Commands

From the project directory:

```powershell
scripts\bootstrap_generator_v2.ps1

generator-v2\target\release\sigma-generator-v2.exe count `
  --config configs\generator_v2_billion.json

scripts\run_generator_v2.ps1 `
  -Threads 8 `
  -CheckpointDirectory runs\generator-v2\checkpoints-reproduction

$env:PYTHONPATH = "$PWD\src"
python -m sigma_theory_compiler crosscheck-v2 `
  --manifest runs\generator-v2\reproduction.json `
  --config configs\generator_v2_billion.json `
  --output runs\generator-v2\reproduction-crosscheck.json
```

The Rust bootstrap is repository-local under the ignored `tmp/rust-local/` directory and does not modify the global PATH.

The milestone was also compiled independently for `x86_64-pc-windows-gnu` and `x86_64-unknown-linux-gnu` under Ubuntu 24.04 WSL2. Their million-action basis hashes, hash-random samples, gate counts, and block commitment roots are identical.

## Scientific isolation

Generator tiers do not open observational data. The manifest records `observational_data_opened=false`. The project’s measured-evidence policy still prohibits invisible-halo targets or rescues, redshift-derived distances by default, and unaudited supernova/cosmological inferences.
