# Invariant / Open-Gravity handoff — 2026-09-01

This is the pickup document for the large Invariant/Open-Gravity research session completed on
2026-09-01. It is intentionally operational: start here after a fresh clone, then follow the linked
receipts and reports for the scientific detail.

## One-minute state of the project

- The repository is an executable, receipt-driven formula-discovery and falsification system.
- The Open-Gravity program ran nine independent lanes plus a cross-domain synthetic steering
  system and a 400-card TWELL replay.
- The scientifically honest bottom line is **no robust positive evidence for a new gravity law**.
- The strongest observation is a negative result: the registered Galileo E14 memory phase scored
  `+3.073933`, while a frozen wrong phase scored `+13.247642`.
- The strongest theorem is that changing propagation speed alone cannot enhance a stationary
  Poisson-limit field.
- The strongest reusable product is the response-blind, append-only, fail-closed audit method.
- Most surviving ideas are blocked before an interpretable real-response score. A block is not a
  theory rejection.
- Synthetic results are steering evidence only (`SYNTHETIC_DIRECTIONAL_SIGNAL`), never empirical
  support or falsification.

The canonical narrative is
[`work/open-gravity-nine-lane-comprehensive-final-report-draft-2026-08-31.md`](../work/open-gravity-nine-lane-comprehensive-final-report-draft-2026-08-31.md).
The filename retains `draft` for provenance, but the document's own status is `FINAL`.

## Start here

Read these in order:

1. [Final nine-lane report](../work/open-gravity-nine-lane-comprehensive-final-report-draft-2026-08-31.md)
   — bottom line, rank, per-object evidence, countermodels, and next falsifiers.
2. [Completion audit](../work/open-gravity-session-completion-audit.md) — the compact
   requirement-to-evidence map and terminal gate dispositions.
3. [Formula/theory status registry](../work/open-gravity-theory-formula-status-registry-draft.md)
   — every tracked theory/formula, including all 400 TWELL cards.
4. [Final evidence tables](../work/open-gravity-nine-lane-final-evidence-tables-draft-2026-08-31.md)
   — exact lane, object, family, comparator, and blocked-ledger values.
5. [Public-data acquisition audit](../work/open-gravity-public-data-acquisition-audit-lanes-2-6-7-8-9-v1.md)
   — URLs, hashes, what was downloaded, what remained unavailable, and response-access ceilings.

The final cross-document verifier is
[`work/audits/verify_open_gravity_final_deliverable.py`](../work/audits/verify_open_gravity_final_deliverable.py).

## Fresh-clone setup

Prerequisites:

- Git and Git LFS
- Python 3.11 or newer
- a local environment outside the repository (recommended)

```powershell
git lfs install
git clone https://github.com/lrspeiser/Invariant.git
Set-Location Invariant
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python work\audits\verify_open_gravity_final_deliverable.py
```

The verifier checks final report/registry/evidence integrity; it does not open protected response
data. Individual frozen packages expose `build`, `check`, or `status` commands as documented in
their modules and receipts. Run package-specific checks before attempting any successor work.

## Repository map

| Path | Purpose |
|---|---|
| [`README.md`](../README.md) | Full Invariant overview and historical roadmap |
| [`docs/GOALS_AND_MEASURED_OUTCOMES.md`](GOALS_AND_MEASURED_OUTCOMES.md) | Repository-wide measured status and priorities |
| [`docs/COMPREHENSIVE_ALPHA.md`](COMPREHENSIVE_ALPHA.md) | Breadth-first alpha milestone |
| [`src/sigma_theory_compiler/`](../src/sigma_theory_compiler) | Executable compiler, runners, formula adapters, audits, and CLIs |
| [`configs/`](../configs) | Frozen contracts, schemas, parameters, gates, and bindings |
| [`tests/`](../tests) | Unit, adversarial, deterministic-replay, and fail-closed tests |
| [`runs/`](../runs) | Immutable receipts and derived artifacts; large arrays use Git LFS |
| [`work/`](../work) | Audit programs, research reports, task briefs, and retained failure evidence |
| [`docs/provenance/INVARIANT_MIGRATION.md`](provenance/INVARIANT_MIGRATION.md) | Repository migration and history |
| [`docs/NASA_EXOPLANET_TASK1.md`](NASA_EXOPLANET_TASK1.md) | Independent-discovery Task 1 |

## Nine-lane terminal state

| Lane | Topic | Terminal state | What the next person must remember |
|---:|---|---|---|
| 1 | Path-accumulated Weyl/redshift | `BLOCK_SOURCE` | Eight lenses were audited; none has the full delay-aligned, calibrated, covariant response product. |
| 2 | Differential propagation | `AUDITED_PRE_RESPONSE_METHOD_BLOCK` | Coherent v6 independently reproduced the optimizer-validity block. Zero strain and zero real likelihood values were opened. |
| 3 | Persistent time wells | `COMPLETE_NEGATIVE` for the registered law | The real Galileo E14 phase law failed its frozen wrong-phase control. Do not generalize this to every memory model. |
| 4 | Dissipative capture/clumping | `AUDITED_SYNTHETIC_REPAIR__REAL_SOURCE_BLOCKED` | V2 passes synthetic conservation/recovery; no simulation structures, merger outcomes, or real responses were opened. |
| 5 | Dynamic source memory | `COMPLETE_NEGATIVE` for the registered experiment | The K03 family recovered 113/128 at SNR 20, below the frozen 90% gate; GW150914 strain stayed sealed. |
| 6 | Full 3-D nonspherical laws | `PARTIAL_WITH_AUDITED_SYNTHETIC_STEERING` | Eight laws run on common grids; response fits and a covariant photon/multisector closure are still absent. |
| 7 | Same-law matter/light | `BLOCK_SOURCE_AND_METHOD__SYNTHETIC_PASS` | The source-free V5 method audit is blocked; the separate lens/ray/clock synthetic matrix passes only at synthetic scope. |
| 8 | Quantum entity/wave | `PARTIAL_METHODS` | The typed atlas exposes equivalences, but no current dataset discriminates the ontology branches. |
| 9 | Void load/correlation | `RETAINED_ZERO_SCORE_NULL_FAILURE__SYNTHETIC_PASS` | One authorized development run decoded source rows and failed before scoring because absolute-length permutation could produce `L_void > D`. Never retry V6. |

Exact per-object values and all allowed claim language are in the
[final report](../work/open-gravity-nine-lane-comprehensive-final-report-draft-2026-08-31.md)
and [evidence tables](../work/open-gravity-nine-lane-final-evidence-tables-draft-2026-08-31.md).

## Current audited endpoints

These receipts are useful anchors when checking whether a future successor really advances the
state rather than replaying an already-closed gate:

- Lane 2 coherent-GW V6 independent audit:
  [`runs/gravity/open-gravity-differential-propagation-gw170817-coherent-v6-independent-audit-45fb2814/receipt.json`](../runs/gravity/open-gravity-differential-propagation-gw170817-coherent-v6-independent-audit-45fb2814/receipt.json)
- Lane 4 hydro/DMO V2 independent audit:
  [`runs/gravity/open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v2-independent-audit-06a81d78/receipt.json`](../runs/gravity/open-gravity-hydro-dmo-capture-clumping-source-shaped-synthetic-injection-matrix-v2-independent-audit-06a81d78/receipt.json)
- GW/time-series V3 independent audit:
  [`runs/gravity/open-gravity-gw-timeseries-source-anchored-synthetic-injection-matrix-v3-independent-audit-1ec33971/receipt.json`](../runs/gravity/open-gravity-gw-timeseries-source-anchored-synthetic-injection-matrix-v3-independent-audit-1ec33971/receipt.json)
- Void/cosmology V3 independent audit:
  [`runs/gravity/open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v3-independent-audit-d210b617/receipt.json`](../runs/gravity/open-gravity-void-cosmology-source-shaped-synthetic-injection-matrix-v3-independent-audit-d210b617/receipt.json)
- PHANGS full-3D synthetic matrix audit:
  [`runs/gravity/open-gravity-full3d-phangs-synthetic-injection-matrix-v2-independent-audit-498d313c/receipt.json`](../runs/gravity/open-gravity-full3d-phangs-synthetic-injection-matrix-v2-independent-audit-498d313c/receipt.json)
- X-COP source-shaped matrix audit:
  [`runs/gravity/open-gravity-xcop-real-source-shaped-synthetic-injection-matrix-v1-independent-audit-7ad3c4e1/receipt.json`](../runs/gravity/open-gravity-xcop-real-source-shaped-synthetic-injection-matrix-v1-independent-audit-7ad3c4e1/receipt.json)
- Lens/ray/clock audit evidence:
  [`work/audits/open-gravity-lens-ray-clock-source-shaped-synthetic-injection-matrix-v2-independent-audit-ce29c0cf.json`](../work/audits/open-gravity-lens-ray-clock-source-shaped-synthetic-injection-matrix-v2-independent-audit-ce29c0cf.json)
- TWELL 400 format-only successor receipt:
  [`runs/gravity/open-gravity-twell-400-source-shaped-rebind-replay-v2/receipt.json`](../runs/gravity/open-gravity-twell-400-source-shaped-rebind-replay-v2/receipt.json)

## Formula registry and synthetic steering system

The registry records the active laws, formula hashes, architectures, drivers, cells, lane mapping,
and audit status. It covers all 400 exact TWELL cards, 19 architecture families, GP01, QG01–QG13,
Q00–Q15, the dynamic-memory and void families, conventional comparators, and historical items.

Use:

- [Theory/formula registry](../work/open-gravity-theory-formula-status-registry-draft.md)
- [TWELL 400 source-rebind task brief](../work/open-gravity-twell-400-source-rebind-task-brief.md)
- [Next temporal/multidomain slice](../work/open-gravity-twell-temporal-and-multidomain-rebind-next-slice.md)
- [Generic synthetic runner V2 audit](../work/audits/open-gravity-synthetic-discovery-runner-v2-independent-audit-d67225ff.json)

The current broad synthetic system has audited galaxy, cluster, Solar/planetary, lens/ray/clock,
GW/time-series, void/cosmology, and hydro/DMO slices. It is intended to steer which formulas and
data features deserve real-response work. It cannot establish a discovery, and it must never be
tuned using a dataset later presented as independent confirmation.

## Data and Git LFS policy

The repository commits unique code, configs, tests, receipts, audit programs, reports, ledgers, and
derived arrays. Binary derived artifacts (`.npz`, `.npy`, HDF5/FITS where retained, databases,
archives, and similar formats) are tracked through Git LFS.

The local session also contains more than 20 GiB of external/reacquirable scientific payloads,
caches, duplicated public archives, and a local LALSuite runtime. They are intentionally ignored
rather than uploaded:

| Ignored local class | Why it is not in GitHub | Reproduction pointer |
|---|---|---|
| `work/private/` | External source bytes, duplicated predictions, and a rebuildable local runtime; includes a 7.0 GiB MUSE FITS file above [GitHub's 5 GiB maximum LFS object size](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) | [Public-data acquisition audit](../work/open-gravity-public-data-acquisition-audit-lanes-2-6-7-8-9-v1.md) and bound source receipts |
| `work/gravity/*cache/`, `work/item*-raw/`, raw subfolders | Reacquirable survey/archive caches | Per-item manifests and audit files under `work/` and `runs/gravity/` |
| `runs/**/source/`, `runs/**/sources/` | Public raw payload copies | The surrounding committed receipt, download manifest, URL, size, and SHA-256 table |
| `*.sqlite-wal`, `*.sqlite-shm` | Live/transient database sidecars, not durable state | The committed database/receipts and replay commands |

The most complete acquisition ledger is
[`work/open-gravity-public-data-acquisition-audit-lanes-2-6-7-8-9-v1.md`](../work/open-gravity-public-data-acquisition-audit-lanes-2-6-7-8-9-v1.md).
For the 157-file lens acquisition, use
[`runs/gravity/open-gravity-path-accumulated-weyl-redshift-source-preflight-v1/download-manifest.json`](../runs/gravity/open-gravity-path-accumulated-weyl-redshift-source-preflight-v1/download-manifest.json)
and
[`downloaded-file-sha256.csv`](../runs/gravity/open-gravity-path-accumulated-weyl-redshift-source-preflight-v1/downloaded-file-sha256.csv).

Do not substitute a similarly named archive file for an exact frozen URL/hash. Reacquire, stream
hash, and stop before scientific decoding whenever the corresponding receipt says the response is
sealed.

## Safe continuation rules

1. Preserve every frozen predecessor byte-for-byte. Repairs are append-only successors.
2. Keep source, development response, validation, confirmation, and synthetic truth roles distinct.
3. Bind exact bytes, schemas, units, coordinates, nuisance rules, seeds, thresholds, countermodels,
   and output paths before opening a protected response.
4. Run target-free recovery, identifiability, numerical-health, and failure-retention gates first.
5. Require a distinct read-only audit before one-use response authorization.
6. If an authorized run fails closed, retain the failure and do not retry the same version.
7. Never translate `SOURCE_BLOCKED`, `NUMERICAL_INVALID`, or optimizer failure into a physical
   rejection.
8. Keep empirical fit, identifiability, physical consistency, novelty, and publication value as
   separate judgments.

## Highest-value next work

The fastest useful continuations are:

1. Lane 9: define a geometry-preserving null using bounded exposure fractions or regenerated
   intersections, prove every draw satisfies `0 <= L_void <= D`, then freeze and independently
   audit a new successor. Never retry V6.
2. Lane 2: build a new target-free optimizer successor that resolves the iteration-cap failures
   symmetrically across all models before considering any strain access.
3. Lane 7: acquire or independently reduce exact PSF/LSF/astrometry/kinematic covariance inputs;
   repair holdout leakage, amplitude-blind convergence, Yukawa zero mode, and incomplete LPD/
   pseudo-NFW execution before opening ESO/SLACS responses.
4. TWELL: extend the audited static X-COP replay to the temporal and multidomain adapters described
   in the [next-slice note](../work/open-gravity-twell-temporal-and-multidomain-rebind-next-slice.md).
5. Keep broadening source-shaped synthetic populations for steering, but require real independent
   holdouts for any empirical claim.

The final report names three fast falsifiers and their exact stop rules; use those before starting a
new unregistered data search.

## Independent-discovery Task 2 automation

Task 2 is currently gated at `BLOCKED_FUTURE_RELEASE_NOT_PUBLISHED`. The current metadata-only
check is [`work/broken-arxiv-task2-source-check-current.json`](../work/broken-arxiv-task2-source-check-current.json),
and the authorization is
[`runs/math/broken-arxiv-task2/authorization-v4.json`](../runs/math/broken-arxiv-task2/authorization-v4.json).

Allowed monitor command:

```powershell
python -m sigma_theory_compiler.broken_arxiv_task2 check-source `
  --authorization runs/math/broken-arxiv-task2/authorization-v4.json `
  --output work/broken-arxiv-task2-source-check-current.json
```

While blocked: read/download zero problem rows and zero answers, make no repository changes, do not
substitute an older release, make no paid calls, and do not start Task 3. If the first eligible raw
`MathArena/brokenarxiv-MMYY` release dated July 2026 or later appears and was published after the
authorization, verify the authorization and source receipt, report the exact release/revision and
maximum cost of the frozen 36-call Anthropic trial, and obtain explicit approval before paid calls.

## Publication boundary

Publishable now: scoped negative results, stop-rule examples, source/audit methodology, the
stationary propagation theorem, and synthetic identifiability maps with explicit claim ceilings.

Not publishable as a discovery: a positive new-gravity claim, universal-law claim, empirical
confirmation from the synthetic matrices, or a Lane 2/Lane 7/Lane 9 response result beyond the exact
frozen terminal dispositions.

When in doubt, use the final report's exact allowed statement and the corresponding immutable
receipt rather than paraphrasing from memory.
