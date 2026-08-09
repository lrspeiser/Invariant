# Sigma Campaign Engine v1

## Scope and scientific claim

Sigma Campaign Engine turns the one-shot gravity compiler into a durable research campaign. It can
run for days or weeks, survive worker and process failures, accept bounded LLM or human proposals,
and continuously preserve evidence and candidate lineage.

It cannot guarantee discovery of the true theory of gravity. "Best" means highest work priority
within a declared grammar and completed hard gates. No historical result, observational fit, or LLM
judgment can rescue a hard-gate rejection.

## Durable state

The SQLite/WAL campaign database contains:

- campaigns, deadlines, task/failure/cycle budgets, pause state, and stop reasons;
- immutable candidates, canonical hashes, parents, families, generations, and Pareto provenance;
- queued/running/succeeded/failed/deferred/blocked/cancelled tasks;
- task dependencies, attempts, worker leases, heartbeat expiry, retry backoff, and idempotency keys;
- versioned gate evidence with hard/soft status, outcomes, margins, units, and evidence class;
- hashed artifacts, bounded proposals, failure clusters, and an append-only event stream.

Workers use `BEGIN IMMEDIATE` task leasing. A crashed worker's lease expires and the task returns to
the queue until `max_attempts` is reached. Long workers renew their leases from a heartbeat thread.
Workers may now declare a task-type allowlist at lease time, so separately sized symbolic, LLM, and
housekeeping pools cannot consume one another's queues. The measured 5090 workstation pool sizes and
launcher are documented in [`PARALLEL_EXECUTION.md`](PARALLEL_EXECUTION.md).

## Worker types

- `generator_export`: Rust exhaustive generator adapter.
- `gpu_dense_screen`: CuPy/RTX dense-static adapter.
- `knowledge_build`: historical evidence-graph adapter.
- `pipeline_artifact_audit`: checks previous exhaustive and GPU evidence before reuse.
- `measurement_policy` / `policy_validate`: prohibited-evidence and universal-policy gate.
- `covariant_lift`: creates explicitly versioned scalar and constrained-vector hypotheses.
- `symbolic_proxy`: compiles the reduced invariant formula and derivative/ablation information.
- `proposal_compile`: validates structured bounded human/LLM actions.
- `constraint_analysis`: rejects missing higher-derivative degeneracy declarations or defers to ADM.
- `reference_control`: runs the Einstein-Hilbert GR and Solar-System golden controls.
- `failure_cluster`: groups terminal failures by gate and mechanism.
- `llm_research`: invokes a trusted JSON adapter or writes a durable offline proposal packet.
- `candidate_dossier`: explains lineage, gates, margins, failures, ablations, comparisons, provenance,
  remaining claims, and why the candidate is receiving work.

The symbolic proxy is not presented as covariant tensor variation. ADM/Dirac constraints,
Hamiltonians, degrees of freedom, and characteristic cones remain unresolved until their formal
backends exist.

## Current live campaign

Database: `runs/campaigns/campaign-v1-live.sqlite`

Campaign: `CMP-55b43fa111e201988d9f2922`

The initial execution registered six generated leaders and an Einstein-Hilbert control, created 12
covariant lifts, and accepted one schema-valid bounded example proposal. After enforcing the
covariant field contract and running formal-control version 25, its current accounting is:

- 23 immutable candidates;
- 2 active controls/static parents, 4 deferred lifts/proposals, and 17 terminal rejections;
- 132 completed hard-gate passes, 22 hard rejections, and 9 unresolved hard gates;
- 188 succeeded tasks, 10 deferred tasks, and 2 queued research-cycle tasks;
- 101 versioned candidate dossier builds queued or completed across the refreshes;
- four metered Claude research calls at $1.104242 total: one proposal failed the
  universal-coupling schema, while one passed schema/policy and was then hard-rejected because its
  claimed higher-derivative degeneracy relation was prose rather than machine-declared; the third
  and fourth passed bounded intake and remain deferred pending their actual scalar-tensor or
  constrained-vector formal gates;
- zero infrastructure task failures.

Legacy static candidates and lifts that depended on baryon-specific `z_b` were rejected by the
universal-matter field contract. One constrained-vector lift and two bounded proposals remain
deferred at formal constraint stages. This is a useful failure, not a discovery claim.

## Commands

Initialize a new campaign:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m sigma_theory_compiler.campaign_cli init `
  --config configs\campaign_v1.json `
  --database runs\campaigns\my-campaign.sqlite
```

Drain currently available work:

```powershell
python -m sigma_theory_compiler.campaign_cli run `
  --database runs\campaigns\my-campaign.sqlite
```

Run continuously for two weeks, polling for proposals and hourly research cycles:

```powershell
python -m sigma_theory_compiler.campaign_cli run `
  --database runs\campaigns\my-campaign.sqlite `
  --duration 14d --follow
```

For automatic process restart, use:

```powershell
scripts\run_campaign_watchdog.ps1 `
  -Database runs\campaigns\my-campaign.sqlite `
  -Slice 6h
```

For reboot persistence, configure Windows Task Scheduler to start that watchdog at login or boot.
Create the configured `STOP` file to end it cleanly.

Operational commands:

```powershell
python -m sigma_theory_compiler.campaign_cli status --database DATABASE
python -m sigma_theory_compiler.campaign_cli pause --database DATABASE --reason "maintenance"
python -m sigma_theory_compiler.campaign_cli resume --database DATABASE
python -m sigma_theory_compiler.campaign_cli recover --database DATABASE
python -m sigma_theory_compiler.campaign_cli report --database DATABASE --output REPORT_DIRECTORY
python -m sigma_theory_compiler.campaign_cli refresh-dossiers --database DATABASE --version 2
```

## LLM adapter contract

The engine does not grant an LLM direct database or evidence authority. With `llm.command` empty, it
writes research packets under the campaign's `llm-outbox` directory. To connect Claude Code or
another model, configure a trusted command as a JSON array. The command receives one JSON research
packet on stdin and must return one metered adapter envelope containing a proposal JSON object on
stdout.

Required proposal fields include the action, fields, symmetries, universal constants, derivative
order, one matter metric, claimed static limit, expected degrees of freedom, evasion rationale,
falsification tests, literature overlap, and a bounded grammar with basis, maximum terms, and a
coefficient alphabet.

Claude Code 2.1.217 is installed on this workstation. The included
`scripts/claude_proposal_adapter.py` invokes it in non-interactive safe mode with no tools, no
session persistence, JSON Schema enforcement, and a per-call dollar cap. Enable it through the
campaign CLI so the per-call cap is paired with an atomic aggregate ledger:

```powershell
python -m sigma_theory_compiler.campaign_cli configure-claude `
  --database DATABASE `
  --total-budget-usd 500 `
  --max-calls 250 `
  --per-call-budget-usd 2 `
  --model sonnet
```

Every worker reserves its maximum call cost transactionally before execution. Provider-reported
cost settles the reservation afterward; missing or failed metering consumes the full reservation.
The call-count limit bounds the worst case, and a live budget cannot be silently raised. The default
configuration remains offline so a long campaign cannot incur model usage without an explicit
enablement command.

Submit a human-reviewed or offline-LLM proposal with:

```powershell
python -m sigma_theory_compiler.campaign_cli submit-proposal `
  --database DATABASE `
  --file configs\proposal_example.json
```

Proposals containing dark-matter targets, NFW, redshift-derived distances, supernova distances, or
derived GR/NFW lensing targets are rejected before compilation. A proposal may schedule tests; it
never counts as evidence merely because an LLM produced it.

## Recovery demonstration

`scripts/demonstrate_campaign_recovery.py` leases a task to a simulated crashed worker, expires the
lease, recovers it, resumes it on a replacement worker at attempt two, and verifies database
integrity. The executed result is `runs/campaigns/recovery-demo.json`.

## Next scientific backend

The engine is operational, and the static-lift bottleneck is now machine classified. Exact static
tensor reduction derives the unit-Aether acceleration mapping for `x`; baryonic `z` is rejected as
a nonuniversal matter action atom; and every `q` formula is deferred to a missing projected
Aether-acceleration-gradient invariant. In the 124-family dense work queue, 104 families reject for
`z`; 20 are clean of `z`. The `q` and nonlinear-`x` action atoms now export six exact candidates,
all of which fail necessary uncompleted-action kinetic/gradient checks. A static-null `K1/K4`
completion of the best mixed candidate preserves the generator shape and is the sole deeper formal
lead. The next scientific backend is therefore its connection-dependent metric variation, ADM/Dirac,
Hamiltonian, and principal-symbol analysis—not another analogy-based lift. Only a completed lift can
reach GR/Solar eligibility, and direct observations remain separately manifest-gated.
