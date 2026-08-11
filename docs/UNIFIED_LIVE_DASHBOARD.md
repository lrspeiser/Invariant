# Unified live dashboard service

The checked live-dashboard service refreshes a **volatile**, read-only view of the running campaign
without rewriting the immutable published status snapshot. It reads the live campaign SQLite file
in the same fail-closed mode as the unified exporter and writes only beneath
`runs/engine/unified-live-dashboard-safety-service/`, which is excluded from Git.

```powershell
$env:PYTHONPATH='src'
python -m sigma_theory_compiler.unified_engine_live_service_safety start --config configs/unified_engine_live_service_safety.json
python -m sigma_theory_compiler.unified_engine_live_service_safety status --config configs/unified_engine_live_service_safety.json
python -m sigma_theory_compiler.unified_engine_live_service_safety stop --config configs/unified_engine_live_service_safety.json
```

The checked configuration refreshes every five minutes for at most 4,032 successful refreshes
(two weeks), stops after 12 consecutive failures, and caps each JSON or HTML output at 3 MiB. The
checkpoint binds the exact configuration hash and uses atomic temp-file, `fsync`, and replace
semantics. A source or artifact binding mismatch counts as a failure; the service never silently
renders a mixed evidence epoch.

The live dashboard preserves the same human-readable master formulas and separate proof/test
hierarchies as the immutable dashboard. It does not open observations, dark-matter/halo inputs,
redshift-distance inputs, or paid LLM calls, and it never writes the campaign database.

## Hardened service status

The active implementation is
`sigma_theory_compiler.unified_engine_live_service_safety`. The legacy worker was gracefully stopped
and its exact PID/argv absence verified before the hardened worker acquired the cutover lease. It adds
Windows-safe list-form worker arguments,
PID/command identity checks, a fixed runtime epoch with counter-preserving compatible reloads,
final checkpointing on reload failure, 0.25-second control polling with guarded refresh phases,
exclusive unpredictable temporary files with symlink rejection, and pre/post dependency manifests
that prohibit stale publication. Start also requires exact legacy-worker absence, holds an exclusive
cross-start lease, writes an atomic `starting` checkpoint before spawn, and rejects repeated starts.

Its remaining limits are explicit: it cannot interrupt the interior of a third-party snapshot or
leaderboard builder call; JSON and dashboard replacements are individually atomic rather than one
cross-file transaction; and PID command verification fails closed when `psutil` is unavailable.
The volatile status panel is authoritative for current PID, argv identity, refresh count, and config
currency. The immutable readiness artifact records the construction lane and therefore truthfully
retains `service_started=false`; it is not the volatile activation record.
