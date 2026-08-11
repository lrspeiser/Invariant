# Unified live dashboard service

The checked live-dashboard service refreshes a **volatile**, read-only view of the running campaign
without rewriting the immutable published status snapshot. It reads the live campaign SQLite file
in the same fail-closed mode as the unified exporter and writes only beneath
`runs/engine/unified-live-dashboard-service/`, which is excluded from Git.

```powershell
$env:PYTHONPATH='src'
python -m sigma_theory_compiler.unified_engine_live_service start
python -m sigma_theory_compiler.unified_engine_live_service status
python -m sigma_theory_compiler.unified_engine_live_service stop
```

The checked configuration refreshes every five minutes for at most 4,032 successful refreshes
(two weeks), stops after 12 consecutive failures, and caps each JSON or HTML output at 3 MiB. The
checkpoint binds the exact configuration hash and uses atomic temp-file, `fsync`, and replace
semantics. A source or artifact binding mismatch counts as a failure; the service never silently
renders a mixed evidence epoch.

The live dashboard preserves the same human-readable master formulas and separate proof/test
hierarchies as the immutable dashboard. It does not open observations, dark-matter/halo inputs,
redshift-distance inputs, or paid LLM calls, and it never writes the campaign database.
