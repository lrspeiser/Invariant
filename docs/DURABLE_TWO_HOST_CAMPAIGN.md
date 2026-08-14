# Durable two-logical-host campaign

This contract supplies the operational evidence that the older persistent engine did not: two
independently opened logical host sessions can share one durable queue, heartbeat their own leases,
fence a dead session, and resume its work without silently accepting a stale result. It also applies
an explicit byte ceiling to the combined persistent sizes of `campaign.sqlite`,
`campaign.sqlite-wal`, and `campaign.sqlite-shm`.

This is deliberately a **two-logical-host** contract. The host IDs can be run by two terminals or
processes and can later be placed on separate machines only if they share a correctly supported
SQLite filesystem. A receipt does not claim that two physical machines were observed. The byte
ceiling is enforced by serialized application reservations, SQLite `max_page_count`, bounded WAL
checkpointing, and post-write measurement; it is not an operating-system filesystem quota.

## Durable mechanics

- The config names exactly `host-a` and `host-b`. Each registration advances a host generation.
- Every claim binds the host, session, generation-mediated active state, attempt, random lease
  token, and expiry. A fenced session cannot heartbeat, finish, or overwrite recovered work.
- A stale host is marked dead before all its running work is either queued for the next attempt or
  terminally failed. An expired lease on a still-live host follows the same retry limit.
- Queue payloads, results, and mutations are source-sealed through a chained event ledger. Startup
  binds the complete config, and status replay rejects changed payload/result/event seals.
- Each write takes a SQLite immediate lock before measuring remaining storage. It must leave a full
  transaction reserve under the configured 512 MiB combined SQLite/WAL/SHM ceiling.

## Launch and resume a real campaign

Run from the repository root. The state directory is runtime data and must not be committed.

```powershell
$env:PYTHONPATH = "src"
$config = "configs/durable_two_host_campaign.json"
$state = "runs/runtime/durable-two-host-001"
python -m sigma_theory_compiler.durable_two_host_campaign --config $config --state-directory $state init
python -m sigma_theory_compiler.durable_two_host_campaign --config $config --state-directory $state enqueue --input packets.jsonl
```

Open two terminals against that same state directory:

```powershell
# terminal one
python -m sigma_theory_compiler.durable_two_host_campaign --config $config --state-directory $state run-host --host-id host-a --maximum-slice-seconds 21600

# terminal two
python -m sigma_theory_compiler.durable_two_host_campaign --config $config --state-directory $state run-host --host-id host-b --maximum-slice-seconds 21600
```

Either command may be stopped cleanly by creating the path supplied with `--stop-path`. Restarting
the same command creates a new fenced session and resumes queued work. If a process dies, wait past
the configured 90-second dead-host timeout and run `recover`, or let the next registration do so.

```powershell
python -m sigma_theory_compiler.durable_two_host_campaign --config $config --state-directory $state recover
python -m sigma_theory_compiler.durable_two_host_campaign --config $config --state-directory $state status
python -m sigma_theory_compiler.durable_two_host_campaign --config $config --state-directory $state receipt --output six-hour-receipt.json
```

## Honest duration gate

The repository contains no manufactured six-hour receipt. Only cleanly closed sessions receive
credit, using the smaller of monotonic elapsed time and UTC wall elapsed time. Dead sessions receive
zero. Overlapping intervals from the two hosts are unioned rather than added, each logical host must
contribute at least 60 seconds, at least one work packet must succeed, and no queued, running, or
failed work may remain. `receipt` exits 20 with `BLOCK` until real credited union time reaches
21,600 seconds and all gates close. Short tests validate the mechanics and the premature-receipt
block; the real-duration campaign must actually run to promote the operational claim.
