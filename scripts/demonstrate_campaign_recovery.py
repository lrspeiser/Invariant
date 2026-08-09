from __future__ import annotations

import argparse
import json
from pathlib import Path

from sigma_theory_compiler.campaign import CampaignStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = CampaignStore(args.database)
    store.initialize()
    config = {
        "name": "lease recovery demonstration",
        "budget": {
            "duration_days": 1,
            "max_tasks": 10,
            "max_failures": 3,
            "max_cycles": 0,
        },
        "scientific_contract": {"purpose": "infrastructure demonstration only"},
    }
    campaign_id = store.create_campaign(config)
    task_id = store.add_task(
        campaign_id,
        "recovery_witness",
        stage=0,
        payload={"claim": "task survives worker loss"},
    )
    crashed = store.claim_task(campaign_id, "simulated-crashed-worker", lease_seconds=-1)
    recovered = store.recover_expired_leases(campaign_id)
    resumed = store.claim_task(campaign_id, "replacement-worker", lease_seconds=60)
    if not crashed or not resumed:
        raise RuntimeError("recovery demonstration could not lease the task")
    store.finish_task(
        resumed,
        "replacement-worker",
        "succeeded",
        {"resumed_attempt": resumed.attempt},
    )
    report = {
        "schema_version": "sigma-campaign-recovery-demo-1.0",
        "campaign_id": campaign_id,
        "task_id": task_id,
        "first_worker": "simulated-crashed-worker",
        "first_attempt": crashed.attempt,
        "recovery": recovered,
        "replacement_worker": "replacement-worker",
        "resumed_attempt": resumed.attempt,
        "final_task_counts": store.task_counts(campaign_id),
        "database_integrity": store.integrity_check(),
        "demonstration_pass": (
            recovered == {"recovered": 1, "failed": 0}
            and resumed.attempt == 2
            and store.task_counts(campaign_id) == {"succeeded": 1}
            and store.integrity_check() == "ok"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["demonstration_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
