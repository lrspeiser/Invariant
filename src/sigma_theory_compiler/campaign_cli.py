from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from .campaign import CampaignStore
from .campaign_engine import CampaignEngine, initialize_campaign
from .campaign_report import build_campaign_report


def _duration(value: str) -> float:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    suffix = value[-1].casefold()
    if suffix in units:
        return float(value[:-1]) * units[suffix]
    return float(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sigma-campaign",
        description="Persistent falsification-first campaign controller for Sigma Gravity.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create and seed a durable campaign database")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--database", type=Path, required=True)
    init.add_argument("--source-queue", type=Path)

    run = sub.add_parser("run", help="Lease and execute restartable campaign tasks")
    run.add_argument("--database", type=Path, required=True)
    run.add_argument("--campaign-id")
    run.add_argument("--worker-id", default=f"{socket.gethostname()}-main")
    run.add_argument("--duration", type=_duration)
    run.add_argument("--max-tasks", type=int)
    run.add_argument("--follow", action="store_true", help="Keep polling for new work")
    run.add_argument("--poll-seconds", type=float, default=5.0)
    run.add_argument(
        "--task-types",
        help="Comma-separated allowlist so a worker pool only leases its assigned resource lane",
    )

    status = sub.add_parser("status", help="Show durable campaign state and budgets")
    status.add_argument("--database", type=Path, required=True)
    status.add_argument("--campaign-id")
    status.add_argument("--output", type=Path)

    for command in ("pause", "resume"):
        item = sub.add_parser(command, help=f"{command.title()} a campaign")
        item.add_argument("--database", type=Path, required=True)
        item.add_argument("--campaign-id")
        item.add_argument("--reason")

    report = sub.add_parser("report", help="Write campaign and candidate evidence reports")
    report.add_argument("--database", type=Path, required=True)
    report.add_argument("--campaign-id")
    report.add_argument("--output", type=Path, required=True)

    recover = sub.add_parser("recover", help="Recover expired worker leases")
    recover.add_argument("--database", type=Path, required=True)
    recover.add_argument("--campaign-id")

    proposal = sub.add_parser(
        "submit-proposal", help="Validate and enqueue a bounded LLM or human research proposal"
    )
    proposal.add_argument("--database", type=Path, required=True)
    proposal.add_argument("--campaign-id")
    proposal.add_argument("--file", type=Path, required=True)
    proposal.add_argument("--worker-id", default=f"{socket.gethostname()}-proposal")
    refresh = sub.add_parser("refresh-dossiers", help="Queue versioned evidence dossier rebuilds")
    refresh.add_argument("--database", type=Path, required=True)
    refresh.add_argument("--campaign-id")
    refresh.add_argument("--version", default="2")
    refresh.add_argument("--worker-id", default=f"{socket.gethostname()}-dossier")
    formal = sub.add_parser(
        "queue-formal-controls", help="Queue versioned formal known-answer controls"
    )
    formal.add_argument("--database", type=Path, required=True)
    formal.add_argument("--campaign-id")
    formal.add_argument("--version", default="1")
    formal.add_argument("--worker-id", default=f"{socket.gethostname()}-formal")
    field_contract = sub.add_parser(
        "enforce-field-contract",
        help="Fail-closed reclassification of legacy candidates using forbidden baryonic z",
    )
    field_contract.add_argument("--database", type=Path, required=True)
    field_contract.add_argument("--campaign-id")
    field_contract.add_argument("--version", default="1.0")
    field_contract.add_argument("--worker-id", default=f"{socket.gethostname()}-field-contract")
    llm = sub.add_parser(
        "configure-claude",
        help="Enable the metered no-tools Claude proposal adapter with aggregate budget controls",
    )
    llm.add_argument("--database", type=Path, required=True)
    llm.add_argument("--campaign-id")
    llm.add_argument("--total-budget-usd", type=float, required=True)
    llm.add_argument("--max-calls", type=int, required=True)
    llm.add_argument("--per-call-budget-usd", type=float, required=True)
    llm.add_argument("--model", default="sonnet")
    llm.add_argument("--effort", default="high")
    llm.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def _campaign_id(store: CampaignStore, value: str | None) -> str:
    return value or store.campaign()["campaign_id"]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = CampaignStore(args.database)
    if args.command == "init":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config_path_root = args.config.resolve().parent.parent
        project_root = Path(config.get("project_root", config_path_root))
        if not project_root.is_absolute():
            project_root = (config_path_root / project_root).resolve()
        config["project_root"] = str(project_root)
        output_root = Path(config.get("output_root", "runs/campaigns"))
        if not output_root.is_absolute():
            output_root = (project_root / output_root).resolve()
        config["output_root"] = str(output_root)
        source = args.source_queue or Path(config["seed"]["source_queue"])
        if not source.is_absolute():
            source = project_root / source
        campaign_id = initialize_campaign(store, config, source)
        print(f"campaign_id={campaign_id}")
        print(f"database={store.database}")
        print(json.dumps(store.status(campaign_id), indent=2, sort_keys=True))
        return 0

    store.initialize()
    campaign_id = _campaign_id(store, args.campaign_id)
    if args.command == "configure-claude":
        campaign = store.campaign(campaign_id)
        config = json.loads(campaign["config_json"])
        project_root = Path(config["project_root"])
        adapter = project_root / "scripts" / "claude_proposal_adapter.py"
        if not adapter.is_file():
            raise FileNotFoundError(adapter)
        if args.per_call_budget_usd > args.total_budget_usd:
            raise ValueError("per-call budget cannot exceed total budget")
        budget = store.configure_llm_budget(
            campaign_id,
            total_budget_usd=args.total_budget_usd,
            max_calls=args.max_calls,
        )
        llm_config = {
            "command": [
                sys.executable,
                str(adapter),
                "--model",
                args.model,
                "--effort",
                args.effort,
                "--max-budget-usd",
                str(args.per_call_budget_usd),
            ],
            "provider": "claude-code-oauth-or-api-key",
            "model": args.model,
            "per_call_budget_usd": args.per_call_budget_usd,
            "total_budget_usd": args.total_budget_usd,
            "max_calls": args.max_calls,
            "timeout_seconds": args.timeout_seconds,
            "role": "proposal and failure-analysis only; never a truth judge",
        }
        store.configure_llm_runtime(campaign_id, llm_config)
        print(json.dumps({"campaign_id": campaign_id, "llm": llm_config, "budget": budget}, indent=2))
        return 0
    if args.command == "run":
        allowed_task_types = (
            None
            if not args.task_types
            else {item.strip() for item in args.task_types.split(",") if item.strip()}
        )
        engine = CampaignEngine(
            store,
            campaign_id,
            args.worker_id,
            allowed_task_types=allowed_task_types,
        )
        result = engine.run(
            max_tasks=args.max_tasks,
            duration_seconds=args.duration,
            wait_for_work=args.follow or args.duration is not None,
            poll_seconds=args.poll_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        result = store.status(campaign_id)
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    if args.command == "pause":
        store.set_campaign_state(campaign_id, "paused", args.reason or "paused by user")
        print(f"campaign_id={campaign_id}\nstate=paused")
        return 0
    if args.command == "resume":
        store.set_campaign_state(campaign_id, "active", None)
        print(f"campaign_id={campaign_id}\nstate=active")
        return 0
    if args.command == "report":
        report = build_campaign_report(store, campaign_id, args.output)
        print(f"campaign_id={campaign_id}")
        print(f"json={report['report_files']['json']}")
        print(f"markdown={report['report_files']['markdown']}")
        return 0
    if args.command == "recover":
        result = store.recover_expired_leases(campaign_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "submit-proposal":
        proposal = json.loads(args.file.read_text(encoding="utf-8"))
        engine = CampaignEngine(store, campaign_id, args.worker_id)
        proposal_id, validation = engine.submit_proposal(proposal)
        print(f"proposal_id={proposal_id}")
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation["valid"] else 1
    if args.command == "refresh-dossiers":
        engine = CampaignEngine(store, campaign_id, args.worker_id)
        queued = engine.queue_dossier_refresh(args.version)
        print(f"campaign_id={campaign_id}")
        print(f"dossiers_queued={queued}")
        return 0
    if args.command == "queue-formal-controls":
        engine = CampaignEngine(store, campaign_id, args.worker_id)
        queued = engine.queue_formal_reference_controls(args.version)
        print(f"campaign_id={campaign_id}")
        print(f"formal_controls_queued={queued}")
        return 0
    if args.command == "enforce-field-contract":
        engine = CampaignEngine(store, campaign_id, args.worker_id)
        result = engine.enforce_covariant_field_contract(args.version)
        print(f"campaign_id={campaign_id}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
