"""Bind downloaded GitHub-hosted reproduction artifacts into a durable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .external_creativity_lean_bridge import validate_receipt as validate_lean_receipt
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/external_creativity_multi_host_artifacts.json"
OUTPUT_PATH = "runs/math/external-creativity-validation/multi-host-reproduction.json"
SCHEMA_VERSION = "invariant-external-creativity-multi-host-reproduction-1.0"
SOURCE_SCHEMA = "invariant-external-creativity-multi-host-source-1.0"
CAMPAIGN_SCHEMAS = frozenset(
    {
        "invariant-external-creativity-validation-result-1.0",
        "invariant-external-creativity-validation-result-1.1",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


class MultiHostReproductionError(ValueError):
    """The downloaded multi-host evidence failed closed."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MultiHostReproductionError(f"{label} keys changed")


def _load_source(root: Path) -> dict[str, Any]:
    value = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    _strict_keys(
        value,
        {
            "artifacts",
            "expected_campaign_content_sha256",
            "expected_lean_content_sha256",
            "head_sha",
            "repository",
            "schema_version",
            "workflow_run_id",
            "workflow_run_url",
        },
        "multi-host source",
    )
    if value["schema_version"] != SOURCE_SCHEMA or value["repository"] != "lrspeiser/Invariant":
        raise MultiHostReproductionError("multi-host source identity changed")
    if not _GIT_SHA.fullmatch(value["head_sha"]):
        raise MultiHostReproductionError("multi-host source commit is invalid")
    if any(
        _SHA256.fullmatch(value[key]) is None
        for key in ("expected_campaign_content_sha256", "expected_lean_content_sha256")
    ):
        raise MultiHostReproductionError("multi-host expected content seal is invalid")
    if not isinstance(value["workflow_run_id"], int) or value["workflow_run_id"] <= 0:
        raise MultiHostReproductionError("multi-host workflow run id is invalid")
    if not isinstance(value["artifacts"], list) or len(value["artifacts"]) < 3:
        raise MultiHostReproductionError("multi-host source has too few artifacts")
    return value


def _artifact_path(artifact_root: Path, row: Mapping[str, Any]) -> Path:
    path = (artifact_root / row["artifact_name"] / row["file_name"]).resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as error:
        raise MultiHostReproductionError("artifact path escapes acquisition root") from error
    return path


def build_receipt(root: Path, artifact_root: Path) -> dict[str, Any]:
    root = root.resolve()
    artifact_root = artifact_root.resolve()
    source = _load_source(root)
    hosts: list[dict[str, Any]] = []
    lean: dict[str, Any] | None = None
    for row in source["artifacts"]:
        _strict_keys(
            row,
            {
                "archive_digest",
                "artifact_id",
                "artifact_name",
                "file_name",
                "file_sha256",
                "job_id",
                "operating_system",
                "python_version",
                "runner_id",
                "runner_name",
            },
            "multi-host artifact",
        )
        path = _artifact_path(artifact_root, row)
        if not path.is_file() or _file_sha256(path) != row["file_sha256"]:
            raise MultiHostReproductionError(f"downloaded artifact hash changed: {row['artifact_name']}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if row["artifact_name"] == "external-creativity-lean":
            validate_lean_receipt(value)
            if value.get("content_sha256") != source["expected_lean_content_sha256"]:
                raise MultiHostReproductionError("Lean artifact content seal changed")
            lean = {
                "artifact_id": row["artifact_id"],
                "content_sha256": value["content_sha256"],
                "file_sha256": row["file_sha256"],
                "job_id": row["job_id"],
                "kernel_checked": (
                    value.get("status") == "PASS"
                    and value.get("claims", {}).get(
                        "known_formula_normal_forms_kernel_checked"
                    )
                    is True
                ),
                "runner_id": row["runner_id"],
            }
            continue
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if (
            value.get("schema_version") not in CAMPAIGN_SCHEMAS
            or value.get("content_sha256") != canonical_sha256(body)
            or value.get("content_sha256") != source["expected_campaign_content_sha256"]
        ):
            raise MultiHostReproductionError("campaign artifact semantic seal changed")
        reproduction = value.get("independent_reproduction", {})
        if reproduction.get("received_implementations", 0) < 2:
            raise MultiHostReproductionError("host artifact lacks two evaluator implementations")
        hosts.append(
            {
                "artifact_id": row["artifact_id"],
                "artifact_name": row["artifact_name"],
                "campaign_content_sha256": value["content_sha256"],
                "file_sha256": row["file_sha256"],
                "independent_implementations": reproduction["received_implementations"],
                "job_id": row["job_id"],
                "operating_system": row["operating_system"],
                "python_version": row["python_version"],
                "runner_id": row["runner_id"],
                "runner_name": row["runner_name"],
            }
        )
    if lean is None:
        raise MultiHostReproductionError("Lean artifact is missing")
    runner_ids = {item["runner_id"] for item in hosts}
    operating_systems = {item["operating_system"] for item in hosts}
    campaigns = {item["campaign_content_sha256"] for item in hosts}
    if len(hosts) < 4 or len(runner_ids) < 2 or len(operating_systems) < 2 or len(campaigns) != 1:
        raise MultiHostReproductionError("cross-host agreement policy failed")
    body = {
        "schema_version": SCHEMA_VERSION,
        "acquisition": {
            "artifact_bytes_downloaded_and_hashed": True,
            "archive_digests_bound": [row["archive_digest"] for row in source["artifacts"]],
            "source_config_sha256": _normalized_file_sha256(root / CONFIG_PATH),
            "workflow_run_id": source["workflow_run_id"],
            "workflow_run_url": source["workflow_run_url"],
        },
        "head_sha": source["head_sha"],
        "hosts": hosts,
        "lean": lean,
        "reproduction": {
            "campaign_content_sha256": next(iter(campaigns)),
            "distinct_operating_systems": sorted(operating_systems),
            "distinct_runner_ids": len(runner_ids),
            "independent_implementations_per_host": min(
                item["independent_implementations"] for item in hosts
            ),
            "machine_kind": "github_hosted_ephemeral_vm",
            "minimum_distinct_machines": 2,
            "received_machines": len(runner_ids),
            "status": "PASS_MULTI_HOST_REPRODUCTION",
        },
        "claim_boundary": {
            "hardware_serials_available": False,
            "host_distinction_basis": "distinct GitHub Actions runner IDs and job IDs",
            "novel_formula_established": False,
            "physical_bare_metal_identity_claimed": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body)
    return body


def validate_receipt(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise MultiHostReproductionError("multi-host receipt content seal changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise MultiHostReproductionError("multi-host receipt schema changed")
    reproduction = value.get("reproduction", {})
    hosts = value.get("hosts", [])
    if (
        reproduction.get("status") != "PASS_MULTI_HOST_REPRODUCTION"
        or reproduction.get("received_machines", 0) < 2
        or reproduction.get("independent_implementations_per_host", 0) < 2
        or len({item.get("runner_id") for item in hosts}) < 2
        or len({item.get("operating_system") for item in hosts}) < 2
        or value.get("lean", {}).get("kernel_checked") is not True
    ):
        raise MultiHostReproductionError("multi-host receipt policy changed")
    boundary = value.get("claim_boundary", {})
    if (
        boundary.get("novel_formula_established") is not False
        or boundary.get("physical_bare_metal_identity_claimed") is not False
    ):
        raise MultiHostReproductionError("multi-host claim boundary changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    receipt = build_receipt(args.root, args.artifact_root)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
