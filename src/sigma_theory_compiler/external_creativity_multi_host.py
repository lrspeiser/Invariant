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
SCHEMA_VERSION = "invariant-external-creativity-multi-host-reproduction-1.2"
SOURCE_SCHEMA = "invariant-external-creativity-multi-host-source-1.1"
CAMPAIGN_SCHEMAS = frozenset(
    {
        "invariant-external-creativity-validation-result-1.0",
        "invariant-external-creativity-validation-result-1.1",
        "invariant-external-creativity-validation-result-1.2",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARCHIVE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_EXPECTED_ARTIFACT_NAMES = {
    "external-creativity-lean",
    "external-creativity-ubuntu-latest-3.11",
    "external-creativity-ubuntu-latest-3.12",
    "external-creativity-windows-latest-3.11",
    "external-creativity-windows-latest-3.12",
}


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
            "expected_core_projection_sha256",
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
        for key in (
            "expected_campaign_content_sha256",
            "expected_core_projection_sha256",
            "expected_lean_content_sha256",
        )
    ):
        raise MultiHostReproductionError("multi-host expected content seal is invalid")
    if not isinstance(value["workflow_run_id"], int) or value["workflow_run_id"] <= 0:
        raise MultiHostReproductionError("multi-host workflow run id is invalid")
    if not isinstance(value["artifacts"], list) or len(value["artifacts"]) < 3:
        raise MultiHostReproductionError("multi-host source has too few artifacts")
    artifact_ids: set[int] = set()
    artifact_names: set[str] = set()
    archive_digests: set[str] = set()
    job_ids: set[int] = set()
    runner_ids: set[int] = set()
    for row in value["artifacts"]:
        if not isinstance(row, Mapping):
            raise MultiHostReproductionError("multi-host artifact source changed")
        for key, seen in (
            ("artifact_id", artifact_ids),
            ("job_id", job_ids),
            ("runner_id", runner_ids),
        ):
            item = row.get(key)
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise MultiHostReproductionError(f"multi-host {key} changed")
            seen.add(item)
        artifact_name = row.get("artifact_name")
        archive_digest = row.get("archive_digest")
        if not isinstance(artifact_name, str) or not artifact_name:
            raise MultiHostReproductionError("multi-host artifact name changed")
        if not isinstance(archive_digest, str) or _ARCHIVE_DIGEST.fullmatch(archive_digest) is None:
            raise MultiHostReproductionError("multi-host archive digest changed")
        artifact_names.add(artifact_name)
        archive_digests.add(archive_digest)
    expected_count = len(value["artifacts"])
    if any(
        len(items) != expected_count
        for items in (
            artifact_ids,
            artifact_names,
            archive_digests,
            job_ids,
            runner_ids,
        )
    ):
        raise MultiHostReproductionError("multi-host artifact identity collapsed")
    if artifact_names != _EXPECTED_ARTIFACT_NAMES:
        raise MultiHostReproductionError("multi-host artifact topology changed")
    return value


def _artifact_path(
    artifact_root: Path, row: Mapping[str, Any], file_key: str = "file_name"
) -> Path:
    path = (artifact_root / row["artifact_name"] / row[file_key]).resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as error:
        raise MultiHostReproductionError("artifact path escapes acquisition root") from error
    return path


def build_receipt(root: Path, artifact_root: Path) -> dict[str, Any]:
    # Imported lazily because the core receipt validator itself binds this multi-host receipt.
    from .core_creative_ci_reproduction import validate_receipt as validate_core_ci_receipt

    root = root.resolve()
    artifact_root = artifact_root.resolve()
    source = _load_source(root)
    hosts: list[dict[str, Any]] = []
    lean: dict[str, Any] | None = None
    for row in source["artifacts"]:
        is_lean = row.get("artifact_name") == "external-creativity-lean"
        expected_keys = {
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
        }
        if not is_lean:
            expected_keys.update({"core_file_name", "core_file_sha256"})
        _strict_keys(
            row,
            expected_keys,
            "multi-host artifact",
        )
        path = _artifact_path(artifact_root, row)
        if not path.is_file() or _file_sha256(path) != row["file_sha256"]:
            raise MultiHostReproductionError(
                f"downloaded artifact hash changed: {row['artifact_name']}"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if is_lean:
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
                    and value.get("claims", {}).get("known_formula_normal_forms_kernel_checked")
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
        core_path = _artifact_path(artifact_root, row, "core_file_name")
        if not core_path.is_file() or _file_sha256(core_path) != row["core_file_sha256"]:
            raise MultiHostReproductionError(
                f"downloaded core artifact hash changed: {row['artifact_name']}"
            )
        core_value = json.loads(core_path.read_text(encoding="utf-8"))
        validate_core_ci_receipt(core_value, require_ci_provenance=True)
        provenance = core_value["ci_provenance"]
        if (
            provenance.get("run_id") != source["workflow_run_id"]
            or provenance.get("head_sha") != source["head_sha"]
            or provenance.get("artifact_name") != row["artifact_name"]
            or provenance.get("operating_system") != row["operating_system"]
            or provenance.get("python_version") != row["python_version"]
            or provenance.get("runner_name") != row["runner_name"]
            or provenance.get("event_name") != "push"
        ):
            raise MultiHostReproductionError("core CI artifact provenance changed")
        if (
            core_value["llm_evidence_projection_sha256"]
            != source["expected_core_projection_sha256"]
        ):
            raise MultiHostReproductionError("core LLM evidence projection changed")
        hosts.append(
            {
                "artifact_id": row["artifact_id"],
                "artifact_name": row["artifact_name"],
                "campaign_content_sha256": value["content_sha256"],
                "core_reproduction": {
                    "content_sha256": core_value["content_sha256"],
                    "file_sha256": row["core_file_sha256"],
                    "llm_evidence_projection_sha256": core_value["llm_evidence_projection_sha256"],
                    "new_provider_calls": core_value["verification"]["new_provider_calls"],
                    "provider_credential_available_on_reproduction_host": core_value[
                        "verification"
                    ]["provider_credential_available_on_reproduction_host"],
                    "status": core_value["verification"]["status"],
                },
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
    all_runner_ids = runner_ids | {lean["runner_id"]}
    all_artifact_ids = {item["artifact_id"] for item in hosts} | {lean["artifact_id"]}
    all_job_ids = {item["job_id"] for item in hosts} | {lean["job_id"]}
    operating_systems = {item["operating_system"] for item in hosts}
    campaigns = {item["campaign_content_sha256"] for item in hosts}
    core_projections = {
        item["core_reproduction"]["llm_evidence_projection_sha256"] for item in hosts
    }
    if (
        len(hosts) != 4
        or len(runner_ids) != 4
        or len(all_runner_ids) != 5
        or len(all_artifact_ids) != 5
        or len(all_job_ids) != 5
        or len(operating_systems) < 2
        or len(campaigns) != 1
        or core_projections != {source["expected_core_projection_sha256"]}
    ):
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
            "core_llm_evidence_projection_sha256": next(iter(core_projections)),
            "core_llm_evidence_reproductions": len(hosts),
            "core_new_provider_calls": sum(
                item["core_reproduction"]["new_provider_calls"] for item in hosts
            ),
            "core_reproduction_machines": len(hosts),
            "distinct_operating_systems": sorted(operating_systems),
            "distinct_runner_ids": len(all_runner_ids),
            "independent_implementations_per_host": min(
                item["independent_implementations"] for item in hosts
            ),
            "lean_kernel_machines": 1,
            "machine_kind": "github_hosted_ephemeral_vm",
            "minimum_distinct_machines": 2,
            "received_machines": len(all_runner_ids),
            "status": "PASS_MULTI_HOST_CORE_LLM_EVIDENCE_REPRODUCTION",
        },
        "claim_boundary": {
            "authenticated_live_llm_evidence_replayed": True,
            "hardware_serials_available": False,
            "host_distinction_basis": "five distinct GitHub Actions runner IDs and job IDs",
            "new_provider_calls_required": False,
            "novel_formula_established": False,
            "physical_bare_metal_identity_claimed": False,
            "provider_credential_present_on_reproduction_hosts": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root)
    return body


def validate_receipt(value: Mapping[str, Any], root: Path | None = None) -> None:
    _strict_keys(
        value,
        {
            "acquisition",
            "claim_boundary",
            "content_sha256",
            "head_sha",
            "hosts",
            "lean",
            "reproduction",
            "schema_version",
        },
        "multi-host receipt",
    )
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise MultiHostReproductionError("multi-host receipt content seal changed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise MultiHostReproductionError("multi-host receipt schema changed")
    reproduction = value.get("reproduction", {})
    _strict_keys(
        reproduction,
        {
            "campaign_content_sha256",
            "core_llm_evidence_projection_sha256",
            "core_llm_evidence_reproductions",
            "core_new_provider_calls",
            "core_reproduction_machines",
            "distinct_operating_systems",
            "distinct_runner_ids",
            "independent_implementations_per_host",
            "lean_kernel_machines",
            "machine_kind",
            "minimum_distinct_machines",
            "received_machines",
            "status",
        },
        "multi-host reproduction",
    )
    hosts = value.get("hosts", [])
    for host in hosts:
        _strict_keys(
            host,
            {
                "artifact_id",
                "artifact_name",
                "campaign_content_sha256",
                "core_reproduction",
                "file_sha256",
                "independent_implementations",
                "job_id",
                "operating_system",
                "python_version",
                "runner_id",
                "runner_name",
            },
            "multi-host host",
        )
        _strict_keys(
            host["core_reproduction"],
            {
                "content_sha256",
                "file_sha256",
                "llm_evidence_projection_sha256",
                "new_provider_calls",
                "provider_credential_available_on_reproduction_host",
                "status",
            },
            "multi-host core reproduction",
        )
    _strict_keys(
        value.get("lean", {}),
        {
            "artifact_id",
            "content_sha256",
            "file_sha256",
            "job_id",
            "kernel_checked",
            "runner_id",
        },
        "multi-host Lean reproduction",
    )
    core_reproductions = [item.get("core_reproduction", {}) for item in hosts]
    core_projections = {item.get("llm_evidence_projection_sha256") for item in core_reproductions}
    host_runner_ids = {item.get("runner_id") for item in hosts}
    all_runner_ids = host_runner_ids | {value.get("lean", {}).get("runner_id")}
    all_artifact_ids = {item.get("artifact_id") for item in hosts} | {
        value.get("lean", {}).get("artifact_id")
    }
    all_job_ids = {item.get("job_id") for item in hosts} | {value.get("lean", {}).get("job_id")}
    archive_digests = value.get("acquisition", {}).get("archive_digests_bound", [])
    _strict_keys(
        value.get("acquisition", {}),
        {
            "archive_digests_bound",
            "artifact_bytes_downloaded_and_hashed",
            "source_config_sha256",
            "workflow_run_id",
            "workflow_run_url",
        },
        "multi-host acquisition",
    )
    if (
        reproduction.get("status") != "PASS_MULTI_HOST_CORE_LLM_EVIDENCE_REPRODUCTION"
        or reproduction.get("received_machines", 0) < 2
        or reproduction.get("received_machines") != len(hosts) + 1
        or reproduction.get("core_reproduction_machines") != len(hosts)
        or reproduction.get("lean_kernel_machines") != 1
        or reproduction.get("core_llm_evidence_reproductions") != len(hosts)
        or reproduction.get("core_new_provider_calls") != 0
        or reproduction.get("independent_implementations_per_host", 0) < 2
        or len(hosts) != 4
        or len(host_runner_ids) != 4
        or len(all_runner_ids) != 5
        or len(all_artifact_ids) != 5
        or len(all_job_ids) != 5
        or reproduction.get("distinct_runner_ids") != len(all_runner_ids)
        or not isinstance(archive_digests, list)
        or len(archive_digests) != 5
        or len(set(archive_digests)) != 5
        or any(
            not isinstance(item, str) or _ARCHIVE_DIGEST.fullmatch(item) is None
            for item in archive_digests
        )
        or len({item.get("operating_system") for item in hosts}) < 2
        or len(core_projections) != 1
        or reproduction.get("core_llm_evidence_projection_sha256")
        != next(iter(core_projections), None)
        or any(
            item.get("status") != "PASS_CORE_LLM_EVIDENCE_REPRODUCTION"
            or item.get("new_provider_calls") != 0
            or item.get("provider_credential_available_on_reproduction_host") is not False
            or _SHA256.fullmatch(str(item.get("content_sha256", ""))) is None
            or _SHA256.fullmatch(str(item.get("file_sha256", ""))) is None
            or _SHA256.fullmatch(str(item.get("llm_evidence_projection_sha256", ""))) is None
            for item in core_reproductions
        )
        or value.get("lean", {}).get("kernel_checked") is not True
        or reproduction.get("machine_kind") != "github_hosted_ephemeral_vm"
        or reproduction.get("minimum_distinct_machines") != 2
    ):
        raise MultiHostReproductionError("multi-host receipt policy changed")
    boundary = value.get("claim_boundary", {})
    _strict_keys(
        boundary,
        {
            "authenticated_live_llm_evidence_replayed",
            "hardware_serials_available",
            "host_distinction_basis",
            "new_provider_calls_required",
            "novel_formula_established",
            "physical_bare_metal_identity_claimed",
            "provider_credential_present_on_reproduction_hosts",
        },
        "multi-host claim boundary",
    )
    if (
        boundary.get("authenticated_live_llm_evidence_replayed") is not True
        or boundary.get("new_provider_calls_required") is not False
        or boundary.get("provider_credential_present_on_reproduction_hosts") is not False
        or boundary.get("novel_formula_established") is not False
        or boundary.get("physical_bare_metal_identity_claimed") is not False
        or boundary.get("host_distinction_basis")
        != "five distinct GitHub Actions runner IDs and job IDs"
    ):
        raise MultiHostReproductionError("multi-host claim boundary changed")
    if root is None:
        return
    root = root.resolve()
    source = _load_source(root)
    expected_acquisition = {
        "archive_digests_bound": [row["archive_digest"] for row in source["artifacts"]],
        "artifact_bytes_downloaded_and_hashed": True,
        "source_config_sha256": _normalized_file_sha256(root / CONFIG_PATH),
        "workflow_run_id": source["workflow_run_id"],
        "workflow_run_url": source["workflow_run_url"],
    }
    if value["acquisition"] != expected_acquisition or value["head_sha"] != source["head_sha"]:
        raise MultiHostReproductionError("multi-host authoritative source binding changed")
    expected_hosts = {
        row["artifact_name"]: row
        for row in source["artifacts"]
        if row["artifact_name"] != "external-creativity-lean"
    }
    if set(expected_hosts) != {host["artifact_name"] for host in hosts}:
        raise MultiHostReproductionError("multi-host source artifact coverage changed")
    for host in hosts:
        expected = expected_hosts[host["artifact_name"]]
        if (
            any(
                host[key] != expected[key]
                for key in (
                    "artifact_id",
                    "artifact_name",
                    "file_sha256",
                    "job_id",
                    "operating_system",
                    "python_version",
                    "runner_id",
                    "runner_name",
                )
            )
            or host["core_reproduction"]["file_sha256"] != expected["core_file_sha256"]
            or host["campaign_content_sha256"] != source["expected_campaign_content_sha256"]
            or host["core_reproduction"]["llm_evidence_projection_sha256"]
            != source["expected_core_projection_sha256"]
        ):
            raise MultiHostReproductionError("multi-host source host binding changed")
    lean_source = next(
        row for row in source["artifacts"] if row["artifact_name"] == "external-creativity-lean"
    )
    lean = value["lean"]
    if (
        any(
            lean[key] != lean_source[key]
            for key in ("artifact_id", "file_sha256", "job_id", "runner_id")
        )
        or lean["content_sha256"] != source["expected_lean_content_sha256"]
    ):
        raise MultiHostReproductionError("multi-host source Lean binding changed")


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
