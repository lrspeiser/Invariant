"""Fail-closed admission of independently reproduced bounded level-5 successes.

The discovery campaign's local process counter is useful diagnostic evidence, but it is not an
open-problem authorization.  This module recomputes the bounded-success criteria, binds downloaded
multi-host and Lean evidence, requires authenticated LLM participation, and admits a success only
after a distinct principal with a named-human-reviewed Ed25519 key signs the exact evidence payload.
An empty signer registry is therefore a normal blocked state, never an implicit pass.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .external_creativity_multi_host import validate_receipt as validate_multi_host_receipt
from .external_creativity_validation import validate_receipt as validate_campaign_receipt
from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/level5_success_admission.json"
REGISTRY_PATH = "configs/level5_success_signers.json"
OUTPUT_PATH = "runs/math/level5-success-admission/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/level5_success_admission.py"
TEST_PATH = "tests/test_level5_success_admission.py"
SCHEMA_VERSION = "invariant-level5-success-admission-receipt-1.0"
CONFIG_SCHEMA = "invariant-level5-success-admission-config-1.0"
REGISTRY_SCHEMA = "invariant-level5-success-signer-registry-1.0"
CERTIFICATE_SCHEMA = "invariant-level5-success-certificate-1.0"
PAYLOAD_SCHEMA = "invariant-level5-success-payload-1.0"
SIGNING_NAMESPACE = "invariant-level5-success-v1"
GENERATOR_PRINCIPAL_ID = "invariant.discovery-engine"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PRINCIPAL = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}\Z")


class Level5SuccessAdmissionError(ValueError):
    """Level-5 evidence or authorization failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Level5SuccessAdmissionError(f"{label} keys changed")


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Level5SuccessAdmissionError(f"{label} path escaped project root") from error
    if not path.is_file():
        raise Level5SuccessAdmissionError(f"{label} is missing: {relative}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Level5SuccessAdmissionError(f"{label} must be an object")
    return value


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _sealed(value: Mapping[str, Any], schema: str, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("schema_version") != schema or value.get("content_sha256") != canonical_sha256(
        body
    ):
        raise Level5SuccessAdmissionError(f"{label} seal changed")


def _load_config(root: Path) -> dict[str, Any]:
    value = _read_json(_under(root, CONFIG_PATH, "level-5 config"), "level-5 config")
    _strict(
        value,
        {
            "admission_certificates",
            "campaign_receipt_path",
            "minimum_independent_successes_before_open_problem",
            "multi_host_receipt_path",
            "required_process_criteria",
            "schema_version",
            "signer_registry_path",
        },
        "level-5 config",
    )
    expected_criteria = {
        "authenticated_llm_participation": True,
        "detached_external_admission_signature": True,
        "matched_candidate_count": True,
        "matched_evaluation_runtime_budget": True,
        "matched_grammar_depth": True,
        "matched_verifier_budget": True,
        "minimum_families_outperforming_random": 5,
        "minimum_unique_behaviors": 3,
        "minimum_unique_proof_mechanisms": 2,
        "sealed_holdout_loss": "0",
        "target_blind_chronology": True,
        "training_loss": "0",
        "two_independent_exact_implementations": True,
        "two_reproduction_machines": True,
    }
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["minimum_independent_successes_before_open_problem"] < 3
        or value["required_process_criteria"] != expected_criteria
        or not isinstance(value["admission_certificates"], list)
    ):
        raise Level5SuccessAdmissionError("level-5 admission policy weakened")
    for item in value["admission_certificates"]:
        _strict(item, {"benchmark_id", "certificate_path"}, "level-5 certificate binding")
        if not isinstance(item["benchmark_id"], str) or not item["benchmark_id"]:
            raise Level5SuccessAdmissionError("level-5 benchmark id changed")
    ids = [item["benchmark_id"] for item in value["admission_certificates"]]
    if len(ids) != len(set(ids)):
        raise Level5SuccessAdmissionError("duplicate level-5 certificate binding")
    return value


def _validate_public_key(public_key: str) -> None:
    parts = public_key.split()
    if len(parts) != 2 or parts[0] != "ssh-ed25519":
        raise Level5SuccessAdmissionError("level-5 signer key must be bare ssh-ed25519")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except (binascii.Error, ValueError) as error:
        raise Level5SuccessAdmissionError("level-5 signer key encoding changed") from error
    if len(blob) < 19 or blob[:15] != b"\x00\x00\x00\x0bssh-ed25519":
        raise Level5SuccessAdmissionError("level-5 signer key blob changed")


def validate_registry(value: Mapping[str, Any]) -> None:
    _strict(
        value,
        {"content_sha256", "namespace", "policy", "schema_version", "signers"},
        "level-5 signer registry",
    )
    _sealed(value, REGISTRY_SCHEMA, "level-5 signer registry")
    if value["namespace"] != SIGNING_NAMESPACE or value["policy"] != {
        "accepted_key_types": ["ssh-ed25519"],
        "minimum_distinct_signers_per_success": 1,
        "named_human_key_identity_review_required": True,
        "signer_must_differ_from_generator": True,
        "signer_must_differ_from_source_principal": True,
    }:
        raise Level5SuccessAdmissionError("level-5 signer registry policy changed")
    if not isinstance(value["signers"], list):
        raise Level5SuccessAdmissionError("level-5 signer registry changed")
    principals: set[str] = set()
    for signer in value["signers"]:
        _strict(
            signer,
            {"identity_review", "principal_id", "public_key", "public_key_sha256"},
            "level-5 signer",
        )
        principal = signer["principal_id"]
        review = signer["identity_review"]
        _strict(
            review,
            {
                "reviewed_utc",
                "reviewer_affiliation_or_independent",
                "reviewer_name",
                "signer_identity_source_uri",
                "status",
            },
            "level-5 signer identity review",
        )
        if (
            not isinstance(principal, str)
            or _PRINCIPAL.fullmatch(principal) is None
            or principal in principals
            or review["status"] != "COMPLETED_NAMED_HUMAN_KEY_IDENTITY_REVIEW"
            or not all(isinstance(review[key], str) and review[key] for key in review)
        ):
            raise Level5SuccessAdmissionError("level-5 signer identity review changed")
        _validate_public_key(signer["public_key"])
        if signer["public_key_sha256"] != hashlib.sha256(
            signer["public_key"].encode("ascii")
        ).hexdigest():
            raise Level5SuccessAdmissionError("level-5 signer key hash changed")
        principals.add(principal)


def _signature_text(value: Any) -> str:
    if not isinstance(value, str):
        raise Level5SuccessAdmissionError("level-5 signature changed")
    normalized = value.replace("\r\n", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    if (
        not normalized.startswith("-----BEGIN SSH SIGNATURE-----\n")
        or not normalized.endswith("-----END SSH SIGNATURE-----\n")
        or len(normalized) > 20_000
    ):
        raise Level5SuccessAdmissionError("level-5 signature armor changed")
    return normalized


def _blind_chronology_pass(campaign: Mapping[str, Any]) -> bool:
    rows = campaign.get("blind_chronology", [])
    if not isinstance(rows, list) or not rows:
        return False
    opened = [row for row in rows if row.get("event") == "sealed_targets_opened_after_proposal_and_critique"]
    sealed = [row for row in rows if row.get("event") == "proposal_roots_and_train_evidence_sealed"]
    if len(opened) != 1 or len(sealed) != 1:
        return False
    open_sequence = opened[0].get("sequence")
    seal_sequence = sealed[0].get("sequence")
    return (
        isinstance(open_sequence, int)
        and isinstance(seal_sequence, int)
        and seal_sequence < open_sequence
        and all(
            row.get("target_reads") == (0 if row.get("sequence", open_sequence) < open_sequence else 1)
            for row in rows
        )
    )


def _process_evidence(
    benchmark: Mapping[str, Any],
    campaign: Mapping[str, Any],
    multi_host: Mapping[str, Any],
    minimum_families: int,
) -> dict[str, Any]:
    ranked = benchmark.get("ranked_candidates", [])
    best = ranked[0] if isinstance(ranked, list) and ranked else {}
    policy = benchmark.get("matched_control_policy", {})
    family_metrics = benchmark.get("family_metrics", [])
    outperformed = sum(item.get("outperformed_random") is True for item in family_metrics)
    reproduction = multi_host.get("reproduction", {})
    criteria = {
        "authenticated_llm_participation": campaign.get("claims", {}).get(
            "claude_used_throughout"
        )
        is True,
        "capability_level_5": benchmark.get("capability_level") == 5,
        "independent_exact_implementation_match": benchmark.get(
            "independent_exact_reproduction", {}
        ).get("match")
        is True,
        "matched_candidate_count": policy.get("candidate_count_matched") is True,
        "matched_evaluation_runtime_budget": policy.get("evaluation_runtime_budget_matched")
        is True,
        "matched_grammar_depth": policy.get("grammar_depth_matched") is True,
        "matched_verifier_budget": policy.get("verifier_budget_matched") is True,
        "minimum_families_outperforming_random": outperformed >= minimum_families,
        "minimum_unique_behaviors": benchmark.get("unique_behaviors", 0) >= 3,
        "minimum_unique_proof_mechanisms": benchmark.get("unique_proof_mechanisms", 0) >= 2,
        "multi_host_campaign_binding": reproduction.get("campaign_content_sha256")
        == campaign.get("content_sha256"),
        "sealed_holdout_loss": best.get("holdout_loss") == "0",
        "target_blind_chronology": _blind_chronology_pass(campaign),
        "training_loss": best.get("train_loss") == "0",
        "two_independent_exact_implementations": reproduction.get(
            "independent_implementations_per_host", 0
        )
        >= 2,
        "two_reproduction_machines": reproduction.get("received_machines", 0) >= 2,
    }
    process_pass = all(criteria.values())
    source = benchmark.get("external_authorship", {})
    return {
        "benchmark_id": benchmark.get("benchmark_id"),
        "blind_id": benchmark.get("blind_id"),
        "criteria": criteria,
        "family_metrics_sha256": canonical_sha256(family_metrics),
        "local_campaign_process_pass_reported": benchmark.get(
            "bounded_unknown_process_pass"
        )
        is True,
        "outperformed_random_families": outperformed,
        "process_pass_before_external_signature": process_pass,
        "proposal_root_sha256": benchmark.get("proposal_root_sha256"),
        "source_principal_id": source.get("authoring_principal_id"),
        "target_commitment": benchmark.get("target_commitment_opened"),
    }


def _payload(
    campaign: Mapping[str, Any],
    multi_host: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PAYLOAD_SCHEMA,
        "benchmark_id": evidence["benchmark_id"],
        "blind_id": evidence["blind_id"],
        "campaign_content_sha256": campaign["content_sha256"],
        "campaign_id": campaign["campaign_id"],
        "core_llm_evidence_projection_sha256": multi_host["reproduction"][
            "core_llm_evidence_projection_sha256"
        ],
        "multi_host_receipt_content_sha256": multi_host["content_sha256"],
        "process_evidence_sha256": canonical_sha256(evidence),
        "source_principal_id": evidence["source_principal_id"],
        "target_commitment": evidence["target_commitment"],
        "workflow_run_id": multi_host["acquisition"]["workflow_run_id"],
    }


def _verify_certificate(
    certificate: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    ssh_keygen: str | Path | None = None,
) -> str:
    validate_registry(registry)
    _validate_payload(expected_payload)
    _strict(
        certificate,
        {
            "claims",
            "content_sha256",
            "payload",
            "payload_sha256",
            "schema_version",
            "signature",
            "signer",
        },
        "level-5 certificate",
    )
    _sealed(certificate, CERTIFICATE_SCHEMA, "level-5 certificate")
    if certificate["payload"] != expected_payload or certificate["payload_sha256"] != canonical_sha256(
        expected_payload
    ):
        raise Level5SuccessAdmissionError("level-5 certificate payload binding changed")
    if certificate["claims"] != {
        "bounded_rediscovery_only": True,
        "literature_novelty_established": False,
        "open_problem_solved": False,
    }:
        raise Level5SuccessAdmissionError("level-5 certificate claim boundary changed")
    signer_evidence = certificate["signer"]
    _strict(
        signer_evidence,
        {"principal_id", "registry_content_sha256"},
        "level-5 certificate signer",
    )
    principal = signer_evidence["principal_id"]
    matches = [item for item in registry["signers"] if item["principal_id"] == principal]
    if len(matches) != 1:
        raise Level5SuccessAdmissionError("level-5 signer is not uniquely registered")
    signer = matches[0]
    if (
        principal in {GENERATOR_PRINCIPAL_ID, expected_payload["source_principal_id"]}
        or signer_evidence["registry_content_sha256"] != registry["content_sha256"]
    ):
        raise Level5SuccessAdmissionError("level-5 signer is not independent")
    signature = _signature_text(certificate["signature"])
    executable = str(ssh_keygen) if ssh_keygen is not None else shutil.which("ssh-keygen")
    if not executable:
        raise Level5SuccessAdmissionError("OpenSSH ssh-keygen is unavailable")
    with tempfile.TemporaryDirectory(prefix="invariant-level5-") as directory:
        temporary = Path(directory)
        allowed = temporary / "allowed_signers"
        signature_path = temporary / "payload.sig"
        allowed.write_text(
            f'{principal} namespaces="{SIGNING_NAMESPACE}" {signer["public_key"]}\n',
            encoding="ascii",
        )
        signature_path.write_text(signature, encoding="ascii", newline="\n")
        completed = subprocess.run(
            [
                executable,
                "-Y",
                "verify",
                "-f",
                str(allowed),
                "-I",
                principal,
                "-n",
                SIGNING_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=canonical_json_bytes(expected_payload),
            capture_output=True,
            check=False,
            timeout=30,
        )
    if completed.returncode != 0:
        raise Level5SuccessAdmissionError("level-5 detached signature is invalid")
    return principal


def _validate_payload(value: Mapping[str, Any]) -> None:
    _strict(
        value,
        {
            "benchmark_id",
            "blind_id",
            "campaign_content_sha256",
            "campaign_id",
            "core_llm_evidence_projection_sha256",
            "multi_host_receipt_content_sha256",
            "process_evidence_sha256",
            "schema_version",
            "source_principal_id",
            "target_commitment",
            "workflow_run_id",
        },
        "level-5 signing payload",
    )
    if (
        value.get("schema_version") != PAYLOAD_SCHEMA
        or not all(
            isinstance(value.get(key), str) and value.get(key)
            for key in (
                "benchmark_id",
                "blind_id",
                "campaign_id",
                "source_principal_id",
            )
        )
        or any(
            _HEX64.fullmatch(str(value.get(key, ""))) is None
            for key in (
                "campaign_content_sha256",
                "core_llm_evidence_projection_sha256",
                "multi_host_receipt_content_sha256",
                "process_evidence_sha256",
                "target_commitment",
            )
        )
        or not isinstance(value.get("workflow_run_id"), int)
        or isinstance(value.get("workflow_run_id"), bool)
        or value.get("workflow_run_id", 0) <= 0
    ):
        raise Level5SuccessAdmissionError("level-5 signing payload changed")


def build_certificate(
    payload: Mapping[str, Any],
    registry: Mapping[str, Any],
    principal_id: str,
    signature: str | bytes,
    *,
    ssh_keygen: str | Path | None = None,
) -> dict[str, Any]:
    _validate_payload(payload)
    validate_registry(registry)
    if isinstance(signature, bytes):
        try:
            signature = signature.decode("ascii")
        except UnicodeDecodeError as error:
            raise Level5SuccessAdmissionError("level-5 signature is not ASCII") from error
    normalized = _signature_text(signature)
    body = {
        "claims": {
            "bounded_rediscovery_only": True,
            "literature_novelty_established": False,
            "open_problem_solved": False,
        },
        "payload": dict(payload),
        "payload_sha256": canonical_sha256(payload),
        "schema_version": CERTIFICATE_SCHEMA,
        "signature": normalized,
        "signer": {
            "principal_id": principal_id,
            "registry_content_sha256": registry["content_sha256"],
        },
    }
    certificate = {**body, "content_sha256": canonical_sha256(body)}
    _verify_certificate(
        certificate,
        payload,
        registry,
        ssh_keygen=ssh_keygen,
    )
    return certificate


def _load_sources(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    config = _load_config(root)
    campaign_path = _under(root, config["campaign_receipt_path"], "campaign receipt")
    multi_host_path = _under(root, config["multi_host_receipt_path"], "multi-host receipt")
    registry_path = _under(root, config["signer_registry_path"], "signer registry")
    campaign = _read_json(campaign_path, "campaign receipt")
    multi_host = _read_json(multi_host_path, "multi-host receipt")
    registry = _read_json(registry_path, "signer registry")
    validate_campaign_receipt(campaign, root)
    validate_multi_host_receipt(multi_host, root)
    validate_registry(registry)
    minimum_families = config["required_process_criteria"][
        "minimum_families_outperforming_random"
    ]
    process_rows = [
        _process_evidence(benchmark, campaign, multi_host, minimum_families)
        for benchmark in campaign["benchmarks"]
        if benchmark.get("capability_level") == 5
    ]
    return config, campaign, multi_host, registry, process_rows


def export_payload(root: Path, benchmark_id: str) -> dict[str, Any]:
    _, campaign, multi_host, _, process_rows = _load_sources(root)
    matches = [row for row in process_rows if row["benchmark_id"] == benchmark_id]
    if len(matches) != 1:
        raise Level5SuccessAdmissionError("level-5 benchmark is not uniquely available")
    if matches[0]["process_pass_before_external_signature"] is not True:
        raise Level5SuccessAdmissionError("failed level-5 process cannot be exported for signing")
    payload = _payload(campaign, multi_host, matches[0])
    _validate_payload(payload)
    return payload


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config, campaign, multi_host, registry, process_rows = _load_sources(root)
    process_by_id = {row["benchmark_id"]: row for row in process_rows}
    admitted: list[dict[str, Any]] = []
    for binding in config["admission_certificates"]:
        evidence = process_by_id.get(binding["benchmark_id"])
        if evidence is None or evidence["process_pass_before_external_signature"] is not True:
            raise Level5SuccessAdmissionError(
                "external certificate cannot promote a failed level-5 process"
            )
        certificate = _read_json(
            _under(root, binding["certificate_path"], "level-5 certificate"),
            "level-5 certificate",
        )
        payload = _payload(campaign, multi_host, evidence)
        principal = _verify_certificate(certificate, payload, registry)
        admitted.append(
            {
                "benchmark_id": evidence["benchmark_id"],
                "certificate_content_sha256": certificate["content_sha256"],
                "process_evidence_sha256": canonical_sha256(evidence),
                "signer_principal_id": principal,
                "target_commitment": evidence["target_commitment"],
            }
        )
    commitments = [row["target_commitment"] for row in admitted]
    benchmark_ids = [row["benchmark_id"] for row in admitted]
    if len(commitments) != len(set(commitments)) or len(benchmark_ids) != len(set(benchmark_ids)):
        raise Level5SuccessAdmissionError("level-5 admissions are not independent")
    required = config["minimum_independent_successes_before_open_problem"]
    authorized = len(admitted) >= required
    body = {
        "schema_version": SCHEMA_VERSION,
        "source_bindings": {
            "campaign_receipt": {
                "content_sha256": campaign["content_sha256"],
                "path": config["campaign_receipt_path"],
            },
            "config": {
                "path": CONFIG_PATH,
                "sha256": _normalized_file_sha256(root / CONFIG_PATH),
            },
            "multi_host_receipt": {
                "content_sha256": multi_host["content_sha256"],
                "path": config["multi_host_receipt_path"],
            },
            "module": {
                "path": SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / SOURCE_PATH),
            },
            "signer_registry": {
                "content_sha256": registry["content_sha256"],
                "path": config["signer_registry_path"],
            },
            "tests": {
                "path": TEST_PATH,
                "sha256": _normalized_file_sha256(root / TEST_PATH),
            },
        },
        "process_evidence": process_rows,
        "admitted_successes": admitted,
        "summary": {
            "admitted_independently_reproduced_level5_successes": len(admitted),
            "campaign_local_level5_process_passes": campaign["open_problem_gate"][
                "level5_process_passes"
            ],
            "capability_level5_benchmarks": len(process_rows),
            "minimum_required_before_open_problem": required,
            "process_passes_before_external_signature": sum(
                row["process_pass_before_external_signature"] is True for row in process_rows
            ),
            "status": (
                "PASS_THREE_INDEPENDENT_LEVEL5_SUCCESSES"
                if authorized
                else "BLOCKED_INSUFFICIENT_ADMITTED_LEVEL5_SUCCESSES"
            ),
        },
        "release_gate": {
            "open_problem_authorized": authorized,
            "success_count_source": "detached_signed_multi_host_level5_admission_ledger",
        },
        "claims": {
            "local_process_pass_is_admitted_success": False,
            "literature_novelty_established": False,
            "open_problem_solved": False,
        },
    }
    receipt = {**body, "content_sha256": canonical_sha256(body)}
    return receipt


def validate_receipt(value: Mapping[str, Any], root: Path) -> None:
    _strict(
        value,
        {
            "admitted_successes",
            "claims",
            "content_sha256",
            "process_evidence",
            "release_gate",
            "schema_version",
            "source_bindings",
            "summary",
        },
        "level-5 admission receipt",
    )
    _sealed(value, SCHEMA_VERSION, "level-5 admission receipt")
    expected = build_receipt(root)
    if value != expected:
        raise Level5SuccessAdmissionError("level-5 admission receipt does not replay")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "export-payload", "certify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--benchmark")
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--principal")
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--registry", type=Path, default=Path(REGISTRY_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.command == "build":
        receipt = build_receipt(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "validate":
        receipt = _read_json(output, "level-5 admission receipt")
        validate_receipt(receipt, root)
    elif args.command == "export-payload":
        if not args.benchmark:
            parser.error("export-payload requires --benchmark")
        payload = export_payload(root, args.benchmark)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(payload))
        print(
            json.dumps(
                {
                    "benchmark_id": payload["benchmark_id"],
                    "bytes": output.stat().st_size,
                    "payload_sha256": canonical_sha256(payload),
                    "status": "EXPORTED_LEVEL5_SIGNING_PAYLOAD",
                },
                sort_keys=True,
            )
        )
        return 0
    else:
        if not args.payload or not args.principal or not args.signature:
            parser.error("certify requires --payload, --principal, and --signature")
        payload_path = args.payload if args.payload.is_absolute() else root / args.payload
        signature_path = args.signature if args.signature.is_absolute() else root / args.signature
        registry_path = args.registry if args.registry.is_absolute() else root / args.registry
        payload = _read_json(payload_path, "level-5 signing payload")
        registry = _read_json(registry_path, "level-5 signer registry")
        certificate = build_certificate(
            payload,
            registry,
            args.principal,
            signature_path.read_bytes(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "content_sha256": certificate["content_sha256"],
                    "principal_id": args.principal,
                    "status": "PASS_LEVEL5_ADMISSION_SIGNATURE_VERIFIED",
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            {"content_sha256": receipt["content_sha256"], **receipt["summary"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
