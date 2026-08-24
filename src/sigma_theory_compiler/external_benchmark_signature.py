"""Create and verify distinct-principal signatures for external structured benchmarks.

The unsigned coordinator receipt binds exact upstream HTTPS bytes, but HTTPS is not a detached
attestation by a benchmark principal.  This module emits the exact canonical payload that an
approved external signer must sign and verifies OpenSSH ``SSHSIG`` signatures without network
access.  A signer is accepted only after its public key has a named-human identity review in the
trusted registry and differs from both the generator and every source principal.

A valid certificate establishes source-pack provenance only.  It never establishes novelty,
mathematical correctness, a level-5 success, or permission to attempt a famous open problem.
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
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .external_structured_benchmarks import (
    GENERATION_PATH,
    RECEIPT_PATH,
    TARGET_PATH,
    load_stored_pack,
    validate_pack,
)
from .external_structured_benchmarks import SOURCE_PATH as PACK_SOURCE_PATH
from .sigma_core import canonical_json_bytes, canonical_sha256

SOURCE_PATH = "src/sigma_theory_compiler/external_benchmark_signature.py"
REGISTRY_PATH = "configs/external_benchmark_signers.json"
REQUEST_PATH = "runs/math/external-structured-benchmarks/2026-08-23-002-signature-request.json"
REGISTRY_SCHEMA = "invariant-external-benchmark-signer-registry-1.0"
REQUEST_SCHEMA = "invariant-external-benchmark-signature-request-1.0"
CERTIFICATE_SCHEMA = "invariant-external-benchmark-signature-certificate-1.0"
SIGNING_NAMESPACE = "invariant-external-structured-benchmark-v1"
GENERATOR_PRINCIPAL_ID = "invariant.discovery-engine"
_KEY_TYPE = "ssh-ed25519"
_HEX = frozenset("0123456789abcdef")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class ExternalBenchmarkSignatureError(ValueError):
    """A signer registry, request, detached signature, or certificate failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ExternalBenchmarkSignatureError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ExternalBenchmarkSignatureError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ExternalBenchmarkSignatureError(f"{label} is not a stable identifier")
    return value


def _text(value: Any, label: str, *, maximum: int = 500) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
    ):
        raise ExternalBenchmarkSignatureError(f"{label} is missing or malformed")
    return value


def _utc(value: Any, label: str) -> str:
    text = _text(value, label, maximum=100)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ExternalBenchmarkSignatureError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExternalBenchmarkSignatureError(f"{label} lacks a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _under(root: Path, value: str | Path, label: str) -> Path:
    root = root.resolve()
    candidate = value if isinstance(value, Path) else Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ExternalBenchmarkSignatureError(f"{label} escaped the repository root") from error
    return candidate


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalBenchmarkSignatureError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ExternalBenchmarkSignatureError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    value["content_sha256"] = canonical_sha256(value)
    return value


def _validate_seal(value: Mapping[str, Any], schema: str, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("schema_version") != schema or value.get("content_sha256") != canonical_sha256(
        body
    ):
        raise ExternalBenchmarkSignatureError(f"{label} identity or content seal changed")


def _public_key(value: Any) -> str:
    text = _text(value, "registered public key", maximum=1000)
    parts = text.split()
    if len(parts) != 2 or parts[0] != _KEY_TYPE or text != " ".join(parts):
        raise ExternalBenchmarkSignatureError(
            "registered key must be canonical ssh-ed25519 public key"
        )
    try:
        decoded = base64.b64decode(parts[1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ExternalBenchmarkSignatureError("registered public key encoding changed") from error
    prefix = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"
    if len(decoded) != len(prefix) + 32 or not decoded.startswith(prefix):
        raise ExternalBenchmarkSignatureError("registered public key is not an Ed25519 key blob")
    return text


def _registry_policy() -> dict[str, Any]:
    return {
        "accepted_key_types": [_KEY_TYPE],
        "minimum_distinct_signers": 1,
        "named_human_key_identity_review_required": True,
        "signer_must_differ_from_generator": True,
        "signer_must_differ_from_source_principals": True,
    }


def validate_registry(value: Mapping[str, Any]) -> None:
    _validate_seal(value, REGISTRY_SCHEMA, "external benchmark signer registry")
    _strict(
        value,
        {"content_sha256", "namespace", "policy", "schema_version", "signers"},
        "external benchmark signer registry",
    )
    if value.get("namespace") != SIGNING_NAMESPACE or value.get("policy") != _registry_policy():
        raise ExternalBenchmarkSignatureError("external benchmark signer policy changed")
    signers = value.get("signers")
    if not isinstance(signers, list):
        raise ExternalBenchmarkSignatureError("external benchmark signer registry changed")
    principal_ids: set[str] = set()
    public_key_hashes: set[str] = set()
    for signer in signers:
        _strict(
            signer,
            {"identity_review", "principal_id", "public_key", "public_key_sha256"},
            "external benchmark signer",
        )
        principal_id = _identifier(signer["principal_id"], "external signer principal")
        public_key = _public_key(signer["public_key"])
        key_sha256 = hashlib.sha256(public_key.encode()).hexdigest()
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
            "external signer identity review",
        )
        source_uri = urllib.parse.urlparse(
            _text(review["signer_identity_source_uri"], "signer identity source URI")
        )
        if (
            signer["public_key_sha256"] != key_sha256
            or principal_id in principal_ids
            or key_sha256 in public_key_hashes
            or review["status"] != "COMPLETED_NAMED_HUMAN_KEY_IDENTITY_REVIEW"
            or review["reviewer_name"].casefold() == "pending"
            or review["reviewer_affiliation_or_independent"].casefold() == "pending"
            or _utc(review["reviewed_utc"], "signer identity review time") != review["reviewed_utc"]
            or source_uri.scheme != "https"
            or not source_uri.hostname
        ):
            raise ExternalBenchmarkSignatureError("external signer identity review changed")
        principal_ids.add(principal_id)
        public_key_hashes.add(key_sha256)


def load_registry(root: Path, path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = _read_json(
        _under(root, path, "external signer registry"), "external signer registry"
    )
    validate_registry(registry)
    return registry


def _signer_policy(source_principals: Sequence[str]) -> dict[str, Any]:
    return {
        "accepted_key_types": [_KEY_TYPE],
        "disallowed_principal_ids": sorted({GENERATOR_PRINCIPAL_ID, *source_principals}),
        "minimum_distinct_signers": 1,
        "named_human_key_identity_review_required": True,
        "trusted_registry_path": REGISTRY_PATH,
    }


def _payload(
    generation: Mapping[str, Any],
    targets: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    sources = [
        {
            "external_principal_id": item["external_principal_id"],
            "source_id": item["source_id"],
            "source_response_sha256": item["source_response_sha256"],
            "source_uri": item["source_uri"],
        }
        for item in sorted(receipt["source_evidence"], key=lambda row: row["source_id"])
    ]
    return {
        "coordinator_receipt_content_sha256": receipt["content_sha256"],
        "generation_packet_content_sha256": generation["content_sha256"],
        "generator_principal_id": GENERATOR_PRINCIPAL_ID,
        "pack_id": receipt["pack_id"],
        "rotation_epoch": receipt["rotation_epoch"],
        "signing_namespace": SIGNING_NAMESPACE,
        "sources": sources,
        "target_packet_content_sha256": targets["content_sha256"],
    }


def build_request(
    root: Path,
    generation: Mapping[str, Any],
    targets: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    root = root.resolve()
    validate_pack(generation, targets, receipt, root)
    payload = _payload(generation, targets, receipt)
    source_principals = sorted(
        {item["external_principal_id"] for item in receipt["source_evidence"]}
    )
    body = {
        "claims": {
            "signature_establishes_level5_success": False,
            "signature_establishes_mathematical_correctness": False,
            "signature_establishes_novelty": False,
        },
        "pack_id": receipt["pack_id"],
        "payload": payload,
        "payload_sha256": canonical_sha256(payload),
        "release_gate": {
            "signature_received": False,
            "source_provenance_eligible_for_level5": False,
            "status": "BLOCKED_DISTINCT_PRINCIPAL_SIGNATURE_REQUIRED",
        },
        "request_id": f"{receipt['pack_id']}.{receipt['rotation_epoch']}.source-signature",
        "rotation_epoch": receipt["rotation_epoch"],
        "schema_version": REQUEST_SCHEMA,
        "signature_contract": {
            "format": "openssh-sshsig-armored",
            "namespace": SIGNING_NAMESPACE,
            "payload_encoding": "sigma-core-canonical-json-utf8",
        },
        "signer_policy": _signer_policy(source_principals),
        "source_bindings": {
            "coordinator_receipt": {
                "content_sha256": receipt["content_sha256"],
                "path": RECEIPT_PATH,
            },
            "generation_packet": {
                "content_sha256": generation["content_sha256"],
                "path": GENERATION_PATH,
            },
            "pack_builder_source": {
                "path": PACK_SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / PACK_SOURCE_PATH),
            },
            "signature_protocol_source": {
                "path": SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / SOURCE_PATH),
            },
            "target_packet": {
                "content_sha256": targets["content_sha256"],
                "path": TARGET_PATH,
            },
        },
    }
    request = _seal(body)
    validate_request(request, generation, targets, receipt, root)
    return request


def validate_request(
    value: Mapping[str, Any],
    generation: Mapping[str, Any],
    targets: Mapping[str, Any],
    receipt: Mapping[str, Any],
    root: Path | None = None,
) -> None:
    validate_pack(generation, targets, receipt, root)
    _validate_seal(value, REQUEST_SCHEMA, "external benchmark signature request")
    _strict(
        value,
        {
            "claims",
            "content_sha256",
            "pack_id",
            "payload",
            "payload_sha256",
            "release_gate",
            "request_id",
            "rotation_epoch",
            "schema_version",
            "signature_contract",
            "signer_policy",
            "source_bindings",
        },
        "external benchmark signature request",
    )
    payload = _payload(generation, targets, receipt)
    source_principals = sorted(
        {item["external_principal_id"] for item in receipt["source_evidence"]}
    )
    if (
        value.get("pack_id") != receipt["pack_id"]
        or value.get("rotation_epoch") != receipt["rotation_epoch"]
        or value.get("request_id")
        != f"{receipt['pack_id']}.{receipt['rotation_epoch']}.source-signature"
        or value.get("payload") != payload
        or value.get("payload_sha256") != canonical_sha256(payload)
        or value.get("signature_contract")
        != {
            "format": "openssh-sshsig-armored",
            "namespace": SIGNING_NAMESPACE,
            "payload_encoding": "sigma-core-canonical-json-utf8",
        }
        or value.get("signer_policy") != _signer_policy(source_principals)
        or value.get("release_gate")
        != {
            "signature_received": False,
            "source_provenance_eligible_for_level5": False,
            "status": "BLOCKED_DISTINCT_PRINCIPAL_SIGNATURE_REQUIRED",
        }
        or value.get("claims")
        != {
            "signature_establishes_level5_success": False,
            "signature_establishes_mathematical_correctness": False,
            "signature_establishes_novelty": False,
        }
    ):
        raise ExternalBenchmarkSignatureError("external benchmark signing request changed")
    bindings = value.get("source_bindings", {})
    _strict(
        bindings,
        {
            "coordinator_receipt",
            "generation_packet",
            "pack_builder_source",
            "signature_protocol_source",
            "target_packet",
        },
        "external benchmark signing request source bindings",
    )
    expected_packet_bindings = {
        "coordinator_receipt": (RECEIPT_PATH, receipt["content_sha256"]),
        "generation_packet": (GENERATION_PATH, generation["content_sha256"]),
        "target_packet": (TARGET_PATH, targets["content_sha256"]),
    }
    for key, (path, content_sha256) in expected_packet_bindings.items():
        if bindings.get(key) != {"content_sha256": content_sha256, "path": path}:
            raise ExternalBenchmarkSignatureError("external benchmark packet binding changed")
    if root is not None:
        root = root.resolve()
        for key, path in (
            ("pack_builder_source", PACK_SOURCE_PATH),
            ("signature_protocol_source", SOURCE_PATH),
        ):
            if bindings.get(key) != {
                "path": path,
                "sha256": _normalized_file_sha256(root / path),
            }:
                raise ExternalBenchmarkSignatureError("external benchmark signer source changed")


def load_stored_request(
    root: Path, request_path: str | Path = REQUEST_PATH
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    generation, targets, receipt = load_stored_pack(root)
    request = _read_json(
        _under(root, request_path, "external benchmark signing request"),
        "external benchmark signing request",
    )
    validate_request(request, generation, targets, receipt, root)
    return generation, targets, receipt, request


def _signature_text(raw: bytes | str) -> str:
    try:
        text = raw.decode("ascii") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as error:
        raise ExternalBenchmarkSignatureError("detached signature is not ASCII armored") from error
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    if (
        not normalized.startswith("-----BEGIN SSH SIGNATURE-----\n")
        or not normalized.endswith("-----END SSH SIGNATURE-----\n")
        or len(normalized) > 20_000
    ):
        raise ExternalBenchmarkSignatureError("detached signature armor changed")
    return normalized


def _registered_signer(
    registry: Mapping[str, Any], principal_id: str, request: Mapping[str, Any]
) -> Mapping[str, Any]:
    validate_registry(registry)
    payload = request.get("payload")
    if not isinstance(payload, Mapping):
        raise ExternalBenchmarkSignatureError("external benchmark signing payload changed")
    sources = payload.get("sources")
    if not isinstance(sources, list) or any(not isinstance(item, Mapping) for item in sources):
        raise ExternalBenchmarkSignatureError("external benchmark signing sources changed")
    source_principals = [
        _identifier(item.get("external_principal_id"), "external source principal")
        for item in sources
    ]
    if request.get("signer_policy") != _signer_policy(source_principals):
        raise ExternalBenchmarkSignatureError("external benchmark signer policy changed")
    principal_id = _identifier(principal_id, "external signer principal")
    signers = [item for item in registry["signers"] if item["principal_id"] == principal_id]
    if len(signers) != 1:
        raise ExternalBenchmarkSignatureError("signature principal is not uniquely registered")
    if principal_id in request["signer_policy"]["disallowed_principal_ids"]:
        raise ExternalBenchmarkSignatureError("signature principal is not independent")
    return signers[0]


def _verify_sshsig(
    payload: Mapping[str, Any],
    principal_id: str,
    public_key: str,
    signature: str,
    *,
    ssh_keygen: str | Path | None = None,
) -> None:
    executable = str(ssh_keygen) if ssh_keygen is not None else shutil.which("ssh-keygen")
    if not executable:
        raise ExternalBenchmarkSignatureError("OpenSSH ssh-keygen is unavailable")
    with tempfile.TemporaryDirectory(prefix="invariant-benchmark-signature-") as directory:
        temporary = Path(directory)
        allowed_signers = temporary / "allowed_signers"
        signature_path = temporary / "payload.sig"
        allowed_signers.write_text(
            f'{principal_id} namespaces="{SIGNING_NAMESPACE}" {public_key}\n',
            encoding="ascii",
        )
        signature_path.write_text(signature, encoding="ascii", newline="\n")
        completed = subprocess.run(
            [
                executable,
                "-Y",
                "verify",
                "-f",
                str(allowed_signers),
                "-I",
                principal_id,
                "-n",
                SIGNING_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=canonical_json_bytes(payload),
            capture_output=True,
            check=False,
            timeout=30,
        )
    if completed.returncode != 0:
        raise ExternalBenchmarkSignatureError("detached external benchmark signature is invalid")


def build_certificate(
    request: Mapping[str, Any],
    registry: Mapping[str, Any],
    principal_id: str,
    signature: bytes | str,
    *,
    ssh_keygen: str | Path | None = None,
) -> dict[str, Any]:
    _validate_seal(request, REQUEST_SCHEMA, "external benchmark signature request")
    signer = _registered_signer(registry, principal_id, request)
    normalized_signature = _signature_text(signature)
    _verify_sshsig(
        request["payload"],
        principal_id,
        signer["public_key"],
        normalized_signature,
        ssh_keygen=ssh_keygen,
    )
    body = {
        "claims": {
            "signature_establishes_level5_success": False,
            "signature_establishes_mathematical_correctness": False,
            "signature_establishes_novelty": False,
        },
        "pack_id": request["pack_id"],
        "release_gate": {
            "distinct_principal_verified": True,
            "pack_level5_success_authorized": False,
            "source_provenance_eligible_for_level5": True,
            "status": "PASS_DISTINCT_PRINCIPAL_SOURCE_SIGNATURE",
        },
        "request_content_sha256": request["content_sha256"],
        "rotation_epoch": request["rotation_epoch"],
        "schema_version": CERTIFICATE_SCHEMA,
        "signature": {
            "armored": normalized_signature,
            "format": "openssh-sshsig-armored",
            "namespace": SIGNING_NAMESPACE,
            "sha256": hashlib.sha256(normalized_signature.encode("ascii")).hexdigest(),
        },
        "signer": {
            "identity_review_sha256": canonical_sha256(signer["identity_review"]),
            "principal_id": principal_id,
            "public_key_sha256": signer["public_key_sha256"],
            "registry_content_sha256": registry["content_sha256"],
        },
        "verification": {
            "cryptographic_signature_verified": True,
            "payload_sha256": request["payload_sha256"],
            "reverification_required_on_load": True,
            "status": "PASS_OPENSSH_SSHSIG_VERIFIED",
        },
    }
    certificate = _seal(body)
    validate_certificate(
        certificate,
        request,
        registry,
        ssh_keygen=ssh_keygen,
    )
    return certificate


def validate_certificate(
    value: Mapping[str, Any],
    request: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    ssh_keygen: str | Path | None = None,
) -> None:
    _validate_seal(request, REQUEST_SCHEMA, "external benchmark signature request")
    validate_registry(registry)
    _validate_seal(value, CERTIFICATE_SCHEMA, "external benchmark signature certificate")
    _strict(
        value,
        {
            "claims",
            "content_sha256",
            "pack_id",
            "release_gate",
            "request_content_sha256",
            "rotation_epoch",
            "schema_version",
            "signature",
            "signer",
            "verification",
        },
        "external benchmark signature certificate",
    )
    signature = value.get("signature", {})
    signer_evidence = value.get("signer", {})
    verification = value.get("verification", {})
    principal_id = signer_evidence.get("principal_id")
    signer = _registered_signer(registry, principal_id, request)
    normalized_signature = _signature_text(signature.get("armored", ""))
    if (
        value.get("pack_id") != request["pack_id"]
        or value.get("rotation_epoch") != request["rotation_epoch"]
        or value.get("request_content_sha256") != request["content_sha256"]
        or signature
        != {
            "armored": normalized_signature,
            "format": "openssh-sshsig-armored",
            "namespace": SIGNING_NAMESPACE,
            "sha256": hashlib.sha256(normalized_signature.encode("ascii")).hexdigest(),
        }
        or signer_evidence
        != {
            "identity_review_sha256": canonical_sha256(signer["identity_review"]),
            "principal_id": principal_id,
            "public_key_sha256": signer["public_key_sha256"],
            "registry_content_sha256": registry["content_sha256"],
        }
        or verification
        != {
            "cryptographic_signature_verified": True,
            "payload_sha256": request["payload_sha256"],
            "reverification_required_on_load": True,
            "status": "PASS_OPENSSH_SSHSIG_VERIFIED",
        }
        or value.get("release_gate")
        != {
            "distinct_principal_verified": True,
            "pack_level5_success_authorized": False,
            "source_provenance_eligible_for_level5": True,
            "status": "PASS_DISTINCT_PRINCIPAL_SOURCE_SIGNATURE",
        }
        or value.get("claims")
        != {
            "signature_establishes_level5_success": False,
            "signature_establishes_mathematical_correctness": False,
            "signature_establishes_novelty": False,
        }
    ):
        raise ExternalBenchmarkSignatureError("external benchmark certificate boundary changed")
    _verify_sshsig(
        request["payload"],
        principal_id,
        signer["public_key"],
        normalized_signature,
        ssh_keygen=ssh_keygen,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, default=Path.cwd())
    prepare.add_argument("--output", type=Path, default=Path(REQUEST_PATH))
    validate_request_parser = subparsers.add_parser("validate-request")
    validate_request_parser.add_argument("--root", type=Path, default=Path.cwd())
    validate_request_parser.add_argument("--request", type=Path, default=Path(REQUEST_PATH))
    export = subparsers.add_parser("export-payload")
    export.add_argument("--root", type=Path, default=Path.cwd())
    export.add_argument("--request", type=Path, default=Path(REQUEST_PATH))
    export.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, default=Path.cwd())
    verify.add_argument("--request", type=Path, default=Path(REQUEST_PATH))
    verify.add_argument("--registry", type=Path, default=Path(REGISTRY_PATH))
    verify.add_argument("--principal", required=True)
    verify.add_argument("--signature", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    validate_certificate_parser = subparsers.add_parser("validate-certificate")
    validate_certificate_parser.add_argument("--root", type=Path, default=Path.cwd())
    validate_certificate_parser.add_argument("--request", type=Path, default=Path(REQUEST_PATH))
    validate_certificate_parser.add_argument("--registry", type=Path, default=Path(REGISTRY_PATH))
    validate_certificate_parser.add_argument("--certificate", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    generation, targets, receipt = load_stored_pack(root)
    if args.command == "prepare":
        request = build_request(root, generation, targets, receipt)
        _write_json(_under(root, args.output, "signing request output"), request)
        result = {
            "content_sha256": request["content_sha256"],
            "payload_sha256": request["payload_sha256"],
            "status": request["release_gate"]["status"],
        }
    else:
        request = _read_json(
            _under(root, args.request, "external benchmark signing request"),
            "external benchmark signing request",
        )
        validate_request(request, generation, targets, receipt, root)
        if args.command == "validate-request":
            result = {
                "content_sha256": request["content_sha256"],
                "payload_sha256": request["payload_sha256"],
                "status": request["release_gate"]["status"],
            }
        elif args.command == "export-payload":
            output = _under(root, args.output, "signing payload output")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(canonical_json_bytes(request["payload"]))
            result = {
                "bytes": output.stat().st_size,
                "payload_sha256": request["payload_sha256"],
                "status": "EXPORTED_CANONICAL_SIGNING_PAYLOAD",
            }
        else:
            registry = load_registry(root, args.registry)
            if args.command == "verify":
                signature_path = _under(root, args.signature, "detached signature")
                certificate = build_certificate(
                    request,
                    registry,
                    args.principal,
                    signature_path.read_bytes(),
                )
                _write_json(_under(root, args.output, "signature certificate output"), certificate)
            else:
                certificate = _read_json(
                    _under(root, args.certificate, "signature certificate"),
                    "signature certificate",
                )
                validate_certificate(certificate, request, registry)
            result = {
                "content_sha256": certificate["content_sha256"],
                "principal_id": certificate["signer"]["principal_id"],
                "status": certificate["release_gate"]["status"],
            }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
