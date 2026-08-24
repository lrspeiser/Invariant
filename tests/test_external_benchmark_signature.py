from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from sigma_theory_compiler.external_benchmark_signature import (
    GENERATOR_PRINCIPAL_ID,
    SIGNING_NAMESPACE,
    ExternalBenchmarkSignatureError,
    _seal,
    build_certificate,
    load_registry,
    load_stored_request,
    validate_certificate,
    validate_registry,
    validate_request,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _reviewed_registry(public_key: str, principal_id: str = "external.test-reviewer") -> dict:
    signer = {
        "identity_review": {
            "reviewed_utc": "2026-08-24T12:00:00Z",
            "reviewer_affiliation_or_independent": "independent test fixture",
            "reviewer_name": "Test Registry Reviewer",
            "signer_identity_source_uri": "https://example.test/keys/external-test-reviewer",
            "status": "COMPLETED_NAMED_HUMAN_KEY_IDENTITY_REVIEW",
        },
        "principal_id": principal_id,
        "public_key": public_key,
        "public_key_sha256": hashlib.sha256(public_key.encode()).hexdigest(),
    }
    return _seal(
        {
            "namespace": SIGNING_NAMESPACE,
            "policy": {
                "accepted_key_types": ["ssh-ed25519"],
                "minimum_distinct_signers": 1,
                "named_human_key_identity_review_required": True,
                "signer_must_differ_from_generator": True,
                "signer_must_differ_from_source_principals": True,
            },
            "schema_version": "invariant-external-benchmark-signer-registry-1.0",
            "signers": [signer],
        }
    )


def _generate_key_and_signature(
    tmp_path: Path,
    payload: bytes,
    namespace: str = SIGNING_NAMESPACE,
    key_path: Path | None = None,
) -> tuple[str, bytes, Path]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is unavailable")
    tmp_path.mkdir(parents=True, exist_ok=True)
    if key_path is None:
        key_path = tmp_path / "signing-key"
        subprocess.run(
            [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
            check=True,
            capture_output=True,
        )
    public_key = " ".join(Path(f"{key_path}.pub").read_text(encoding="ascii").split()[:2])
    payload_path = tmp_path / f"payload-{namespace.replace('.', '-')}"
    payload_path.write_bytes(payload)
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-f",
            str(key_path),
            "-n",
            namespace,
            str(payload_path),
        ],
        check=True,
        capture_output=True,
    )
    return public_key, Path(f"{payload_path}.sig").read_bytes(), key_path


def test_committed_registry_and_request_are_fail_closed() -> None:
    registry = load_registry(ROOT)
    generation, targets, receipt, request = load_stored_request(ROOT)

    assert registry["signers"] == []
    assert request["release_gate"] == {
        "signature_received": False,
        "source_provenance_eligible_for_level5": False,
        "status": "BLOCKED_DISTINCT_PRINCIPAL_SIGNATURE_REQUIRED",
    }
    assert request["claims"] == {
        "signature_establishes_level5_success": False,
        "signature_establishes_mathematical_correctness": False,
        "signature_establishes_novelty": False,
    }
    assert request["signer_policy"]["disallowed_principal_ids"] == [
        "external.sympy-project",
        GENERATOR_PRINCIPAL_ID,
    ]
    validate_registry(registry)
    validate_request(request, generation, targets, receipt, ROOT)


def test_request_substitution_fails_even_when_resealed() -> None:
    generation, targets, receipt, request = load_stored_request(ROOT)
    changed = copy.deepcopy(request)
    changed["payload"]["sources"][0]["source_response_sha256"] = "0" * 64
    changed["payload_sha256"] = canonical_sha256(changed["payload"])
    changed = _seal({key: value for key, value in changed.items() if key != "content_sha256"})

    with pytest.raises(ExternalBenchmarkSignatureError):
        validate_request(changed, generation, targets, receipt, ROOT)


def test_empty_registry_cannot_issue_a_certificate() -> None:
    registry = load_registry(ROOT)
    _, _, _, request = load_stored_request(ROOT)

    with pytest.raises(ExternalBenchmarkSignatureError, match="not uniquely registered"):
        build_certificate(
            request,
            registry,
            "external.test-reviewer",
            "-----BEGIN SSH SIGNATURE-----\n-----END SSH SIGNATURE-----\n",
        )


def test_openssh_signature_round_trip_and_tamper_boundaries(tmp_path: Path) -> None:
    _, _, _, request = load_stored_request(ROOT)
    public_key, signature, key_path = _generate_key_and_signature(
        tmp_path, canonical_json_bytes(request["payload"])
    )
    registry = _reviewed_registry(public_key)

    certificate = build_certificate(request, registry, "external.test-reviewer", signature)
    validate_certificate(certificate, request, registry)

    assert certificate["release_gate"] == {
        "distinct_principal_verified": True,
        "pack_level5_success_authorized": False,
        "source_provenance_eligible_for_level5": True,
        "status": "PASS_DISTINCT_PRINCIPAL_SOURCE_SIGNATURE",
    }
    assert not any(certificate["claims"].values())

    _, wrong_signature, _ = _generate_key_and_signature(
        tmp_path / "wrong", b"wrong payload", key_path=key_path
    )
    with pytest.raises(ExternalBenchmarkSignatureError, match="signature is invalid"):
        build_certificate(request, registry, "external.test-reviewer", wrong_signature)

    for disallowed_principal in ("external.sympy-project", GENERATOR_PRINCIPAL_ID):
        disallowed_registry = _reviewed_registry(public_key, disallowed_principal)
        with pytest.raises(ExternalBenchmarkSignatureError, match="not independent"):
            build_certificate(request, disallowed_registry, disallowed_principal, signature)

    changed_registry = copy.deepcopy(registry)
    changed_registry["signers"][0]["identity_review"]["reviewer_name"] = "Different Reviewer"
    changed_registry = _seal(
        {key: value for key, value in changed_registry.items() if key != "content_sha256"}
    )
    with pytest.raises(ExternalBenchmarkSignatureError):
        validate_certificate(certificate, request, changed_registry)


def test_invalid_ed25519_blob_is_rejected() -> None:
    invalid = _reviewed_registry("ssh-ed25519 YWJj")
    with pytest.raises(ExternalBenchmarkSignatureError, match="not an Ed25519 key blob"):
        validate_registry(invalid)
