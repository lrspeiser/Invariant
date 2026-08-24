from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sigma_theory_compiler.level5_success_admission import (
    CERTIFICATE_SCHEMA,
    OUTPUT_PATH,
    PAYLOAD_SCHEMA,
    REGISTRY_SCHEMA,
    SIGNING_NAMESPACE,
    Level5SuccessAdmissionError,
    _authenticated_llm_evidence,
    _blind_chronology_pass,
    _verify_certificate,
    build_certificate,
    build_receipt,
    export_payload,
    validate_receipt,
    validate_registry,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _seal(body: dict) -> dict:
    return {**body, "content_sha256": canonical_sha256(body)}


def _key_and_signature(tmp_path: Path, payload: dict) -> tuple[str, str]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is unavailable")
    key = tmp_path / "level5-signing-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
        capture_output=True,
    )
    public_key = " ".join(Path(f"{key}.pub").read_text(encoding="ascii").split()[:2])
    payload_path = tmp_path / "payload.json"
    payload_path.write_bytes(canonical_json_bytes(payload))
    subprocess.run(
        [ssh_keygen, "-Y", "sign", "-f", str(key), "-n", SIGNING_NAMESPACE, str(payload_path)],
        check=True,
        capture_output=True,
    )
    signature = Path(f"{payload_path}.sig").read_text(encoding="ascii")
    return public_key, signature


def _registry(public_key: str, principal: str = "external.level5-reviewer") -> dict:
    body = {
        "namespace": SIGNING_NAMESPACE,
        "policy": {
            "accepted_key_types": ["ssh-ed25519"],
            "minimum_distinct_signers_per_success": 1,
            "named_human_key_identity_review_required": True,
            "signer_must_differ_from_generator": True,
            "signer_must_differ_from_source_principal": True,
        },
        "schema_version": REGISTRY_SCHEMA,
        "signers": [
            {
                "identity_review": {
                    "reviewed_utc": "2026-08-24T19:00:00Z",
                    "reviewer_affiliation_or_independent": "independent test fixture",
                    "reviewer_name": "Test Human Reviewer",
                    "signer_identity_source_uri": "https://example.test/level5-key",
                    "status": "COMPLETED_NAMED_HUMAN_KEY_IDENTITY_REVIEW",
                },
                "principal_id": principal,
                "public_key": public_key,
                "public_key_sha256": hashlib.sha256(public_key.encode("ascii")).hexdigest(),
            }
        ],
    }
    return _seal(body)


def _payload() -> dict:
    return {
        "schema_version": PAYLOAD_SCHEMA,
        "benchmark_id": "external.test.level5",
        "blind_id": "blind.test",
        "campaign_content_sha256": "1" * 64,
        "campaign_id": "test-campaign",
        "core_llm_evidence_projection_sha256": "2" * 64,
        "live_api_evidence_content_sha256": "9" * 64,
        "multi_host_receipt_content_sha256": "3" * 64,
        "process_evidence_sha256": "4" * 64,
        "source_principal_id": "external.test-source",
        "target_commitment": "5" * 64,
        "workflow_run_id": 12345,
    }


def _certificate(payload: dict, principal: str, signature: str, registry: dict) -> dict:
    return _seal(
        {
            "claims": {
                "bounded_rediscovery_only": True,
                "literature_novelty_established": False,
                "open_problem_solved": False,
            },
            "payload": payload,
            "payload_sha256": canonical_sha256(payload),
            "schema_version": CERTIFICATE_SCHEMA,
            "signature": signature,
            "signer": {
                "principal_id": principal,
                "registry_content_sha256": registry["content_sha256"],
            },
        }
    )


def test_committed_admission_ledger_replays_and_stays_zero() -> None:
    stored = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    rebuilt = build_receipt(ROOT)
    assert rebuilt == stored
    validate_receipt(stored, ROOT)
    summary = stored["summary"]
    assert summary["capability_level5_benchmarks"] == 2
    assert summary["authenticated_llm_benchmarks"] == 2
    assert summary["campaign_local_level5_process_passes"] == 0
    assert summary["process_passes_before_external_signature"] == 0
    assert summary["admitted_independently_reproduced_level5_successes"] == 0
    assert summary["minimum_required_before_open_problem"] == 3
    assert stored["release_gate"] == {
        "open_problem_authorized": False,
        "success_count_source": "detached_signed_multi_host_level5_admission_ledger",
    }
    assert all(
        row["criteria"]["two_reproduction_machines"] is True
        and row["criteria"]["two_independent_exact_implementations"] is True
        and row["criteria"]["authenticated_llm_participation"] is True
        and row["llm_evidence"]["authenticated_llm_call_count"] == 2
        and row["llm_evidence"]["authenticated_llm_call_roles"]
        == ["critic", "proposer"]
        and row["llm_evidence"]["cross_host_live_evidence_binding"] is True
        and row["process_pass_before_external_signature"] is False
        for row in stored["process_evidence"]
    )


def test_benchmark_llm_evidence_is_call_level_and_cross_host_bound() -> None:
    campaign = json.loads(
        (ROOT / "runs/math/external-creativity-validation/campaign.json").read_text(
            encoding="utf-8"
        )
    )
    live = json.loads(
        (ROOT / "runs/math/external-creativity-validation/live-api-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    multi_host = json.loads(
        (ROOT / "runs/math/external-creativity-validation/multi-host-reproduction.json").read_text(
            encoding="utf-8"
        )
    )
    benchmark = next(
        row
        for row in campaign["benchmarks"]
        if row["benchmark_id"] == "external.authority-oeis-005132"
    )
    assert _authenticated_llm_evidence(benchmark, live, multi_host)["passed"] is True

    changed = copy.deepcopy(live)
    matching = next(
        call for call in changed["calls"] if call["benchmark_id"] == benchmark["blind_id"]
    )
    matching["credential_persisted"] = True
    assert _authenticated_llm_evidence(benchmark, changed, multi_host)["passed"] is False

    changed_host = copy.deepcopy(multi_host)
    changed_host["reproduction"]["core_live_evidence_content_sha256"] = "0" * 64
    assert _authenticated_llm_evidence(benchmark, live, changed_host)["passed"] is False


def test_resealed_local_counter_or_authorization_tamper_fails() -> None:
    stored = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    for mutation in (
        lambda value: value["summary"].__setitem__(
            "admitted_independently_reproduced_level5_successes", 3
        ),
        lambda value: value["release_gate"].__setitem__("open_problem_authorized", True),
        lambda value: value["process_evidence"][0].__setitem__(
            "process_pass_before_external_signature", True
        ),
    ):
        changed = copy.deepcopy(stored)
        mutation(changed)
        changed = _seal({key: item for key, item in changed.items() if key != "content_sha256"})
        with pytest.raises(Level5SuccessAdmissionError):
            validate_receipt(changed, ROOT)


def test_target_blind_chronology_rejects_early_target_read() -> None:
    campaign = json.loads(
        (ROOT / "runs/math/external-creativity-validation/campaign.json").read_text(
            encoding="utf-8"
        )
    )
    assert _blind_chronology_pass(campaign)
    changed = copy.deepcopy(campaign)
    changed["blind_chronology"][2]["target_reads"] = 1
    assert not _blind_chronology_pass(changed)


def test_named_reviewed_independent_signature_round_trip(tmp_path: Path) -> None:
    payload = _payload()
    public_key, signature = _key_and_signature(tmp_path, payload)
    registry = _registry(public_key)
    validate_registry(registry)
    certificate = build_certificate(
        payload, registry, "external.level5-reviewer", signature
    )
    assert (
        _verify_certificate(certificate, payload, registry)
        == "external.level5-reviewer"
    )

    changed_payload = {**payload, "target_commitment": "6" * 64}
    with pytest.raises(Level5SuccessAdmissionError, match="payload binding"):
        _verify_certificate(certificate, changed_payload, registry)


def test_failed_current_process_cannot_be_exported_for_signature() -> None:
    with pytest.raises(Level5SuccessAdmissionError, match="failed level-5 process"):
        export_payload(ROOT, "external.authority-oeis-005132")


@pytest.mark.parametrize(
    "principal",
    ("invariant.discovery-engine", "external.test-source"),
)
def test_generator_or_source_principal_cannot_admit_success(
    tmp_path: Path, principal: str
) -> None:
    payload = _payload()
    public_key, signature = _key_and_signature(tmp_path, payload)
    registry = _registry(public_key, principal)
    certificate = _certificate(payload, principal, signature, registry)
    with pytest.raises(Level5SuccessAdmissionError, match="not independent"):
        _verify_certificate(certificate, payload, registry)


def test_empty_registry_cannot_admit_even_a_well_formed_certificate(tmp_path: Path) -> None:
    payload = _payload()
    public_key, signature = _key_and_signature(tmp_path, payload)
    reviewed = _registry(public_key)
    certificate = _certificate(payload, "external.level5-reviewer", signature, reviewed)
    empty = _seal(
        {
            "namespace": SIGNING_NAMESPACE,
            "policy": reviewed["policy"],
            "schema_version": REGISTRY_SCHEMA,
            "signers": [],
        }
    )
    with pytest.raises(Level5SuccessAdmissionError, match="not uniquely registered"):
        _verify_certificate(certificate, payload, empty)
