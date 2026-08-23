from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.claim_specific_prior_art import HTTPResponse
from sigma_theory_compiler.rotating_external_benchmarks import (
    RotationError,
    build_pack,
    validate_pack,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _transport(uri: str, _headers: object, _timeout: int, _maximum: int) -> HTTPResponse:
    offset = sum(uri.encode()) % 7
    body = "".join(f"{index} {index * index + offset}\n" for index in range(100)).encode()
    return HTTPResponse(200, {"content-type": "text/plain", "content-range": ""}, body)


@pytest.fixture(scope="module")
def pack() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return build_pack(ROOT, transport=_transport, retrieved_utc="2026-08-23T06:00:00Z")


def test_pack_has_24_live_external_tasks_and_balanced_representation_families(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    generation, targets, receipt = pack
    validate_pack(generation, targets, receipt, ROOT)
    assert receipt["coverage"]["tasks"] == 24
    assert set(receipt["coverage"]["representation_counts"].values()) == {4}
    assert len(generation["tasks"]) == len(targets["targets"]) == 24


def test_generation_view_excludes_sources_and_holdouts_but_binds_targets(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    generation, targets, _ = pack
    assert all("source_uri" not in task for task in generation["tasks"])
    assert all("source_id" not in task for task in generation["tasks"])
    target_by_id = {item["task_id"]: item for item in targets["targets"]}
    for task in generation["tasks"]:
        target = target_by_id[task["task_id"]]
        target_body = {key: value for key, value in target.items() if key != "task_id"}
        assert task["target_commitment"] == canonical_sha256(target_body)
        assert not ({row["index"] for row in task["training"]} & {row["index"] for row in target["holdout"]})


def test_unsigned_https_pack_is_useful_but_cannot_count_as_level5(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    _, _, receipt = pack
    assert receipt["source_signature"] == {
        "cryptographic_signature_verified": False,
        "external_https_origin_hash_bound": True,
        "status": "PENDING_DISTINCT_PRINCIPAL_SIGNATURE",
    }
    assert receipt["release_gate"]["level5_eligible"] is False
    assert not any(receipt["claims"].values())


def test_commitment_or_release_tamper_fails_closed(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    generation, targets, receipt = pack
    changed = json.loads(json.dumps(generation))
    changed["tasks"][0]["target_commitment"] = "0" * 64
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(RotationError):
        validate_pack(changed, targets, receipt)
    changed_receipt = json.loads(json.dumps(receipt))
    changed_receipt["release_gate"]["level5_eligible"] = True
    changed_receipt["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed_receipt.items() if key != "content_sha256"}
    )
    with pytest.raises(RotationError):
        validate_pack(generation, targets, changed_receipt)
