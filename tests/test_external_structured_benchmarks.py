from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.claim_specific_prior_art import HTTPResponse
from sigma_theory_compiler.external_structured_benchmarks import (
    StructuredBenchmarkError,
    build_pack,
    load_stored_pack,
    reproduce_pack,
    validate_pack,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]

_TENSOR_FUNCTIONS = (
    "test_canonicalize_no_slot_sym",
    "test_canonicalize_no_dummies",
    "test_no_metric_symmetry",
    "test_canonicalize1",
)
_VARIATIONAL_FUNCTIONS = (
    "test_euler_pendulum",
    "test_euler_henonheiles",
    "test_euler_sineg",
    "test_euler_high_order",
)
_TRANSFORM_FUNCTIONS = (
    "test_apply_finite_diff",
    "test_finite_diff_weights",
    "test_as_finite_diff",
    "test_differentiate_finite",
)


def _tensor_source(*, changed: bool = False) -> bytes:
    blocks = ["raise RuntimeError('AST-only fixture must never execute')"]
    for index, name in enumerate(_TENSOR_FUNCTIONS):
        target = f"canonical-{index}{'-changed' if changed and index == 0 else ''}"
        blocks.append(
            f"""
def {name}():
    raw = 'tensor-input-{index}'
    canonical = '{target}'
    assert str(canonical) == '{target}'
""".strip()
        )
    return ("\n\n".join(blocks) + "\n").encode()


def _variational_source(*, changed: bool = False) -> bytes:
    blocks = ["raise RuntimeError('AST-only fixture must never execute')"]
    for index, name in enumerate(_VARIATIONAL_FUNCTIONS):
        target = f"equation-{index}{'-changed' if changed and index == 0 else ''}"
        blocks.append(
            f"""
def {name}():
    functional = 'functional-{index}'
    derived = ['{target}']
    assert derive(functional) == ['{target}']
""".strip()
        )
    return ("\n\n".join(blocks) + "\n").encode()


def _transform_source(*, changed: bool = False) -> bytes:
    blocks = ["raise RuntimeError('AST-only fixture must never execute')"]
    for index, name in enumerate(_TRANSFORM_FUNCTIONS):
        target = f"stencil-{index}{'-changed' if changed and index == 0 else ''}"
        blocks.append(
            f"""
def {name}():
    samples = 'shifted-samples-{index}'
    transformed = '{target}'
    assert transformed == '{target}'
""".strip()
        )
    return ("\n\n".join(blocks) + "\n").encode()


def _fixture_source(uri: str, *, changed: bool = False) -> bytes:
    if "/tensor/" in uri:
        return _tensor_source(changed=changed)
    if "finite_diff" in uri:
        return _transform_source(changed=changed)
    return _variational_source(changed=changed)


def _transport(uri: str, _headers: object, _timeout: int, _maximum: int) -> HTTPResponse:
    body = _fixture_source(uri)
    return HTTPResponse(200, {"content-type": "text/plain; charset=utf-8"}, body)


def _changed_transport(uri: str, _headers: object, _timeout: int, _maximum: int) -> HTTPResponse:
    body = _fixture_source(uri, changed=True)
    return HTTPResponse(200, {"content-type": "text/plain; charset=utf-8"}, body)


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


@pytest.fixture(scope="module")
def pack() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return build_pack(
        ROOT,
        transport=_transport,
        retrieved_utc="2026-08-23T17:00:00Z",
    )


def test_ast_only_pack_builds_balanced_blind_tasks_and_refetches_exactly(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    generation, targets, receipt = pack
    validate_pack(generation, targets, receipt, ROOT)
    reproduce_pack(ROOT, generation, targets, receipt, transport=_transport)
    assert receipt["coverage"] == {
        "external_principals": 1,
        "representation_counts": {
            "tensor_identity": 4,
            "transform_relation": 4,
            "variational_functional": 4,
        },
        "tasks": 12,
        "unique_external_response_hashes": 3,
    }
    assert len(generation["tasks"]) == len(targets["targets"]) == 12


def test_generation_packet_hides_source_and_expected_expression(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    generation, targets, _ = pack
    target_by_id = {row["task_id"]: row for row in targets["targets"]}
    for task in generation["tasks"]:
        assert "source_id" not in task
        assert "source_uri" not in task
        assert "test_function" not in task
        assert "target_expression" not in task
        target = target_by_id[task["task_id"]]
        target_body = {key: value for key, value in target.items() if key != "task_id"}
        assert task["target_commitment"] == canonical_sha256(target_body)
    public_text = json.dumps(generation, sort_keys=True)
    for forbidden in (
        "external.sympy-project",
        "raw.githubusercontent.com",
        "sympy-euler-tests",
        "sympy-finite-difference-tests",
        "sympy-tensor-tests",
        "test_canonicalize",
        "test_euler",
        "test_finite_diff",
    ):
        assert forbidden not in public_text


def test_target_leak_fails_even_after_generation_reseal(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    generation, targets, receipt = pack
    changed = copy.deepcopy(generation)
    changed["tasks"][0]["target_expression"] = "leaked"
    _reseal(changed)
    with pytest.raises(StructuredBenchmarkError, match="keys changed|leaked"):
        validate_pack(changed, targets, receipt)


def test_target_substitution_fails_against_upstream_function_after_full_reseal(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    generation, targets, receipt = pack
    changed_generation = copy.deepcopy(generation)
    changed_targets = copy.deepcopy(targets)
    changed_targets["targets"][0]["target_expression"] = "'fabricated-target'"
    target = changed_targets["targets"][0]
    target_body = {key: value for key, value in target.items() if key != "task_id"}
    commitment = canonical_sha256(target_body)
    task_id = (
        "blind."
        + canonical_sha256(
            {"commitment": commitment, "epoch": changed_generation["rotation_epoch"]}
        )[:24]
    )
    changed_generation["tasks"][0]["target_commitment"] = commitment
    changed_generation["tasks"][0]["task_id"] = task_id
    target["task_id"] = task_id
    _reseal(changed_generation)
    _reseal(changed_targets)
    changed_receipt = copy.deepcopy(receipt)
    changed_receipt["source_bindings"]["generation_packet_content_sha256"] = changed_generation[
        "content_sha256"
    ]
    changed_receipt["source_bindings"]["target_packet_content_sha256"] = changed_targets[
        "content_sha256"
    ]
    _reseal(changed_receipt)
    with pytest.raises(StructuredBenchmarkError, match="upstream assertion"):
        validate_pack(changed_generation, changed_targets, changed_receipt)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("release_gate", "level5_eligible", True),
        ("source_signature", "cryptographic_signature_verified", True),
        ("claims", "targets_are_literature_novel", True),
    ),
)
def test_claim_or_signature_promotion_fails_closed(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
    section: str,
    key: str,
    value: object,
) -> None:
    generation, targets, receipt = pack
    changed = copy.deepcopy(receipt)
    changed[section][key] = value
    _reseal(changed)
    with pytest.raises(StructuredBenchmarkError, match="release boundary"):
        validate_pack(generation, targets, changed)


def test_valid_unsigned_source_substitution_requires_live_refetch(
    pack: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    changed = build_pack(
        ROOT,
        transport=_changed_transport,
        retrieved_utc="2026-08-23T17:00:00Z",
    )
    validate_pack(*changed, ROOT)
    with pytest.raises(StructuredBenchmarkError, match="did not reproduce"):
        reproduce_pack(ROOT, *changed, transport=_transport)


def test_missing_upstream_selector_fails_closed() -> None:
    def missing(uri: str, _headers: object, _timeout: int, _maximum: int) -> HTTPResponse:
        body = b"def unrelated():\n    assert 1 == 1\n"
        return HTTPResponse(200, {"content-type": "text/plain"}, body)

    with pytest.raises(StructuredBenchmarkError, match="missing or duplicated"):
        build_pack(ROOT, transport=missing, retrieved_utc="2026-08-23T17:00:00Z")


def test_unavailable_upstream_source_fails_closed() -> None:
    def unavailable(uri: str, _headers: object, _timeout: int, _maximum: int) -> HTTPResponse:
        return HTTPResponse(503, {"content-type": "text/plain"}, b"unavailable")

    with pytest.raises(StructuredBenchmarkError, match="unavailable"):
        build_pack(
            ROOT,
            transport=unavailable,
            retrieved_utc="2026-08-23T17:00:00Z",
        )


def test_stored_live_pack_is_source_bound_and_valid() -> None:
    generation, targets, receipt = load_stored_pack(ROOT)
    assert len(generation["tasks"]) == len(targets["targets"]) == 12
    assert receipt["release_gate"]["level5_eligible"] is False
