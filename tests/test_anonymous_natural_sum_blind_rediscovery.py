from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.anonymous_natural_sum_blind_rediscovery import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    _write_immutable,
    run,
    validate_result,
)

WITHHELD_KNOWN_THEOREM = {
    "anonymous_function": "S",
    "domain": "nonnegative_integers",
    "basis": ["square", "linear", "constant"],
    "coefficients": {
        "square": {"numerator": 1, "denominator": 2},
        "linear": {"numerator": 1, "denominator": 2},
        "constant": {"numerator": 0, "denominator": 1},
    },
}

ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.fixture(scope="module")
def checked_result() -> dict[str, object]:
    result = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_result(result, ROOT)
    assert result == run(ROOT)
    return result


def test_checked_artifact_is_blind_replayable_and_bound(checked_result: dict[str, object]) -> None:
    assert checked_result["decision"] == (
        "pass_blind_bounded_grammar_rediscovery_independently_proved_before_unseal"
    )
    assert (
        checked_result["blinded_pre_unseal_root_sha256"]
        == checked_result["pre_unseal"]["content_sha256"]
    )
    assert checked_result["content_sha256"] == _canonical_sha(
        {key: value for key, value in checked_result.items() if key != "content_sha256"}
    )
    bindings = checked_result["bindings"]
    for label, relative in (
        ("config", CONFIG_PATH),
        ("source", SOURCE_PATH),
        ("test_and_withheld_reference", TEST_PATH),
    ):
        assert bindings[label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }


def test_bounded_search_falsification_and_induction(checked_result: dict[str, object]) -> None:
    assert checked_result["enumeration"] == {
        "raw_cartesian_candidates": 46_656,
        "canonical_coefficient_classes": 12_167,
        "exact_example_survivors": 1,
        "counterexample_tests_per_survivor": 59,
        "fully_tested_survivors": 1,
    }
    assert checked_result["winner"]["coefficients"] == WITHHELD_KNOWN_THEOREM["coefficients"]
    proof = checked_result["induction_proof"]
    assert proof["base_case"]["proved"] is True
    assert proof["successor_identity_proved"] is True
    assert proof["forbidden_premises_used"] == []
    assert set(proof["allowed_premises_used"]) == {
        "definition_base_case",
        "definition_successor_rule",
        "exact_rational_arithmetic",
        "polynomial_identity",
    }


def test_read_boundary_dependency_closure_and_post_seal_reference(
    checked_result: dict[str, object],
) -> None:
    pre_unseal = checked_result["pre_unseal"]
    boundary = pre_unseal["io_boundary"]
    assert boundary["owned_threads"] == 1
    assert boundary["allowed_paths"] == [CONFIG_PATH, SOURCE_PATH]
    assert boundary["denied_paths"] == [TEST_PATH]
    assert boundary["denied_access_count"] == 1
    assert boundary["denied_content_bytes_exposed"] == 0
    assert boundary["network_calls"] == boundary["llm_calls"] == boundary["subprocesses"] == 0
    audit = pre_unseal["dependency_audit"]
    assert audit["unexpected_import_roots"] == []
    assert audit["forbidden_import_roots_present"] == []
    assert audit["forbidden_dependency_closure_empty"] is True
    post_unseal = checked_result["post_unseal"]
    assert post_unseal["pre_unseal_seal_verified"] is True
    assert post_unseal["comparison_permitted_only_after_pre_unseal_seal"] is True
    assert post_unseal["withheld_theorem"] == WITHHELD_KNOWN_THEOREM
    source = (ROOT / SOURCE_PATH).read_text(encoding="utf-8").replace(" ", "")
    assert "n*(n+1)/2" not in source
    assert "(n**2+n)/2" not in source
    assert "WITHHELD_KNOWN_THEOREM=" not in source


def test_negative_controls_and_resealed_tamper_fail_closed(
    checked_result: dict[str, object],
) -> None:
    controls = {row["control_id"]: row for row in checked_result["negative_controls"]}
    assert set(controls) == {
        "wrong_formula_square_only",
        "example_lookup_overfit",
        "forbidden_withheld_theorem_premise",
        "winner_with_unproved_generalization",
    }
    assert all(row["eligible"] is False for row in controls.values())
    assert controls["wrong_formula_square_only"]["first_counterexample"] == 2
    assert controls["example_lookup_overfit"]["first_counterexample"] == 6

    tampered = copy.deepcopy(checked_result)
    tampered["claims"]["novel_theorem_claimed"] = True
    tampered["content_sha256"] = _canonical_sha(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="contract changed"):
        validate_result(tampered, ROOT)

    tampered = copy.deepcopy(checked_result)
    tampered["winner"]["candidate_id"] = "forged"
    tampered["content_sha256"] = _canonical_sha(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="immutable replay mismatch"):
        validate_result(tampered, ROOT)


def test_artifact_writer_is_idempotent_but_refuses_replacement(tmp_path: Path) -> None:
    target = tmp_path / "sealed.json"
    _write_immutable(target, {"result": "sealed"})
    original = target.read_bytes()
    _write_immutable(target, {"result": "sealed"})
    assert target.read_bytes() == original
    with pytest.raises(FileExistsError, match="different bytes"):
        _write_immutable(target, {"result": "changed"})
