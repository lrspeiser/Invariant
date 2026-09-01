from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_split_gate_range_source_publication_candidate as publication,
)


def test_config_identity_and_decision() -> None:
    config = publication.load_config()
    assert config["artifact_id"] == publication.ARTIFACT_ID
    assert config["publication_adjudication"]["decision"] == publication.DECISION


def test_theorem_predecessor_is_committed_and_exact() -> None:
    receipt = publication.build_receipt()
    assert receipt["theorem_binding"]["commit"] == ("ab54cf4a1eadec793df8cba61ee0fb70002bfb0e")
    assert receipt["theorem_binding"]["receipt_content_sha256"] == (
        "50ec6d65d3dc09b993c19e57638ace6b6585054ba1e80d8300b93833d4402b2d"
    )


def test_novelty_benchmark_is_exactly_bound() -> None:
    receipt = publication.build_receipt()
    assert receipt["novelty_binding"]["config_sha256"] == (
        "5bcc5003c367fc35f68a1641ea9a0eb56699e2bff92e1895d890d03bdee760a5"
    )
    assert receipt["novelty_binding"]["receipt_content_sha256"] == (
        "78776b0530e015859991285c555c2901e4bd216f16c1cc89702052d9342e726a"
    )


def test_symbolic_derivation_passes_independently() -> None:
    checks = publication._symbolic_checks()
    assert len(checks) == 6
    assert all(checks.values())


def test_draft_contains_falsification_and_scope_boundaries() -> None:
    root = publication._repo_root()
    text = (root / publication.DRAFT_PATH).read_text(encoding="utf-8")
    assert "What would falsify or demote the result" in text
    assert "Claims not made" in text
    assert "not a universal law of gravity" in text
    assert "Historical novelty" in text


def test_publication_value_is_positive_but_model_claim_is_negative() -> None:
    adjudication = publication.load_config()["publication_adjudication"]
    assert adjudication["scientifically_interesting"] is True
    assert adjudication["worth_independent_expert_review"] is True
    assert adjudication["worth_preparing_as_narrow_theory_note"] is True
    assert adjudication["worth_claiming_as_successful_gravity_model"] is False


def test_claim_ceiling_is_restrictive() -> None:
    claims = publication.load_config()["claim_boundary"]
    assert claims["exact_restricted_theorem"] is True
    assert claims["narrow_note_candidate"] is True
    assert claims["historical_novelty_established"] is False
    assert claims["radiative_stability_established"] is False
    assert claims["full_coupled_solution"] is False
    assert claims["observational_support"] is False
    assert claims["publication_ready"] is False


def test_zero_observational_access() -> None:
    receipt = publication.build_receipt()
    assert not any(receipt["access_ledger"].values())


def test_receipt_is_deterministic_self_hashed_and_complete() -> None:
    first = publication.build_receipt()
    second = publication.build_receipt()
    assert first == second
    assert first["content_sha256"] == publication._self_hash(first)
    assert first["checks_passed"] == 12
    assert all(first["checks"].values())


def test_coherent_overclaim_forgery_differs_from_rebuild() -> None:
    forged = copy.deepcopy(publication.build_receipt())
    forged["claim_boundary"]["publication_ready"] = True
    forged["content_sha256"] = publication._self_hash(forged)
    assert forged["content_sha256"] == publication._self_hash(forged)
    assert forged != publication.build_receipt()


def test_config_byte_mutation_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = publication._repo_root()
    path = tmp_path / publication.CONFIG_PATH
    path.parent.mkdir(parents=True)
    config = json.loads((root / publication.CONFIG_PATH).read_text(encoding="utf-8"))
    config["publication_adjudication"]["worth_claiming_as_successful_gravity_model"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(publication, "EXPECTED_CONFIG_RAW_SHA256", "0" * 64)
    with pytest.raises(
        publication.SplitGateRangeSourcePublicationError, match="config semantics changed"
    ):
        publication.load_config(tmp_path)


def test_local_integrity_pins_module_and_tests() -> None:
    binding = publication._local_integrity(publication._repo_root())
    assert binding["config_raw_sha256"] == publication.EXPECTED_CONFIG_RAW_SHA256
    assert binding["module_semantic_sha256"] == publication.EXPECTED_MODULE_SEMANTIC_SHA256
    assert binding["test_raw_sha256"] == publication.EXPECTED_TEST_RAW_SHA256


def test_write_replay_and_check_are_no_clobber() -> None:
    assert publication.write_receipt() == "EXISTING_IDENTICAL"
    assert publication.validate_receipt() == publication.build_receipt()


def test_maximal_claim_uses_architecture_class_wording() -> None:
    claim = publication.load_config()["maximal_claim"]
    assert "architecture-class" not in claim["scalings"]["architecture_class_product"]
    assert claim["scope"].startswith("Exact only for the frozen")
