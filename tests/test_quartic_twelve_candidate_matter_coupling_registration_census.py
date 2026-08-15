from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_twelve_candidate_matter_coupling_registration_census import (
    QuarticMatterCouplingCensusError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_twelve_candidate_matter_coupling_registration_census.json"
OUTPUT = ROOT / (
    "runs/math/quartic-twelve-candidate-matter-coupling-registration-census/receipt.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_twelve_candidate_six_item_census() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "TYPED_BLOCK_CENSUS_NO_CANDIDATE_COUPLED_REGISTRATION"
    assert receipt["counts"] == {
        "candidates_audited": 12,
        "prerequisite_passes": 72,
        "contract_items_audited": 72,
        "contract_passes": 0,
        "typed_blocks": 72,
        "candidates_fully_registered": 0,
        "candidates_blocked_at_first_item": 12,
        "rejects": 0,
    }
    candidates = receipt["candidate_results"]
    assert len(candidates) == len({item["candidate_id"] for item in candidates}) == 12
    for candidate in candidates:
        assert set(candidate["prerequisite_census"].values()) == {"PASS"}
        assert len(candidate["prerequisite_census"]) == 6
        assert [item["outcome"] for item in candidate["contract_results"]] == ["BLOCK"] * 6
        assert candidate["first_blocker"] == "missing_total_matter_action_hash_binding"
        assert candidate["outcome"] == "BLOCK"


def test_claims_are_bounded_and_content_sealed() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    claims = dict(receipt["claims"])
    assert claims.pop("exact_candidate_specific_prerequisite_census_complete") is True
    assert not any(claims.values())
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_checked_receipt_is_current_and_path_free() -> None:
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == build_receipt(CONFIG, root=ROOT)
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize(
    "binding",
    [
        "predecessor",
        "candidate_action_symbol",
        "vacuum_euler",
        "vacuum_first_order_constraints",
        "vacuum_full_symmetrizer",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    target = (
        config["predecessor"] if binding == "predecessor" else config["evidence_bindings"][binding]
    )
    target["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticMatterCouplingCensusError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["predecessor"]["path"] = "runs/math/absent-coupling-receipt.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticMatterCouplingCensusError, match="cannot read bound file"):
        build_receipt(candidate, root=ROOT)


def test_candidate_dropout_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = ROOT / config["evidence_bindings"]["vacuum_euler"]["path"]
    altered = json.loads(source.read_text(encoding="utf-8"))
    altered["certificates"] = altered["certificates"][:-1]
    artifact = ROOT / ".pytest-census-altered.json"
    artifact.write_text(json.dumps(altered), encoding="utf-8")
    try:
        binding = config["evidence_bindings"]["vacuum_euler"]
        binding["path"] = artifact.relative_to(ROOT).as_posix()
        binding["file_sha256"] = _sha(artifact)
        candidate = tmp_path / "config.json"
        candidate.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(QuarticMatterCouplingCensusError, match="candidate set mismatch"):
            build_receipt(candidate, root=ROOT)
    finally:
        artifact.unlink(missing_ok=True)


def test_broadened_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_coupled_principal_system"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticMatterCouplingCensusError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
