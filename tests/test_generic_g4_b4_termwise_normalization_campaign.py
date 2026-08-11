from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.generic_g4_b4_termwise_normalization_campaign import (
    B4_COEFFICIENTS,
    build_generic_g4_b4_termwise_normalization_campaign,
    validate_generic_g4_b4_termwise_normalization_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "generic_g4_b4_termwise_normalization_campaign.json"
ARTIFACT = ROOT / "runs" / "engine" / "generic-g4-b4-termwise-normalization-campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(artifact: dict) -> None:
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_generic_g4_b4_termwise_normalization_campaign(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    validate_generic_g4_b4_termwise_normalization_campaign(committed, ROOT)
    assert (
        committed["content_sha256"]
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in committed.items() if key != "content_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
    )


def test_all_24_b4_coefficients_match_exactly(rebuilt: dict) -> None:
    assert len(B4_COEFFICIENTS) == 24
    assert rebuilt["canonical_term_count"] == 24
    assert rebuilt["matched_term_count"] == 24
    assert rebuilt["nonzero_residual_count"] == 0
    assert {record["term_id"] for record in rebuilt["term_records"]} == set(B4_COEFFICIENTS)
    assert all(record["residual"] == "0" for record in rebuilt["term_records"])


def test_scope_remains_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["metric_variation_normalization_pass"] is True
    assert rebuilt["scalar_equation_or_noether_rederived_here"] is False
    assert rebuilt["full_candidate_formal_pass_inferred"] is False
    assert rebuilt["global_energy_inferred"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0


def test_fragment_coefficient_and_source_tamper_fail_closed() -> None:
    config = _load(CONFIG)
    fragment = copy.deepcopy(config)
    fragment["cadabra_canonical_fragments"]["G4_XX_QQ_h"] = "+ C Q^{a} Q^{b} h_{a b} sqrtg"
    with pytest.raises(ValueError, match="missing or duplicated"):
        build_generic_g4_b4_termwise_normalization_campaign(fragment, ROOT)

    source = copy.deepcopy(config)
    source["campaign_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        build_generic_g4_b4_termwise_normalization_campaign(source, ROOT)

    paper = copy.deepcopy(config)
    paper["primary_source"]["equation"] = "B.8"
    with pytest.raises(ValueError, match="primary KYY"):
        build_generic_g4_b4_termwise_normalization_campaign(paper, ROOT)

    receipt = copy.deepcopy(config)
    receipt["formal_controls_artifact"]["control_stdout_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="execution receipt"):
        build_generic_g4_b4_termwise_normalization_campaign(receipt, ROOT)

    transcription = copy.deepcopy(config)
    transcription["primary_source_transcription"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="transcription hash"):
        build_generic_g4_b4_termwise_normalization_campaign(transcription, ROOT)


def test_validator_rejects_rehashed_false_promotion(rebuilt: dict) -> None:
    tampered = copy.deepcopy(rebuilt)
    tampered["full_candidate_formal_pass_inferred"] = True
    _rehash(tampered)
    with pytest.raises(ValueError, match="artifact is invalid"):
        validate_generic_g4_b4_termwise_normalization_campaign(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scalar_equation_or_noether_rederived_here", True),
        ("global_energy_inferred", True),
        ("dark_matter_or_halo_inputs", True),
        ("redshift_distance_inputs", True),
        ("paid_llm_spend_usd", 1.0),
    ],
)
def test_validator_rejects_rehashed_scope_or_seal_tamper(
    rebuilt: dict, field: str, value: object
) -> None:
    tampered = copy.deepcopy(rebuilt)
    tampered[field] = value
    _rehash(tampered)
    with pytest.raises(ValueError, match="artifact is invalid"):
        validate_generic_g4_b4_termwise_normalization_campaign(tampered)


def test_validator_rejects_rehashed_controls_coefficient_and_source_tamper(
    rebuilt: dict,
) -> None:
    controls = copy.deepcopy(rebuilt)
    controls["negative_controls"]["flip_R_pp_sign_rejected"] = False
    _rehash(controls)
    with pytest.raises(ValueError, match="artifact is invalid"):
        validate_generic_g4_b4_termwise_normalization_campaign(controls)

    coefficient = copy.deepcopy(rebuilt)
    record = next(row for row in coefficient["term_records"] if row["term_id"] == "F_Ricci_h")
    record["cadabra_coefficient"] = "2"
    record["B4_coefficient"] = "2"
    record_body = {key: value for key, value in record.items() if key != "content_sha256"}
    record["content_sha256"] = hashlib.sha256(
        json.dumps(
            record_body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    coefficient["term_registry_root_sha256"] = hashlib.sha256(
        json.dumps(
            coefficient["term_records"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    _rehash(coefficient)
    with pytest.raises(ValueError, match="term record is invalid"):
        validate_generic_g4_b4_termwise_normalization_campaign(coefficient)

    source = copy.deepcopy(rebuilt)
    source["primary_source"]["equation"] = "B.8"
    _rehash(source)
    with pytest.raises(ValueError, match="artifact is invalid"):
        validate_generic_g4_b4_termwise_normalization_campaign(source)
