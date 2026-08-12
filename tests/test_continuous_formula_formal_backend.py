import json
from pathlib import Path

import pytest

from sigma_theory_compiler.continuous_formula_formal_backend import (
    MANIFEST_SCHEMA,
    _sealed,
    build_formal_evidence,
    load_backend_config,
    validate_candidate_manifest,
    validate_formal_evidence,
)
from sigma_theory_compiler.high_throughput import (
    build_basis,
    candidate_id,
    correction_expression,
    decode_ordinal,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/continuous_formula_formal_backend.json"


def _candidate(ordinal: int, generator: dict) -> dict:
    decoded = decode_ordinal(generator["basis_count"], generator["max_action_terms"], ordinal)
    return {
        "candidate_id": candidate_id(generator["protocol_version"], decoded),
        "ordinal": ordinal,
        "term_ids": decoded["term_ids"],
        "signs": decoded["signs"],
        "correction_expression": correction_expression(
            decoded, build_basis(generator["basis_count"])
        ),
        "sampled_static_margin": 1.0,
    }


def _manifest(ordinals: list[int], generator: dict) -> dict:
    records = [_candidate(ordinal, generator) for ordinal in ordinals]
    candidate_count = max(ordinals) - min(ordinals) + 1
    return _sealed(
        {
            "schema_version": MANIFEST_SCHEMA,
            "batch": {
                "start_ordinal": min(ordinals),
                "end_ordinal_exclusive": max(ordinals) + 1,
                "candidate_count": candidate_count,
            },
            "candidate_root_sha256": "b" * 64,
            "screen_counts": {
                "reject": candidate_count - len(records),
                "pass": len(records),
                "ambiguous": 0,
            },
            "all_survivor_ordinals_root_sha256": "a" * 64,
            "survivor_records": records,
            "survivor_record_count": len(records),
            "sample_complete": True,
            "observations_opened": False,
            "forbidden_target_inputs_opened": False,
        }
    )


def _receipt() -> dict:
    return _sealed(
        {
            "candidate_root_sha256": "b" * 64,
            "screen_decision": "pass",
            "unique_formula_count": 678,
            "theory_pass_claimed": False,
            "observations_opened": False,
            "rank_eligible": False,
        }
    )


def test_candidate_bound_covariant_health_is_deterministic_and_fail_closed(
    tmp_path: Path,
) -> None:
    config = load_backend_config(ROOT, CONFIG)
    generator = json.loads((ROOT / config["generator_config_path"]).read_text())
    manifest = _manifest([7, 677, 0], generator)
    first_receipt, first = build_formal_evidence(
        _receipt(), manifest, config, root=ROOT, output_root=tmp_path / "first"
    )
    second_receipt, second = build_formal_evidence(
        _receipt(), manifest, config, root=ROOT, output_root=tmp_path / "second"
    )
    validate_formal_evidence(first)
    assert first == second and first_receipt == second_receipt
    assert first["decision"] == "block"
    assert first["covariant_action_mapped_count"] == 1
    assert first["action_health_execution_count"] == 1
    assert first["hard_reject_count"] == 2
    records = {row["ordinal"]: row for row in first["candidate_records"]}
    assert records[7]["covariant_mapping_decision"] == "mapped"
    assert records[7]["semantic_action_health"]["status"] == "reject"
    assert records[7]["semantic_action_health"]["promotion_allowed"] is False
    assert records[7]["decision"] == "reject"
    assert records[677]["covariant_mapping_decision"] == "blocked"
    assert records[677]["decision"] == "block"
    assert records[0]["covariant_mapping_decision"] == "reject"
    assert records[0]["first_blocker"] == "forbidden_baryonic_action_atom"
    assert first["complete_comparable_evidence"] is False
    assert first["direct_rank_assignment"] is False
    assert first["theory_rejected"] is False


def test_manifest_and_evidence_tamper_fail_closed(tmp_path: Path) -> None:
    config = load_backend_config(ROOT, CONFIG)
    generator = json.loads((ROOT / config["generator_config_path"]).read_text())
    manifest = _manifest([7], generator)
    broken = json.loads(json.dumps(manifest))
    broken["survivor_records"][0]["candidate_id"] = "forged"
    broken = _sealed({key: value for key, value in broken.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="lineage"):
        build_formal_evidence(_receipt(), broken, config, root=ROOT, output_root=tmp_path / "bad")

    _, evidence = build_formal_evidence(
        _receipt(), manifest, config, root=ROOT, output_root=tmp_path / "good"
    )
    tampered = json.loads(json.dumps(evidence))
    tampered["candidate_records"][0]["semantic_action_health"]["promotion_allowed"] = True
    tampered = _sealed({key: value for key, value in tampered.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="formal evidence"):
        validate_formal_evidence(tampered)


def test_config_and_manifest_contracts_are_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["formal_controls_file_sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="formal_controls_path"):
        load_backend_config(ROOT, path)

    generator = json.loads((ROOT / "configs/generator_v2_billion.json").read_text())
    manifest = _manifest([7], generator)
    manifest["extra"] = False
    manifest = _sealed({key: value for key, value in manifest.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="manifest contract"):
        validate_candidate_manifest(manifest)
