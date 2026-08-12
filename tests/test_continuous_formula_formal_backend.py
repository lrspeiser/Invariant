import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from sigma_theory_compiler.continuous_formula_formal_backend import (
    MANIFEST_SCHEMA,
    _sealed,
    _sha,
    build_formal_evidence,
    combine_candidate_manifests,
    extract_candidate_manifest,
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
from sigma_theory_compiler.real_formula_execution import cpu_formula_batch_evaluator

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
    ordinals = sorted(ordinals)
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
            "all_survivor_ordinals_root_sha256": _sha(ordinals),
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


def _formula_payload(config: dict, *, start: int = 0, count: int = 16) -> dict:
    generator_path = (ROOT / config["generator_config_path"]).resolve()
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    return {
        "generator_config_path": str(generator_path),
        "generator_config_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
        "start_ordinal": start,
        "end_ordinal_exclusive": start + count,
        "candidate_count": count,
        "basis_count": generator["basis_count"],
        "max_action_terms": generator["max_action_terms"],
        "protocol_version": generator["protocol_version"],
        "ambiguity_guard": 1e-12,
        "data_eligibility": {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
        },
    }


def _reseal_evidence(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    body["candidate_records_root_sha256"] = _sha(body["candidate_records"])
    return _sealed(body)


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
    with pytest.raises(ValueError, match="manifest record"):
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


def test_extraction_recomputes_exact_status_root_and_rejects_bool_counts() -> None:
    config = load_backend_config(ROOT, CONFIG)
    payload = _formula_payload(config)
    result = cpu_formula_batch_evaluator(SimpleNamespace(payload=payload))
    manifest = extract_candidate_manifest(payload, result)
    validate_candidate_manifest(manifest)
    assert manifest["candidate_root_sha256"] == result["status_root_sha256"]

    forged_root = json.loads(json.dumps(result))
    forged_root["status_root_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="status root"):
        extract_candidate_manifest(payload, forged_root)

    forged_count = json.loads(json.dumps(result))
    nonzero = next(key for key, count in forged_count["counts"].items() if count > 0)
    forged_count["counts"][nonzero] = True
    with pytest.raises(ValueError, match="accounting"):
        extract_candidate_manifest(payload, forged_count)


def test_manifest_count_root_and_combination_replay_are_exact() -> None:
    generator = json.loads((ROOT / "configs/generator_v2_billion.json").read_text())
    first = _manifest([0], generator)
    second = _manifest([1], generator)
    combined = combine_candidate_manifests([second, first], 32)
    validate_candidate_manifest(combined)
    assert [row["ordinal"] for row in combined["survivor_records"]] == [0, 1]
    assert combined["all_survivor_ordinals_root_sha256"] == _sha([0, 1])

    bool_count = json.loads(json.dumps(first))
    bool_count["batch"]["candidate_count"] = True
    bool_count = _sealed({key: item for key, item in bool_count.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="count contract"):
        validate_candidate_manifest(bool_count)

    wrong_root = json.loads(json.dumps(first))
    wrong_root["all_survivor_ordinals_root_sha256"] = "0" * 64
    wrong_root = _sealed({key: item for key, item in wrong_root.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="ordinal root"):
        validate_candidate_manifest(wrong_root)

    with pytest.raises(ValueError, match="at least one shard"):
        combine_candidate_manifests([], 32)
    with pytest.raises(ValueError, match="bound"):
        combine_candidate_manifests([first], True)


def test_formal_evidence_rederives_counts_decisions_and_reject_evidence(
    tmp_path: Path,
) -> None:
    config = load_backend_config(ROOT, CONFIG)
    generator = json.loads((ROOT / config["generator_config_path"]).read_text())
    manifest = _manifest([0, 7, 677], generator)
    _, evidence = build_formal_evidence(
        _receipt(), manifest, config, root=ROOT, output_root=tmp_path / "evidence"
    )

    bool_count = json.loads(json.dumps(evidence))
    bool_count["hard_reject_count"] = True
    with pytest.raises(ValueError, match="contract"):
        validate_formal_evidence(_reseal_evidence(bool_count))

    forged_count = json.loads(json.dumps(evidence))
    forged_count["hard_reject_count"] = 0
    forged_count["candidate_hard_reject_count"] = 0
    with pytest.raises(ValueError, match="derived count"):
        validate_formal_evidence(_reseal_evidence(forged_count))

    forged_decision = json.loads(json.dumps(evidence))
    rejected = next(
        row for row in forged_decision["candidate_records"] if row["decision"] == "reject"
    )
    rejected["decision"] = "block"
    rejected["first_blocker"] = "forged"
    with pytest.raises(ValueError, match="candidate decision"):
        validate_formal_evidence(_reseal_evidence(forged_decision))

    missing_mapping_reject_evidence = json.loads(json.dumps(evidence))
    mapping_reject = next(
        row
        for row in missing_mapping_reject_evidence["candidate_records"]
        if row["covariant_mapping_decision"] == "reject"
    )
    mapping_reject["covariant_mapping_payload"]["reason"] = ""
    mapping_reject["covariant_mapping_payload_sha256"] = _sha(
        mapping_reject["covariant_mapping_payload"]
    )
    with pytest.raises(ValueError, match="rejection lacks evidence"):
        validate_formal_evidence(_reseal_evidence(missing_mapping_reject_evidence))

    missing_reject_evidence = json.loads(json.dumps(evidence))
    mapped_reject = next(
        row
        for row in missing_reject_evidence["candidate_records"]
        if row["covariant_mapping_decision"] == "mapped"
    )
    health = mapped_reject["semantic_action_health"]
    health["gate_statuses"] = {
        key: "unresolved" if status == "reject" else status
        for key, status in health["gate_statuses"].items()
    }
    health = _sealed({key: item for key, item in health.items() if key != "content_sha256"})
    mapped_reject["semantic_action_health"] = health
    mapped_reject["semantic_action_health_sha256"] = health["content_sha256"]
    with pytest.raises(ValueError, match="status evidence|lacks evidence"):
        validate_formal_evidence(_reseal_evidence(missing_reject_evidence))


def test_semantic_health_cache_replays_and_rejects_changed_or_transplanted_files(
    tmp_path: Path,
) -> None:
    config = load_backend_config(ROOT, CONFIG)
    generator = json.loads((ROOT / config["generator_config_path"]).read_text())
    first_manifest = _manifest([7], generator)
    second_manifest = _manifest([689], generator)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _, first = build_formal_evidence(
        _receipt(), first_manifest, config, root=ROOT, output_root=first_root
    )
    _, replay = build_formal_evidence(
        _receipt(), first_manifest, config, root=ROOT, output_root=first_root
    )
    assert replay == first

    first_dir = first_root / first_manifest["survivor_records"][0]["candidate_id"]
    q_path = first_dir / "action-health/q-operator-ir.json"
    q_payload = json.loads(q_path.read_text(encoding="utf-8"))
    q_payload["tampered"] = True
    q_path.write_text(json.dumps(q_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bound file changed"):
        build_formal_evidence(_receipt(), first_manifest, config, root=ROOT, output_root=first_root)

    build_formal_evidence(_receipt(), second_manifest, config, root=ROOT, output_root=second_root)
    second_dir = second_root / second_manifest["survivor_records"][0]["candidate_id"]
    shutil.copyfile(
        first_dir / "semantic-action-health.json",
        second_dir / "semantic-action-health.json",
    )
    with pytest.raises(ValueError, match="stale or transplanted|cache binding"):
        build_formal_evidence(
            _receipt(), second_manifest, config, root=ROOT, output_root=second_root
        )
