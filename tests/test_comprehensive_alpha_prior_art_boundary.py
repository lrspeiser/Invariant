from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.comprehensive_alpha_prior_art_boundary import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    PriorArtBoundaryError,
    build_boundary,
    validate_boundary,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_legacy(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _reseal(value: dict[str, object]) -> None:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)


def _reseal_legacy(value: dict[str, object]) -> None:
    value.pop("content_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["content_sha256"] = hashlib.sha256(encoded).hexdigest()


def _snapshot_sha(config: dict[str, object]) -> str:
    return canonical_sha256(
        {
            "source_bindings": config["source_bindings"],
            "expected_survivors": config["expected_survivors"],
            "canonicalizer": config["canonicalizer"],
            "expected_equation_import": config["expected_equation_import"],
        }
    )


def _copy_boundary_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config = _load(ROOT / CONFIG_PATH)
    bindings = config["source_bindings"]
    assert isinstance(bindings, dict)
    paths = {CONFIG_PATH, SOURCE_PATH, TEST_PATH}
    paths.update(str(item["path"]) for item in bindings.values())
    for relative in paths:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path, config


def _update_bound_json(
    root: Path,
    config: dict[str, object],
    role: str,
    value: dict[str, object],
) -> None:
    bindings = config["source_bindings"]
    assert isinstance(bindings, dict)
    descriptor = bindings[role]
    assert isinstance(descriptor, dict)
    path = root / str(descriptor["path"])
    if role == "dossier_artifact":
        _write_legacy(path, value)
    else:
        _write(path, value)
    descriptor["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if descriptor["content_sha256"] is not None:
        descriptor["content_sha256"] = value["content_sha256"]
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write(root / CONFIG_PATH, config)


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return build_boundary(root=ROOT)


def test_clean_boundary_passes_with_exact_classification(result: dict[str, object]) -> None:
    assert result["decision"] == "pass"
    assert result["first_blocker"] is None
    assert result["counts"] == {
        "survivors": 4,
        "present_in_corpus": 0,
        "absent_from_this_corpus": 4,
        "ambiguous": 0,
        "survivor_equivalence_classes": 2,
        "equation_universe_records": 18,
        "scalable_dossier_records": 163,
        "total_corpus_records": 181,
        "comparable_corpus_records": 0,
        "leakage_hits": 0,
    }


def test_registered_equivalence_groups_three_x_survivors(
    result: dict[str, object],
) -> None:
    classes = result["survivor_equivalence_classes"]
    assert isinstance(classes, list)
    cardinalities = sorted(item["cardinality"] for item in classes)
    assert cardinalities == [1, 3]
    x_class = next(item for item in classes if item["cardinality"] == 3)
    assert x_class["survivor_artifact_ids"] == [
        "sig-12478959ec4d761f4daee94e",
        "sig-4577ed4e2ef2252a4e84b389",
        "sig-4b6939f52ab0aa46e1df878a",
    ]


def test_x_plus_two_is_a_distinct_registered_class(result: dict[str, object]) -> None:
    survivors = result["survivors"]
    assert isinstance(survivors, list)
    row = next(item for item in survivors if item["artifact_id"].startswith("sig-0d97"))
    assert row["canonical_form"]["normal_form"] == {
        "coefficient": 1,
        "integer_offset": 2,
        "variable": "x",
    }
    assert row["equivalent_survivor_artifact_ids"] == []


def test_absence_is_explicitly_not_novelty(result: dict[str, object]) -> None:
    assert result["claims"]["absence_means_novelty"] is False
    assert result["claims"]["external_search_complete"] is False
    assert result["claims"]["promotion_authorized"] is False
    assert all(item["absence_is_novelty"] is False for item in result["survivors"])
    assert "Absence is not novelty" in result["scope"]


def test_snapshot_replays_equation_graph_and_dossier_registry(
    result: dict[str, object],
) -> None:
    snapshot = result["corpus_snapshot"]
    assert snapshot["equation_universe"] == {
        "content_sha256": "184a77ab27dd95d34a26c592ab4cce20cfb485c7ac4c1151ca57faf7d7791a92",
        "graph_content_sha256": "3cc26065d80f543e59619b8945dfb6ed94640a35e9cde9eab807f24ac5dbd558",
        "record_count": 18,
        "equivalence_edge_count": 1,
        "canonical_namespace": "equation_universe/domain/semantic_hash",
    }
    assert snapshot["scalable_dossier"]["record_count"] == 163
    assert snapshot["scalable_dossier"]["dossier_registry_root_sha256"] == (
        "d6716bc8ad06d82bceed5b92dbca661e221d1f31b45a8d3a60d2b2e615f2f45f"
    )


def test_live_validation_replays_every_binding(result: dict[str, object]) -> None:
    validate_boundary(result, root=ROOT)


def test_checked_receipt_has_exact_live_parity(result: dict[str, object]) -> None:
    checked = _load(ROOT / OUTPUT_PATH)
    assert canonical_json_bytes(checked) == canonical_json_bytes(result)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("counts", "absent_from_this_corpus"), 3),
        (("decision",), "block"),
        (("claims", "absence_means_novelty"), True),
        (("survivors", 0, "status"), "present_in_corpus"),
    ],
)
def test_receipt_tampering_fails_exact_replay(
    result: dict[str, object], path: tuple[object, ...], replacement: object
) -> None:
    tampered = copy.deepcopy(result)
    cursor: object = tampered
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(PriorArtBoundaryError, match="exact live replay"):
        validate_boundary(tampered, root=ROOT)


def test_unknown_config_key_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_boundary_root(tmp_path)
    config["unregistered"] = True
    _write(root / CONFIG_PATH, config)
    with pytest.raises(PriorArtBoundaryError, match="config keys changed"):
        build_boundary(root=root)


def test_boolean_resource_limit_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_boundary_root(tmp_path)
    config["limits"]["maximum_corpus_records"] = True
    _write(root / CONFIG_PATH, config)
    with pytest.raises(PriorArtBoundaryError, match="resource limits changed"):
        build_boundary(root=root)


def test_bound_source_hash_tampering_fails_before_classification(tmp_path: Path) -> None:
    root, _ = _copy_boundary_root(tmp_path)
    path = root / "configs/equation_universe/source_policy.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(PriorArtBoundaryError, match="file hash changed"):
        build_boundary(root=root)


def test_resealed_survivor_identity_tamper_fails_sigma_core_replay(tmp_path: Path) -> None:
    root, config = _copy_boundary_root(tmp_path)
    campaign = _load(root / "runs/math/comprehensive-alpha-cross-generator/campaign.json")
    campaign["pareto"]["candidates"][0]["representation"]["variant"] = 9
    _reseal(campaign)
    _update_bound_json(root, config, "cross_campaign_artifact", campaign)
    with pytest.raises(PriorArtBoundaryError, match="Sigma Core replay"):
        build_boundary(root=root)


def test_resealed_missing_pareto_survivor_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_boundary_root(tmp_path)
    campaign = _load(root / "runs/math/comprehensive-alpha-cross-generator/campaign.json")
    campaign["pareto"]["candidates"].pop()
    _reseal(campaign)
    _update_bound_json(root, config, "cross_campaign_artifact", campaign)
    with pytest.raises(PriorArtBoundaryError, match="survivor set changed"):
        build_boundary(root=root)


def test_dossier_content_tamper_without_reseal_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_boundary_root(tmp_path)
    dossier = _load(root / "runs/engine/scalable-candidate-explanation-dossier-bridge.json")
    dossier["interpretation"] = "tampered"
    bindings = config["source_bindings"]
    descriptor = bindings["dossier_artifact"]
    path = root / descriptor["path"]
    _write_legacy(path, dossier)
    descriptor["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config["snapshot_sha256"] = _snapshot_sha(config)
    _write(root / CONFIG_PATH, config)
    with pytest.raises(PriorArtBoundaryError, match="content seal changed"):
        build_boundary(root=root)


def test_resealed_corpus_identity_leakage_classifies_ambiguous_and_blocks(
    tmp_path: Path,
) -> None:
    root, config = _copy_boundary_root(tmp_path)
    dossier = _load(root / "runs/engine/scalable-candidate-explanation-dossier-bridge.json")
    dossier["interpretation"] += " sig-0d97ae9fd3ccfe78b67f30c6"
    _reseal_legacy(dossier)
    _update_bound_json(root, config, "dossier_artifact", dossier)
    blocked = build_boundary(root=root)
    assert blocked["decision"] == "block"
    assert blocked["counts"]["ambiguous"] == 1
    assert blocked["counts"]["absent_from_this_corpus"] == 3
    assert blocked["leakage_audit"]["hit_count"] == 1
    validate_boundary(blocked, root=root)


def test_equation_seed_reseal_cannot_bypass_adapter_audit(tmp_path: Path) -> None:
    root, config = _copy_boundary_root(tmp_path)
    seed = _load(root / "configs/equation_universe/gravity_seed_v1.json")
    seed["equations"].pop()
    _update_bound_json(root, config, "equation_seed", seed)
    with pytest.raises((PriorArtBoundaryError, ValueError)):
        build_boundary(root=root)


def test_tournament_and_d2_drafts_are_outside_dependency_closure(
    result: dict[str, object],
) -> None:
    snapshot = result["source_bindings"]["snapshot"]
    bound_paths = [item["path"] for item in snapshot.values()]
    assert all("prospective" not in path for path in bound_paths)
    assert all("quartic_registered_direction" not in path for path in bound_paths)
    assert result["claims"]["prospective_tournament_used"] is False


def test_boundary_never_opens_sqlite(result: dict[str, object]) -> None:
    scanned = result["leakage_audit"]["scanned_roles"]
    snapshot = result["source_bindings"]["snapshot"]
    assert all(not snapshot[role]["path"].endswith(".sqlite") for role in scanned)


def test_domain_and_artifact_kind_are_not_erased(result: dict[str, object]) -> None:
    compatibility = result["corpus_snapshot"]["compatibility_boundary"]
    assert compatibility["comparable_corpus_record_count"] == 0
    assert "do not erase domain or artifact-kind boundaries" in compatibility["reason"]
    assert result["canonicalizer"]["forbidden_rules"] == [
        "approximate_numeric_equality",
        "field_relabeling",
        "integration_by_parts",
        "physical_family_aliasing",
        "statement_text_similarity",
    ]
