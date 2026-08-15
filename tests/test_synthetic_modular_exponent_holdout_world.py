from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.synthetic_modular_exponent_holdout_world import (
    BENCHMARK_ID,
    CONFIG_PATH,
    OUTPUT_PATH,
    _discover,
    _evaluate_exponent,
    _load_config,
    _select_modulus,
    _write_immutable,
    run,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = _canonical_sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


@pytest.fixture(scope="module")
def checked() -> dict[str, object]:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_campaign(value, ROOT)
    assert value == run(ROOT)
    return value


def test_seeded_prime_world_and_public_discovery_boundary_are_exact(
    checked: dict[str, object],
) -> None:
    config = _load_config(ROOT)
    assert _select_modulus(config) == 11
    world = checked["pre_unseal"]["public_input"]["world"]
    assert world["modulus"] == 11
    assert world["residues"] == list(range(11))
    assert world["nonzero_residues"] == list(range(1, 11))
    assert world["candidate_exponents"] == list(range(1, 11))
    assert len(world["multiplication_table"]) == 11
    assert all(len(row) == 11 for row in world["multiplication_table"])
    assert list(inspect.signature(_discover).parameters) == ["public_world"]
    assert checked["pre_unseal"]["reference_payload_supplied_to_discovery"] is False


def test_bounded_enumeration_finds_unique_least_exponent_and_exact_proof(
    checked: dict[str, object],
) -> None:
    discovery = checked["pre_unseal"]["discovery"]
    assert discovery["grammar"] == {
        "candidate_kind": "positive_integer_exponent",
        "minimum": 1,
        "maximum": 10,
        "candidate_count": 10,
    }
    assert discovery["passing_exponents"] == [10]
    winner = discovery["winner"]
    assert winner["exponent"] == 10
    assert winner["residues_checked"] == 10
    assert winner["passed"] is True
    assert winner["first_failing_residue"] is None
    assert all(row["terminal_residue"] == 1 and row["passed"] for row in winner["rows"])
    for exponent in range(1, 10):
        evaluation = _evaluate_exponent(11, exponent)
        assert evaluation["passed"] is False
        assert evaluation["first_failing_residue"] is not None
    assert checked["proof"] == {
        "method": "exhaustive_nonzero_residue_replay",
        "candidate_exponents_checked": 10,
        "residues_per_candidate": 10,
        "modular_power_obligations": 100,
        "winning_exponent": 10,
        "winning_residues_checked": 10,
        "winning_counterexample_count": 0,
        "minimality_counterexamples": [
            {
                "exponent": exponent,
                "first_failing_residue": _evaluate_exponent(11, exponent)["first_failing_residue"],
            }
            for exponent in range(1, 10)
        ],
    }


def test_chronology_post_unseal_equivalence_and_controls_are_closed(
    checked: dict[str, object],
) -> None:
    assert [row["phase"] for row in checked["chronology"]] == [
        "public_modular_world_generated",
        "reference_theorem_sealed",
        "public_discovery_input_sealed",
        "bounded_exponent_grammar_enumerated",
        "winner_and_exact_proof_sealed",
        "reference_unsealed_and_compared",
    ]
    assert [row["ordinal"] for row in checked["chronology"]] == list(range(1, 7))
    comparison = checked["post_unseal"]["comparison"]
    assert comparison["performed_after_winner_seal"] is True
    assert comparison["reference_exponent"] == comparison["rediscovered_exponent"] == 10
    assert comparison["exact_match"] is True
    controls = {row["control_id"]: row for row in checked["negative_controls"]}
    assert set(controls) == {
        "neighboring_exponent_rejected",
        "truncated_residue_evidence_rejected",
        "composite_modulus_fermat_shape_rejected",
    }
    assert controls["neighboring_exponent_rejected"]["rejected"] is True
    assert controls["truncated_residue_evidence_rejected"] == {
        "control_id": "truncated_residue_evidence_rejected",
        "candidate_exponent": 9,
        "truncated_residues": [1],
        "truncated_check_would_pass": True,
        "full_replay_rejected": True,
    }
    assert controls["composite_modulus_fermat_shape_rejected"]["rejected"] is True


def test_claims_scope_bindings_and_broad_boundaries_fail_closed(
    checked: dict[str, object],
) -> None:
    assert checked["benchmark_id"] == BENCHMARK_ID
    assert checked["decision_counts"] == {"pass": 1, "reject": 0, "blocked": 0}
    claims = checked["claims"]
    for name in (
        "general_number_theory_completeness_established",
        "unbounded_exponent_discovery_established",
        "historical_novelty_established",
        "formal_proof_assistant_kernel_checked",
        "hostile_process_isolation_established",
        "external_mathematical_significance_established",
    ):
        assert claims[name] is False
    assert "one deterministic anonymous prime residue world modulo 11" in checked["scope"]
    assert set(checked["source_bindings"]) == {"config", "source", "test"}
    for binding in checked["source_bindings"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["file_sha256"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("proof", "modular_power_obligations"), 99),
        (("pre_unseal", "reference_payload_supplied_to_discovery"), True),
        (("post_unseal", "comparison", "exact_match"), False),
        (("claims", "historical_novelty_established"), True),
        (("source_bindings", "test", "path"), "tests/forged.py"),
        (("negative_controls", 0, "rejected"), False),
    ],
)
def test_resealed_semantic_mutations_fail_replay(
    checked: dict[str, object], path: tuple[object, ...], replacement: object
) -> None:
    forged = copy.deepcopy(checked)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _reseal(forged)
    with pytest.raises(ValueError, match="contract changed|immutable replay mismatch"):
        validate_campaign(forged, ROOT)


def test_unknown_top_level_key_and_config_mutation_fail_closed(
    checked: dict[str, object], tmp_path: Path
) -> None:
    forged = copy.deepcopy(checked)
    forged["unsupported"] = True
    _reseal(forged)
    with pytest.raises(ValueError, match="result keys changed"):
        validate_campaign(forged, ROOT)

    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    config["world_generator"]["prime_candidates"].append(23)
    config_path = tmp_path / CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config contract changed"):
        _load_config(tmp_path)


def test_immutable_writer_is_idempotent_and_refuses_replacement(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    _write_immutable(path, {"state": "bounded_pass"})
    before = path.read_bytes()
    _write_immutable(path, {"state": "bounded_pass"})
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError, match="differs"):
        _write_immutable(path, {"state": "false_success"})
