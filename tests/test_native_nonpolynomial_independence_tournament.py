from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.native_nonpolynomial_independence_tournament import (
    CLAIMS,
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TARGET_PATH,
    NativeIndependenceError,
    _unseal_targets,
    build_campaign,
    run_generator,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checked() -> dict:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    rebuilt = build_campaign(ROOT)
    validate_campaign(value, root=ROOT)
    assert value == rebuilt
    return value


def _world(value: dict, world_class: str) -> dict:
    return next(row for row in value["world_results"] if row["class"] == world_class)


def _result(world: dict, family: str) -> dict:
    return next(row for row in world["candidate_results"] if row["family"] == family)


def _reseal(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_two_nonpolynomial_worlds_have_multiple_independent_exact_passes(checked: dict) -> None:
    rational = _world(checked, "rational_linear_fractional")
    exponential = _world(checked, "shifted_exponential")

    assert rational["counts"] == {"block": 0, "pass": 2, "reject": 1}
    assert exponential["counts"] == {"block": 0, "pass": 2, "reject": 1}
    assert {row["family"] for row in rational["candidate_results"] if row["status"] == "PASS"} == {
        "reciprocal_elimination",
        "bounded_structure_enumerator",
    }
    assert {
        row["family"] for row in exponential["candidate_results"] if row["status"] == "PASS"
    } == {"difference_ratio", "bounded_structure_enumerator"}

    reciprocal = _result(rational, "reciprocal_elimination")["candidate"]
    geometric = _result(exponential, "difference_ratio")["candidate"]
    assert reciprocal["parameters"] == {
        "a": {"p": 3, "q": 1},
        "b": {"p": 5, "q": 1},
        "c": {"p": 2, "q": 1},
    }
    assert geometric["parameters"] == {
        "a": {"p": 2, "q": 1},
        "b": {"p": 5, "q": 1},
        "r": {"p": 3, "q": 1},
    }


def test_every_pass_has_exact_certificate_and_reject_has_counterexample(checked: dict) -> None:
    results = [row for world in checked["world_results"] for row in world["candidate_results"]]
    passes = [row for row in results if row["status"] == "PASS"]
    rejects = [row for row in results if row["status"] == "REJECT"]
    assert len(passes) == 4
    assert len(rejects) == 2
    assert all(
        row["proof_certificate"]["decision"] == "proved_exact_structured_identity" for row in passes
    )
    for row in passes:
        assert all(
            residual == {"p": 0, "q": 1}
            for residual in row["proof_certificate"]["parameter_residuals"].values()
        )
        assert row["counterexample"] is None
    assert _result(_world(checked, "rational_linear_fractional"), "difference_ratio")[
        "counterexample"
    ] == {
        "candidate_value": {"p": 5, "q": 2},
        "point": 4,
        "residual": {"p": -1, "q": 3},
        "target_value": {"p": 17, "q": 6},
    }
    assert _result(_world(checked, "shifted_exponential"), "reciprocal_elimination")[
        "counterexample"
    ] == {
        "candidate_value": {"p": 7, "q": 1},
        "point": 4,
        "residual": {"p": -160, "q": 1},
        "target_value": {"p": 167, "q": 1},
    }


def test_leave_one_family_out_preserves_every_class(checked: dict) -> None:
    ablation = checked["leave_one_family_out"]
    assert ablation["decision"] == "PASS"
    assert [row["excluded_family"] for row in ablation["rows"]] == [
        "reciprocal_elimination",
        "difference_ratio",
        "bounded_structure_enumerator",
    ]
    assert all(row["all_declared_classes_covered"] for row in ablation["rows"])
    assert all(
        row["class_coverage"] == {"rational_linear_fractional": True, "shifted_exponential": True}
        for row in ablation["rows"]
    )


def test_generic_solver_and_delegation_count_are_explicitly_zero(checked: dict) -> None:
    assert checked["counts"]["generic_exact_linear_solver_calls"] == 0
    assert checked["counts"]["formula_discovery_job_delegations"] == 0
    assert checked["counts"]["bayesian_generator_calls"] == 0
    assert checked["phase_a"]["generic_exact_linear_solver_calls"] == 0
    assert checked["phase_a"]["bayesian_generator_calls"] == 0
    assert checked["claims"] == CLAIMS
    for outcomes in checked["phase_a"]["frozen_generator_outcomes"].values():
        for outcome in outcomes:
            candidate = outcome["candidate"]
            if candidate is not None:
                assert candidate["construction_audit"] == {
                    "bayesian_generator_calls": 0,
                    "formula_discovery_job_delegations": 0,
                    "generic_exact_linear_solver_calls": 0,
                    "target_fields_read": [],
                }


def test_source_has_no_generic_or_formula_job_import_or_call() -> None:
    source = (ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("formula_discovery" in name for name in imports)
    assert not any("linear_solver" in name or name == "sympy" for name in imports)
    assert "exact_linear_basis" not in source
    assert "linsolve(" not in source


def test_candidate_set_is_frozen_before_exactly_one_target_read(
    checked: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = (ROOT / TARGET_PATH).resolve()
    original = Path.read_bytes
    reads = 0

    def audited_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path.resolve() == target:
            reads += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", audited_read_bytes)
    rebuilt = build_campaign(ROOT)
    assert rebuilt == checked
    assert reads == 2
    assert checked["phase_a"]["target_records_read"] == 0
    assert checked["phase_a"]["target_access_enforcement"] == {
        "attempted_reads": 1,
        "denied_content_bytes_exposed": 0,
        "denied_reads": 1,
    }
    assert checked["chronology"][2]["event"] == "phase_a_receipt_sealed"
    assert checked["chronology"][3] == {
        "event": "atomic_two_record_target_unseal",
        "target_reads": 1,
    }


def test_unseal_rejects_missing_or_tampered_phase_a_seal() -> None:
    phase_a = copy.deepcopy(build_campaign(ROOT)["phase_a"])
    phase_a["candidate_generation_complete"] = False
    _reseal(phase_a)
    with pytest.raises(NativeIndependenceError, match="before candidate freeze"):
        _unseal_targets(ROOT, CONFIG, phase_a)


def test_hard_budget_stops_enumerator_without_partial_candidate() -> None:
    world = CONFIG["worlds"][1]
    contract = copy.deepcopy(CONFIG["family_contracts"][2])
    contract["work_budget"] = 1
    outcome = run_generator(world, contract)
    assert outcome.status == "BLOCK"
    assert outcome.blocker == "work_budget_exhausted"
    assert outcome.candidate is None
    assert outcome.work_units == 1


def test_exact_replay_is_deterministic(checked: dict) -> None:
    assert build_campaign(ROOT) == checked
    assert build_campaign(ROOT) == build_campaign(ROOT)
    assert checked["decision"] == "PASS"
    assert checked["counts"] == {
        "bayesian_generator_calls": 0,
        "candidate_blocks": 0,
        "candidate_passes": 4,
        "candidate_rejects": 2,
        "declared_benchmark_classes": 2,
        "exact_counterexamples": 2,
        "exact_identity_certificates": 4,
        "formula_discovery_job_delegations": 0,
        "generator_families": 3,
        "generic_exact_linear_solver_calls": 0,
        "post_unseal_generation_events": 0,
        "target_fixture_reads": 1,
        "target_fixture_reads_denied_before_unseal": 1,
        "worlds": 2,
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["counts"].__setitem__("generic_exact_linear_solver_calls", 1),
        lambda value: value["phase_a"].__setitem__("target_records_read", 1),
        lambda value: value["world_results"][0]["candidate_results"][0]["candidate"][
            "parameters"
        ].__setitem__("a", {"p": 4, "q": 1}),
        lambda value: value["world_results"][1]["candidate_results"][0][
            "counterexample"
        ].__setitem__("residual", {"p": 0, "q": 1}),
        lambda value: value["leave_one_family_out"].__setitem__("decision", "BLOCK"),
        lambda value: value.__setitem__("unknown", True),
    ],
)
def test_resealed_receipt_tamper_fails_exact_replay(checked: dict, mutator) -> None:
    tampered = copy.deepcopy(checked)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(NativeIndependenceError):
        validate_campaign(tampered, root=ROOT)


def test_target_commitment_and_family_identity_tamper_fail_closed(tmp_path: Path) -> None:
    commitment_tamper = copy.deepcopy(CONFIG)
    commitment_tamper["worlds"][0]["sealed_target_sha256"] = "0" * 64
    commitment_path = tmp_path / "commitment.json"
    commitment_path.write_text(json.dumps(commitment_tamper), encoding="utf-8")
    with pytest.raises(NativeIndependenceError, match="commitment did not open"):
        build_campaign(ROOT, commitment_path)

    identity_tamper = copy.deepcopy(CONFIG)
    identity_tamper["family_contracts"][0]["code_identity_sha256"] = "0" * 64
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity_tamper), encoding="utf-8")
    with pytest.raises(NativeIndependenceError, match="code identity"):
        build_campaign(ROOT, identity_path)
