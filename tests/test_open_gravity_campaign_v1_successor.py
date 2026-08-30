from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_campaign_v1_successor as successor

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_predecessor_are_exact() -> None:
    config = successor.load_config()
    verified = successor.verify_predecessor(ROOT, config)
    assert len(verified["verified"]) == 8
    assert config["continuation_semantics"]["second_formula_campaign"] is False
    assert config["continuation_semantics"]["response_access_attempt_ordinal"] == 2


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("continuation_semantics", "post_freeze_formula_changes"), 1),
        (("continuation_semantics", "second_formula_campaign"), True),
        (("failure_retention_policy", "invalid_cell_can_rank"), True),
        (("accounting", "cumulative_conservative_cell_charge"), 1848),
        (("scope", "network_calls"), 1),
        (("claim_ceiling", "new_theory_claim"), True),
        (("output_paths", "result"), "work/forged.json"),
    ],
)
def test_config_mutations_fail(path: tuple[str, ...], value: object) -> None:
    config = successor.load_config()
    mutated = copy.deepcopy(config)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(successor.OpenGravityCampaignSuccessorError):
        successor.validate_config(mutated)


def test_candidate_gate_failure_is_sanitized_and_unranked() -> None:
    error = successor.base.adapter.OpenGravityStaticRadialAdapterError("PRIVATE SOURCE VALUE")
    row = successor._failure_row(error, "S1", "OBJECT-1")
    assert row == {
        "scenario_id": "S1",
        "object": "OBJECT-1",
        "failure_code": "OPEN_GRAVITY_STATIC_RADIAL_ADAPTER_ERROR",
        "raw_message_retained": False,
        "raw_exception_class_retained": False,
    }
    assert "PRIVATE" not in str(row)


def test_unexpected_implementation_error_remains_terminal() -> None:
    error = RuntimeError("implementation defect")
    with pytest.raises(RuntimeError, match="implementation defect"):
        successor._failure_row(error, "S1", "OBJECT-1")


def test_sparc_scoring_retains_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    galaxies = [SimpleNamespace(name="G1"), SimpleNamespace(name="G2")]
    scenarios = [{"cell_id": "S1"}, {"cell_id": "S2"}]
    candidate = {
        "candidate_id": "CANDIDATE",
        "anonymous_formula_id": "ANON",
        "lane": "CORE",
    }
    cell = {"cell_id": "CELL"}
    context = {"packet": {"twell_rows": []}}
    monkeypatch.setattr(successor.base, "_eligible_cells", lambda *_args: [(candidate, cell)])

    def synthetic_source(galaxy: object, scenario: object) -> dict[str, object]:
        return {
            "bundle": {"name": f"{galaxy.name}-{scenario['cell_id']}"},
            "radius_m": np.asarray([1.0]),
            "vbar2": np.asarray([1.0]),
        }

    monkeypatch.setattr(successor.base, "_sparc_source", synthetic_source)
    monkeypatch.setattr(
        successor.base,
        "_factor_for_cell",
        lambda *_args: (
            (_ for _ in ()).throw(
                successor.base.adapter.OpenGravityStaticRadialAdapterError("private")
            )
            if _args[3]["name"] == "G1-S1"
            else (np.asarray([1.0]), {"PASS": True})
        ),
    )
    monkeypatch.setattr(successor.base, "_factor_on_radii", lambda factor, *_args: factor)
    monkeypatch.setattr(
        successor.base,
        "_sparc_score",
        lambda *_args: {
            "loss": 1.0,
            "row_count": 1,
            "worst_radius": 1.0,
            "worst_standardized_square": 1.0,
        },
    )

    rows = successor._score_sparc_resilient({}, context, galaxies, scenarios)
    assert len(rows) == 1
    assert rows[0]["valid"] is False
    assert rows[0]["robust_loss"] is None
    assert rows[0]["gate_failure_count"] == 1
    assert sum(len(row["objects"]) for row in rows[0]["scenario_results"]) == 3


def test_invalid_cell_cannot_pass_adjudication() -> None:
    result = {
        "cell_id": "CELL",
        "concept_id": "CONCEPT",
        "valid": False,
        "gate_failure_count": 1,
        "gate_failures": [
            {
                "scenario_id": "S1",
                "object": "G1",
                "failure_code": "OPEN_GRAVITY_STATIC_RADIAL_ADAPTER_ERROR",
                "raw_message_retained": False,
                "raw_exception_class_retained": False,
            }
        ],
        "scenario_results": [],
    }
    rows = successor._adjudicate_resilient("GALAXIES", [result], {}, {}, {})
    assert rows[0]["passes"] is False
    assert rows[0]["gates"]["SOURCE_OPERATOR_VALID"] is False
    assert rows[0]["minimum_loo_improvement"] is None


def test_cross_domain_invalid_cell_is_not_a_survivor() -> None:
    row = {
        "cell_id": "CELL",
        "concept_id": "CONCEPT",
        "passes": False,
        "scenario_evidence": [],
    }
    cross = successor._cross_resilient([row], [row])
    assert cross[0]["cross_domain_pass"] is False
    assert cross[0]["combined_worst_fractional_improvement"] is None
    assert cross[0]["invalid_domains"] == ["GALAXIES", "CLUSTERS"]


def test_preflight_is_zero_new_access_and_no_clobber() -> None:
    preflight = successor.build_preflight()
    assert preflight["zero_new_response_access_at_freeze"] == successor.base.ZERO_ACCESS
    assert preflight["manifest"]["candidate_count"] == 407
    assert preflight["manifest"]["parameter_cell_count"] == 2486
    assert preflight["accounting"]["cumulative_conservative_cell_charge"] == 3696


def test_access_intent_precedes_compute() -> None:
    source = successor.MODULE_PATH.read_text(encoding="utf-8")
    execute = source[source.index("def execute()") : source.index("def check_result()")]
    assert execute.index("_atomic_no_clobber(root / ACCESS_INTENT_PATH") < execute.index(
        "computed = _compute()"
    )


def test_artifact_path_contract_is_exact() -> None:
    _manifest, context = successor.base.build_manifest(ROOT)
    paths = successor._artifact_paths(context)
    assert len(paths) == len(set(paths)) == 156
    assert all(path.startswith(successor.ARTIFACT_DIRECTORY.as_posix()) for path in paths)
