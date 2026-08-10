import json
from pathlib import Path

from sigma_theory_compiler.quartic_universal_source_dag_campaign import (
    generic_exact_operator_dag_control,
    run_quartic_universal_source_dag_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
LOWER = RUNS / "quartic-lower-source-remainder-campaign" / "campaign.json"
NONLINEAR = RUNS / "quartic-nonlinear-evolution-campaign" / "campaign.json"
SOLVED = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_universal_source_dag_campaign.json"
ARTIFACT = RUNS / "quartic-universal-source-dag-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_operator_semantics_known_answer() -> None:
    passed, control = generic_exact_operator_dag_control()
    assert passed
    assert control["affine_split_residual"] == "0"
    assert set(control["derivative_residuals"].values()) == {"0"}
    assert control["negative_control"]["rejected"]


def test_all_candidates_emit_bounded_unspecialized_checkpoints() -> None:
    result = run_quartic_universal_source_dag_campaign(
        _load(LOWER), _load(NONLINEAR), _load(SOLVED), _load(CONFIG)
    )
    assert result["status"] == (
        "partial_all_12_exact_universal_source_operator_dag_checkpoints"
    )
    assert result["counts"]["pure_derivative_component_roots_per_candidate"] == 88
    assert result["counts"]["affine_splits_proved"] == 0
    for certificate in result["certificates"]:
        assert certificate["expression_dag"]["node_count"] > 153
        assert certificate["exact_component_derivative_roots_emitted"] == 88
        assert certificate["evidence"]["universal_input"][
            "no_coordinate_atom_substitution"
        ]
        assert certificate["evidence"]["negative_controls"][
            "rational_witness_specialization"
        ]["rejected"]
        assert not certificate["universal_acceleration_affine_split_proved"]
        assert not certificate["paralinearization_remainder_bound_proved"]
        assert not certificate["full_H7_commutator_closed"]
    assert result == _load(ARTIFACT)


def test_false_affine_claim_and_corrupt_provenance_reject() -> None:
    campaigns = tuple(map(_load, (LOWER, NONLINEAR, SOLVED)))
    config = _load(CONFIG)
    false_claim = dict(config)
    false_claim["declare_affine_split_proved"] = True
    result = run_quartic_universal_source_dag_campaign(*campaigns, false_claim)
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[1]))
    corrupt["certificates"][0]["source_geometric_formula_contract_sha256"] = "corrupt"
    result = run_quartic_universal_source_dag_campaign(
        campaigns[0], corrupt, campaigns[2], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
