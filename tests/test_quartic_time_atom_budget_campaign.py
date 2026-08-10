import json
from pathlib import Path

from sigma_theory_compiler.quartic_time_atom_budget_campaign import (
    generic_coordinate_atom_time_evolution_control,
    generic_marked_time_chain_control,
    run_quartic_time_atom_budget_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
LOW_PATH = RUNS / "quartic-low-frequency-symbol-extension-campaign" / "campaign.json"
R3_PATH = RUNS / "quartic-r3-sobolev-calculus-campaign" / "campaign.json"
SOURCE_PATH = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"
PDE_PATH = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
CONFIG_PATH = ROOT / "configs" / "backgrounds" / "quartic_time_atom_budget_campaign.json"
ARTIFACT_PATH = RUNS / "quartic-time-atom-budget-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_coordinate_atom_evolution_and_marked_chain_controls_are_exact() -> None:
    atom_passed, atom = generic_coordinate_atom_time_evolution_control()
    chain_passed, chain = generic_marked_time_chain_control()
    assert atom_passed
    assert chain_passed
    assert atom["coordinate_atom_counts"]["total"] == 153
    assert set(atom["commuting_partial_residuals"].values()) == {"0"}
    assert atom["minimal_integer_state_sobolev_order"] == 7
    assert atom["insufficient_H6_negative"]["rejected"]
    assert atom["negative_control"]["rejected"]
    assert set(chain["source_spatial_residuals"].values()) == {"0"}
    assert set(chain["marked_time_residuals"].values()) == {"0"}
    assert chain["negative_control"]["rejected"]


def test_all_candidates_close_time_atoms_and_time_k55_from_h7_state() -> None:
    result = run_quartic_time_atom_budget_campaign(
        _load(LOW_PATH),
        _load(R3_PATH),
        _load(SOURCE_PATH),
        _load(PDE_PATH),
        _load(CONFIG_PATH),
    )
    assert result["status"] == "pass_all_12_H7_closed_coordinate_atom_time_budgets"
    assert result["counts"] == {
        "selected": 12,
        "H7_time_atom_budgets_passed": 12,
        "rejected": 0,
    }
    assert all(
        len(item["closed_coordinate_atom_time_jets"]) == 4
        and len(item["closed_time_K55_bounds"]) == 10
        and len(item["source_spatial_chain_bounds"]) == 5
        and set(item["published_R3_placeholder_residuals"].values()) == {"0"}
        and not item["derivative_accounting"]["undefined_partial_t_Y_norm_remaining"]
        and item["sufficient_H7_state_radius_for_coordinate_tube"]["numeric"] > 2e-12
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_insufficient_state_order_and_corrupt_provenance_reject() -> None:
    low = _load(LOW_PATH)
    r3 = _load(R3_PATH)
    source = _load(SOURCE_PATH)
    pde = _load(PDE_PATH)
    config = _load(CONFIG_PATH)

    insufficient = dict(config)
    insufficient["state_sobolev_order"] = 6
    result = run_quartic_time_atom_budget_campaign(
        low, r3, source, pde, insufficient
    )
    assert result["status"] == "reject"
    assert "state H7" in result["errors"][0]

    corrupt = json.loads(json.dumps(pde))
    corrupt["certificates"][0]["candidate_id"] = "corrupt"
    result = run_quartic_time_atom_budget_campaign(
        low, r3, source, corrupt, config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
