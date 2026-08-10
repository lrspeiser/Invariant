import json
from pathlib import Path

from sigma_theory_compiler.quartic_unspecialized_source_jacobian_campaign import (
    generic_unspecialized_source_jacobian_control,
    run_quartic_unspecialized_source_jacobian_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
CONTRACT = RUNS / "quartic-component-jacobian-contract-campaign" / "campaign.json"
FIRST_ORDER = RUNS / "quartic-first-order-reduction-campaign" / "campaign.json"
EVOLUTION = RUNS / "quartic-evolution-symbol-campaign" / "campaign.json"
PDE = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_unspecialized_source_jacobian_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-unspecialized-source-jacobian-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_block_multiplicities_and_unspecialized_extraction_are_exact() -> None:
    passed, control = generic_unspecialized_source_jacobian_control()
    assert passed
    assert control["source_derivative_chunks"]["completed_chunk_count"] == 9
    assert control["source_derivative_chunks"]["completed_source_entries"] == 1089
    assert control["known_answer_two_field_control"]["zero_entry_count"] == 100
    assert control["unspecialized_block_extraction"]["B_reconstruction_zero"]
    assert control["unspecialized_block_extraction"]["C_reconstruction_zero"]
    assert all(
        item["rejected"] for item in control["negative_controls"].values()
    )


def test_all_candidates_complete_principal_chunks_but_not_remainder() -> None:
    result = run_quartic_unspecialized_source_jacobian_campaign(
        _load(CONTRACT),
        _load(FIRST_ORDER),
        _load(EVOLUTION),
        _load(PDE),
        _load(CONFIG),
    )
    assert result["status"] == (
        "pass_all_12_complete_unspecialized_principal_source_jacobians_"
        "remainder_fail_closed"
    )
    assert result["counts"]["principal_composed_identities_proved"] == 12
    assert result["counts"]["full_source_jacobians_completed"] == 0
    assert result["counts"]["remainder_bounds_proved"] == 0
    assert all(
        item["completion"]["exact_entries_completed"] == 1089
        and item["completion"]["lower_atom_columns_unresolved"] == 54
        and item["principal_composed_identity"]["entry_residuals_proved_zero"]
        == 3025
        and item["principal_composed_identity"]["proved"]
        and not item["paralinearization_remainder_bound_proved"]
        and not item["H7_derivative_loss_resolved"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_false_remainder_and_corrupt_provenance_reject() -> None:
    campaigns = tuple(map(_load, (CONTRACT, FIRST_ORDER, EVOLUTION, PDE)))
    config = _load(CONFIG)
    false_remainder = dict(config)
    false_remainder["declare_remainder_bound_proved"] = True
    result = run_quartic_unspecialized_source_jacobian_campaign(
        *campaigns, false_remainder
    )
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[1]))
    corrupt["certificates"][0]["source_spatial_block_sha256"] = "corrupt"
    result = run_quartic_unspecialized_source_jacobian_campaign(
        campaigns[0], corrupt, *campaigns[2:], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
