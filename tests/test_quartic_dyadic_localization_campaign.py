import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_dyadic_localization_campaign import (
    generic_dyadic_localization_control,
    run_quartic_dyadic_localization_campaign,
    validate_quartic_dyadic_localization_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
R3 = RUNS / "quartic-r3-sobolev-calculus-campaign" / "campaign.json"
EVOLUTION = RUNS / "quartic-evolution-symbol-campaign" / "campaign.json"
FIRST_ORDER = RUNS / "quartic-first-order-reduction-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_dyadic_localization_campaign.json"
ARTIFACT = RUNS / "quartic-dyadic-localization-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_partition_energy_bernstein_and_loss_controls_are_exact() -> None:
    passed, control = generic_dyadic_localization_control()
    assert passed
    assert control["partition"]["maximum_nonzero_ordinary_multipliers"] == 2
    assert control["partition"]["maximum_simultaneous_enlarged_multiplier_overlap"] == 4
    assert control["partition"]["ordinary_shells_interacting_with_one_enlarged_shell"] == 5
    assert control["partition"]["finite_telescoping_residual"] == "0"
    assert control["H7_equivalence"]["lower"] == "2^-15"
    assert control["H7_equivalence"]["upper"] == "2^14"
    assert control["H7_equivalence"]["lower_base_inequality_residual"] == "3/4"
    assert control["H7_equivalence"]["upper_base_inequality_residual"] == "3"
    assert control["shell_local_commutator"]["integration_by_parts_integrand_residual"] == "0"
    assert control["derivative_loss_negative"]["growth_exponent"] == 1
    assert control["derivative_loss_negative"]["rejected"]
    assert control["derivative_loss_negative"]["R3_Schwartz_Fourier_support_counterexample_encoded"]
    assert control["derivative_loss_negative"]["low_shell_separation_margin_at_N4"] == "7/32"


def test_all_candidates_pass_local_framework_and_fail_closed_globally() -> None:
    result = run_quartic_dyadic_localization_campaign(
        _load(R3), _load(EVOLUTION), _load(FIRST_ORDER), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed"
    )
    assert result["counts"]["dyadic_local_frameworks_passed"] == 12
    assert result["counts"]["full_H7_commutators_closed"] == 0
    assert all(
        item["shell_local_commutator_bound_certified"]
        and item["shell_local_commutator_bound"]["uniform_in_shell_index"]
        and item["shell_local_commutator_bound"]["per_coordinate_to_Euclidean_gradient_factor"]
        == "sqrt(3)"
        and not item["conditional_monotone_dyadic_summation"]["applied"]
        and not item["full_H7_commutator_closed"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_incompatible_regularity_contract_or_corrupting_hash_rejects() -> None:
    r3, evolution, first_order, config = map(_load, (R3, EVOLUTION, FIRST_ORDER, CONFIG))
    insufficient = dict(config)
    insufficient["coefficient_sobolev_order"] = 7
    result = run_quartic_dyadic_localization_campaign(r3, evolution, first_order, insufficient)
    assert result["status"] == "reject"

    corrupt = json.loads(json.dumps(evolution))
    corrupt["certificates"][0]["candidate_id"] = "corrupt"
    result = run_quartic_dyadic_localization_campaign(r3, corrupt, first_order, config)
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]


def test_public_validator_rejects_resealed_false_global_closure() -> None:
    artifact, config = _load(ARTIFACT), _load(CONFIG)
    validate_quartic_dyadic_localization_artifact(artifact, ROOT, config)
    promoted = json.loads(json.dumps(artifact))
    promoted["counts"]["full_H7_commutators_closed"] = 12
    body = {key: value for key, value in promoted.items() if key != "content_sha256"}
    promoted["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        validate_quartic_dyadic_localization_artifact(promoted, ROOT, config)
