import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_global_h7_energy_campaign import (
    generic_global_h7_energy_control,
    run_quartic_global_h7_energy_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
NAMES = (
    "quartic-annular-k55-c6-campaign",
    "quartic-frequency-localized-evolution-campaign",
    "quartic-finite-low-operator-campaign",
    "quartic-reference-equilibrium-campaign",
    "quartic-paradifferential-good-unknown-campaign",
    "quartic-unspecialized-source-jacobian-campaign",
    "quartic-universal-source-dag-campaign",
    "quartic-lower-source-remainder-campaign",
    "quartic-dyadic-localization-campaign",
    "quartic-time-atom-budget-campaign",
    "quartic-nonlinear-evolution-campaign",
    "quartic-coordinate-jet-tube-campaign",
)
PATHS = tuple(RUNS / name / "campaign.json" for name in NAMES)
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_global_h7_energy_campaign.json"
ARTIFACT = RUNS / "quartic-global-h7-energy-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def test_neighbor_sum_conditional_lifespan_and_negatives_are_exact() -> None:
    passed, control = generic_global_h7_energy_control()
    assert passed
    neighbor = control["dyadic_neighbor_summation"]
    assert neighbor["base_ratio_residual"] == "15"
    assert neighbor["maximum_weight_ratio"] == str(2**28)
    assert neighbor["exact_neighbor_sum_constant"] == str(5 * 2**28)
    assert control["finite_ordinary_shells"]["indices"] == list(range(7))
    assert control["finite_ordinary_shells"]["maximum_support_radius"] == 128
    assert control["conditional_lifespan"]["riccati_residual"] == "0"
    assert control["global_remainder_functional"]["required_bound"] == (
        "B7(t)<=C_L(R)*sqrt(Q7(t))+C_B(R)*Q7(t)"
    )
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_sum_known_terms_but_fail_closed_on_B7() -> None:
    result = run_quartic_global_h7_energy_campaign(*_inputs(), _load(CONFIG))
    assert result["status"] == (
        "audit_all_12_global_H7_energies_single_source_remainder_"
        "lifespans_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "global_energy_equivalences_certified": 12,
        "global_nonremainder_summations_certified": 12,
        "leading_good_unknown_bindings_verified": 12,
        "closed_global_H7_inequalities": 0,
        "global_H7_sums_applied": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    assert all(
        item["global_H7_energy_equivalence_certified"]
        and item["global_nonremainder_dyadic_summation_certified"]
        and item["summed_certified_terms"][
            "ordinary_shells_0_through_6_included"
        ]
        and item["good_unknown_and_source"][
            "leading_good_unknown_symbol_binding_verified"
        ]
        and item["strongest_global_differential_inequality"][
            "proved_with_explicit_remainder"
        ]
        and not item["strongest_global_differential_inequality"][
            "closed_Gronwall_inequality"
        ]
        and not item["global_H7_differential_inequality_closed"]
        and not item["global_H7_dyadic_sum_applied"]
        and not item["nonlinear_lifespan_proved"]
        and "C_L(R_tube)*sqrt(Q7(t))"
        in item["bootstrap_and_conditional_lifespan"]["missing_hypothesis"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_omission_corruption_and_false_closure_policies_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    omitted = list(inputs)
    omitted[6] = {}
    result = run_quartic_global_h7_energy_campaign(*omitted, config)
    assert result["status"] == "reject"
    assert "source_dag prerequisite status mismatch" in result["errors"][0]

    for policy in ("source_remainder_policy", "global_H7_policy", "lifespan_policy"):
        false_closure = dict(config)
        false_closure[policy] = "pass"
        result = run_quartic_global_h7_energy_campaign(*inputs, false_closure)
        assert result["status"] == "reject"

    corrupt = json.loads(json.dumps(inputs[6]))
    corrupt["upstream_sha256"]["lower_source_remainder"] = "0" * 64
    _rehash(corrupt)
    inputs[6] = corrupt
    result = run_quartic_global_h7_energy_campaign(*inputs, config)
    assert result["status"] == "reject"
    assert "source-DAG provenance mismatch" in result["errors"][0]
