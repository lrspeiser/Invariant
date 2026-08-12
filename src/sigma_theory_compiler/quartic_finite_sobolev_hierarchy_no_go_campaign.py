"""Finite-order Sobolev derivative-loss no-go for the quartic recovery chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-finite-sobolev-hierarchy-no-go-config-1.0"
RESULT_SCHEMA = "sigma-quartic-finite-sobolev-hierarchy-no-go-campaign-1.0"
CAMPAIGN_ID = "quartic-finite-sobolev-hierarchy-no-go-001"
FIRST_BLOCKER = (
    "candidate_bound_full_tensor_paradifferential_cancellation_or_derivative_loss_"
    "evolution_theorem_for_the_coefficient_high_state_low_branch"
)
EXPECTED_PREDECESSORS = {
    "anti_wick_composition": {
        "path": "runs/physics-language/quartic-anti-wick-composition-campaign/campaign.json",
        "file_sha256": "9a9cb443ee86a5b5d45ba29ea1287442b101f8c675681f6eedaa927d33f41f1e",
        "content_sha256": "02c98ac16a6cd4bc3871003fb77918e21666a60fec65bc28c85484a6011c541d",
    },
    "annular_c6": {
        "path": "runs/physics-language/quartic-annular-k55-c6-campaign/campaign.json",
        "file_sha256": "bcc2b4184e5bcfb64d9a8a24ca095aa4067c18502c0c2f4956dcd8ad6f7fc527",
        "content_sha256": "55fa580fb91e37f48a8e6bd39c4c172c9aa4b3960336d42b88ceb211331b4e2f",
    },
    "bounded_frequency_defect": {
        "path": "runs/physics-language/quartic-bounded-frequency-defect-campaign/campaign.json",
        "file_sha256": "e2dd669e0a939558d7379ac3600032eb7bca22e550d6965816ceca5e2724187a",
        "content_sha256": "56ea95a21af505cf1a75f1fe757b947ee13039a241e91b1853e305fbfee7514a",
    },
    "dyadic_localization": {
        "path": "runs/physics-language/quartic-dyadic-localization-campaign/campaign.json",
        "file_sha256": "859b472f666cae9175aa7da8bc90ef175ca16f1987b967d65c71f5cc14139c94",
        "content_sha256": "ce7afcaf428144cc7149dfdd67be5139d09a2e33d3d2bb8a19b867799313f3b5",
    },
    "high_atom_d2": {
        "path": "runs/physics-language/quartic-high-atom-d2-good-unknown-campaign/campaign.json",
        "file_sha256": "5848e62c811baf4a005e821d73c3dcc6d29a285fa2be57cfbe6842b56dfd3513",
        "content_sha256": "5b6a5c43d9e22c2780f3987e3271b8c863c802129b3837777da246a5d635b466",
    },
}
EXPECTED_STATUSES = {
    "anti_wick_composition": "pass_exact_anti_wick_composition_prerequisite_audit_C6_required",
    "annular_c6": "pass_all_12_targeted_annular_K55_C6_principal_composition_constants",
    "bounded_frequency_defect": "pass_all_12_actual_P55_compact_frequency_defect_KN_L2_lemmas",
    "dyadic_localization": "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
    "high_atom_d2": (
        "pass_all_12_exact_representative_D2_obstructions_named_good_unknown_"
        "cancellation_refuted_global_H7_fail_closed"
    ),
}
EXPECTED_COUNTS = {
    "selected": 12,
    "candidate_pass": 0,
    "candidate_reject": 0,
    "candidate_blocked": 12,
    "finite_order_direct_hierarchy_no_go_certificates": 12,
    "nonzero_candidate_D2_slices": 12,
    "named_good_unknown_slice_cancellations_refuted": 12,
    "conditional_one_derivative_remedies_replayed": 12,
    "autonomous_finite_Sobolev_closures": 0,
    "full_tensor_cancellations_proved": 0,
    "global_H7_closures": 0,
    "lifespans_proved": 0,
}
EXPECTED_CLAIM_SEALS = {
    "full_tensor_good_unknown_proved": False,
    "all_possible_modified_energies_refuted": False,
    "Nash_Moser_or_derivative_loss_wellposedness_proved": False,
    "analytic_or_Gevrey_closure_proved": False,
    "autonomous_H7_energy_closed": False,
    "autonomous_H8_energy_closed": False,
    "global_dyadic_energy_closed": False,
    "nonlinear_lifespan_proved": False,
    "candidate_theory_rejected": False,
    "observational_claim_made": False,
}
EXPECTED_DATA_SEALS = {
    "observations_opened": False,
    "solar_system_inputs_opened": False,
    "cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
EXPECTED_RESULT_KEYS = {
    "schema_version",
    "campaign_id",
    "decision",
    "decision_counts",
    "gate_counts",
    "first_blocker",
    "theorem",
    "recovery_chain_audit",
    "exact_controls",
    "candidate_records",
    "secondary_blockers",
    "claim_seals",
    "data_seals",
    "scope",
    "source_bindings",
    "content_sha256",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("finite-Sobolev path escapes repository") from error
    return path


def _bound_artifact(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("finite-Sobolev predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("finite-Sobolev predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("finite-Sobolev predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "minimum_integer_order",
            "witness_orders",
            "policies",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path")
        != "runs/physics-language/quartic-finite-sobolev-hierarchy-no-go-campaign/campaign.json"
        or config.get("predecessors") != EXPECTED_PREDECESSORS
        or config.get("minimum_integer_order") != 4
        or config.get("witness_orders") != [7, 8, 9, 12]
        or config.get("policies")
        != {
            "theorem_class": "unmodified_componentwise_finite_Sobolev_hierarchy",
            "full_tensor_cancellation": "fail_closed",
            "derivative_loss_evolution": "fail_closed",
            "global_H7": "fail_closed",
            "lifespan": "fail_closed",
            "candidate_rejection": "forbidden",
        }
        or config.get("seals") != EXPECTED_DATA_SEALS
    ):
        raise ValueError("finite-Sobolev config boundary changed")


def _validate_predecessors(predecessors: Mapping[str, Mapping[str, Any]]) -> None:
    if set(predecessors) != set(EXPECTED_PREDECESSORS):
        raise ValueError("finite-Sobolev predecessor set changed")
    if any(
        predecessors[label].get("status") != EXPECTED_STATUSES[label]
        for label in EXPECTED_STATUSES
    ):
        raise ValueError("finite-Sobolev predecessor status changed")
    anti = predecessors["anti_wick_composition"]
    annular = predecessors["annular_c6"]
    defect = predecessors["bounded_frequency_defect"]
    dyadic = predecessors["dyadic_localization"]
    d2 = predecessors["high_atom_d2"]
    if (
        anti.get("counts", {}).get("anti_wick_compositions_closed") != 0
        or annular.get("counts", {}).get("targeted_C6_bounds_passed") != 12
        or annular.get("counts", {}).get("full_dyadic_energies_closed") != 0
        or defect.get("generic_compact_frequency_defect_control", {})
        .get("physical_scale_contract", {})
        .get("high_shell_defect_zero")
        is not True
        or dyadic.get("generic_dyadic_localization_control", {})
        .get("derivative_loss_negative", {})
        .get("growth_exponent")
        != 1
        or dyadic.get("counts", {}).get("full_H7_commutators_closed") != 0
        or d2.get("counts", {}).get("nonzero_obstructions") != 12
        or d2.get("counts", {}).get("named_good_unknown_cancellations_refuted")
        != 12
    ):
        raise ValueError("finite-Sobolev recovery boundary changed")


def _order_ledger(order: int) -> dict[str, Any]:
    if not isinstance(order, int) or order < 4:
        raise ValueError("finite-Sobolev order must be an integer at least four")
    return {
        "order": order,
        "coefficient_packet": f"a_N=N^({1-order}) exp(iNx_1)a_0",
        "coefficient_H_order_minus_1_exponent": (1 - order) + (order - 1),
        "fixed_low_state_H_order_exponent": 0,
        "high_low_output_H_order_exponent": (1 - order) + order,
        "conditional_next_order_state_packet": f"U_N=N^(-{order + 1}) exp(iNx_1)U_0",
        "conditional_coefficient_H_order_exponent": -order + order,
        "restarted_top_order": order + 1,
        "restarted_high_low_output_exponent": -order + (order + 1),
    }


def _theorem(orders: list[int]) -> dict[str, Any]:
    return {
        "name": "finite_unmodified_Sobolev_hierarchy_derivative_loss_no_go",
        "domain": "R3 compact-frequency Schwartz packets; integer s>=4",
        "operator_class": (
            "one spatial derivative in the coefficient, followed by the uncancelled "
            "coefficient-high/state-low component product or commutator branch"
        ),
        "packet": (
            "a_N=N^(1-s) exp(iNx_1)a_0, N=2^j>=4; u=u_0 fixed low packet"
        ),
        "uniform_inputs": [
            "||a_N||H^(s-1)<=C_(a,s-1) uniformly in N",
            "||u_0||H^s<infinity",
        ],
        "exact_high_shell_lower_bound": (
            "||Delta_j(a_N partial_1 u_0)||H^s >= N*c_packet/2"
        ),
        "candidate_slice_lower_bound": (
            "||D2_slice*Delta_j(a_N partial_1 u_0)||H^s "
            ">=abs(-2*a10)*N*c_packet/2"
        ),
        "conclusion": (
            "no N-uniform direct estimate of this uncancelled branch by "
            "||a||H^(s-1)||u||H^s exists at any finite integer s>=4"
        ),
        "hierarchy_recurrence": (
            "an a priori H^(s+1) state conditionally controls the H^s branch, but "
            "restarting the same unmodified energy at s+1 produces exponent +1 again"
        ),
        "proof_ledger": [_order_ledger(order) for order in orders],
        "scope_limit": (
            "this does not rule out a full tensor cancellation, modified energy, "
            "Nash-Moser/derivative-loss evolution, or analytic/Gevrey closure"
        ),
    }


def _recovery_chain_audit() -> dict[str, Any]:
    return {
        "anti_Wick_composition": {
            "registered_effect": "finite order-zero energy-composition prerequisite",
            "spatial_derivative_gain": 0,
            "can_cancel_candidate_D2_slice_without_identity": False,
        },
        "annular_C6_constants": {
            "registered_effect": "finite principal composition constants on the annulus",
            "spatial_derivative_gain": 0,
            "full_dyadic_energy_closed": False,
        },
        "bounded_frequency_defect": {
            "registered_effect": "compact physical low-frequency defect paid once",
            "high_shell_value_on_witness": 0,
            "can_cancel_high_shell_growth": False,
        },
        "dyadic_localization": {
            "registered_effect": "selects the N=2^j packet with multiplier lower 1/2",
            "witness_growth_exponent": 1,
            "global_commutator_closed": False,
        },
        "net_conclusion": (
            "the registered recovery operations multiply by finite shell-uniform constants "
            "or vanish on the high shell; none supplies a negative derivative or a bound "
            "cancelling the candidate D2 coefficient-high/state-low slice"
        ),
    }


def _controls(orders: list[int]) -> dict[str, Any]:
    ledgers = [_order_ledger(order) for order in orders]
    return {
        "positive_exponent_replay": {
            "orders": orders,
            "input_exponents_zero": all(
                row["coefficient_H_order_minus_1_exponent"] == 0 for row in ledgers
            ),
            "output_exponents_one": all(
                row["high_low_output_H_order_exponent"] == 1 for row in ledgers
            ),
            "restarted_output_exponents_one": all(
                row["restarted_high_low_output_exponent"] == 1 for row in ledgers
            ),
        },
        "promote_conditional_H8_to_autonomous_H8": {
            "s8_restarted_growth_exponent": _order_ledger(8)[
                "restarted_high_low_output_exponent"
            ],
            "rejected": True,
        },
        "promote_annular_C6_to_spatial_smoothing": {
            "registered_spatial_derivative_gain": 0,
            "rejected": True,
        },
        "use_compact_defect_to_cancel_high_shell": {
            "defect_high_shell_value": 0,
            "rejected": True,
        },
        "erase_candidate_D2_coupling": {
            "registered_zero_D2_slices": 0,
            "rejected": True,
        },
        "promote_slice_no_go_to_all_modified_energies": {
            "full_tensor_identity_available": False,
            "rejected": True,
        },
    }


def _candidate_records(d2: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for source in d2.get("certificates", []):
        candidate_id = source.get("candidate_id")
        coefficients = source.get("coefficients")
        slice_data = source.get("representative_slice", {})
        value = str(slice_data.get("component_D2_value"))
        a10 = str(coefficients.get("a10")) if isinstance(coefficients, Mapping) else ""
        if (
            not isinstance(candidate_id, str)
            or not isinstance(coefficients, Mapping)
            or value not in {"-2", "-1", "1", "2"}
            or a10 not in {"-1", "-1/2", "1/2", "1"}
            or slice_data.get("expected_formula") != "-2*a10"
        ):
            raise ValueError("finite-Sobolev candidate D2 slice changed")
        records.append(
            {
                "candidate_id": candidate_id,
                "coefficients": dict(coefficients),
                "representative_D2_value": value,
                "absolute_growth_multiplier": str(abs(int(value))),
                "uncancelled_slice_Hs_lower_bound": (
                    f"{abs(int(value))}*N*c_packet/2 for every integer s>=4"
                ),
                "direct_finite_Sobolev_hierarchy_closure": False,
                "full_tensor_cancellation_proved": False,
                "candidate_rejection_authorized": False,
                "decision": "blocked",
                "first_blocker": FIRST_BLOCKER,
            }
        )
    records.sort(key=lambda row: row["candidate_id"])
    if len(records) != 12 or len({row["candidate_id"] for row in records}) != 12:
        raise ValueError("finite-Sobolev candidate set changed")
    return records


def _expected_body(
    config: Mapping[str, Any], root: Path, config_path: Path, d2: Mapping[str, Any]
) -> dict[str, Any]:
    source_path = (
        root
        / "src/sigma_theory_compiler/quartic_finite_sobolev_hierarchy_no_go_campaign.py"
    ).resolve()
    test_path = root / "tests/test_quartic_finite_sobolev_hierarchy_no_go_campaign.py"
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "finite_unmodified_Sobolev_hierarchy_refuted_candidates_blocked",
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": EXPECTED_COUNTS,
        "first_blocker": FIRST_BLOCKER,
        "theorem": _theorem(list(config["witness_orders"])),
        "recovery_chain_audit": _recovery_chain_audit(),
        "exact_controls": _controls(list(config["witness_orders"])),
        "candidate_records": _candidate_records(d2),
        "secondary_blockers": [
            "full_tensor_variable_TC2_constraint_compatibility_not_proved",
            "initial_H7_energy_bound_not_registered",
            "global_dyadic_summation_and_lifespan_substitution_not_closed",
        ],
        "claim_seals": EXPECTED_CLAIM_SEALS,
        "data_seals": EXPECTED_DATA_SEALS,
        "scope": (
            "exact candidate-bound obstruction to unmodified finite-order Sobolev "
            "closure of one coefficient-high/state-low slice; no full H7 energy, "
            "lifespan, theory rejection, or exclusion of structured remedies"
        ),
        "source_bindings": {
            **EXPECTED_PREDECESSORS,
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": source_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }


def _validate_source_bindings(result: Mapping[str, Any], root: Path) -> None:
    bindings = result.get("source_bindings", {})
    expected_labels = {*EXPECTED_PREDECESSORS, "config", "source", "test"}
    if not isinstance(bindings, Mapping) or set(bindings) != expected_labels:
        raise ValueError("finite-Sobolev source binding set changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings.get(label) != expected:
            raise ValueError("finite-Sobolev predecessor binding changed")
        _bound_artifact(root, expected)
    paths = {
        "config": "configs/backgrounds/quartic_finite_sobolev_hierarchy_no_go_campaign.json",
        "source": (
            "src/sigma_theory_compiler/"
            "quartic_finite_sobolev_hierarchy_no_go_campaign.py"
        ),
        "test": "tests/test_quartic_finite_sobolev_hierarchy_no_go_campaign.py",
    }
    for label, relative in paths.items():
        binding = bindings.get(label, {})
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or binding.get("path") != relative
            or _file_sha(_inside(root, relative)) != binding.get("file_sha256")
        ):
            raise ValueError("finite-Sobolev local source binding changed")


def _validate_result(result: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("finite-Sobolev content hash changed")
    _validate_source_bindings(result, validation_root)
    config_path = _inside(
        validation_root,
        "configs/backgrounds/quartic_finite_sobolev_hierarchy_no_go_campaign.json",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    d2 = _bound_artifact(validation_root, EXPECTED_PREDECESSORS["high_atom_d2"])
    expected = _expected_body(config, validation_root, config_path, d2)
    if set(result) != EXPECTED_RESULT_KEYS or {
        key: value for key, value in result.items() if key != "content_sha256"
    } != expected:
        raise ValueError("finite-Sobolev result boundary changed")


def build_campaign(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _bound_artifact(root, binding)
        for label, binding in config["predecessors"].items()
    }
    _validate_predecessors(predecessors)
    body = _expected_body(config, root, config_path, predecessors["high_atom_d2"])
    result = {**body, "content_sha256": _sha(body)}
    _validate_result(result, root=root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_campaign(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output or args.config.resolve().parents[2] / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
