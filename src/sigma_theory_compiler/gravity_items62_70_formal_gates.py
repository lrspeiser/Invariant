"""Items 62-70: formal-readiness and external-domain gates."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_file,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_items62_70_formal_gates_v1.json")
ITEM59_PATH = Path("runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json")


class GravityItems62To70Error(RuntimeError):
    """Raised when the frozen Items 62-70 contract changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-items62-70-formal-gates-config-1.0"
        or config.get("items") != list(range(62, 71))
        or config.get("status")
        != "scientific_freeze_before_formal_and_external_domain_audit"
    ):
        raise GravityItems62To70Error("unsupported Items 62-70 config")
    freeze = str(config.get("scientific_freeze_commit", ""))
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItems62To70Error("Items 62-70 scientific freeze is not bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItems62To70Error("invalid Items 62-70 freeze marker")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItems62To70Error(f"scientific dependency changed: {relative}")
    item59 = _read_json(root / ITEM59_PATH)
    selected = item59["selection"]["selected_qualifying"]["variant"]
    target = config["target"]
    if (
        selected["variant_id"] != target["variant_id"]
        or selected["family_id"] != target["family_id"]
        or selected["parameters"] != target["parameters"]
        or target["post_item59_refit_allowed"]
    ):
        raise GravityItems62To70Error("target representation changed")
    running = config["gates"]["64"]
    best_running = item59["selection"]["best_by_family"]["distance_running_gravity"]
    if (
        best_running["variant"]["variant_id"] != running["item59_variant_id"]
        or best_running["variant"]["parameters"] != running["parameters"]
        or not math.isclose(best_running["training_score"], running["training_score"])
    ):
        raise GravityItems62To70Error("distance-running evidence changed")
    policy = config["counterexample_policy"]
    if (
        policy["single_empirical_counterexample_terminal"]
        or policy["counterexample_count_alone_terminal"]
        or policy["finite_empirical_sample_may_prune_family"]
        or not policy[
            "verified_hard_theoretical_witness_may_veto_exact_representation_in_declared_domain"
        ]
        or policy["hard_witness_may_prune_broader_family_without_family_scope_proof"]
        or policy["missing_theory_structure_is_empirical_counterexample"]
    ):
        raise GravityItems62To70Error("counterexample policy changed")


def _base(config: Mapping[str, Any], item: int) -> dict[str, Any]:
    gate = config["gates"][str(item)]
    return {
        "schema_version": "invariant-gravity-roadmap-formal-gate-receipt-1.0",
        "goal": f"GRAVITY_ROADMAP_ITEM_{item}_{str(gate['name']).upper()}_GATE",
        "item": item,
        "gate_name": gate["name"],
        "scientific_freeze_commit": config["scientific_freeze_commit"],
        "target": config["target"],
        "test": gate["test"],
        "compute": {"backend": "deterministic_formal_audit", "gpu_used": False, "paid_api_cost_usd": 0.0},
    }


def _static_gate(config: Mapping[str, Any], item: int) -> dict[str, Any]:
    row = _base(config, item)
    if item == 62:
        row.update(
            {
                "result_class": "INCONCLUSIVE",
                "decision": config["gates"]["62"]["expected_if_absent"],
                "gate_passed": False,
                "audit": {
                    "explicit_time_or_epoch_variables": 0,
                    "evolving_couplings": 0,
                    "clock_or_orbital_history_predictions": 0,
                    "finding": "The current ansatz is static; time-varying gravity families remain untested.",
                },
                "next_action": "Only reopen this gate for a descendant with a declared evolution equation and unchanged predictions for clocks, orbits, stellar evolution, and cosmology.",
            }
        )
    elif item == 63:
        row.update(
            {
                "result_class": "INCONCLUSIVE",
                "decision": config["gates"]["63"]["expected_if_absent"],
                "gate_passed": False,
                "audit": {
                    "carrier_mass_parameters": 0,
                    "dispersion_relations": 0,
                    "polarization_predictions": 0,
                    "finding": "The current ansatz contains no massive propagating degree of freedom.",
                },
                "next_action": "Test massive-mode descendants only after their mass, range, coupling, dispersion, and polarizations are derived.",
            }
        )
    else:
        raise GravityItems62To70Error("unknown static gate")
    return _content_hashed(row)


def _distance_running_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    row = _base(config, 64)
    gate = config["gates"]["64"]
    relative_loss = gate["training_score"] / gate["empirical_rar_training_score"] - 1.0
    row.update(
        {
            "result_class": "REJECT",
            "decision": "ITEM64_DISTANCE_RUNNING_GATE_NOT_PASSED_EXACT_SCAFFOLD_RETAINED",
            "gate_passed": False,
            "audit": {
                "variant_id": gate["item59_variant_id"],
                "parameters": gate["parameters"],
                "item59_development_training_score": gate["training_score"],
                "empirical_rar_development_training_score": gate[
                    "empirical_rar_training_score"
                ],
                "relative_loss_vs_rar": relative_loss,
                "uses_object_specific_r500": gate["uses_object_specific_r500"],
                "universal_physical_length_supplied": False,
                "laboratory_solar_galaxy_cosmology_joint_curve_evaluable": False,
            },
            "claims": {
                "exact_preregistered_scaffold_promotable": False,
                "distance_running_family_pruned": False,
                "single_empirical_counterexample_used_as_veto": False,
            },
            "next_action": "Generate a running curve with a derived universal scale or field-dependent scale, then freeze it across laboratory through cosmological regimes.",
        }
    )
    return _content_hashed(row)


def _missing_structure_gate(
    config: Mapping[str, Any], item: int, missing: list[str], decision: str, next_action: str
) -> dict[str, Any]:
    row = _base(config, item)
    row.update(
        {
            "result_class": "BLOCKED",
            "decision": decision,
            "gate_passed": False,
            "audit": {
                "required_primitives": missing,
                "available_primitives": config["target"]["available_primitives"],
                "missing_primitives": missing,
                "empirical_counterexamples": [],
                "formal_witnesses": [],
            },
            "claims": {
                "missing_structure_is_empirical_counterexample": False,
                "exact_representation_rejected_by_missing_structure": False,
                "formula_family_pruned": False,
            },
            "next_action": next_action,
        }
    )
    return _content_hashed(row)


def _strong_field_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    row = _base(config, 69)
    gate = config["gates"]["69"]
    constants = gate["constants"]
    beta = float(config["target"]["parameters"]["beta"])
    a0 = float(config["target"]["transition_acceleration_m_s2"])
    radii = []
    for radius_au in gate["solar_radii_au"]:
        radius_m = float(radius_au) * float(constants["astronomical_unit_m"])
        gbar = (
            float(constants["gravity_si"])
            * float(constants["solar_mass_kg"])
            / radius_m**2
        )
        occupancy = (gbar / a0) / (gbar / a0 + 0.1)
        ratio = 1.0 + beta * occupancy * (1.0 + a0 / gbar)
        radii.append(
            {
                "radius_au": float(radius_au),
                "gbar_m_s2": gbar,
                "occupancy": occupancy,
                "predicted_acceleration_ratio": ratio,
                "predicted_fractional_change": ratio - 1.0,
                "coarse_gate_passed": abs(ratio - 1.0)
                <= float(gate["maximum_allowed_fractional_acceleration_change_for_coarse_gate"]),
            }
        )
    row.update(
        {
            "result_class": "REJECT",
            "decision": "ITEM69_EXACT_UNIVERSAL_REPRESENTATION_VETOED_IN_SOLAR_SYSTEM_DOMAIN_FAMILY_RETAINED",
            "gate_passed": False,
            "audit": {
                "declared_domain_includes_solar_system": True,
                "screening_rule_available": gate["screening_rule_available"],
                "coarse_fractional_tolerance": gate[
                    "maximum_allowed_fractional_acceleration_change_for_coarse_gate"
                ],
                "solar_radius_checks": radii,
                "high_acceleration_asymptotic_ratio": 1.0 + beta,
                "all_checks_fail": all(not value["coarse_gate_passed"] for value in radii),
                "hard_theoretical_witness_category": "verified_local_or_strong_field_violation",
            },
            "claims": {
                "exact_unscreened_universal_representation_vetoed_in_tested_domain": True,
                "item59_cluster_fit_rejected": False,
                "boundary_or_nonlocal_family_pruned": False,
                "screened_descendant_may_remain_viable": True,
                "single_empirical_counterexample_used_as_veto": False,
            },
            "next_action": "A descendant must derive a screening or high-acceleration decoupling limit before any further universal-gravity claim; the screen cannot be fitted per object.",
        }
    )
    return _content_hashed(row)


def build_receipts(root: Path) -> dict[int, dict[str, Any]]:
    config = load_config(root)
    return {
        62: _static_gate(config, 62),
        63: _static_gate(config, 63),
        64: _distance_running_gate(config),
        65: _missing_structure_gate(
            config,
            65,
            ["dynamical_potential", "lensing_potential", "derived_gravitational_slip", "photon_coupling"],
            "ITEM65_LENSING_SLIP_GATE_BLOCKED_NO_COMMON_METRIC_FIELDS",
            "Derive massive and null motion from one action or explicit metric-field system before direct lensing data are opened.",
        ),
        66: _missing_structure_gate(
            config,
            66,
            ["action", "matter_coupling", "stress_energy_identity", "constraint_and_gauge_identities"],
            "ITEM66_CONSERVATION_GATE_BLOCKED_NO_ACTION_OR_FIELD_IDENTITIES",
            "Construct an action-level completion and symbolically verify its conservation and gauge identities.",
        ),
        67: _missing_structure_gate(
            config,
            67,
            ["degrees_of_freedom", "quadratic_action", "hamiltonian", "perturbation_spectrum"],
            "ITEM67_STABILITY_GATE_BLOCKED_NO_DYNAMICAL_COMPLETION",
            "Derive perturbations and Hamiltonian for each action descendant; do not infer stability from a good radial fit.",
        ),
        68: _missing_structure_gate(
            config,
            68,
            ["evolution_equations", "principal_symbol", "characteristic_cones", "initial_value_problem"],
            "ITEM68_CAUSALITY_GATE_BLOCKED_NO_EVOLUTION_SYSTEM",
            "Supply local or explicitly causal memory equations and prove hyperbolicity and well-posed evolution.",
        ),
        69: _strong_field_gate(config),
        70: _missing_structure_gate(
            config,
            70,
            ["friedmann_background", "radiation_and_bbn", "cmb_perturbations", "bao", "structure_growth", "cosmological_lensing"],
            "ITEM70_COSMOLOGY_GATE_BLOCKED_NO_BACKGROUND_OR_PERTURBATION_THEORY",
            "Derive the homogeneous background and perturbations, then test expansion and growth without an implicit fitted dark component.",
        ),
    }


def build_matrix(root: Path) -> dict[str, Any]:
    config = load_config(root)
    receipts = build_receipts(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-items62-70-formal-gate-matrix-1.0",
            "goal": "GRAVITY_ROADMAP_ITEMS_62_THROUGH_70",
            "items": list(receipts),
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "decisions": {str(item): row["decision"] for item, row in receipts.items()},
            "result_classes": {str(item): row["result_class"] for item, row in receipts.items()},
            "passed_items": [item for item, row in receipts.items() if row["gate_passed"]],
            "blocked_items": [item for item, row in receipts.items() if row["result_class"] == "BLOCKED"],
            "not_applicable_items": [62, 63],
            "exact_representation_rejected_items": [64, 69],
            "formula_families_pruned": [],
            "empirical_singletons_used_as_veto": 0,
            "compute": {"backend": "deterministic_formal_audit", "gpu_used": False, "paid_api_cost_usd": 0.0},
            "claims": {
                "roadmap_items_62_70_attempts_complete": True,
                "current_candidate_is_complete_gravity_theory": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
            },
            "next_action": "Items 71-72 must independently confirm only claims that survived and adjudicate novelty; action-level descendants remain future work.",
        }
    )


def _receipt_path(root: Path, config: Mapping[str, Any], item: int) -> Path:
    gate = config["gates"][str(item)]
    relative = str(config["paths"]["result_template"]).format(item=item, name=gate["name"])
    return root / relative


def write_results(root: Path) -> list[Path]:
    config = load_config(root)
    receipts = build_receipts(root)
    paths = []
    for item, receipt in receipts.items():
        path = _receipt_path(root, config, item)
        _write_json(path, receipt)
        paths.append(path)
    matrix_path = root / str(config["paths"]["matrix"])
    _write_json(matrix_path, build_matrix(root))
    paths.append(matrix_path)
    return paths


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    receipts = build_receipts(root)
    checks = {
        f"item_{item}": _receipt_path(root, config, item).is_file()
        and _read_json(_receipt_path(root, config, item)) == receipt
        for item, receipt in receipts.items()
    }
    matrix_path = root / str(config["paths"]["matrix"])
    checks["matrix"] = matrix_path.is_file() and _read_json(matrix_path) == build_matrix(root)
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "replay"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "evaluate":
        print(json.dumps({"paths": [str(path) for path in write_results(root)]}, sort_keys=True))
        return 0
    result = replay(root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
