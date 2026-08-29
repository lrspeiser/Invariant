"""Bind and replay the exact cluster-nuisance observable quotient audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_audit_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/nuisance-quotient-audit-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-audit-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-audit-receipt-1.0"
DECISION = "EXACT_TEN_DIMENSIONAL_NUISANCE_QUOTIENT_IDENTIFIED_COMPOSITE_POSTERIOR_NOT_CONVERGED"
COMPOSITES = (
    "outer_nonthermal_fraction",
    "nonthermal_radial_power",
    "outer_pressure_boundary_sigma",
    "density_error_sigma",
    "missing_stellar_to_gas_mass_ratio",
    "clumping_amplitude",
    "spherical_acceleration_scale",
    "projected_gas_geometry_scale",
    "published_stellar_acceleration_scale",
    "temperature_density_calibration_scale",
)


class GravityClusterNuisanceQuotientError(RuntimeError):
    """Raised when the quotient, prior, sample seal, or claim boundary changes."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterNuisanceQuotientError(f"expected JSON object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterNuisanceQuotientError(f"{label} keys changed")


def _content_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    actual = _sha(body)
    if expected != actual:
        raise GravityClusterNuisanceQuotientError("bound content hash changed")
    return actual


def _validate_sources(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if tuple(row.get("source_id") for row in rows) != (
        "uncertainty_config",
        "predecessor_identifiability_audit",
    ):
        raise GravityClusterNuisanceQuotientError("source order changed")
    for row in rows:
        _strict(
            row,
            {"source_id", "path", "file_sha256", "content_sha256"},
            "source binding",
        )
        path = (root / str(row["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise GravityClusterNuisanceQuotientError("source escaped repository root") from error
        if not path.is_file() or _file_sha(path) != row["file_sha256"]:
            raise GravityClusterNuisanceQuotientError("source file changed")
        if (
            row["content_sha256"] is not None
            and _content_sha(_read_json(path)) != row["content_sha256"]
        ):
            raise GravityClusterNuisanceQuotientError("source content changed")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root.resolve() / CONFIG_PATH)
    validate_config(config, root.resolve())
    return config


def validate_config(config: Mapping[str, Any], root: Path) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "audit_id",
            "purpose",
            "source_bindings",
            "sample_seal",
            "primitive_parameters",
            "exact_composite_coordinates",
            "exact_null_structure",
            "induced_prior_rule",
            "rank_protocol",
            "forward_invariance_protocol",
            "unchanged_posterior_thresholds",
            "observed_results",
            "adjudication",
            "required_next_actions",
            "output_path",
        },
        "quotient config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_development_only_no_target_access"
        or config["audit_id"] != "gravity-cluster-nuisance-quotient-audit-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterNuisanceQuotientError("quotient audit identity changed")
    _validate_sources(root, config["source_bindings"])
    if config["sample_seal"] != {
        "likelihood_split": "development_train",
        "development_holdout_used_for_selection": False,
        "same_release_confirmation_rows_used": False,
        "independent_source_rows_used": False,
        "target_rows_opened": 0,
        "paid_model_calls": 0,
    }:
        raise GravityClusterNuisanceQuotientError("sample seal changed")
    if (
        config["primitive_parameters"] != 17
        or tuple(row.get("coordinate") for row in config["exact_composite_coordinates"])
        != COMPOSITES
    ):
        raise GravityClusterNuisanceQuotientError("quotient coordinates changed")
    nulls = config["exact_null_structure"]
    if (
        nulls["stellar_product_null_dimensions"] != 5
        or nulls["centering_triaxial_decomposition_null_dimensions"] != 1
        or nulls["coupled_scale_orbit_null_dimensions"] != 1
        or nulls["total_null_dimensions"] != 7
    ):
        raise GravityClusterNuisanceQuotientError("null dimension changed")
    prior = config["induced_prior_rule"]
    if (
        prior["primitive_priors_changed"] is not False
        or prior["stellar_clip_mixture_retained"] is not True
        or prior["primitive_orbit_labels_reportable_as_separately_identified"] is not False
    ):
        raise GravityClusterNuisanceQuotientError("induced prior rule changed")
    rank = config["rank_protocol"]
    observed_rank = config["observed_results"]["rank"]
    if (
        rank["anchors"] != 16
        or rank["required_rank_at_every_anchor"] != 10
        or observed_rank["ranks"] != [10] * 16
        or observed_rank["minimum_tenth_relative_singular_value"]
        < rank["minimum_tenth_relative_singular_value"]
        or observed_rank["maximum_first_null_relative_singular_value"]
        > rank["maximum_first_null_relative_singular_value"]
    ):
        raise GravityClusterNuisanceQuotientError("rank boundary changed")
    invariance = config["forward_invariance_protocol"]
    observed_invariance = config["observed_results"]["coupled_scale_orbit"]
    if (
        invariance["coupled_scale_orbit_cases"] != 32
        or invariance["stellar_product_redistribution_cases"] != 40
        or invariance["centering_triaxial_decomposition_cases"] != 16
        or observed_invariance["maximum_absolute_log_prediction_difference"]
        > invariance["maximum_absolute_log_prediction_difference"]
    ):
        raise GravityClusterNuisanceQuotientError("invariance boundary changed")
    thresholds = config["unchanged_posterior_thresholds"]
    posterior = config["observed_results"]["composite_posterior"]
    if (
        thresholds["maximum_rhat"] != 1.2
        or thresholds["minimum_effective_samples"] != 50
        or thresholds["maximum_standardized_between_replicate_median_spread"] != 0.25
        or thresholds["maximum_standardized_smc_to_rejuvenated_median_shift"] != 0.25
        or posterior["maximum_rhat"] <= thresholds["maximum_rhat"]
        or posterior["minimum_effective_samples"] < thresholds["minimum_effective_samples"]
        or posterior["maximum_standardized_between_replicate_median_spread"]
        <= thresholds["maximum_standardized_between_replicate_median_spread"]
        or posterior["maximum_standardized_smc_to_rejuvenated_median_shift"]
        > thresholds["maximum_standardized_smc_to_rejuvenated_median_shift"]
        or posterior["coordinates_above_rhat_threshold"] != 10
        or posterior["converged"] is not False
    ):
        raise GravityClusterNuisanceQuotientError("posterior boundary changed")
    adjudication = config["adjudication"]
    if (
        adjudication["decision"] != DECISION
        or adjudication["exact_observable_quotient_identified"] is not True
        or adjudication["primitive_dimension_convergence_is_valid_completion_gate"] is not False
        or adjudication["composite_posterior_converged"] is not False
        or adjudication["CP5_7_through_CP5_10_complete"] is not False
        or adjudication["newtonian_control_run"] is not False
        or adjudication["quotient_aware_sampler_required"] is not True
        or adjudication["do_not_weaken_thresholds"] is not True
        or adjudication["do_not_open_confirmation_or_independent_rows"] is not True
    ):
        raise GravityClusterNuisanceQuotientError("adjudication changed")


def _primitive_values(unit: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    return lows + unit * (highs - lows)


def _unit_values(values: Mapping[str, float], config: Mapping[str, Any]) -> np.ndarray:
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    physical = np.asarray([float(values[name]) for name in uncertainty.PARAMETERS])
    return (physical - lows) / (highs - lows)


def _training_runtime(root: Path) -> tuple[Any, Any, Any, Any, list[str]]:
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][0]
    row_ids = [str(row["row_id"]) for row in item59._rows(packets, "development_train")]
    return config, config59, packets, family, row_ids


def _log_predictions(
    unit: np.ndarray,
    packets: Any,
    family: Any,
    config: Any,
    config59: Any,
    row_ids: Sequence[str],
) -> np.ndarray:
    _likelihood, predictions = uncertainty._evaluate_unit(unit, packets, family, config, config59)
    return np.log(np.asarray([predictions[row_id] for row_id in row_ids]))


def replay(root: Path) -> dict[str, Any]:
    audit = load_config(root)
    config, config59, packets, family, row_ids = _training_runtime(root.resolve())
    rank_protocol = audit["rank_protocol"]
    engine = qmc.Sobol(d=len(uncertainty.PARAMETERS), scramble=True, seed=596001)
    anchors = 0.2 + 0.6 * engine.random_base2(m=4)
    rank_rows = []
    for anchor in anchors:
        matrix = np.empty((len(row_ids), len(anchor)))
        for dimension in range(len(anchor)):
            low = anchor.copy()
            high = anchor.copy()
            low[dimension] -= float(rank_protocol["central_difference_step_in_unit_coordinates"])
            high[dimension] += float(rank_protocol["central_difference_step_in_unit_coordinates"])
            matrix[:, dimension] = (
                _log_predictions(high, packets, family, config, config59, row_ids)
                - _log_predictions(low, packets, family, config, config59, row_ids)
            ) / (2.0 * float(rank_protocol["central_difference_step_in_unit_coordinates"]))
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank_rows.append(
            {
                "rank": int(np.sum(singular > singular[0] * 1e-8)),
                "tenth_relative": float(singular[9] / singular[0]),
                "first_null_relative": float(singular[10] / singular[0]),
            }
        )

    midpoint = _primitive_values(np.full(len(uncertainty.PARAMETERS), 0.5), config)
    base = dict(zip(uncertainty.PARAMETERS, midpoint, strict=True))
    base_unit = _unit_values(base, config)
    base_predictions = _log_predictions(base_unit, packets, family, config, config59, row_ids)
    differences: dict[str, list[float]] = {
        "coupled_scale_orbit": [],
        "stellar_product_redistribution": [],
        "centering_triaxial_decomposition": [],
    }
    for scale in np.linspace(0.98, 1.02, 32):
        changed = dict(base)
        changed["triaxial_radius_scale"] = float(scale)
        changed["projection_density_scale"] = 1.0 / float(scale)
        changed["bcg_mass_scale"] = float(scale) ** 2
        changed["xray_temperature_cross_calibration"] = 1.0 / float(scale)
        predictions = _log_predictions(
            _unit_values(changed, config),
            packets,
            family,
            config,
            config59,
            row_ids,
        )
        differences["coupled_scale_orbit"].append(
            float(np.max(np.abs(predictions - base_predictions)))
        )
    stellar_factors = (
        "satellite_mass_scale",
        "missing_member_fraction",
        "intracluster_light_fraction",
        "imf_mass_scale",
        "mass_to_light_scale",
    )
    for parameter in stellar_factors:
        for scale in np.linspace(0.99, 1.01, 8):
            changed = dict(base)
            changed["bcg_mass_scale"] = 1.0 / float(scale)
            if parameter in {
                "missing_member_fraction",
                "intracluster_light_fraction",
            }:
                changed[parameter] = (1.0 + base[parameter]) * float(scale) - 1.0
            else:
                changed[parameter] = base[parameter] * float(scale)
            predictions = _log_predictions(
                _unit_values(changed, config),
                packets,
                family,
                config,
                config59,
                row_ids,
            )
            differences["stellar_product_redistribution"].append(
                float(np.max(np.abs(predictions - base_predictions)))
            )
    for shift in np.linspace(-0.02, 0.02, 16):
        changed = dict(base)
        changed["centering_radius_shift"] = float(shift)
        changed["triaxial_radius_scale"] = 1.0 / (1.0 + float(shift))
        predictions = _log_predictions(
            _unit_values(changed, config),
            packets,
            family,
            config,
            config59,
            row_ids,
        )
        differences["centering_triaxial_decomposition"].append(
            float(np.max(np.abs(predictions - base_predictions)))
        )
    result = {
        "rank": {
            "ranks": [row["rank"] for row in rank_rows],
            "minimum_tenth_relative_singular_value": min(
                row["tenth_relative"] for row in rank_rows
            ),
            "maximum_first_null_relative_singular_value": max(
                row["first_null_relative"] for row in rank_rows
            ),
            "forward_evaluations": 544,
        },
        "forward_invariance": {
            name: {
                "cases": len(values),
                "maximum_absolute_log_prediction_difference": max(values),
            }
            for name, values in differences.items()
        },
        "forward_evaluations": 544 + 1 + sum(map(len, differences.values())),
    }
    if (
        result["rank"]["ranks"] != [10] * 16
        or result["rank"]["minimum_tenth_relative_singular_value"]
        < rank_protocol["minimum_tenth_relative_singular_value"]
        or result["rank"]["maximum_first_null_relative_singular_value"]
        > rank_protocol["maximum_first_null_relative_singular_value"]
        or max(
            row["maximum_absolute_log_prediction_difference"]
            for row in result["forward_invariance"].values()
        )
        > audit["forward_invariance_protocol"]["maximum_absolute_log_prediction_difference"]
    ):
        raise GravityClusterNuisanceQuotientError("quotient replay failed")
    return result


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "audit_id": config["audit_id"],
        "decision": config["adjudication"]["decision"],
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "content_sha256": _sha(config),
        },
        "source_bindings": config["source_bindings"],
        "sample_seal": config["sample_seal"],
        "exact_composite_coordinates": config["exact_composite_coordinates"],
        "exact_null_structure": config["exact_null_structure"],
        "induced_prior_rule": config["induced_prior_rule"],
        "protocols": {
            "rank": config["rank_protocol"],
            "forward_invariance": config["forward_invariance_protocol"],
            "posterior_thresholds": config["unchanged_posterior_thresholds"],
        },
        "observed_results": config["observed_results"],
        "completed_goal_evidence": {},
        "blocked_goal_evidence": {
            "CP5.7": "ten_coordinate_quotient_defined_but_candidate_composite_posterior_not_converged",
            "CP5.8": "stellar_product_pushforward_defined_but_not_calibrated_or_converged",
            "CP5.9": "boundary_coordinate_retained_but_joint_composite_posterior_not_converged",
            "CP5.10": "geometry_projection_density_quotient_defined_but_not_calibrated_or_converged",
        },
        "claims": {
            "maximum_observable_nuisance_dimension": 10,
            "exact_null_dimensions": 7,
            "rank_ten_at_all_frozen_interior_anchors": True,
            "forward_symmetry_checks_passed": True,
            "primitive_labels_separately_identified": False,
            "joint_induced_composite_prior_preserved": True,
            "composite_posterior_converged": False,
            "CP5_7_through_CP5_10_complete": False,
            "newtonian_control_run": False,
            "independent_replication": False,
            "physical_mechanism_identified": False,
        },
        "counts": {
            "primitive_parameters": 17,
            "exact_composite_coordinates": 10,
            "exact_null_dimensions": 7,
            "rank_anchors": 16,
            "rank_forward_evaluations": 544,
            "frozen_invariance_cases": 88,
            "source_composite_posterior_draws": 131072,
            "target_rows_opened": 0,
            "paid_model_calls": 0,
        },
        "required_next_actions": config["required_next_actions"],
        "reproduction": {
            "write_command": "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_audit write",
            "check_command": "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_audit check",
            "replay_command": "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_audit replay",
            "cheap_check_reexecutes_forward_models": False,
            "replay_reexecutes_rank_and_invariance_forward_models": True,
        },
        "limitations": [
            "The algebraic factorization gives a ten-dimensional maximum observable quotient; rank can be lower at clipping boundaries or special parameter values.",
            "The 16-anchor rank audit is development-training evidence, not a proof of global statistical identification or independent replication.",
            "The exact induced prior is a dependent pushforward with stellar clipping mixture components; independent priors on the ten composites would change the model.",
            "The existing composite posterior still fails the unchanged convergence thresholds, so no nuisance-marginalization task is complete.",
        ],
        "next_action": config["required_next_actions"][0],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterNuisanceQuotientError("quotient receipt changed")


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status", "replay"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
    elif args.command == "replay":
        output = {"status": "PASS", "result": replay(root)}
    else:
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        if args.command == "check":
            output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
        else:
            output = {
                "decision": receipt["decision"],
                "counts": receipt["counts"],
                "claims": receipt["claims"],
                "next_action": receipt["next_action"],
            }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
