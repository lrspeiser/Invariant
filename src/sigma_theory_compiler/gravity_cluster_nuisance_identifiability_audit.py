"""Bind the development-only SMC nuisance-identifiability boundary into Invariant."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_nuisance_identifiability_audit_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/nuisance-identifiability-audit-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-nuisance-identifiability-audit-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-nuisance-identifiability-audit-receipt-1.0"
RUN_IDS = (
    "TEMPERED_SMC_4X512_V2",
    "FULL_POSTERIOR_REJUVENATION_4X512X64_V3",
)
DECISION = (
    "TEMPERED_SMC_AND_POSTERIOR_REJUVENATION_FAILED_UNCHANGED_CONVERGENCE_"
    "REQUIRES_IDENTIFIABILITY_REDESIGN"
)


class GravityClusterNuisanceIdentifiabilityError(RuntimeError):
    """Raised when the frozen audit, seal, or failed-gate boundary changes."""


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
        raise GravityClusterNuisanceIdentifiabilityError(f"expected JSON object: {path}")
    return value


def _content_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    actual = _sha(body)
    if expected != actual:
        raise GravityClusterNuisanceIdentifiabilityError("predecessor content hash changed")
    return actual


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterNuisanceIdentifiabilityError(f"{label} keys changed")


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
            "predecessor_binding",
            "sample_seal",
            "unchanged_completion_thresholds",
            "diagnostic_runs",
            "adjudication",
            "required_next_actions",
            "reproduction",
            "output_path",
        },
        "nuisance identifiability config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_development_only_no_target_access"
        or config["audit_id"] != "gravity-cluster-nuisance-identifiability-audit-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterNuisanceIdentifiabilityError("audit identity changed")

    predecessor = config["predecessor_binding"]
    _strict(
        predecessor,
        {"path", "file_sha256", "content_sha256"},
        "predecessor binding",
    )
    predecessor_path = (root / str(predecessor["path"])).resolve()
    try:
        predecessor_path.relative_to(root)
    except ValueError as error:
        raise GravityClusterNuisanceIdentifiabilityError(
            "predecessor escaped repository root"
        ) from error
    if (
        not predecessor_path.is_file()
        or _file_sha(predecessor_path) != predecessor["file_sha256"]
        or _content_sha(_read_json(predecessor_path)) != predecessor["content_sha256"]
    ):
        raise GravityClusterNuisanceIdentifiabilityError("predecessor binding changed")

    if config["sample_seal"] != {
        "likelihood_split": "development_train",
        "development_holdout_rows_used": False,
        "same_release_confirmation_rows_used": False,
        "independent_source_rows_used": False,
        "target_rows_opened": 0,
        "paid_model_calls": 0,
    }:
        raise GravityClusterNuisanceIdentifiabilityError("sample seal changed")
    thresholds = config["unchanged_completion_thresholds"]
    if thresholds != {
        "maximum_rhat": 1.2,
        "minimum_effective_samples": 50,
        "maximum_standardized_between_replicate_median_spread": 0.25,
        "all_17_parameters_must_pass": True,
    }:
        raise GravityClusterNuisanceIdentifiabilityError("completion threshold changed")

    runs = config["diagnostic_runs"]
    if tuple(run.get("run_id") for run in runs) != RUN_IDS:
        raise GravityClusterNuisanceIdentifiabilityError("run inventory changed")
    for run in runs:
        _strict(
            run,
            {"run_id", "engine", "settings", "result", "prototype_evidence"},
            "diagnostic run",
        )
        if run["result"]["converged"] is not False:
            raise GravityClusterNuisanceIdentifiabilityError("failed run promoted")
    smc = runs[0]["result"]
    rejuvenated = runs[1]["result"]
    if (
        smc["maximum_rhat"] > thresholds["maximum_rhat"]
        or smc["minimum_effective_samples"] < thresholds["minimum_effective_samples"]
        or smc["maximum_standardized_between_replicate_median_spread"]
        <= thresholds["maximum_standardized_between_replicate_median_spread"]
        or rejuvenated["maximum_rhat"] <= thresholds["maximum_rhat"]
        or rejuvenated["minimum_effective_samples"] < thresholds["minimum_effective_samples"]
        or rejuvenated["maximum_standardized_between_replicate_median_spread"]
        <= thresholds["maximum_standardized_between_replicate_median_spread"]
        or rejuvenated["maximum_standardized_source_to_rejuvenated_median_shift"]
        > thresholds["maximum_standardized_between_replicate_median_spread"]
        or rejuvenated["parameters_above_rhat_threshold"] != 17
    ):
        raise GravityClusterNuisanceIdentifiabilityError("diagnostic result boundary changed")
    adjudication = config["adjudication"]
    if (
        adjudication["decision"] != DECISION
        or adjudication["more_sampling_alone_supported"] is not False
        or adjudication["CP5_7_through_CP5_10_complete"] is not False
        or adjudication["newtonian_control_run"] is not False
        or adjudication["do_not_weaken_thresholds"] is not True
        or adjudication["do_not_use_holdout_for_sampler_selection"] is not True
        or adjudication["do_not_open_confirmation_or_independent_rows"] is not True
    ):
        raise GravityClusterNuisanceIdentifiabilityError("adjudication changed")
    if len(config["required_next_actions"]) != 6:
        raise GravityClusterNuisanceIdentifiabilityError("next actions changed")


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    runs = config["diagnostic_runs"]
    new_evaluations = sum(int(run["settings"]["evaluations"]) for run in runs)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "audit_id": config["audit_id"],
        "decision": config["adjudication"]["decision"],
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "content_sha256": _sha(config),
        },
        "predecessor_binding": config["predecessor_binding"],
        "sample_seal": config["sample_seal"],
        "unchanged_completion_thresholds": config["unchanged_completion_thresholds"],
        "diagnostic_runs": runs,
        "completed_goal_evidence": {},
        "blocked_goal_evidence": {
            "CP5.7": "continuous_nonthermal_and_calibration_coordinates_not_jointly_identified",
            "CP5.8": "primitive_stellar_latent_factors_not_jointly_identified",
            "CP5.9": "outer_boundary_coordinate_not_jointly_identified",
            "CP5.10": "density_clumping_centering_projection_geometry_coordinates_not_jointly_identified",
        },
        "claims": {
            "tempered_smc_mechanics_passed": True,
            "full_posterior_rejuvenation_completed": True,
            "more_sampling_alone_supported": False,
            "finite_particle_noise_is_sufficient_explanation": False,
            "posterior_sampler_converged": False,
            "development_nuisance_marginalization_complete": False,
            "CP5_7_through_CP5_10_complete": False,
            "newtonian_control_run": False,
            "independent_replication": False,
            "physical_mechanism_identified": False,
        },
        "counts": {
            "diagnostic_runs": len(runs),
            "new_candidate_forward_evaluations": new_evaluations,
            "cumulative_candidate_forward_evaluations_with_predecessor": 501636 + new_evaluations,
            "largest_posterior_draws": 131072,
            "nuisance_dimensions": 17,
            "parameters_passing_rejuvenated_rhat": 0,
            "target_rows_opened": 0,
            "paid_model_calls": 0,
        },
        "required_next_actions": config["required_next_actions"],
        "reproduction": config["reproduction"],
        "limitations": [
            "Both runs used only the X-COP development-training likelihood and cannot establish independent replication.",
            "The first SMC R-hat treats an unordered posterior particle cloud as draws; the trajectory-based rejuvenation diagnostic is the stronger convergence result.",
            "Failure of more sampling to pass is evidence for an identifiability redesign, not proof of one unique degeneracy or a physical mechanism.",
            "The raw NPZ archives remain machine-local prototype artifacts; the committed receipt binds their hashes and frozen aggregate observations but does not embed them.",
        ],
        "next_action": config["required_next_actions"][0],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterNuisanceIdentifiabilityError("audit receipt changed")


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
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
