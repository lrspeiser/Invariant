from __future__ import annotations

import copy
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import sympy as sp

from .flrw_background import (
    BackgroundCertificationError,
    _compile_bundle,
    certify_flrw_background,
)

SCHEMA_VERSION = "sigma-flrw-background-campaign-1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _candidate_id(coefficients: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(coefficients).encode()).hexdigest()
    return f"flrw-{digest[:16]}"


def _mutation_assignments(ir: dict[str, Any], maximum: int) -> list[dict[str, str]]:
    mutation_space = ir.get("mutation_space")
    if not isinstance(mutation_space, dict):
        raise BackgroundCertificationError("IR has no mutation space")
    axes = mutation_space.get("axes")
    if not isinstance(axes, list):
        raise BackgroundCertificationError("mutation axes must be a list")
    parsed_axes: list[tuple[str, list[str]]] = []
    cardinality = 1
    for axis in axes:
        if not isinstance(axis, dict) or not isinstance(axis.get("coefficient"), str):
            raise BackgroundCertificationError("every mutation axis needs a coefficient")
        values = axis.get("values")
        if not isinstance(values, list) or not values:
            raise BackgroundCertificationError("every mutation axis needs nonempty values")
        parsed_values = [str(value) for value in values]
        parsed_axes.append((axis["coefficient"], parsed_values))
        cardinality *= len(parsed_values)
    if cardinality > maximum:
        raise BackgroundCertificationError(
            f"mutation cardinality {cardinality} exceeds campaign maximum {maximum}"
        )
    return [
        {coefficient: value for (coefficient, _), value in zip(parsed_axes, values, strict=True)}
        for values in product(*(values for _, values in parsed_axes))
    ]


def _complete_eligible_partition(ir: dict[str, Any]) -> set[str] | None:
    formulation = ir.get("formulation_classification")
    if not isinstance(formulation, dict):
        return None
    partition = formulation.get("mutation_axis_partition")
    if not isinstance(partition, dict):
        return None
    examples = partition.get("eligible_examples")
    eligible_count = partition.get("generalized_harmonic_eligible")
    if (
        partition.get("status") != "exact_axis_partition"
        or partition.get("count_residual") != 0
        or not isinstance(examples, list)
        or eligible_count != len(examples)
        or not all(isinstance(example, dict) for example in examples)
    ):
        return None
    return {_canonical_json(example) for example in examples}


def _complete_partition_index(ir: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    formulation = ir.get("formulation_classification")
    if not isinstance(formulation, dict):
        return None
    partition = formulation.get("mutation_axis_partition")
    if not isinstance(partition, dict):
        return None
    classifications = partition.get("assignment_classifications")
    if (
        partition.get("status") != "exact_axis_partition"
        or partition.get("count_residual") != 0
        or not isinstance(classifications, list)
    ):
        return None
    index: dict[str, dict[str, Any]] = {}
    for classification in classifications:
        if not isinstance(classification, dict) or not isinstance(
            classification.get("assignment"), dict
        ):
            return None
        index[_canonical_json(classification["assignment"])] = classification
    expected = partition.get("generalized_harmonic_eligible", 0) + partition.get(
        "modified_harmonic_required", 0
    )
    return index if len(index) == expected else None


def _solve_initial_x(
    bundle: dict[str, Any],
    initial_state: dict[str, Any],
    solver: dict[str, Any],
) -> tuple[str, str]:
    precision = int(solver.get("precision_digits", 70))
    if precision < 30:
        raise BackgroundCertificationError("root precision_digits must be at least 30")
    u_symbol, x_symbol, h_symbol = bundle["state_symbols"]
    u_value = sp.Float(str(initial_state["u"]), precision)
    h_value = sp.Float(str(initial_state["h"]), precision)
    seed = sp.Float(str(initial_state["x"]), precision)
    reduced = sp.factor(
        bundle["constraint_expression"].subs({u_symbol: u_value, h_symbol: h_value})
    )
    tolerance = sp.Float(10, precision) ** (-(precision - 10))
    root = sp.nsolve(
        reduced,
        x_symbol,
        seed,
        tol=tolerance,
        maxsteps=int(solver.get("max_steps", 100)),
        prec=precision,
    )
    root = sp.N(root, precision)
    if root.is_real is not True or not math.isfinite(float(root)) or root <= 0:
        raise BackgroundCertificationError("constraint root is not finite, real, and positive")
    residual = sp.N(reduced.subs(x_symbol, root), precision)
    if abs(residual) > sp.sqrt(tolerance):
        raise BackgroundCertificationError("constraint root residual exceeds tolerance")
    return str(root), str(residual)


def _certify_with_step_refinement(
    ir: dict[str, Any],
    candidate_config: dict[str, Any],
    maximum_refinements: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    config = copy.deepcopy(candidate_config)
    retry_fragments = ("constraint-drift enclosure", "Picard enclosure failed")
    for refinement in range(maximum_refinements + 1):
        certificate = certify_flrw_background(ir, config)
        if certificate["status"] == "pass_interval_certified":
            return certificate, config, refinement
        retryable = any(
            fragment in error
            for error in certificate.get("errors", [])
            for fragment in retry_fragments
        )
        if not retryable or refinement == maximum_refinements:
            return certificate, config, refinement
        config["step"] = float(config["step"]) / 2
    raise AssertionError("unreachable interval refinement state")


def run_flrw_background_campaign(
    ir: dict[str, Any], campaign_config: dict[str, Any]
) -> dict[str, Any]:
    """Enumerate a bounded family and certify every generalized-harmonic candidate."""

    errors: list[str] = []
    certificates: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    try:
        if campaign_config.get("schema_version") != SCHEMA_VERSION:
            raise BackgroundCertificationError("unsupported FLRW campaign schema")
        template = campaign_config.get("background_template")
        if not isinstance(template, dict):
            raise BackgroundCertificationError("background_template must be an object")
        if template.get("schema_version") != "sigma-flrw-background-run-1.0":
            raise BackgroundCertificationError("background_template has unsupported schema")
        base_coefficients = template.get("coefficients")
        if not isinstance(base_coefficients, dict):
            raise BackgroundCertificationError("template coefficients must be an object")
        initial_state = template.get("initial_state")
        if not isinstance(initial_state, dict) or set(initial_state) != {"u", "x", "h"}:
            raise BackgroundCertificationError("template initial_state must contain u, x, h")
        solver = campaign_config.get("constraint_root_solver", {})
        if not isinstance(solver, dict):
            raise BackgroundCertificationError("constraint_root_solver must be an object")
        assignments = _mutation_assignments(
            ir, int(campaign_config.get("maximum_assignments", 10000))
        )
        exact_eligible_partition = _complete_eligible_partition(ir)
        exact_partition_index = _complete_partition_index(ir)
        root_radius = float(solver.get("initial_enclosure_radius", 1e-12))
        maximum_step_refinements = int(
            campaign_config.get("maximum_step_refinements", 2)
        )
        if root_radius <= 0:
            raise BackgroundCertificationError("initial_enclosure_radius must be positive")
        if maximum_step_refinements < 0:
            raise BackgroundCertificationError(
                "maximum_step_refinements must be non-negative"
            )

        for assignment in assignments:
            coefficients = {**base_coefficients, **assignment}
            candidate_id = _candidate_id(coefficients)
            record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "mutation_assignment": assignment,
                "coefficients": coefficients,
            }
            try:
                exact_classification = (
                    exact_partition_index.get(_canonical_json(assignment))
                    if exact_partition_index is not None
                    else None
                )
                if exact_partition_index is not None and exact_classification is None:
                    raise BackgroundCertificationError(
                        "assignment missing from exact formulation partition"
                    )
                if (
                    exact_classification is not None
                    and exact_classification["obstruction_class"] != "generalized_harmonic_kessence"
                ):
                    record["formulation_route"] = "modified_harmonic_uniform_bound_required"
                    record["obstruction_class"] = exact_classification["obstruction_class"]
                    record["active_obstructions"] = exact_classification["active_obstructions"]
                    record["proof_route"] = exact_classification.get(
                        "proof_route",
                        "general_horndeski_modified_harmonic_weak_coupling",
                    )
                    record["identity_residuals"] = exact_classification["identity_residuals"]
                    if (
                        exact_classification["obstruction_class"]
                        != "modified_harmonic_G3_only"
                    ):
                        record["status"] = (
                            "unresolved_modified_harmonic_uniform_bound_required"
                        )
                        records.append(record)
                        continue
                if (
                    exact_partition_index is None
                    and exact_eligible_partition is not None
                    and _canonical_json(assignment) not in exact_eligible_partition
                ):
                    record["formulation_route"] = "modified_harmonic_uniform_bound_required"
                    record["status"] = "unresolved_modified_harmonic_uniform_bound_required"
                    records.append(record)
                    continue
                bundle = _compile_bundle(ir, coefficients)
                route = bundle["formulation"]["route"]
                record["formulation_route"] = route
                record["obstruction_class"] = (
                    exact_classification["obstruction_class"]
                    if exact_classification is not None
                    else "generalized_harmonic_kessence"
                    if route == "generalized_harmonic_kessence"
                    else "modified_harmonic_unclassified"
                )
                record["proof_route"] = (
                    exact_classification.get("proof_route")
                    if exact_classification is not None
                    else route
                )
                cubic_g3_only = (
                    record["obstruction_class"] == "modified_harmonic_G3_only"
                )
                if route != "generalized_harmonic_kessence" and not cubic_g3_only:
                    record["status"] = "unresolved_modified_harmonic_uniform_bound_required"
                    records.append(record)
                    continue

                candidate_config = copy.deepcopy(template)
                candidate_config["coefficients"] = coefficients
                root, root_residual = _solve_initial_x(
                    bundle, candidate_config["initial_state"], solver
                )
                candidate_config["initial_state"]["x"] = root
                candidate_config["initial_radius"] = max(
                    float(candidate_config.get("initial_radius", 0.0)), root_radius
                )
                if any(sp.Rational(value) != 0 for value in assignment.values()):
                    candidate_config.pop("analytic_reference", None)
                certificate, certified_config, refinements = _certify_with_step_refinement(
                    ir, candidate_config, maximum_step_refinements
                )
                certificates[candidate_id] = certificate
                record.update(
                    {
                        "status": (
                            "pass_flrw_interval_cubic_weak_field_bounds_unresolved"
                            if cubic_g3_only
                            and certificate["status"] == "pass_interval_certified"
                            else "reject_flrw_background_screen"
                            if cubic_g3_only
                            else certificate["status"]
                        ),
                        "constraint_root_x": root,
                        "constraint_root_residual": root_residual,
                        "certificate": f"certificates/{candidate_id}.json",
                        "certificate_sha256": certificate["content_sha256"],
                        "step_refinements": refinements,
                        "certified_step": certified_config["step"],
                        "certificate_errors": certificate.get("errors", []),
                    }
                )
                cubic_diagnostic = certificate["formulation_certificate"].get(
                    "cubic_bssn_homogeneous_diagnostic"
                )
                if cubic_g3_only and isinstance(cubic_diagnostic, dict):
                    record["cubic_bssn_diagnostic_summary"] = {
                        "maximum_derivative_ratio": cubic_diagnostic[
                            "maximum_derivative_ratio"
                        ],
                        "weak_field_scale_E_upper_bound": cubic_diagnostic[
                            "weak_field_scale_E_upper_bound"
                        ],
                        "scalar_slicing_cone_gap_min_abs": cubic_diagnostic[
                            "scalar_slicing_cone_gap_min_abs"
                        ],
                        "ranking_interpretation": (
                            "smaller derivative ratio and larger cone gap are preferred; no "
                            "source-backed universal pass threshold exists"
                        ),
                    }
            except (BackgroundCertificationError, KeyError, TypeError, ValueError) as error:
                cubic_background_reject = (
                    record.get("obstruction_class") == "modified_harmonic_G3_only"
                )
                record.update(
                    {
                        "formulation_route": record.get(
                            "formulation_route", "classification_failed"
                        ),
                        "status": (
                            "reject_flrw_background_screen"
                            if cubic_background_reject
                            else "reject"
                        ),
                        "errors": [str(error)],
                    }
                )
            records.append(record)

        obstruction_class_counts: dict[str, int] = {}
        for record in records:
            obstruction_class = str(record.get("obstruction_class", "unclassified"))
            obstruction_class_counts[obstruction_class] = (
                obstruction_class_counts.get(obstruction_class, 0) + 1
            )
        counts = {
            "total": len(records),
            "generalized_harmonic_eligible": sum(
                record.get("formulation_route") == "generalized_harmonic_kessence"
                for record in records
            ),
            "interval_certified": sum(
                record.get("status") == "pass_interval_certified" for record in records
            ),
            "cubic_G3_only_flrw_screened": sum(
                record.get("status")
                == "pass_flrw_interval_cubic_weak_field_bounds_unresolved"
                for record in records
            ),
            "cubic_G3_only_flrw_rejected": sum(
                record.get("status") == "reject_flrw_background_screen"
                for record in records
            ),
            "modified_harmonic_unresolved": sum(
                record.get("formulation_route")
                == "modified_harmonic_uniform_bound_required"
                for record in records
            ),
            "modified_harmonic_not_background_screened": sum(
                record.get("status")
                == "unresolved_modified_harmonic_uniform_bound_required"
                for record in records
            ),
            "rejected": sum(record.get("status") == "reject" for record in records),
            "obstruction_classes": dict(sorted(obstruction_class_counts.items())),
        }
        all_eligible_certified = (
            counts["generalized_harmonic_eligible"] > 0
            and counts["interval_certified"] == counts["generalized_harmonic_eligible"]
            and counts["rejected"] == 0
        )
        cubic_diagnostic_ranking = sorted(
            [
                {
                    "candidate_id": record["candidate_id"],
                    "mutation_assignment": record["mutation_assignment"],
                    **record["cubic_bssn_diagnostic_summary"],
                }
                for record in records
                if "cubic_bssn_diagnostic_summary" in record
            ],
            key=lambda item: (
                item["maximum_derivative_ratio"],
                -item["scalar_slicing_cone_gap_min_abs"],
            ),
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_generalized_harmonic_candidates_interval_certified"
                if all_eligible_certified
                else "incomplete_or_reject"
            ),
            "errors": errors,
            "source_ir_sha256": ir.get("content_sha256"),
            "campaign_config_sha256": hashlib.sha256(
                _canonical_json(campaign_config).encode()
            ).hexdigest(),
            "counts": counts,
            "claim": (
                "Every mutation assignment was formulation-classified; every generalized-"
                "harmonic k-essence assignment received an automatically constraint-rooted, "
                "outward-rounded FLRW interval run. G3-only cubic-Horndeski assignments also "
                "received adaptive local background screens while remaining conditional on "
                "uniform BSSN weak-field/cone bounds. G4_X assignments remain unresolved and "
                "were not falsely promoted or rejected."
            ),
            "classification_execution": (
                "exact_compiler_obstruction_partition_with_eligible_assignments_rebound"
                if exact_partition_index is not None
                else "exact_compiler_partition_with_eligible_assignments_rebound"
                if exact_eligible_partition is not None
                else "exhaustive_candidate_rebinding_fallback"
            ),
            "cubic_G3_only_diagnostic_ranking": cubic_diagnostic_ranking,
            "scope": (
                "Finite declared mutation axes and one shared local FLRW seed/time template. "
                "This is not an exhaustive initial-data search, an inhomogeneous theorem, or a "
                "global positive-energy proof."
            ),
            "candidates": records,
        }
    except (BackgroundCertificationError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "source_ir_sha256": ir.get("content_sha256"),
            "counts": {
                "total": 0,
                "generalized_harmonic_eligible": 0,
                "interval_certified": 0,
                "cubic_G3_only_flrw_screened": 0,
                "cubic_G3_only_flrw_rejected": 0,
                "modified_harmonic_unresolved": 0,
                "modified_harmonic_not_background_screened": 0,
                "rejected": 0,
                "obstruction_classes": {},
            },
            "candidates": [],
            "cubic_G3_only_diagnostic_ranking": [],
        }
    manifest = {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }
    return {"manifest": manifest, "certificates": certificates}


def write_flrw_background_campaign(result: dict[str, Any], output: Path) -> tuple[Path, list[Path]]:
    output.mkdir(parents=True, exist_ok=True)
    certificate_directory = output / "certificates"
    certificate_directory.mkdir(parents=True, exist_ok=True)
    certificate_paths: list[Path] = []
    for candidate_id, certificate in sorted(result["certificates"].items()):
        path = certificate_directory / f"{candidate_id}.json"
        path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        certificate_paths.append(path)
    manifest_path = output / "campaign.json"
    manifest_path.write_text(
        json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, certificate_paths
