from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class Quartic85StateSourcedGravityConstraintError(RuntimeError):
    """Raised when the sourced gravity-constraint gate fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateSourcedGravityConstraintError(
            f"cannot read bound file: {path}"
        ) from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateSourcedGravityConstraintError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateSourcedGravityConstraintError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateSourcedGravityConstraintError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateSourcedGravityConstraintError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateSourcedGravityConstraintError(f"bound content hash mismatch: {path}")
    return path, value


def _source_cancellation() -> dict[str, Any]:
    normalization = Fraction(-1, 2)
    sector_coefficients = [Fraction(1), Fraction(1), Fraction(1)]
    normalized = [normalization * coefficient for coefficient in sector_coefficients]
    on_shell = [Fraction(0), Fraction(0), Fraction(0)]
    residual = sum(
        (coefficient * equation for coefficient, equation in zip(normalized, on_shell)),
        Fraction(0),
    )
    omitted_fluid = normalized[:2] + [Fraction(0)]
    coefficient_residual = [
        omitted - expected for omitted, expected in zip(omitted_fluid, normalized)
    ]
    if residual != 0 or coefficient_residual != [Fraction(0), Fraction(0), Fraction(1, 2)]:
        raise Quartic85StateSourcedGravityConstraintError(
            "exact matter-source cancellation replay changed"
        )
    return {
        "metric_source": "S_mu_nu=-T_total_mu_nu/2",
        "stress_divergence_identity": (
            "registered total-stress Euler-force decomposition with sector coefficients [1,1,1]"
        ),
        "sector_euler_force_coefficients": ["1", "1", "1"],
        "normalized_source_divergence_coefficients": ["-1/2", "-1/2", "-1/2"],
        "on_shell_equation_values": ["0", "0", "0"],
        "on_shell_source_divergence_residual": "0",
        "candidate_count_with_same_source_cancellation": 12,
        "omitted_fluid_coefficient_negative": {
            "mutation": "omit -T_fluid/2 from the registered metric source",
            "coefficient_residual": ["0", "0", "1/2"],
            "rejected": True,
            "scope": (
                "source-completeness negative before imposing matter equations; it is not "
                "a gravity subsidiary-system residual"
            ),
        },
    }


def _materialize(
    symmetrizer: dict[str, Any],
    matter: dict[str, Any],
    sourced: dict[str, Any],
    first_order: dict[str, Any],
) -> dict[str, Any]:
    if symmetrizer.get("decision") != "PASS_EXACT_FLAT_SPHERE_FULL_SYMMETRIZER_BOUNDED_B":
        raise Quartic85StateSourcedGravityConstraintError("symmetrizer predecessor changed")
    if matter.get("decision") != "BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK":
        raise Quartic85StateSourcedGravityConstraintError("matter predecessor changed")
    if sourced.get("decision") != "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY":
        raise Quartic85StateSourcedGravityConstraintError("sourced Euler predecessor changed")
    if sourced.get("counts", {}).get("sourced_metric_euler_bindings_passed") != 12:
        raise Quartic85StateSourcedGravityConstraintError("twelve-candidate source binding changed")
    generic = first_order.get("generic_reduction_control", {})
    constraints = generic.get("constraints", {})
    if (
        first_order.get("status")
        != ("pass_all_12_exact_55_variable_principal_first_order_reductions")
        or constraints.get("passed") is not True
    ):
        raise Quartic85StateSourcedGravityConstraintError(
            "vacuum definition/curl predecessor changed"
        )
    matter_certificate = matter.get("combined_matter_certificate", {})
    conservation = matter_certificate.get("combined_stress_conservation", {})
    internal = matter_certificate.get("internal_matter_constraint_closure", {})
    if conservation.get("on_shell_total_residual") != [0, 0, 0]:
        raise Quartic85StateSourcedGravityConstraintError("matter conservation replay changed")
    if internal.get("subsidiary_equation") != ("box_g C=0 for the source-free Lorenz-gauge system"):
        raise Quartic85StateSourcedGravityConstraintError("Maxwell subsidiary changed")
    source_cancellation = _source_cancellation()
    missing = [
        {
            "registration": "candidate_gravity_constraint_basis",
            "required": (
                "exact modified-harmonic gauge constraint C_mu and any Hamiltonian/momentum "
                "constraint maps in the registered 85-state ordering"
            ),
        },
        {
            "registration": "candidate_gauge_fixed_euler_divergence_identity",
            "required": (
                "for each of 12 candidates, an off-shell exact divergence of every gauge-fixed "
                "metric Euler row, including gravitational-scalar Euler terms and conventions"
            ),
        },
        {
            "registration": "flat_subsidiary_factorization",
            "required": (
                "exact factorization after total matter-source cancellation into a closed "
                "homogeneous operator on the registered gravity constraints"
            ),
        },
        {
            "registration": "constraint_surface_initial_data_map",
            "required": (
                "exact implication from Hamiltonian/momentum plus gauge constraints on initial "
                "data to vanishing subsidiary data"
            ),
        },
        {
            "registration": "sourced_constraint_corruption_witness",
            "required": (
                "wrong source normalization or omitted-sector mutation with an exact nonzero "
                "gravity subsidiary-system residual"
            ),
        },
    ]
    return {
        "closed_flat_reference_subgates": {
            "completed_85_state_symmetrizer_bound": True,
            "total_matter_source_divergence_cancels_on_shell": True,
            "candidates_with_identical_source_cancellation": 12,
            "maxwell_subsidiary_equation": "box_g C=0",
            "scalar_internal_constraint_count": 0,
            "fluid_internal_constraint_count": 0,
            "vacuum_definition_constraints_per_field": constraints[
                "definition_constraints_per_field"
            ],
            "vacuum_independent_spatial_curl_constraints_per_field": constraints[
                "independent_spatial_curl_constraints_per_field"
            ],
            "vacuum_definition_time_residuals": constraints["definition_time_residuals"],
            "vacuum_curl_time_residuals": constraints["curl_time_residuals_in_coordinate_chart"],
        },
        "source_cancellation_replay": source_cancellation,
        "gravity_constraint_inference": {
            "premises": {
                "bounded_flat_85_state_symmetrizer": True,
                "sourced_metric_rows_bound_for_all_12_candidates": True,
                "total_matter_source_divergence_zero_on_shell": True,
                "candidate_gravity_constraint_basis_registered": False,
                "candidate_gauge_fixed_euler_divergence_identity_registered": False,
                "flat_subsidiary_factorization_registered": False,
            },
            "conclusion": "BLOCK",
            "reason_code": "missing_candidate_gravity_constraint_jet_divergence_registration",
            "scientific_boundary": (
                "a symmetrizer controls the evolution principal part and cannot supply the "
                "missing off-shell Noether/Bianchi divergence identity"
            ),
        },
        "minimal_registration_contract": missing,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-sourced-gravity-constraint-propagation-config-1.0"
    ):
        raise Quartic85StateSourcedGravityConstraintError("unsupported config schema")
    if config.get("flat_reference_source_normalization") != "-1/2":
        raise Quartic85StateSourcedGravityConstraintError("source normalization changed")
    expected_policy = {
        "matter_source_divergence_cancellation": True,
        "maxwell_subsidiary_closure": True,
        "vacuum_definition_curl_constraint_propagation": True,
        "sourced_gravity_constraint_propagation": False,
        "candidate_jet_uniformity": False,
        "nonlinear_global_closure": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateSourcedGravityConstraintError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    expected_bindings = {
        "bounded_flat_symmetrizer",
        "matter_interface",
        "sourced_metric_euler",
        "vacuum_first_order_constraints",
    }
    if set(bound) != expected_bindings:
        raise Quartic85StateSourcedGravityConstraintError("closed binding manifest changed")
    materialization = _materialize(
        bound["bounded_flat_symmetrizer"][1],
        bound["matter_interface"][1],
        bound["sourced_metric_euler"][1],
        bound["vacuum_first_order_constraints"][1],
    )
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_85_state_sourced_gravity_constraint_propagation_gate.py"
    )
    body: dict[str, Any] = {
        "schema_version": (
            "invariant-quartic-85-state-sourced-gravity-constraint-propagation-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "TYPED_BLOCK_CANDIDATE_GRAVITY_CONSTRAINT_JET_DIVERGENCE_UNREGISTERED",
        "materialization": materialization,
        "counts": {
            "candidates_with_sourced_metric_rows": 12,
            "candidates_with_exact_matter_source_cancellation": 12,
            "closed_matter_subsidiary_systems": 1,
            "closed_vacuum_definition_curl_controls": 1,
            "missing_gravity_constraint_registrations": 5,
            "sourced_gravity_constraint_propagation_passes": 0,
            "negative_controls": 1,
        },
        "claims": {
            "flat_reference_matter_source_divergence_cancellation_closed": True,
            "maxwell_subsidiary_closure_closed": True,
            "vacuum_definition_curl_constraint_propagation_closed": True,
            "sourced_gravity_constraint_propagation_closed": False,
            "candidate_jet_uniformity_closed": False,
            "nonlinear_global_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "The exact flat-reference matter forcing cancels on all registered matter equations "
            "for each of the 12 sourced candidate metric systems. Maxwell subsidiary closure and "
            "the vacuum first-order definition/curl identities also replay. This does not prove "
            "sourced gravity-constraint propagation: the candidate-specific constraint basis, "
            "off-shell gauge-fixed Euler divergence identity, and subsidiary factorization are "
            "not registered. Candidate-jet, nonlinear/global, H7, universal-matter, and promotion "
            "claims remain false."
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
