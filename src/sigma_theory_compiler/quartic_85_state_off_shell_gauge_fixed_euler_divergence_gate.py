from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


class Quartic85StateOffShellEulerDivergenceError(RuntimeError):
    """Raised when the off-shell sourced Euler-divergence gate fails closed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateOffShellEulerDivergenceError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateOffShellEulerDivergenceError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateOffShellEulerDivergenceError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateOffShellEulerDivergenceError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateOffShellEulerDivergenceError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateOffShellEulerDivergenceError(f"bound content hash mismatch: {path}")
    return path, value


def _formal_check(report: dict[str, Any], name: str) -> dict[str, Any]:
    checks = report.get("semantic_report", {}).get("checks", [])
    matches = [item for item in checks if item.get("name") == name]
    if len(matches) != 1 or matches[0].get("status") != "pass":
        raise Quartic85StateOffShellEulerDivergenceError(f"required formal check changed: {name}")
    return matches[0]


def _algebraic_assembly_replay() -> dict[str, Any]:
    # Atomic off-shell values replay only normalization and signs in
    # E_sourced=H+Q-T/2; the tensor identities are bound predecessor evidence.
    scalar_force = Fraction(6)
    gauge_divergence = Fraction(5)
    matter_force = Fraction(7)
    ungauged_divergence = -scalar_force / 2
    sourced_divergence = ungauged_divergence + gauge_divergence - matter_force / 2
    residual = 2 * sourced_divergence + scalar_force + matter_force - 2 * gauge_divergence
    wrong_sign_divergence = ungauged_divergence + gauge_divergence + matter_force / 2
    wrong_sign_residual = (
        2 * wrong_sign_divergence + scalar_force + matter_force - 2 * gauge_divergence
    )
    if residual != 0 or wrong_sign_residual != 14:
        raise Quartic85StateOffShellEulerDivergenceError(
            "off-shell assembly normalization replay changed"
        )
    return {
        "atomic_witness": {
            "E_phi_g*nabla_nu(phi_g)": "6",
            "nabla_mu(Q^mu_nu)": "5",
            "F_total_nu=nabla_mu(T_total^mu_nu)": "7",
            "nabla_mu(H^mu_nu)": "-3",
            "nabla_mu(E_sourced^mu_nu)": "-3/2",
        },
        "assembled_identity_residual": "0",
        "wrong_source_sign_negative": {
            "mutation": "replace -T_total/2 by +T_total/2",
            "residual": "14",
            "expected_symbolic_residual": "2*F_total_nu",
            "rejected": True,
        },
    }


def _materialize(
    basis: dict[str, Any],
    sourced: dict[str, Any],
    gauge_fixed: dict[str, Any],
    formal: dict[str, Any],
    matter: dict[str, Any],
) -> dict[str, Any]:
    if basis.get("decision") != (
        "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP"
    ):
        raise Quartic85StateOffShellEulerDivergenceError(
            "constraint-coordinate predecessor changed"
        )
    if sourced.get("decision") != "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY":
        raise Quartic85StateOffShellEulerDivergenceError("sourced Euler predecessor changed")
    if gauge_fixed.get("status") != (
        "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
    ):
        raise Quartic85StateOffShellEulerDivergenceError("gauge-fixed Euler predecessor changed")
    nonlinear = gauge_fixed.get("nonlinear_evolution_control", {})
    formula_contract = nonlinear.get("formula_contract", {})
    expected_gauge_constraint = "C_beta=tilde_g^rho_sigma(Delta Gamma)_beta_rho_sigma-H_beta"
    expected_gauge_completion = "-M2/2 hat_P_alpha^(gamma mu nu) g^(alpha beta) nabla_gamma C_beta"
    if (
        formula_contract.get("gauge_constraint") != expected_gauge_constraint
        or formula_contract.get("gauge_completion") != expected_gauge_completion
    ):
        raise Quartic85StateOffShellEulerDivergenceError(
            "modified-harmonic formula contract changed"
        )
    g2_check = _formal_check(formal, "generic_g2_variation_noether_identity")
    g4_check = _formal_check(formal, "generic_g4_curved_symbolic_all_jet_noether")
    cadabra_check = _formal_check(formal, "quartic_horndeski_metric_variation_and_noether")
    if g4_check.get("evidence", {}).get("verified_identity") != (
        "2 nabla^mu H_mu_nu+E_phi nabla_nu(phi)=0"
    ):
        raise Quartic85StateOffShellEulerDivergenceError("quartic Noether identity changed")
    conservation = matter.get("combined_matter_certificate", {}).get(
        "combined_stress_conservation", {}
    )
    if conservation.get("sector_euler_force_coefficients") != [1, 1, 1]:
        raise Quartic85StateOffShellEulerDivergenceError("matter Euler-force decomposition changed")
    common_formula = {
        "definitions": {
            "ungauged_metric_euler": "H_mu_nu",
            "gauge_completion": "Q_mu_nu",
            "total_matter_stress": "T_total_mu_nu",
            "sourced_gauge_fixed_euler": ("E_sourced_mu_nu=H_mu_nu+Q_mu_nu-T_total_mu_nu/2"),
            "matter_euler_force": (
                "F_total_nu=nabla^mu T_total_mu_nu, with registered sector "
                "Euler-force coefficients [1,1,1]"
            ),
        },
        "premise_identities": [
            "2*nabla^mu H_mu_nu+E_phi_g*nabla_nu(phi_g)=0",
            "nabla^mu T_total_mu_nu=F_total_nu",
        ],
        "gauge_formula": {
            "constraint": expected_gauge_constraint,
            "completion": f"Q^mu_nu={expected_gauge_completion}",
        },
        "maximal_common_off_shell_identity": (
            "2*nabla^mu E_sourced_mu_nu+E_phi_g*nabla_nu(phi_g)+F_total_nu-2*nabla^mu Q_mu_nu=0"
        ),
        "scope": (
            "exact covariant assembly common to all 12 candidate coefficients; "
            "nabla Q remains an unevaluated covariant divergence"
        ),
    }
    common_sha = _canonical_sha(common_formula)
    sourced_records = {
        item.get("candidate_id"): item for item in sourced.get("candidate_results", [])
    }
    basis_records = {
        item.get("candidate_id"): item
        for item in basis.get("materialization", {}).get("candidate_results", [])
    }
    if (
        len(sourced_records) != 12
        or set(sourced_records) != set(basis_records)
        or None in sourced_records
    ):
        raise Quartic85StateOffShellEulerDivergenceError("candidate manifest sets changed")
    candidate_results: list[dict[str, Any]] = []
    for candidate_id in sorted(sourced_records):
        manifest = {
            "schema_version": "invariant-candidate-off-shell-euler-divergence-manifest-1.0",
            "candidate_id": candidate_id,
            "sourced_metric_euler_sha256": sourced_records[candidate_id][
                "sourced_metric_euler_sha256"
            ],
            "constraint_coordinate_manifest_sha256": basis_records[candidate_id][
                "constraint_coordinate_manifest_sha256"
            ],
            "common_covariant_formula_sha256": common_sha,
            "common_covariant_identity_closed": True,
            "differentiated_gauge_completion_85_state_map_closed": False,
            "outcome": "TYPED_BLOCK_DIFFERENTIATED_GAUGE_SOURCE_MAP_UNREGISTERED",
        }
        candidate_results.append({**manifest, "manifest_sha256": _canonical_sha(manifest)})
    if len({item["manifest_sha256"] for item in candidate_results}) != 12:
        raise Quartic85StateOffShellEulerDivergenceError(
            "candidate divergence manifests are not one-to-one"
        )
    missing = {
        "reason_code": "missing_differentiated_modified_harmonic_formulation_field_map",
        "available_gauge_jet_order": {
            "hat_inverse_metric": 0,
            "tilde_inverse_metric": 1,
            "reference_connection": 1,
            "gauge_source_H_beta": 1,
            "physical_metric": 2,
        },
        "required_for_nabla_Q": [
            "first derivative of hat_inverse_metric and the hat projector",
            "second derivative of tilde_inverse_metric",
            "second derivative of the prescribed reference connection",
            "second derivative of the prescribed gauge source H_beta",
            (
                "second derivative of the physical connection, equivalently the physical "
                "metric third jet"
            ),
            (
                "an exact chain-rule map from those differentiated formulation fields to "
                "candidate jet and 85-state coordinate-differential rows"
            ),
        ],
        "zero_fill_forbidden": True,
        "why": (
            "the nonlinear Euler builder accepts only the jets needed for Q itself; "
            "differentiating Q introduces independent formulation-field derivatives"
        ),
    }
    return {
        "common_formula": common_formula,
        "common_formula_sha256": common_sha,
        "candidate_results": candidate_results,
        "normalization_replay": _algebraic_assembly_replay(),
        "formal_control_bindings": {
            "generic_g2_noether_sha256": _canonical_sha(g2_check),
            "generic_g4_all_jet_noether_sha256": _canonical_sha(g4_check),
            "quartic_cadabra_noether_sha256": _canonical_sha(cadabra_check),
            "gauge_formula_contract_sha256": nonlinear["formula_contract_sha256"],
        },
        "differentiated_gauge_source_block": missing,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-off-shell-gauge-fixed-euler-divergence-config-1.0"
    ):
        raise Quartic85StateOffShellEulerDivergenceError("unsupported config schema")
    if config.get("sourced_euler_normalization") != "E_sourced=H+Q-T_total/2":
        raise Quartic85StateOffShellEulerDivergenceError("sourced Euler normalization changed")
    expected_policy = {
        "common_off_shell_covariant_identity": True,
        "candidate_specific_formula_hashes": True,
        "differentiated_gauge_completion_in_85_state_variables": False,
        "constraint_propagation": False,
        "candidate_jet_uniformity": False,
        "nonlinear_global_closure": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateOffShellEulerDivergenceError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    expected_bindings = {
        "constraint_coordinate_basis",
        "sourced_metric_euler",
        "vacuum_gauge_fixed_euler",
        "off_shell_noether_controls",
        "matter_divergence_interface",
    }
    if set(bound) != expected_bindings:
        raise Quartic85StateOffShellEulerDivergenceError("closed binding manifest changed")
    materialization = _materialize(
        bound["constraint_coordinate_basis"][1],
        bound["sourced_metric_euler"][1],
        bound["vacuum_gauge_fixed_euler"][1],
        bound["off_shell_noether_controls"][1],
        bound["matter_divergence_interface"][1],
    )
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_85_state_off_shell_gauge_fixed_euler_divergence_gate.py"
    )
    body: dict[str, Any] = {
        "schema_version": (
            "invariant-quartic-85-state-off-shell-gauge-fixed-euler-divergence-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_COMMON_COVARIANT_IDENTITY_TYPED_BLOCK_DIFFERENTIATED_GAUGE_MAP",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "candidate_common_formula_hashes": 12,
            "ungauged_off_shell_noether_controls": 3,
            "source_normalization_replays": 1,
            "differentiated_gauge_completion_85_state_maps": 0,
            "constraint_propagation_claims": 0,
            "negative_controls": 1,
        },
        "claims": {
            "common_off_shell_covariant_sourced_identity_closed": True,
            "all_twelve_candidate_formula_manifests_hash_bound": True,
            "differentiated_gauge_completion_in_85_state_variables_closed": False,
            "constraint_propagation_closed": False,
            "candidate_jet_uniformity_closed": False,
            "nonlinear_global_closure_established": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "The exact ungauged gravity-scalar Noether identity, total matter Euler-force "
            "identity, sourced sign -T/2, and registered modified-harmonic completion assemble "
            "into one common off-shell covariant formula for all 12 candidates. The divergence "
            "of the gauge completion is not expanded into candidate jets or 85-state variables "
            "because the required higher derivatives of the prescribed auxiliary metrics, "
            "reference connection, gauge source, and physical connection are unregistered. "
            "Constraint propagation, candidate-jet, nonlinear/global, H7, universal-matter, "
            "and promotion claims remain false."
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
