from __future__ import annotations

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_twelve_candidate_85_state_first_order_reduction import _generic_reduction


class Quartic85StateSymmetrizerGateError(RuntimeError):
    """Raised when the bounded coupled-symmetrizer audit cannot be replayed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateSymmetrizerGateError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateSymmetrizerGateError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateSymmetrizerGateError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateSymmetrizerGateError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateSymmetrizerGateError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateSymmetrizerGateError(f"bound content hash mismatch: {path}")
    return path, value


@cache
def _nonzero_witness_control() -> dict[str, Any]:
    generic = _generic_reduction()
    data = generic["data"]
    substitution: dict[sp.Symbol, sp.Expr] = {
        data["m2"]: 1,
        data["alpha"]: sp.Rational(-1, 2),
        data["c20"]: -1,
    }
    all_symbols = set(generic["A"].free_symbols)
    for block in generic["B_i"]:
        all_symbols.update(block.free_symbols)
    for row in generic["C_ij"]:
        for block in row:
            all_symbols.update(block.free_symbols)
    for symbol in all_symbols:
        if str(symbol).startswith("B_") or str(symbol).startswith("xi_"):
            continue
        substitution.setdefault(symbol, sp.Integer(0))
    substitution.update(
        {
            sp.Symbol("B_0"): 1,
            sp.Symbol("B_1"): 0,
            sp.Symbol("B_2"): 0,
            sp.Symbol("B_3"): 0,
        }
    )
    coefficient_a = generic["A"].subs(substitution)
    coefficient_b = generic["B_i"][0].subs(substitution)
    coefficient_c = generic["C_ij"][0][0].subs(substitution)
    if coefficient_a.det() == 0:
        raise Quartic85StateSymmetrizerGateError("witness time block is singular")
    operator = sp.zeros(34)
    operator[:17, 17:] = sp.eye(17)
    operator[17:, :17] = -coefficient_a.inv() * coefficient_c
    operator[17:, 17:] = -coefficient_a.inv() * coefficient_b
    rational_roots = [
        sp.Integer(-1),
        sp.Integer(1),
        sp.Rational(-1, 2),
        sp.Rational(1, 2),
        sp.Rational(-1, 3),
        sp.Rational(1, 3),
    ]
    nullities = {str(root): 34 - (operator - root * sp.eye(34)).rank() for root in rational_roots}
    fluid_pair_dimension = 34 - (operator**2 - sp.eye(34) / 3).rank()
    if nullities != {"-1": 8, "1": 8, "-1/2": 4, "1/2": 4, "-1/3": 4, "1/3": 4}:
        raise Quartic85StateSymmetrizerGateError("witness eigenspace census changed")
    if fluid_pair_dimension != 2 or sum(nullities.values()) + fluid_pair_dimension != 34:
        raise Quartic85StateSymmetrizerGateError("witness is not exactly diagonalizable")
    return {
        "scope": (
            "one exact flat-jet, unit-x-direction, B_0=1 witness for a representative "
            "candidate; this is not a uniform candidate-domain symmetrizer"
        ),
        "candidate_coefficients": {"M2": "1", "alpha": "-1/2", "c20": "-1"},
        "background": {"covariant_gravity_jet": "0", "B_mu": [1, 0, 0, 0]},
        "direction": [1, 0, 0],
        "time_block_determinant": str(sp.factor(coefficient_a.det())),
        "rational_root_geometric_multiplicities": nullities,
        "fluid_plus_minus_inverse_sqrt_3_combined_dimension": fluid_pair_dimension,
        "eigenspace_dimension_sum": 34,
        "diagonalizable": True,
        "interpretation": (
            "the nonzero Maxwell coupling does not produce a Jordan defect at this exact "
            "witness; it does not solve the uniform resonant Sylvester problem"
        ),
    }


def _sylvester_contract() -> dict[str, Any]:
    lam, x, forcing = sp.symbols("lambda x forcing")
    resonant_left = sp.expand(lam * x - x * lam)
    incompatible_residual = sp.expand(resonant_left - forcing).subs(forcing, 1)
    if resonant_left != 0 or incompatible_residual != -1:
        raise Quartic85StateSymmetrizerGateError("resonant Sylvester control failed")
    return {
        "block_decomposition": {
            "gravity_state_dimension": 55,
            "matter_state_dimension": 30,
            "total_state_dimension": 85,
            "cross_unknown_X_shape": [30, 55],
            "cross_unknown_entries": 1650,
        },
        "candidate_operator": "L=[[G,0],[C,M]]",
        "symmetrizer_ansatz": "H=[[H_g,X^T],[X,H_m]]",
        "cross_symmetry_equation": "M^T X-X G=H_m C",
        "shared_characteristic_roots": ["-1", "+1"],
        "resonant_compatibility_required": (
            "Pi_M(lambda)^T H_m C Pi_G(lambda)=0 for lambda in {-1,+1}"
        ),
        "resonant_scalar_control": {
            "left_side": str(resonant_left),
            "unit_forcing_residual": str(incompatible_residual),
            "conclusion": "shared eigenvalues require an explicit zero projected forcing proof",
        },
        "positivity_condition": "H_g-X^T H_m^(-1) X is positive definite",
        "minimal_missing_registrations": [
            (
                "materialize the vacuum K55 and its +/-1 Riesz projectors in the same "
                "85-state basis as the coupled reduction"
            ),
            (
                "materialize H_m and the 30x55 first-order Maxwell cross block C(B,n) and "
                "prove both resonant projector compatibilities exactly"
            ),
            (
                "solve M^T X-XG=H_m C on the nonresonant complement with exact uniform "
                "bounds in direction and candidate jet"
            ),
            (
                "register a nonzero bounded Maxwell-potential domain and prove the Schur "
                "complement positivity inequality uniformly on it"
            ),
        ],
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-quartic-85-state-common-time-symmetrizer-gate-config-1.0"
    ):
        raise Quartic85StateSymmetrizerGateError("unsupported config schema")
    expected_policy = {
        "exact_sylvester_registration_contract": True,
        "nonzero_witness_diagonalizability": True,
        "full_coupled_symmetrizer": False,
        "uniform_common_time_domain": False,
        "sourced_constraint_propagation": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateSymmetrizerGateError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "coupled_85_state_reduction",
        "vacuum_K55",
        "matter_common_time",
        "maxwell_mixed_block",
    }:
        raise Quartic85StateSymmetrizerGateError("closed binding manifest changed")
    reduction = bound["coupled_85_state_reduction"][1]
    vacuum = bound["vacuum_K55"][1]
    matter = bound["matter_common_time"][1]
    mixed = bound["maxwell_mixed_block"][1]
    if reduction.get("decision") != "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE":
        raise Quartic85StateSymmetrizerGateError("85-state predecessor changed")
    if vacuum.get("counts", {}).get("full_55_state_symmetrizer_lifts_passed") != 12:
        raise Quartic85StateSymmetrizerGateError("vacuum K55 predecessor changed")
    principal = matter.get("combined_matter_certificate", {}).get(
        "combined_matter_principal_compatibility", {}
    )
    if principal.get("strongly_hyperbolic_matter_direct_sum") is not True:
        raise Quartic85StateSymmetrizerGateError("matter common-time predecessor changed")
    if (
        mixed.get("decision") != "PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL"
        or mixed.get("counts", {}).get("structurally_nonzero_mixed_entries") != 40
    ):
        raise Quartic85StateSymmetrizerGateError("Maxwell cross block predecessor changed")
    reduction_records = reduction.get("candidate_results", [])
    vacuum_records = vacuum.get("certificates", [])
    expected_count = config.get("expected_candidate_count")
    if (
        expected_count != 12
        or len(reduction_records) != expected_count
        or {item.get("candidate_id") for item in reduction_records}
        != {item.get("candidate_id") for item in vacuum_records}
    ):
        raise Quartic85StateSymmetrizerGateError("candidate set mismatch")
    witness = _nonzero_witness_control()
    contract = _sylvester_contract()
    results = [
        {
            "candidate_id": item["candidate_id"],
            "outcome": "BLOCK",
            "reason_codes": [
                "unregistered_resonant_cross_symmetrizer_solution",
                "missing_bounded_maxwell_potential_domain_for_schur_positivity",
            ],
            "vacuum_K55_lower_bound": next(
                record["uniform_bounds"]["K55_2_lower"]
                for record in vacuum_records
                if record["candidate_id"] == item["candidate_id"]
            ),
        }
        for item in reduction_records
    ]
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_quartic_twelve_candidate_85_state_common_time_symmetrizer_gate.py"
    )
    body: dict[str, Any] = {
        "schema_version": "invariant-quartic-85-state-common-time-symmetrizer-gate-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "TYPED_BLOCK_RESONANT_SYLVESTER_AND_SCHUR_DOMAIN_UNREGISTERED",
        "nonzero_coupling_witness": witness,
        "sylvester_registration_contract": contract,
        "candidate_results": results,
        "counts": {
            "candidates": 12,
            "vacuum_K55_prerequisites_passed": 12,
            "matter_common_time_prerequisites_passed": 12,
            "nonzero_coupling_diagonalizable_witnesses": 1,
            "sylvester_unknown_entries_per_candidate": 1650,
            "resonant_roots_requiring_compatibility": 2,
            "coupled_symmetrizers_passed": 0,
            "typed_blocks": 12,
            "constraint_propagation_claims": 0,
            "rejects": 0,
        },
        "claims": {
            "nonzero_coupling_witness_diagonalizable": True,
            "physical_jordan_obstruction_established": False,
            "full_coupled_symmetrizer_any_candidate": False,
            "uniform_common_time_domain_closed": False,
            "sourced_constraint_propagation_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "exact certification audit for coupling vacuum K55 to the 30-state matter energy. "
            "A nonzero Maxwell-potential witness is diagonalizable, so no Jordan no-go is "
            "claimed. The uniform 85-state symmetrizer remains blocked on the resonant +/-1 "
            "Sylvester projections, cross solution, Maxwell-potential domain, and Schur "
            "positivity bound. Constraint propagation and all broader claims remain excluded"
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
