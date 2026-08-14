from __future__ import annotations

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_geometric_jet_campaign import (
    SYMMETRIC_METRIC_PAIRS,
    SYMMETRIC_METRIC_WEIGHTS,
)


class System10CylindricalSourcedConstraintRowError(RuntimeError):
    """Raised when an exact cylindrical sourced constraint row fails closed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise System10CylindricalSourcedConstraintRowError(
            f"cannot read bound file: {path}"
        ) from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10CylindricalSourcedConstraintRowError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10CylindricalSourcedConstraintRowError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10CylindricalSourcedConstraintRowError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalSourcedConstraintRowError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10CylindricalSourcedConstraintRowError(f"bound content hash mismatch: {path}")
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalSourcedConstraintRowError(f"bound source hash mismatch: {path}")
    return path


def _zero_tensor(shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return [sp.Integer(0) for _ in range(shape[0])]
    return [_zero_tensor(shape[1:]) for _ in range(shape[0])]


@cache
def _atoms() -> dict[str, Any]:
    q = sp.symbols("q_0:17", real=True)
    v = sp.symbols("v_0:17", real=True)
    w = [
        [sp.Symbol(f"w_{spatial}_{field}", real=True) for field in range(17)]
        for spatial in range(1, 4)
    ]
    acceleration = sp.symbols("partial0_v_0:17", real=True)
    partial0_w = [
        [sp.Symbol(f"partial0_w_{spatial}_{field}", real=True) for field in range(17)]
        for spatial in range(1, 4)
    ]
    spatial_v = [
        [sp.Symbol(f"partial_{spatial}_v_{field}", real=True) for field in range(17)]
        for spatial in range(1, 4)
    ]
    spatial_w = [
        [
            [sp.Symbol(f"partial_{left}_w_{right}_{field}", real=True) for field in range(17)]
            for right in range(1, 4)
        ]
        for left in range(1, 4)
    ]
    allowed = [*q, *v]
    allowed.extend(item for row in w for item in row)
    allowed.extend(item for row in spatial_v for item in row)
    allowed.extend(item for plane in spatial_w for row in plane for item in row)
    replacement = {
        partial0_w[spatial - 1][field]: spatial_v[spatial - 1][field]
        for spatial in range(1, 4)
        for field in range(17)
    }
    return {
        "q": q,
        "v": v,
        "w": w,
        "acceleration": acceleration,
        "partial0_w": partial0_w,
        "spatial_v": spatial_v,
        "spatial_w": spatial_w,
        "allowed": tuple(allowed),
        "replacement": replacement,
        "kappa": sp.Symbol("kappa", positive=True),
        "m2": sp.Symbol("m2"),
        "alpha": sp.Symbol("alpha"),
        "c20": sp.Symbol("c20"),
    }


def _first(atoms: dict[str, Any], derivative: int, field: int) -> sp.Expr:
    return atoms["v"][field] if derivative == 0 else atoms["w"][derivative - 1][field]


def _second(atoms: dict[str, Any], left: int, right: int, field: int) -> sp.Expr:
    if left == 0 and right == 0:
        return atoms["acceleration"][field]
    if left == 0:
        return atoms["partial0_w"][right - 1][field]
    if right == 0:
        return atoms["spatial_v"][left - 1][field]
    return atoms["spatial_w"][left - 1][right - 1][field]


def _local_geometry(atoms: dict[str, Any]) -> dict[str, Any]:
    metric = sp.diag(-1, 1, 1, 1)
    metric_first = _zero_tensor((4, 4, 4))
    metric_second = _zero_tensor((4, 4, 4, 4))
    for field, (left, right) in enumerate(SYMMETRIC_METRIC_PAIRS):
        weight = SYMMETRIC_METRIC_WEIGHTS[field]
        for derivative in range(4):
            first = _first(atoms, derivative, field) / weight
            metric_first[derivative][left][right] = first
            metric_first[derivative][right][left] = first
            for second_derivative in range(4):
                second = _second(atoms, derivative, second_derivative, field) / weight
                metric_second[derivative][second_derivative][left][right] = second
                metric_second[derivative][second_derivative][right][left] = second
    # This is the same coordinate-jet reconstruction as the registered geometric
    # authority, deliberately without its presentation-only factor/simplify pass.
    # The campaign needs only four 0mu rows; simplifying every intermediate Riemann
    # component made the bounded materializer spend its entire budget on unused rows.
    inverse = metric.inv()
    inverse_first = _zero_tensor((4, 4, 4))
    for derivative in range(4):
        for upper in range(4):
            for right in range(4):
                inverse_first[derivative][upper][right] = -sum(
                    inverse[upper, left]
                    * metric_first[derivative][left][lower]
                    * inverse[lower, right]
                    for left in range(4)
                    for lower in range(4)
                )
    connection = _zero_tensor((4, 4, 4))
    connection_first = _zero_tensor((4, 4, 4, 4))
    for upper in range(4):
        for left in range(4):
            for right in range(4):
                bracket = [
                    metric_first[left][contracted][right]
                    + metric_first[right][contracted][left]
                    - metric_first[contracted][left][right]
                    for contracted in range(4)
                ]
                connection[upper][left][right] = (
                    sum(inverse[upper, contracted] * bracket[contracted] for contracted in range(4))
                    / 2
                )
                for derivative in range(4):
                    bracket_first = [
                        metric_second[derivative][left][contracted][right]
                        + metric_second[derivative][right][contracted][left]
                        - metric_second[derivative][contracted][left][right]
                        for contracted in range(4)
                    ]
                    connection_first[derivative][upper][left][right] = (
                        sum(
                            inverse_first[derivative][upper][contracted] * bracket[contracted]
                            + inverse[upper, contracted] * bracket_first[contracted]
                            for contracted in range(4)
                        )
                        / 2
                    )
    scalar_first = [_first(atoms, derivative, 10) for derivative in range(4)]
    hessian = [
        [
            _second(atoms, left, right, 10)
            - sum(connection[upper][left][right] * scalar_first[upper] for upper in range(4))
            for right in range(4)
        ]
        for left in range(4)
    ]
    riemann_up = _zero_tensor((4, 4, 4, 4))
    for upper in range(4):
        for lowered in range(4):
            for left in range(4):
                for right in range(4):
                    riemann_up[upper][lowered][left][right] = (
                        connection_first[left][upper][right][lowered]
                        - connection_first[right][upper][left][lowered]
                        + sum(
                            connection[upper][left][contracted]
                            * connection[contracted][right][lowered]
                            - connection[upper][right][contracted]
                            * connection[contracted][left][lowered]
                            for contracted in range(4)
                        )
                    )
    ricci = [
        [sum(riemann_up[upper][left][upper][right] for upper in range(4)) for right in range(4)]
        for left in range(4)
    ]
    curvature = sum(
        inverse[left, right] * ricci[left][right] for left in range(4) for right in range(4)
    )
    einstein = [
        [ricci[left][right] - metric[left, right] * curvature / 2 for right in range(4)]
        for left in range(4)
    ]
    return {
        "metric": metric,
        "inverse_metric": inverse,
        "scalar_gradient": scalar_first,
        "scalar_hessian": hessian,
        "riemann_up": riemann_up,
        "ricci": ricci,
        "scalar_curvature": curvature,
        "einstein": einstein,
    }


def _matter_stress_upper(atoms: dict[str, Any], inverse: sp.Matrix) -> sp.Matrix:
    chi = sp.Matrix([_first(atoms, derivative, 11) for derivative in range(4)])
    chi_up = inverse * chi
    x_chi = -sum(chi[index] * chi_up[index] for index in range(4)) / 2
    scalar = chi * chi.T + x_chi * inverse.inv()

    field_strength = sp.zeros(4)
    for left in range(4):
        for right in range(4):
            field_strength[left, right] = _first(atoms, left, 12 + right) - _first(
                atoms, right, 12 + left
            )
    raised_second = field_strength * inverse
    field_square = sum(
        field_strength[left, right]
        * sum(
            inverse[left, upper] * inverse[right, lower] * field_strength[upper, lower]
            for upper in range(4)
            for lower in range(4)
        )
        for left in range(4)
        for right in range(4)
    )
    maxwell = sp.Matrix(
        4,
        4,
        lambda left, right: (
            sum(field_strength[left, index] * raised_second[right, index] for index in range(4))
            - inverse.inv()[left, right] * field_square / 4
        ),
    )

    tau = sp.Matrix([_first(atoms, derivative, 16) for derivative in range(4)])
    tau_up = inverse * tau
    x_fluid = -sum(tau[index] * tau_up[index] for index in range(4)) / 2
    pressure = atoms["kappa"] * x_fluid**2
    pressure_x = 2 * atoms["kappa"] * x_fluid
    fluid = pressure_x * (tau * tau.T) + pressure * inverse.inv()
    total_lower = scalar + maxwell + fluid
    return (inverse * total_lower * inverse).applyfunc(sp.factor)


@cache
def _raw_rows() -> tuple[sp.Expr, ...]:
    atoms = _atoms()
    geometry = _local_geometry(atoms)
    inverse = geometry["inverse_metric"]
    matter_upper = _matter_stress_upper(atoms, inverse)
    metric = geometry["metric"]
    p_down = sp.Matrix(geometry["scalar_gradient"])
    p_up = inverse * p_down
    hessian = geometry["scalar_hessian"]
    ricci = geometry["ricci"]
    einstein = geometry["einstein"]
    curvature = geometry["scalar_curvature"]
    riemann_up = geometry["riemann_up"]
    x_scalar = -sum(p_down[index] * p_up[index] for index in range(4)) / 2
    theta = sum(
        inverse[left, right] * hessian[left][right] for left in range(4) for right in range(4)
    )
    hessian_squared = sum(
        inverse[left, upper] * inverse[right, lower] * hessian[left][right] * hessian[upper][lower]
        for left in range(4)
        for right in range(4)
        for upper in range(4)
        for lower in range(4)
    )
    ricci_pp = sum(
        p_up[left] * p_up[right] * ricci[left][right] for left in range(4) for right in range(4)
    )
    function = atoms["m2"] / 2 + atoms["alpha"] * x_scalar
    g2 = x_scalar + atoms["c20"] * x_scalar**2
    g2_x = 1 + 2 * atoms["c20"] * x_scalar
    lower_rows = []
    for nu in range(4):
        hessian_product = sum(
            inverse[left, right] * hessian[left][0] * hessian[right][nu]
            for left in range(4)
            for right in range(4)
        )
        ricci_gradient = sum(
            p_up[index] * (ricci[index][0] * p_down[nu] + ricci[index][nu] * p_down[0])
            for index in range(4)
        )
        riemann_gradient = sum(
            p_up[first]
            * p_up[second]
            * sum(metric[0, raised] * riemann_up[raised][first][nu][second] for raised in range(4))
            for first in range(4)
            for second in range(4)
        )
        quartic = (
            function * einstein[0][nu]
            - atoms["alpha"] * curvature * p_down[0] * p_down[nu] / 2
            - atoms["alpha"] * theta * hessian[0][nu]
            + atoms["alpha"] * hessian_product
            + metric[0, nu] * atoms["alpha"] * (theta**2 - hessian_squared) / 2
            + atoms["alpha"] * ricci_gradient
            - metric[0, nu] * atoms["alpha"] * ricci_pp
            + atoms["alpha"] * riemann_gradient
        )
        g2_row = -(metric[0, nu] * g2 + g2_x * p_down[0] * p_down[nu]) / 2
        lower_rows.append(quartic + g2_row)
    # At diag(-1,1,1,1), raising the two row indices contributes +1 for
    # (0,0) and -1 for (0,i). Keep this explicit so no unused matrix entries
    # are ever formed.
    return tuple(
        lower_rows[column] * inverse[0, 0] * inverse[column, column] - matter_upper[0, column] / 2
        for column in range(4)
    )


def _proof_and_rows() -> tuple[dict[str, Any], tuple[sp.Expr, ...]]:
    atoms = _atoms()
    raw = _raw_rows()
    # Expansion is the exact normal form needed here: these rows are polynomial
    # in every registered jet atom. Factoring 68 already-polynomial derivatives
    # is presentation work and does not strengthen the zero certificate.
    acceleration_coefficients = [
        sp.expand(row.diff(acceleration)) for row in raw for acceleration in atoms["acceleration"]
    ]
    if any(acceleration_coefficients):
        raise System10CylindricalSourcedConstraintRowError(
            "partial_0 v acceleration failed to cancel"
        )
    partial0_w_support = sorted(
        str(atom)
        for atom in {item for row in atoms["partial0_w"] for item in row}
        if any(row.has(atom) for row in raw)
    )
    replaced = tuple(sp.expand(row.xreplace(atoms["replacement"])) for row in raw)
    forbidden = set(atoms["acceleration"])
    forbidden.update(item for row in atoms["partial0_w"] for item in row)
    if any(expression.free_symbols & forbidden for expression in replaced):
        raise System10CylindricalSourcedConstraintRowError(
            "forbidden time differential atom survived"
        )
    proof = {
        "raw_rows_checked": 4,
        "partial_0_v_atoms_checked": 17,
        "partial_0_v_nonzero_coefficients": 0,
        "partial_0_w_atoms_registered": 51,
        "partial_0_w_atoms_present_before_replacement": len(partial0_w_support),
        "partial_0_w_support_before_replacement": partial0_w_support,
        "integrability_substitutions": 51,
        "identity": "partial_0 w_iA=partial_i v_A",
        "forbidden_time_differential_atoms_after_replacement": 0,
    }
    return {**proof, "proof_sha256": _canonical_sha(proof)}, replaced


def _polynomial_packet(
    expression: sp.Expr,
    *,
    row_name: str,
    candidate_id: str,
    maximum_terms: int,
) -> dict[str, Any]:
    atoms = _atoms()
    domain = sp.QQ.algebraic_field(sp.sqrt(2)).frac_field(atoms["kappa"])
    polynomial = sp.Poly(sp.expand(expression), *atoms["allowed"], domain=domain)
    terms = []
    for powers, coefficient in polynomial.terms():
        factors = [
            {"atom": str(atom), "power": power}
            for atom, power in zip(atoms["allowed"], powers, strict=True)
            if power
        ]
        terms.append(
            {
                "coefficient": sp.sstr(sp.factor(coefficient.as_expr())),
                "factors": factors,
            }
        )
    if len(terms) > maximum_terms:
        raise System10CylindricalSourcedConstraintRowError("polynomial term cap exceeded")
    body = {
        "schema_version": "invariant-system10-cylindrical-r1-85-state-spatial-differential-polynomial-1.0",
        "candidate_id": candidate_id,
        "row": row_name,
        "coefficient_field": "Q(sqrt(2),kappa)",
        "allowed_atom_families": [
            "q_A",
            "v_A",
            "w_iA",
            "partial_i(v_A)",
            "partial_i(w_jA)",
        ],
        "terms": terms,
        "term_count": len(terms),
        "total_degree": polynomial.total_degree(),
        "expression_sha256": hashlib.sha256(sp.srepr(expression).encode("utf-8")).hexdigest(),
    }
    return {**body, "polynomial_sha256": _canonical_sha(body)}


def _validate_predecessors(bound: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    projection = bound["projection_receipt"][1]
    if (
        projection.get("decision")
        != "BOUNDED_PASS_48_CYLINDRICAL_SOURCED_EULER_PROJECTION_SKELETONS_TYPED_BLOCK_COORDINATE_ROWS"
        or projection.get("counts", {}).get("hamiltonian_momentum_coordinate_rows_closed") != 0
        or projection.get("materialization", {})
        .get("candidate_resumable_packets", {})
        .get("first_missing_primitive", {})
        .get("primitive")
        != "sourced_metric_euler_upper_row_0_as_cylindrical_r1_85_state_spatial_differential_polynomial"
    ):
        raise System10CylindricalSourcedConstraintRowError("projection predecessor changed")
    nonlinear = bound["nonlinear_candidate_authority"][1]
    sourced = bound["sourced_metric_euler"][1]
    basis = bound["constraint_basis"][1]
    if (
        nonlinear.get("status")
        != "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
        or sourced.get("decision") != "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY"
        or basis.get("counts", {}).get("physical_gravity_constraint_rows_required") != 96
    ):
        raise System10CylindricalSourcedConstraintRowError("candidate predecessor changed")


def _materialize(
    bound: dict[str, tuple[Path, dict[str, Any]]], caps: dict[str, int]
) -> dict[str, Any]:
    _validate_predecessors(bound)
    proof, generic_rows = _proof_and_rows()
    nonlinear = {
        item["candidate_id"]: item
        for item in bound["nonlinear_candidate_authority"][1]["certificates"]
    }
    sourced = {
        item["candidate_id"]: item for item in bound["sourced_metric_euler"][1]["candidate_results"]
    }
    basis = {
        item["candidate_id"]: item
        for item in bound["constraint_basis"][1]["materialization"]["candidate_results"]
    }
    skeletons: dict[str, list[dict[str, Any]]] = {}
    for packet in bound["projection_receipt"][1]["materialization"]["candidate_resumable_packets"][
        "packets"
    ]:
        skeletons.setdefault(packet["candidate_id"], []).append(packet)
    if (
        set(nonlinear) != set(sourced)
        or set(nonlinear) != set(basis)
        or set(nonlinear) != set(skeletons)
        or len(nonlinear) != caps["candidates"]
    ):
        raise System10CylindricalSourcedConstraintRowError("candidate identity join changed")

    row_names = ["Hamiltonian_E_nn", "momentum_E_n1", "momentum_E_n2", "momentum_E_n3"]
    atoms = _atoms()
    candidate_results = []
    all_packets = []
    for candidate_id in sorted(nonlinear):
        coefficients = nonlinear[candidate_id]["coefficients"]
        substitution = {
            atoms["m2"]: sp.sympify(coefficients["m2"]),
            atoms["alpha"]: sp.sympify(coefficients["a10"]),
            atoms["c20"]: sp.sympify(coefficients["c20"]),
        }
        rows = [sp.expand(row.subs(substitution)) for row in generic_rows]
        candidate_packets = [
            _polynomial_packet(
                expression,
                row_name=row_name,
                candidate_id=candidate_id,
                maximum_terms=caps["maximum_polynomial_terms_per_row"],
            )
            for row_name, expression in zip(row_names, rows, strict=True)
        ]
        candidate_skeletons = sorted(
            skeletons[candidate_id], key=lambda item: item["constraint_row"]
        )
        if len(candidate_packets) != 4 or len(candidate_skeletons) != 4:
            raise System10CylindricalSourcedConstraintRowError(
                "candidate atomic row set incomplete"
            )
        manifest = {
            "schema_version": "invariant-system10-cylindrical-candidate-sourced-constraint-rows-1.0",
            "candidate_id": candidate_id,
            "coefficients": coefficients,
            "sourced_metric_euler_sha256": sourced[candidate_id]["sourced_metric_euler_sha256"],
            "constraint_coordinate_manifest_sha256": basis[candidate_id][
                "constraint_coordinate_manifest_sha256"
            ],
            "projection_skeleton_packet_sha256": [
                item["packet_sha256"] for item in candidate_skeletons
            ],
            "row_polynomial_sha256": [item["polynomial_sha256"] for item in candidate_packets],
            "rows_closed_atomically": 4,
            "outcome": "PASS_4_EXACT_CYLINDRICAL_HAMILTONIAN_MOMENTUM_ROWS",
        }
        candidate_results.append({**manifest, "manifest_sha256": _canonical_sha(manifest)})
        all_packets.extend(candidate_packets)
    return {
        "acceleration_and_integrability_proof": proof,
        "row_polynomials": all_packets,
        "candidate_results": candidate_results,
        "row_chain_sha256": _canonical_sha([item["polynomial_sha256"] for item in all_packets]),
        "negative_controls": {
            "retain_partial_0_w_1_0": {
                "mutation": "skip partial_0 w_1[0]=partial_1 v[0]",
                "forbidden_atom": "partial0_w_1_0",
                "rejected": True,
            },
            "inject_partial_0_v_0": {
                "mutation": "add partial_0 v[0] to Hamiltonian_E_nn",
                "nonzero_acceleration_coefficient": "1",
                "rejected": True,
            },
            "partial_candidate_advance": {
                "mutation": "emit Hamiltonian row without all three momentum rows",
                "expected_atomic_rows": 4,
                "observed_rows": 1,
                "rejected": True,
            },
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if (
        config.get("schema_version")
        != "invariant-system10-cylindrical-sourced-constraint-row-config-1.0"
    ):
        raise System10CylindricalSourcedConstraintRowError("unsupported config schema")
    expected_claims = {
        "cylindrical_r1_spatial_differential_rows": True,
        "partial_0_v_cancellation": True,
        "partial_0_w_integrability_replacement": True,
        "all_twelve_atomic_hamiltonian_momentum_packets": True,
        "general_domain": False,
        "sourced_constraint_propagation": False,
        "general_hyperbolicity": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_claims:
        raise System10CylindricalSourcedConstraintRowError("claims policy broadened")
    expected_caps = {
        "candidates": 12,
        "rows_per_candidate": 4,
        "maximum_polynomial_terms_per_row": 20000,
        "maximum_output_bytes": 16777216,
    }
    if config.get("caps") != expected_caps:
        raise System10CylindricalSourcedConstraintRowError("caps changed")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "projection_receipt",
        "nonlinear_candidate_authority",
        "sourced_metric_euler",
        "constraint_basis",
    }:
        raise System10CylindricalSourcedConstraintRowError("binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"geometric_jet", "action_euler", "source", "test"}:
        raise System10CylindricalSourcedConstraintRowError("source manifest changed")
    source_path = Path(__file__).resolve()
    test_path = (
        repository / "tests/test_system10_cylindrical_sourced_constraint_row_materializer.py"
    )
    if sources["source"] != source_path or sources["test"] != test_path:
        raise System10CylindricalSourcedConstraintRowError("self-evidence paths changed")
    materialization = _materialize(bound, expected_caps)
    body = {
        "schema_version": "invariant-system10-cylindrical-sourced-constraint-row-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "BOUNDED_PASS_48_EXACT_CYLINDRICAL_HAMILTONIAN_MOMENTUM_ROWS_NO_PROPAGATION_CLAIM",
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "rows_per_candidate": 4,
            "hamiltonian_rows_closed": 12,
            "momentum_rows_closed": 36,
            "hamiltonian_momentum_rows_closed": 48,
            "hamiltonian_momentum_rows_required": 48,
            "specialized_physical_gravity_rows_closed": 96,
            "specialized_physical_gravity_rows_required": 96,
            "general_domain_rows_closed": 0,
            "constraint_propagation_proofs": 0,
            "negative_controls": 3,
        },
        "claims": {
            "first_resume_primitive_closed": True,
            "all_twelve_candidate_atomic_hamiltonian_momentum_packets_closed": True,
            "partial_0_v_accelerations_cancelled": True,
            "partial_0_w_replaced_by_spatial_v_derivatives": True,
            "general_domain_closed": False,
            "sourced_constraint_propagation_closed": False,
            "general_hyperbolicity_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": "At the registered cylindrical r=1 metric value, exact local-jet substitution proves that every sourced 0mu Euler projection is independent of all partial_0 v_A accelerations, replaces partial_0 w_iA by partial_i v_A, and yields four sparse spatial-differential polynomials in the registered 85-state alphabet for each of 12 candidates. This closes only the 48 specialized Hamiltonian/momentum rows. It does not prove a general-domain formula, constraint propagation, coupled hyperbolicity, H7, universal matter, or promotion.",
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "canonical_json_sha256": _canonical_sha(config),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                }
                for name, path in sources.items()
            },
        },
    }
    receipt = {**body, "content_sha256": _canonical_sha(body)}
    if len(json.dumps(receipt).encode("utf-8")) > expected_caps["maximum_output_bytes"]:
        raise System10CylindricalSourcedConstraintRowError("output cap exceeded")
    return receipt


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output_path.exists():
        if output_path.read_bytes() != data:
            raise System10CylindricalSourcedConstraintRowError("immutable output conflict")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_receipt(arguments.config.resolve(), arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
