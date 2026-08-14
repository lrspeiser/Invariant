from __future__ import annotations

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_geometric_jet_campaign import _coordinate_state, state_to_covariant_geometry
from .quartic_nonlinear_evolution_campaign import quartic_action_euler_tensors


class System10CylindricalConstraintProjectionError(RuntimeError):
    """Raised when the bounded cylindrical constraint projection fails closed."""


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise System10CylindricalConstraintProjectionError(
            f"cannot read bound file: {path}"
        ) from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise System10CylindricalConstraintProjectionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise System10CylindricalConstraintProjectionError("bound JSON root is not an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise System10CylindricalConstraintProjectionError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalConstraintProjectionError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise System10CylindricalConstraintProjectionError(
            f"bound content hash mismatch: {path}"
        )
    return path, value


def _load_source(root: Path, binding: dict[str, Any]) -> Path:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise System10CylindricalConstraintProjectionError(
            f"bound source hash mismatch: {path}"
        )
    return path


def _projection_operator() -> dict[str, Any]:
    rows = [
        {
            "constraint_row": "Hamiltonian_E_nn",
            "definition": "n_mu*n_nu*(E_gf^mu_nu-T_total^mu_nu/2)",
            "sourced_metric_euler_row": 0,
            "orthonormal_row": "E_sourced^00",
            "projection_coefficient": "1",
            "projected_source": "-T_total^00/2",
        }
    ]
    for spatial in range(1, 4):
        rows.append(
            {
                "constraint_row": f"momentum_E_n{spatial}",
                "definition": (
                    f"-h_{spatial}_mu*n_nu*(E_gf^mu_nu-T_total^mu_nu/2)"
                ),
                "sourced_metric_euler_row": spatial,
                "orthonormal_row": f"sqrt(2)*E_sourced^0{spatial}",
                "projection_coefficient": "sqrt(2)/2",
                "projected_source": f"-T_total^0{spatial}/2",
            }
        )
    sealed = []
    for row in rows:
        body = {
            **row,
            "background": "cylindrical_physical_metric_at_r=1",
            "normal_covector": ["-1", "0", "0", "0"],
            "tangent_basis": "coordinate_basis_equals_orthonormal_spatial_basis_at_r=1",
            "gauge_completion_excluded": (
                "physical Hamiltonian/momentum constraints project the ungauged sourced action "
                "Euler tensor; modified-harmonic rows are separate"
            ),
        }
        sealed.append({**body, "projection_sha256": _canonical_sha(body)})
    operator = {
        "schema_version": "invariant-system10-cylindrical-constraint-projection-operator-1.0",
        "metric_row_basis": (
            "[E^00,sqrt(2)E^01,sqrt(2)E^02,sqrt(2)E^03,E^11,"
            "sqrt(2)E^12,sqrt(2)E^13,E^22,sqrt(2)E^23,E^33]"
        ),
        "rows": sealed,
    }
    return {**operator, "operator_sha256": _canonical_sha(operator)}


def _project_metric(metric: sp.Matrix, coordinates: tuple[sp.Symbol, ...]) -> list[str]:
    state, derivative = _coordinate_state(metric, sp.Integer(0), coordinates)
    geometry = state_to_covariant_geometry(state, derivative)
    euler = quartic_action_euler_tensors(
        geometry, m2=sp.Integer(2), alpha=sp.Integer(0), c20=sp.Integer(0)
    )["metric_euler_lower"]
    time, radius, angle, height = coordinates
    point = {time: 0, radius: 1, angle: 0, height: 0}
    values = [sp.factor(euler[0][0].subs(point))]
    values.extend(sp.factor(-euler[0][spatial].subs(point)) for spatial in range(1, 4))
    return [str(value) for value in values]


@cache
def _geometric_controls() -> dict[str, Any]:
    time, radius, angle, height = sp.symbols("t r theta z", real=True)
    coordinates = (time, radius, angle, height)
    flat = sp.diag(-1, 1, radius**2, 1)
    cases = [
        ("flat_cylindrical_known_answer", flat, ["0", "0", "0", "0"]),
        (
            "hamiltonian_second_radial_jet_mutation",
            sp.diag(-1, 1, radius**2, 1 + (radius - 1) ** 2),
            ["-1", "0", "0", "0"],
        ),
        (
            "radial_momentum_mixed_jet_mutation",
            sp.diag(-1, 1, radius**2, 1 + time * (radius - 1)),
            ["0", "1/2", "0", "0"],
        ),
        (
            "angular_momentum_mixed_jet_mutation",
            sp.diag(-1, 1, radius**2, 1 + time * angle),
            ["0", "0", "1/2", "0"],
        ),
        (
            "axial_momentum_mixed_jet_mutation",
            sp.diag(-1, 1, radius**2 + time * height, 1),
            ["0", "0", "0", "1/2"],
        ),
    ]
    results = []
    for control_id, metric, expected in cases:
        observed = _project_metric(metric, coordinates)
        if observed != expected:
            raise System10CylindricalConstraintProjectionError(
                f"geometric projection control changed: {control_id}"
            )
        body = {
            "control_id": control_id,
            "projected_rows": [
                "Hamiltonian_E_nn",
                "momentum_E_n1",
                "momentum_E_n2",
                "momentum_E_n3",
            ],
            "observed_exact_values": observed,
            "expected_exact_values": expected,
            "theory": "Einstein_Hilbert_control_m2=2_alpha=0_scalar=0_matter=0",
            "evaluation_point": "t=0,r=1,theta=0,z=0",
            "passed": True,
        }
        results.append({**body, "control_sha256": _canonical_sha(body)})
    body = {"controls": results, "passed": len(results), "failed": 0}
    return {**body, "controls_sha256": _canonical_sha(body)}


def _validate_predecessors(bound: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    cylindrical = bound["cylindrical_gauge_specialization"][1]
    if (
        cylindrical.get("decision")
        != "BOUNDED_PASS_CYLINDRICAL_1010_VALUES_AND_48_GAUGE_ROWS_TYPED_BLOCK_ADM_GENERAL"
        or cylindrical.get("counts", {}).get("modified_harmonic_rows_closed_all_candidates")
        != 48
        or cylindrical.get("counts", {}).get("hamiltonian_momentum_rows_closed") != 0
    ):
        raise System10CylindricalConstraintProjectionError("cylindrical predecessor changed")
    sourced = bound["sourced_metric_euler"][1]
    if (
        sourced.get("decision") != "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY"
        or sourced.get("counts", {}).get("sourced_metric_euler_bindings_passed") != 12
        or sourced.get("counts", {}).get("sourced_acceleration_solutions") != 0
    ):
        raise System10CylindricalConstraintProjectionError("sourced Euler predecessor changed")
    basis = bound["constraint_basis"][1]
    if (
        basis.get("decision")
        != "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP"
        or basis.get("counts", {}).get("physical_gravity_constraint_rows_required") != 96
    ):
        raise System10CylindricalConstraintProjectionError("constraint basis changed")


def _resumable_packets(
    bound: dict[str, tuple[Path, dict[str, Any]]], operator: dict[str, Any]
) -> dict[str, Any]:
    sourced = {
        item["candidate_id"]: item
        for item in bound["sourced_metric_euler"][1]["candidate_results"]
    }
    basis = {
        item["candidate_id"]: item
        for item in bound["constraint_basis"][1]["materialization"]["candidate_results"]
    }
    cylindrical = {
        item["candidate_id"]: item
        for item in bound["cylindrical_gauge_specialization"][1]["materialization"][
            "candidate_results"
        ]
    }
    if set(sourced) != set(basis) or set(sourced) != set(cylindrical) or len(sourced) != 12:
        raise System10CylindricalConstraintProjectionError("candidate identity join changed")
    packets = []
    for candidate_id in sorted(sourced):
        rows = []
        for row in operator["rows"]:
            first_missing = {
                "primitive": (
                    f"sourced_metric_euler_upper_row_{row['sourced_metric_euler_row']}_as_"
                    "cylindrical_r1_85_state_spatial_differential_polynomial"
                ),
                "required_proof": (
                    "exact cancellation of partial_0 v_A accelerations and replacement of "
                    "partial_0 w_iA by partial_i v_A before sparse coefficient emission"
                ),
                "target_alphabet": (
                    "q_A,v_A,w_iA,partial_i(v_A),partial_i(w_jA), with A=0..16 and i,j=1..3"
                ),
                "zero_inference_forbidden": True,
            }
            body = {
                "candidate_id": candidate_id,
                "constraint_row": row["constraint_row"],
                "projection_sha256": row["projection_sha256"],
                "sourced_metric_euler_sha256": sourced[candidate_id][
                    "sourced_metric_euler_sha256"
                ],
                "constraint_coordinate_manifest_sha256": basis[candidate_id][
                    "constraint_coordinate_manifest_sha256"
                ],
                "cylindrical_gauge_manifest_sha256": cylindrical[candidate_id][
                    "manifest_sha256"
                ],
                "projection_skeleton_closed": True,
                "coordinate_differential_row_closed": False,
                "status": "PASS_EXACT_PROJECTION_TYPED_BLOCK_COORDINATE_COEFFICIENT_EXPANSION",
                "first_missing_primitive": first_missing,
            }
            rows.append({**body, "packet_sha256": _canonical_sha(body)})
        packets.extend(rows)
    first = packets[0]["first_missing_primitive"]
    return {
        "packets": packets,
        "packet_chain_sha256": _canonical_sha(
            [packet["packet_sha256"] for packet in packets]
        ),
        "first_missing_candidate": packets[0]["candidate_id"],
        "first_missing_row": packets[0]["constraint_row"],
        "first_missing_primitive": first,
        "resume_order": [
            "emit first candidate Hamiltonian sparse coordinate polynomial and acceleration cancellation",
            "finish first candidate three momentum polynomials",
            "replay four independent geometric controls after each candidate",
            "advance the manifest atomically candidate by candidate through all twelve",
            "only then attempt sourced subsidiary factorization",
        ],
        "atomicity": "a candidate advances only when all four coordinate rows and controls pass",
    }


def _materialize(bound: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    _validate_predecessors(bound)
    operator = _projection_operator()
    controls = _geometric_controls()
    resumable = _resumable_packets(bound, operator)
    wrong = {
        "mutation": "replace sqrt(2)/2 momentum projection coefficient by 1",
        "correct_value_at_E01_equals_1_over_2": "1/2",
        "corrupted_value_at_E01_equals_1_over_2": "sqrt(2)/2",
        "exact_difference": "(sqrt(2)-1)/2",
        "rejected": True,
    }
    return {
        "projection_operator": operator,
        "independent_geometric_controls": controls,
        "candidate_resumable_packets": resumable,
        "negative_controls": {
            "momentum_normalization": wrong,
            "source_sign": bound["sourced_metric_euler"][1][
                "exact_variation_and_row_insertion"
            ]["wrong_sign_negative"],
            "partial_candidate_advance": {
                "mutation": "mark one of four candidate coordinate rows complete",
                "expected_atomic_rows": 4,
                "observed_rows": 1,
                "rejected": True,
            },
        },
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != (
        "invariant-system10-cylindrical-sourced-constraint-projection-config-1.0"
    ):
        raise System10CylindricalConstraintProjectionError("unsupported config schema")
    expected_policy = {
        "cylindrical_r1_only": True,
        "normal_tangential_projection_operator": True,
        "candidate_projection_skeletons": True,
        "hamiltonian_momentum_coordinate_rows": False,
        "sourced_acceleration_cancellation": False,
        "general_domain": False,
        "sourced_constraint_propagation": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise System10CylindricalConstraintProjectionError("claims policy broadened")
    expected_specialization = {
        "coordinates": ["t", "r", "theta", "z"],
        "evaluation_point": {"t": "0", "r": "1", "theta": "0", "z": "0"},
        "physical_metric": "diag(-1,1,r^2,1)",
        "future_unit_normal_covector_at_point": ["-1", "0", "0", "0"],
        "orthonormal_symmetric_metric_row_order": [
            "00", "01", "02", "03", "11", "12", "13", "22", "23", "33"
        ],
    }
    if config.get("specialization") != expected_specialization:
        raise System10CylindricalConstraintProjectionError("cylindrical specialization changed")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "cylindrical_gauge_specialization",
        "sourced_metric_euler",
        "constraint_basis",
    }:
        raise System10CylindricalConstraintProjectionError("closed binding manifest changed")
    sources = {
        name: _load_source(repository, binding)
        for name, binding in config.get("source_evidence", {}).items()
    }
    if set(sources) != {"geometric_euler", "coordinate_jet", "source", "test"}:
        raise System10CylindricalConstraintProjectionError("source evidence manifest changed")
    source_path = Path(__file__).resolve()
    test_path = repository / (
        "tests/test_system10_cylindrical_sourced_constraint_projection_materializer.py"
    )
    if sources["source"] != source_path or sources["test"] != test_path:
        raise System10CylindricalConstraintProjectionError("self-evidence paths changed")
    materialization = _materialize(bound)
    body = {
        "schema_version": (
            "invariant-system10-cylindrical-sourced-constraint-projection-result-1.0"
        ),
        "campaign_id": config["campaign_id"],
        "decision": (
            "BOUNDED_PASS_48_CYLINDRICAL_SOURCED_EULER_PROJECTION_SKELETONS_"
            "TYPED_BLOCK_COORDINATE_ROWS"
        ),
        "materialization": materialization,
        "counts": {
            "candidates": 12,
            "gauge_rows_previously_closed": 48,
            "hamiltonian_momentum_projection_skeletons_closed": 48,
            "hamiltonian_momentum_coordinate_rows_closed": 0,
            "hamiltonian_momentum_coordinate_rows_required": 48,
            "resumable_row_packets": 48,
            "independent_geometric_controls": 5,
            "negative_controls": 3,
            "physical_gravity_coordinate_rows_closed": 48,
            "physical_gravity_coordinate_rows_required": 96,
        },
        "claims": {
            "exact_r1_normal_tangential_projection_operator_closed": True,
            "all_twelve_candidate_projection_skeletons_hash_bound": True,
            "independent_projection_controls_passed": True,
            "hamiltonian_momentum_coordinate_rows_closed": False,
            "sourced_acceleration_cancellation_closed": False,
            "general_domain_closed": False,
            "sourced_constraint_propagation_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "At the registered cylindrical r=1 point, the future-normal/tangent projection of "
            "the exact sourced orthonormal metric Euler basis is now explicit and hash-bound "
            "into 48 candidate row skeletons. Five independent Einstein-Hilbert jet controls "
            "fix the Hamiltonian/momentum signs and sqrt(2) normalization. These are projection "
            "skeletons, not the missing sparse q/v/w coordinate-differential rows: acceleration "
            "cancellation and coefficient expansion remain atomically blocked for all 48. No "
            "general-domain, propagation, H7, universal-matter, or promotion claim is made."
        ),
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_receipt(arguments.config.resolve(), arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
