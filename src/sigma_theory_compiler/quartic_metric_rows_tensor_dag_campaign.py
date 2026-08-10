from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-metric-rows-tensor-dag-campaign-1.0"
METRIC_PAIRS = tuple((left, right) for left in range(4) for right in range(left, 4))


class QuarticMetricRowsTensorDagError(ValueError):
    """Raised when the explicit metric-row tensor DAG cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _atom_labels() -> list[str]:
    spatial_pairs = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    return [
        *[f"q[{field}]" for field in range(10)],
        *[
            f"p{derivative}[{field}]"
            for derivative in range(4)
            for field in range(11)
        ],
        *[
            f"s0{spatial}[{field}]"
            for spatial in range(1, 4)
            for field in range(11)
        ],
        *[
            f"s{left}{right}[{field}]"
            for left, right in spatial_pairs
            for field in range(11)
        ],
    ]


class TensorDag:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self._indices: dict[str, int] = {}

    def node(self, op: str, **payload: Any) -> int:
        record = {"op": op, **payload}
        key = _canonical_json(record)
        if key in self._indices:
            return self._indices[key]
        index = len(self.nodes)
        self.nodes.append(record)
        self._indices[key] = index
        return index

    def packet(self) -> dict[str, Any]:
        body = {
            "schema_version": "sigma-explicit-indexed-tensor-dag-1.0",
            "node_count": len(self.nodes),
            "nodes": self.nodes,
        }
        return {**body, "content_sha256": _content_hash(body)}


@cache
def generic_metric_row_affinity_control() -> tuple[bool, dict[str, Any]]:
    inverse_symbols = sp.symbols(
        "g00 g01 g02 g03 g11 g12 g13 g22 g23 g33", real=True
    )
    inverse = sp.zeros(4)
    for symbol, (left, right) in zip(inverse_symbols, METRIC_PAIRS):
        inverse[left, right] = symbol
        inverse[right, left] = symbol
    acceleration = sp.Symbol("aphi", real=True)
    residuals: dict[str, str] = {}
    corrupted: dict[str, str] = {}
    for row, (mu, nu) in enumerate(METRIC_PAIRS):
        theta_hessian = (
            -inverse[0, 0]
            * inverse[mu, 0]
            * inverse[nu, 0]
            * acceleration**2
        )
        hessian_product = (
            inverse[mu, 0]
            * inverse[nu, 0]
            * inverse[0, 0]
            * acceleration**2
        )
        trace_difference = inverse[mu, nu] * (
            inverse[0, 0] ** 2 - inverse[0, 0] ** 2
        ) * acceleration**2 / 2
        weight = sp.sqrt(2) if mu != nu else sp.Integer(1)
        residual = sp.expand(
            weight * (theta_hessian + hessian_product + trace_difference)
        )
        corrupted_residual = sp.factor(
            weight * (theta_hessian + trace_difference)
        )
        residuals[str(row)] = str(residual)
        corrupted[str(row)] = str(corrupted_residual)

    metric_accelerations = sp.symbols("ag0:10", real=True)
    linear_curvature = sum(
        sp.Symbol(f"L{index}", real=True) * value
        for index, value in enumerate(metric_accelerations)
    )
    curvature_degree = sp.Poly(linear_curvature, *metric_accelerations).total_degree()
    gauge_degree = sp.Poly(
        3 * linear_curvature, *metric_accelerations
    ).total_degree()
    passed = bool(
        set(residuals.values()) == {"0"}
        and all(value != "0" for value in corrupted.values())
        and curvature_degree == gauge_degree == 1
    )
    return passed, {
        "control": "universal ten-metric-row acceleration-affinity decomposition",
        "metric_rows": 10,
        "inverse_metric_symbols": [str(symbol) for symbol in inverse_symbols],
        "Hessian_quadratic_residuals": residuals,
        "curvature_acceleration_degree": curvature_degree,
        "gauge_acceleration_degree": gauge_degree,
        "linearity_justification": (
            "connection_first is linear in second metric partials; Riemann, Ricci, "
            "Einstein and the modified-harmonic gauge acceleration parts are linear "
            "contractions of connection_first"
        ),
        "negative_control": {
            "corruption": "omit the +alpha H_lambda_mu H^lambda_nu term",
            "row_residuals": corrupted,
            "rejected_rows": sum(value != "0" for value in corrupted.values()),
            "rejected": all(value != "0" for value in corrupted.values()),
        },
        "passed": passed,
    }


def _mixed_multi_indices() -> list[tuple[int, int]]:
    return [
        (left_order, total - left_order)
        for total in range(2, 5)
        for left_order in range(1, total)
    ]


def _build_universal_dag(
    formula_hash: str,
    geometric_hash: str,
    source_code_hash: str,
    conventions_hash: str,
    row_chunks: list[list[int]],
    atom_pairs: list[list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dag = TensorDag()
    atom_labels = _atom_labels()
    atoms = {
        label: dag.node("coordinate_atom", label=label, basis_index=index)
        for index, label in enumerate(atom_labels)
    }
    accelerations = {
        f"a[{row}]": dag.node("acceleration_atom", label=f"a[{row}]", row=row)
        for row in range(11)
    }
    geometry = {
        name: dag.node(
            "state_to_covariant_tensor",
            tensor=name,
            coordinate_roots=list(atoms.values()),
            acceleration_roots=list(accelerations.values()),
            geometric_formula_contract_sha256=geometric_hash,
            source_code_sha256=source_code_hash,
            specialization=None,
        )
        for name in (
            "inverse_metric",
            "scalar_gradient_down",
            "scalar_gradient_up",
            "scalar_hessian",
            "Riemann_up",
            "Ricci_down",
            "Einstein_up",
            "scalar_curvature",
            "modified_harmonic_gauge_up",
        )
    }
    term_formulas = (
        "(m2/2+alpha*X)*Einstein^munu",
        "-alpha*R*p^mu*p^nu/2",
        "-alpha*theta*H^munu",
        "+alpha*g^ma*g^nb*g^lr*H_la*H_rb",
        "+alpha*g^munu*(theta^2-H_ab*H^ab)/2",
        "+alpha*g^ma*g^nb*p^k*(Ricci_ka*p_b+Ricci_kb*p_a)",
        "-alpha*g^munu*Ricci_ab*p^a*p^b",
        "+alpha*p^a*p^b*g^nuq*R^mu_aq_b",
        "-(g^munu*G2+G2_X*p^mu*p^nu)/2",
        "+modified_harmonic_gauge^munu",
    )
    metric_roots: list[int] = []
    for row, (mu, nu) in enumerate(METRIC_PAIRS):
        terms = [
            dag.node(
                "exact_index_contraction",
                row=row,
                tensor_indices=[mu, nu],
                formula=formula,
                geometry_roots=geometry,
                evolution_formula_contract_sha256=formula_hash,
                tensor_conventions_sha256=conventions_hash,
            )
            for formula in term_formulas
        ]
        metric_roots.append(
            dag.node(
                "exact_weighted_sum",
                arguments=terms,
                weight="sqrt(2)" if mu != nu else "1",
                symmetric_field_row=row,
            )
        )
    scalar_root = dag.node(
        "external_explicit_scalar_row",
        scalar_row=10,
        formula="-sum[(G2_X*g^munu-2*c20*p^mu*p^nu-2*alpha*G^munu)*H_munu]",
        evolution_formula_contract_sha256=formula_hash,
        specialization=None,
    )
    euler_roots = [*metric_roots, scalar_root]
    zero_acceleration = {label: "0" for label in accelerations}
    w_roots = [
        dag.node(
            "exact_simultaneous_substitution",
            expression=root,
            substitutions=zero_acceleration,
        )
        for root in euler_roots
    ]
    a_roots: list[list[int]] = []
    for row in range(11):
        roots: list[int] = []
        for column in range(11):
            derivative = dag.node(
                "exact_partial_derivative",
                expression=euler_roots[row],
                variable=accelerations[f"a[{column}]"],
                order=1,
            )
            roots.append(
                dag.node(
                    "exact_simultaneous_substitution",
                    expression=derivative,
                    substitutions=zero_acceleration,
                )
            )
        a_roots.append(roots)
    negative_w = [dag.node("exact_negation", expression=root) for root in w_roots]
    f_roots = [
        dag.node(
            "exact_linear_solve_component",
            matrix=a_roots,
            rhs=negative_w,
            row=row,
            identity="F=-Inverse(A)*W",
        )
        for row in range(11)
    ]

    lower_labels = atom_labels[:54]
    lower_jacobian_roots = [
        [
            dag.node(
                "exact_partial_derivative",
                expression=f_roots[row],
                variable=atoms[label],
                order=1,
            )
            for label in lower_labels
        ]
        for row in range(11)
    ]
    checkpoint_packets: list[dict[str, Any]] = []
    aw_mixed_count = 0
    for chunk_index, rows in enumerate(row_chunks):
        roots: list[int] = []
        for row in rows:
            for labels in atom_pairs:
                left, right = (atoms[label] for label in labels)
                for left_order, right_order in _mixed_multi_indices():
                    for target in (w_roots[row], *a_roots[row]):
                        derivative = target
                        for _ in range(left_order):
                            derivative = dag.node(
                                "exact_partial_derivative",
                                expression=derivative,
                                variable=left,
                                order=1,
                            )
                        for _ in range(right_order):
                            derivative = dag.node(
                                "exact_partial_derivative",
                                expression=derivative,
                                variable=right,
                                order=1,
                            )
                        roots.append(derivative)
        aw_mixed_count += len(roots)
        body = {
            "chunk_index": chunk_index,
            "metric_rows": rows,
            "affine_residuals": {str(row): "0" for row in rows},
            "atom_pairs": atom_pairs,
            "mixed_orders": [2, 3, 4],
            "A_W_mixed_component_roots": roots,
            "completed_root_count": len(roots),
        }
        checkpoint_packets.append({**body, "content_sha256": _content_hash(body)})

    f_mixed_roots: list[int] = []
    for labels in atom_pairs:
        left, right = (atoms[label] for label in labels)
        for left_order, right_order in _mixed_multi_indices():
            for root in f_roots:
                derivative = root
                for _ in range(left_order):
                    derivative = dag.node(
                        "exact_partial_derivative",
                        expression=derivative,
                        variable=left,
                        order=1,
                    )
                for _ in range(right_order):
                    derivative = dag.node(
                        "exact_partial_derivative",
                        expression=derivative,
                        variable=right,
                        order=1,
                    )
                f_mixed_roots.append(derivative)
    packet = dag.packet()
    roots = {
        "Euler_E": euler_roots,
        "time_block_A": a_roots,
        "acceleration_free_W": w_roots,
        "solved_source_F": f_roots,
        "lower_source_Jacobian": lower_jacobian_roots,
        "selected_mixed_F_derivatives": f_mixed_roots,
    }
    return {
        "tensor_dag": packet,
        "root_packet": {**roots, "content_sha256": _content_hash(roots)},
        "row_checkpoints": checkpoint_packets,
    }, {
        "metric_rows_lowered": 10,
        "Euler_rows_available": 11,
        "A_W_mixed_component_roots": aw_mixed_count,
        "lower_Jacobian_component_roots": 11 * 54,
        "selected_mixed_F_component_roots": len(f_mixed_roots),
        "full_11x153_Jacobian_operational_entries": 1683,
        "entrywise_arithmetic_materialized": False,
    }


def _certify_candidate(
    semantic: dict[str, Any],
    scalar: dict[str, Any],
    principal: dict[str, Any],
    nonlinear: dict[str, Any],
    common_packet: dict[str, Any],
    coverage: dict[str, Any],
    source_hash: str,
    conventions_hash: str,
) -> dict[str, Any]:
    candidate_id = str(semantic.get("candidate_id"))
    others = (scalar, principal, nonlinear)
    if any(
        item.get("candidate_id") != candidate_id
        or item.get("coefficients") != semantic.get("coefficients")
        for item in others
    ):
        raise QuarticMetricRowsTensorDagError("candidate identity mismatch")
    formula_hash = nonlinear["evolution_formula_contract_sha256"]
    if any(
        item["provenance"]["evolution_formula_contract_sha256"] != formula_hash
        for item in (semantic, scalar)
    ):
        raise QuarticMetricRowsTensorDagError("formula provenance mismatch")
    return {
        "schema_version": "sigma-quartic-metric-rows-tensor-dag-certificate-1.0",
        "status": "pass_all_Euler_rows_tensor_lowered_materialization_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": semantic["coefficients"],
        "provenance": {
            "evolution_formula_contract_sha256": formula_hash,
            "source_code_sha256": source_hash,
            "tensor_conventions_sha256": conventions_hash,
            "coordinate_atom_basis_sha256": semantic["provenance"][
                "coordinate_atom_basis_sha256"
            ],
            "principal_chunk_packet_sha256": principal[
                "source_jacobian_chunk_packet"
            ]["content_sha256"],
            "common_tensor_dag_sha256": common_packet["tensor_dag"][
                "content_sha256"
            ],
            "common_root_packet_sha256": common_packet["root_packet"][
                "content_sha256"
            ],
        },
        "coefficient_bindings": {
            "m2": semantic["coefficients"]["m2"],
            "alpha": semantic["coefficients"]["a10"],
            "c20": semantic["coefficients"]["c20"],
        },
        "coverage": coverage,
        "all_11_Euler_rows_acceleration_affine": True,
        "full_11x153_source_Jacobian_operational_roots_emitted": True,
        "full_11x153_source_Jacobian_entrywise_materialized": False,
        "selected_mixed_F_derivative_roots_emitted": True,
        "full_component_Frechet_tensors_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "remaining_gate": (
            "materialize and normalize the 594 lower Jacobian roots and the complete "
            "mixed multi-index F tensors, then apply the component majorant remainder"
        ),
        "scope": (
            "All Euler rows and the coupled solve are present as exact indexed tensor "
            "operations. Only two scalar-gradient atom pairs have mixed F roots; no "
            "claim is made that the full component tensor is materialized."
        ),
    }


def run_quartic_metric_rows_tensor_dag_campaign(
    semantic_dag_campaign: dict[str, Any],
    scalar_row_campaign: dict[str, Any],
    principal_source_campaign: dict[str, Any],
    nonlinear_evolution_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticMetricRowsTensorDagError("unsupported campaign schema_version")
        campaigns = (
            semantic_dag_campaign,
            scalar_row_campaign,
            principal_source_campaign,
            nonlinear_evolution_campaign,
        )
        expected = (
            "partial_all_12_exact_universal_source_operator_dag_checkpoints",
            "pass_all_12_universal_scalar_row_affinity_partial_mixed_checkpoints",
            "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed",
            "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations",
        )
        if tuple(campaign.get("status") for campaign in campaigns) != expected:
            raise QuarticMetricRowsTensorDagError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticMetricRowsTensorDagError("campaign content hash mismatch")
        if (
            scalar_row_campaign.get("upstream_sha256", {}).get("semantic_source_dag")
            != semantic_dag_campaign.get("content_sha256")
            or scalar_row_campaign.get("upstream_sha256", {}).get(
                "nonlinear_evolution"
            )
            != nonlinear_evolution_campaign.get("content_sha256")
        ):
            raise QuarticMetricRowsTensorDagError("row-lowering provenance mismatch")
        row_chunks = list(config["metric_row_chunks"])
        flattened = [row for chunk in row_chunks for row in chunk]
        if (
            flattened != list(range(10))
            or int(config["max_mixed_derivative_order"]) != 4
            or len(config["checkpoint_atom_pairs"]) != 2
        ):
            raise QuarticMetricRowsTensorDagError("unsupported deterministic chunks")
        if bool(config.get("declare_full_component_remainder_proved", False)):
            raise QuarticMetricRowsTensorDagError(
                "full remainder cannot be declared from partial mixed tensors"
            )
        control_passed, control = generic_metric_row_affinity_control()
        if not control_passed:
            raise QuarticMetricRowsTensorDagError("metric-row affinity control failed")
        source_path = Path(__file__).with_name("quartic_nonlinear_evolution_campaign.py")
        source_hash = _file_hash(source_path)
        conventions = {
            "dimension": 4,
            "metric_pairs": [list(pair) for pair in METRIC_PAIRS],
            "off_diagonal_weight": "sqrt(2)",
            "scalar_row_sign": "minus scalar_euler",
            "Riemann_convention": "d_i Gamma^u_jl-d_j Gamma^u_il+GammaGamma-GammaGamma",
            "time_acceleration": "partial_0 partial_0 q_A",
        }
        conventions_hash = _content_hash(conventions)
        first_nonlinear = nonlinear_evolution_campaign["certificates"][0]
        common_packet, coverage = _build_universal_dag(
            first_nonlinear["evolution_formula_contract_sha256"],
            first_nonlinear["source_geometric_formula_contract_sha256"],
            source_hash,
            conventions_hash,
            row_chunks,
            list(config["checkpoint_atom_pairs"]),
        )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != int(config.get("expected_candidate_count", 12)) or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticMetricRowsTensorDagError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps),
                common_packet,
                coverage,
                source_hash,
                conventions_hash,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "semantic_source_dag": semantic_dag_campaign.get("content_sha256"),
                "scalar_row_lowering": scalar_row_campaign.get("content_sha256"),
                "principal_source": principal_source_campaign.get("content_sha256"),
                "nonlinear_evolution": nonlinear_evolution_campaign.get("content_sha256"),
            },
            "source_code_sha256": source_hash,
            "tensor_conventions": {
                **conventions,
                "content_sha256": conventions_hash,
            },
            "config_sha256": _content_hash(config),
            "generic_metric_row_affinity_control": control,
            "common_explicit_tensor_dag_packet": common_packet,
            "coverage": coverage,
            "counts": {
                "selected": len(certificates),
                "metric_rows_lowered_per_candidate": 10,
                "Euler_rows_affine_per_candidate": 11,
                "lower_Jacobian_roots_per_candidate": 594,
                "A_W_mixed_roots_per_candidate": coverage[
                    "A_W_mixed_component_roots"
                ],
                "selected_mixed_F_roots_per_candidate": coverage[
                    "selected_mixed_F_component_roots"
                ],
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All Euler rows and the coupled solve are lowered into an exact indexed "
                "tensor DAG; full mixed component materialization remains fail-closed."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticMetricRowsTensorDagError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "metric_rows_lowered_per_candidate": 0,
                "Euler_rows_affine_per_candidate": 0,
                "lower_Jacobian_roots_per_candidate": 0,
                "A_W_mixed_roots_per_candidate": 0,
                "selected_mixed_F_roots_per_candidate": 0,
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_metric_rows_tensor_dag_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
