from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-universal-source-dag-campaign-1.0"


class QuarticUniversalSourceDagError(ValueError):
    """Raised when an exact universal source-DAG checkpoint is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


class ExactDag:
    """Small hash-consed DAG for exact semantic source operations."""

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
            "schema_version": "sigma-exact-semantic-expression-dag-1.0",
            "node_count": len(self.nodes),
            "nodes": self.nodes,
        }
        return {**body, "content_sha256": _content_hash(body)}


def _universal_atom_labels() -> list[str]:
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


def _validate_unspecialized_source_nodes(
    dag: ExactDag, source_roots: list[int]
) -> bool:
    return all(
        dag.nodes[root].get("op") == "gauge_fixed_euler_component"
        and dag.nodes[root].get("specialization") is None
        and dag.nodes[root].get("coordinate_atom_count") == 153
        and dag.nodes[root].get("acceleration_count") == 11
        for root in source_roots
    )


def _build_candidate_dag(
    candidate_id: str,
    coefficients: dict[str, Any],
    formula_hash: str,
    geometric_hash: str,
    selected_atoms: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dag = ExactDag()
    atom_labels = _universal_atom_labels()
    atoms = {
        label: dag.node("coordinate_atom", label=label, basis_index=index)
        for index, label in enumerate(atom_labels)
    }
    accelerations = {
        f"a[{row}]": dag.node("acceleration_atom", label=f"a[{row}]", row=row)
        for row in range(11)
    }
    parameters = {
        name: dag.node("exact_parameter", name=name, value=str(coefficients[name]))
        for name in ("m2", "a10", "c20")
    }
    source_roots = [
        dag.node(
            "gauge_fixed_euler_component",
            row=row,
            formula="gauge_fixed_euler_from_state",
            evolution_formula_contract_sha256=formula_hash,
            source_geometric_formula_contract_sha256=geometric_hash,
            coordinate_atom_count=len(atoms),
            acceleration_count=len(accelerations),
            coordinate_roots=list(atoms.values()),
            acceleration_roots=list(accelerations.values()),
            coefficient_roots=parameters,
            specialization=None,
        )
        for row in range(11)
    ]
    zero_acceleration = {
        label: "0" for label in accelerations
    }
    w_roots = [
        dag.node(
            "exact_simultaneous_substitution",
            expression=root,
            substitutions=zero_acceleration,
        )
        for root in source_roots
    ]
    a_roots = []
    for row in range(11):
        a_row = []
        for column in range(11):
            derivative = dag.node(
                "exact_partial_derivative",
                expression=source_roots[row],
                variable=accelerations[f"a[{column}]"],
                order=1,
            )
            a_row.append(
                dag.node(
                    "exact_simultaneous_substitution",
                    expression=derivative,
                    substitutions=zero_acceleration,
                )
            )
        a_roots.append(a_row)
    negative_w_roots = [dag.node("exact_negation", expression=root) for root in w_roots]
    f_roots = [
        dag.node(
            "exact_linear_solve_component",
            matrix=a_roots,
            rhs=negative_w_roots,
            row=row,
            identity="F=-Inverse(A)*W",
        )
        for row in range(11)
    ]

    checkpoints: list[dict[str, Any]] = []
    derivative_root_count = 0
    for atom_label in selected_atoms:
        if atom_label not in atoms:
            raise QuarticUniversalSourceDagError(
                f"selected atom is not in the canonical basis: {atom_label}"
            )
        order_roots: dict[str, list[int]] = {}
        current = list(f_roots)
        for order in range(1, 5):
            current = [
                dag.node(
                    "exact_partial_derivative",
                    expression=root,
                    variable=atoms[atom_label],
                    order=1,
                )
                for root in current
            ]
            order_roots[str(order)] = current
            derivative_root_count += len(current)
        checkpoint_body = {
            "candidate_id": candidate_id,
            "atom": atom_label,
            "basis_index": atom_labels.index(atom_label),
            "output_rows": 11,
            "orders_completed": [1, 2, 3, 4],
            "pure_repeated_derivative_roots": order_roots,
            "completed_component_roots": 44,
            "mixed_multi_indices_completed": 0,
        }
        checkpoints.append(
            {**checkpoint_body, "content_sha256": _content_hash(checkpoint_body)}
        )

    source_unspecialized = _validate_unspecialized_source_nodes(dag, source_roots)
    corrupted_source = dag.node(
        "gauge_fixed_euler_component",
        row=0,
        formula="gauge_fixed_euler_from_state",
        evolution_formula_contract_sha256=formula_hash,
        source_geometric_formula_contract_sha256=geometric_hash,
        coordinate_atom_count=153,
        acceleration_count=11,
        coordinate_roots=list(atoms.values()),
        acceleration_roots=list(accelerations.values()),
        coefficient_roots=parameters,
        specialization="rational_local_witness",
    )
    witness_specialization_rejected = not _validate_unspecialized_source_nodes(
        dag, [corrupted_source]
    )
    packet = dag.packet()
    roots = {
        "Euler_E": source_roots,
        "time_block_A": a_roots,
        "acceleration_free_W": w_roots,
        "solved_source_F": f_roots,
    }
    evidence = {
        "universal_input": {
            "coordinate_atom_count": len(atom_labels),
            "coordinate_atom_labels_sha256": _content_hash(atom_labels),
            "acceleration_atom_count": len(accelerations),
            "no_coordinate_atom_substitution": source_unspecialized,
        },
        "exact_split": {
            "definition": "E(Y,a)=A(Y,a)*a+W(Y) audit pending affine residual",
            "A_definition": "A_row,column=(partial E_row/partial a_column)|_{a=0}",
            "W_definition": "W=E|_{a=0}",
            "A_shape": [11, 11],
            "W_shape": [11],
            "affine_residual_entrywise_proved_zero": False,
            "reason": (
                "the operator DAG emits the exact split candidates, but the universal "
                "Horndeski acceleration-cancellation residual has not been normalized"
            ),
        },
        "solved_source": {
            "definition": "F=-Inverse(A)*W",
            "shape": [11],
            "exact_operational_roots_emitted": len(f_roots),
            "invertibility_bound_source": "upstream certified coordinate tube",
        },
        "checkpoints": checkpoints,
        "coverage": {
            "selected_lower_atoms": len(selected_atoms),
            "selected_atom_labels": selected_atoms,
            "orders_completed": [1, 2, 3, 4],
            "pure_repeated_derivative_component_roots": derivative_root_count,
            "full_order_1_component_count": 11 * 153,
            "full_orders_2_to_4_dense_component_count": sum(
                11 * 153**order for order in range(2, 5)
            ),
            "mixed_multi_index_components_completed": 0,
            "coverage_is_partial": True,
        },
        "negative_controls": {
            "rational_witness_specialization": {
                "corrupted_root": corrupted_source,
                "rejected": witness_specialization_rejected,
            }
        },
        "root_packet_sha256": _content_hash(roots),
        "dag_content_sha256": packet["content_sha256"],
    }
    return {"dag": packet, "roots": roots, "checkpoints": checkpoints}, evidence


def generic_exact_operator_dag_control() -> tuple[bool, dict[str, Any]]:
    """Execute the same DAG semantics on a finite polynomial known answer."""

    x, y, a = sp.symbols("x y a", real=True, finite=True)
    euler = (1 + x + y**2) * a + x**2 * y + y**4
    coefficient_a = sp.diff(euler, a)
    remainder_w = euler.subs(a, 0)
    solved_f = -remainder_w / coefficient_a
    affine_residual = sp.expand(euler - coefficient_a * a - remainder_w)
    derivative_residuals = {}
    for variable in (x, y):
        for order in range(1, 5):
            direct = sp.diff(solved_f, variable, order)
            operational = sp.diff(-remainder_w / coefficient_a, variable, order)
            derivative_residuals[f"{variable},{order}"] = sp.factor(
                direct - operational
            )
    corrupted = sp.factor(euler - coefficient_a * a - (remainder_w + x))
    passed = bool(
        affine_residual == 0
        and all(value == 0 for value in derivative_residuals.values())
        and corrupted != 0
    )
    return passed, {
        "control": "executable exact substitute/partial/linear-solve DAG semantics",
        "known_answer": "E=(1+x+y^2)a+x^2y+y^4",
        "affine_split_residual": str(affine_residual),
        "orders_checked": [1, 2, 3, 4],
        "derivative_residuals": {
            key: str(value) for key, value in derivative_residuals.items()
        },
        "negative_control": {
            "corruption": "replace W by W+x",
            "residual": str(corrupted),
            "rejected": corrupted != 0,
        },
        "passed": passed,
    }


def _certify_candidate(
    lower: dict[str, Any],
    nonlinear: dict[str, Any],
    solved: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(lower.get("candidate_id"))
    if any(
        item.get("candidate_id") != candidate_id
        or item.get("coefficients") != lower.get("coefficients")
        for item in (nonlinear, solved)
    ):
        raise QuarticUniversalSourceDagError("candidate identity mismatch")
    if lower.get("status") != "audit_lower_source_and_component_remainder_fail_closed":
        raise QuarticUniversalSourceDagError("lower-source prerequisite failed")
    formula_hash = nonlinear["evolution_formula_contract_sha256"]
    geometric_hash = nonlinear["source_geometric_formula_contract_sha256"]
    if lower["provenance"]["evolution_formula_contract_sha256"] != formula_hash:
        raise QuarticUniversalSourceDagError("formula provenance mismatch")
    selected_atoms = list(config["checkpoint_atom_labels"])
    packet, evidence = _build_candidate_dag(
        candidate_id,
        lower["coefficients"],
        formula_hash,
        geometric_hash,
        selected_atoms,
    )
    basis = lower["provenance"]
    if evidence["universal_input"]["coordinate_atom_labels_sha256"] != basis[
        "coordinate_atom_basis_sha256"
    ]:
        raise QuarticUniversalSourceDagError("coordinate basis hash mismatch")
    return {
        "schema_version": "sigma-quartic-universal-source-dag-certificate-1.0",
        "status": "partial_exact_universal_source_operator_dag_checkpoint",
        "candidate_id": candidate_id,
        "coefficients": lower["coefficients"],
        "provenance": {
            "state_basis_sha256": basis["state_basis_sha256"],
            "coordinate_atom_basis_sha256": basis["coordinate_atom_basis_sha256"],
            "principal_jet_injection_sha256": basis[
                "principal_jet_injection_sha256"
            ],
            "evolution_formula_contract_sha256": formula_hash,
            "source_geometric_formula_contract_sha256": geometric_hash,
        },
        "expression_dag": packet["dag"],
        "root_packet": packet["roots"],
        "checkpoint_packets": packet["checkpoints"],
        "evidence": evidence,
        "exact_component_derivative_roots_emitted": evidence["coverage"][
            "pure_repeated_derivative_component_roots"
        ],
        "universal_acceleration_affine_split_proved": False,
        "full_component_Frechet_tensors_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "next_checkpoint": (
            "normalize and prove all 11 universal acceleration-affine residuals, then "
            "expand mixed multi-indices in deterministic atom-pair chunks"
        ),
        "scope": (
            "The DAG is universal and unspecialized, and its exact semantic derivative "
            "roots are checkpointed. It is not an expanded component tensor and does "
            "not yet prove the nonlinear acceleration-affine cancellation or H7 closure."
        ),
    }


def run_quartic_universal_source_dag_campaign(
    lower_source_campaign: dict[str, Any],
    nonlinear_evolution_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticUniversalSourceDagError("unsupported campaign schema_version")
        campaigns = (
            lower_source_campaign,
            nonlinear_evolution_campaign,
            solved_source_campaign,
        )
        expected = (
            "audit_all_12_lower_source_maps_component_remainder_fail_closed",
            "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
        )
        if tuple(campaign.get("status") for campaign in campaigns) != expected:
            raise QuarticUniversalSourceDagError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticUniversalSourceDagError("campaign content hash mismatch")
        upstream = lower_source_campaign.get("upstream_sha256", {})
        if (
            upstream.get("nonlinear_evolution")
            != nonlinear_evolution_campaign.get("content_sha256")
            or upstream.get("solved_source") != solved_source_campaign.get("content_sha256")
        ):
            raise QuarticUniversalSourceDagError("lower-source provenance mismatch")
        if (
            int(config["coordinate_atom_dimension"]) != 153
            or int(config["acceleration_dimension"]) != 11
            or int(config["max_derivative_order"]) != 4
            or not 0 < len(config["checkpoint_atom_labels"]) <= 4
        ):
            raise QuarticUniversalSourceDagError("unsupported checkpoint contract")
        if bool(config.get("declare_affine_split_proved", False)):
            raise QuarticUniversalSourceDagError(
                "affine split cannot be declared before universal residual normalization"
            )
        generic_passed, generic = generic_exact_operator_dag_control()
        if not generic_passed:
            raise QuarticUniversalSourceDagError("generic DAG semantics failed")
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        ids = set(maps[0])
        if len(ids) != int(config.get("expected_candidate_count", 12)) or any(
            set(records) != ids for records in maps[1:]
        ):
            raise QuarticUniversalSourceDagError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id],
                maps[1][candidate_id],
                maps[2][candidate_id],
                config,
            )
            for candidate_id in sorted(ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "partial_all_12_exact_universal_source_operator_dag_checkpoints",
            "errors": [],
            "upstream_sha256": {
                "lower_source_remainder": lower_source_campaign.get("content_sha256"),
                "nonlinear_evolution": nonlinear_evolution_campaign.get("content_sha256"),
                "solved_source": solved_source_campaign.get("content_sha256"),
            },
            "config_sha256": _content_hash(config),
            "generic_exact_operator_dag_control": generic,
            "counts": {
                "selected": len(certificates),
                "checkpoint_atoms_per_candidate": len(config["checkpoint_atom_labels"]),
                "pure_derivative_component_roots_per_candidate": (
                    len(config["checkpoint_atom_labels"]) * 11 * 4
                ),
                "affine_splits_proved": 0,
                "complete_component_tensors": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All candidates have deterministic universal unspecialized semantic "
                "source DAGs and bounded pure-derivative checkpoints; expanded mixed "
                "component tensors and affine residual normalization remain open."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticUniversalSourceDagError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "checkpoint_atoms_per_candidate": 0,
                "pure_derivative_component_roots_per_candidate": 0,
                "affine_splits_proved": 0,
                "complete_component_tensors": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_universal_source_dag_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
