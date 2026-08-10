from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-scalar-row-lowering-campaign-1.0"
PAIRS = tuple((left, right) for left in range(4) for right in range(left, 4))


class QuarticScalarRowLoweringError(ValueError):
    """Raised when the scalar Euler-row arithmetic lowering is invalid."""


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


class ArithmeticDag:
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

    def from_sympy(self, expression: sp.Expr) -> int:
        expression = sp.sympify(expression)
        if expression.is_Number:
            return self.node("exact_constant", value=str(expression))
        if expression.is_Symbol:
            return self.node("tensor_component_atom", label=str(expression))
        if expression.func is sp.Add:
            return self.node(
                "exact_add",
                arguments=[self.from_sympy(argument) for argument in expression.args],
            )
        if expression.func is sp.Mul:
            return self.node(
                "exact_multiply",
                arguments=[self.from_sympy(argument) for argument in expression.args],
            )
        if expression.func is sp.Pow:
            return self.node(
                "exact_power",
                base=self.from_sympy(expression.args[0]),
                exponent=self.from_sympy(expression.args[1]),
            )
        raise QuarticScalarRowLoweringError(
            f"unsupported arithmetic expression node: {expression.func}"
        )

    def packet(self) -> dict[str, Any]:
        body = {
            "schema_version": "sigma-explicit-arithmetic-tensor-dag-1.0",
            "node_count": len(self.nodes),
            "nodes": self.nodes,
        }
        return {**body, "content_sha256": _content_hash(body)}


def _symmetric_matrix(symbols: tuple[sp.Symbol, ...]) -> sp.Matrix:
    matrix = sp.zeros(4)
    for symbol, (left, right) in zip(symbols, PAIRS):
        matrix[left, right] = symbol
        matrix[right, left] = symbol
    return matrix


@cache
def _universal_scalar_row_data() -> dict[str, Any]:
    inverse_symbols = sp.symbols("ginv00 ginv01 ginv02 ginv03 ginv11 ginv12 ginv13 ginv22 ginv23 ginv33")
    inverse = _symmetric_matrix(inverse_symbols)
    metric_acceleration_symbols = sp.symbols("ag00 ag01 ag02 ag03 ag11 ag12 ag13 ag22 ag23 ag33")
    metric_acceleration = sp.zeros(4)
    for symbol, (left, right) in zip(metric_acceleration_symbols, PAIRS):
        value = symbol if left == right else symbol / sp.sqrt(2)
        metric_acceleration[left, right] = value
        metric_acceleration[right, left] = value
    scalar_acceleration = sp.Symbol("aphi", real=True)

    def metric_second(
        derivative: int, left: int, contracted: int, right: int
    ) -> sp.Expr:
        if derivative == 0 and left == 0:
            return metric_acceleration[contracted, right]
        return sp.Integer(0)

    def connection_first(
        derivative: int, upper: int, left: int, right: int
    ) -> sp.Expr:
        return sum(
            inverse[upper, contracted]
            * (
                metric_second(derivative, left, contracted, right)
                + metric_second(derivative, right, contracted, left)
                - metric_second(derivative, contracted, left, right)
            )
            / 2
            for contracted in range(4)
        )

    def riemann_acceleration(
        upper: int, lowered: int, left: int, right: int
    ) -> sp.Expr:
        return connection_first(left, upper, right, lowered) - connection_first(
            right, upper, left, lowered
        )

    ricci_acceleration = sp.zeros(4)
    for left in range(4):
        for right in range(4):
            ricci_acceleration[left, right] = sp.expand(
                sum(
                    riemann_acceleration(upper, left, upper, right)
                    for upper in range(4)
                )
            )
    scalar_curvature_acceleration = sp.expand(
        sum(
            inverse[left, right] * ricci_acceleration[left, right]
            for left in range(4)
            for right in range(4)
        )
    )
    einstein_upper_acceleration = sp.zeros(4)
    for mu in range(4):
        for nu in range(4):
            einstein_upper_acceleration[mu, nu] = sp.factor(
                sum(
                    inverse[mu, left]
                    * inverse[nu, right]
                    * ricci_acceleration[left, right]
                    for left in range(4)
                    for right in range(4)
                )
                - inverse[mu, nu] * scalar_curvature_acceleration / 2
            )

    connection_symbols = sp.symbols("Gamma0:40")
    connection = [
        _symmetric_matrix(connection_symbols[10 * upper : 10 * (upper + 1)])
        for upper in range(4)
    ]
    scalar_gradient = sp.Matrix(sp.symbols("pphi0:4", real=True))
    scalar_second_symbols = sp.symbols("sphi00 sphi01 sphi02 sphi03 sphi11 sphi12 sphi13 sphi22 sphi23 sphi33")
    scalar_second = _symmetric_matrix(scalar_second_symbols)
    scalar_second[0, 0] = 0
    hessian_bar = sp.zeros(4)
    for left in range(4):
        for right in range(4):
            hessian_bar[left, right] = scalar_second[left, right] - sum(
                connection[upper][left, right] * scalar_gradient[upper]
                for upper in range(4)
            )
    hessian = hessian_bar.copy()
    hessian[0, 0] += scalar_acceleration

    einstein_bar_symbols = sp.symbols("Ebar00 Ebar01 Ebar02 Ebar03 Ebar11 Ebar12 Ebar13 Ebar22 Ebar23 Ebar33")
    einstein_bar = _symmetric_matrix(einstein_bar_symbols)
    einstein_upper = einstein_bar + einstein_upper_acceleration
    alpha, c20 = sp.symbols("alpha c20", real=True)
    gradient_up = inverse * scalar_gradient
    x_scalar = -sum(
        scalar_gradient[index] * gradient_up[index] for index in range(4)
    ) / 2
    g2_x = 1 + 2 * c20 * x_scalar
    coefficient = sp.zeros(4)
    for mu in range(4):
        for nu in range(4):
            coefficient[mu, nu] = (
                g2_x * inverse[mu, nu]
                - 2 * c20 * gradient_up[mu] * gradient_up[nu]
                - 2 * alpha * einstein_upper[mu, nu]
            )
    scalar_row = -sum(
        coefficient[mu, nu] * hessian[mu, nu]
        for mu in range(4)
        for nu in range(4)
    )
    accelerations = (*metric_acceleration_symbols, scalar_acceleration)
    zero_accelerations = {symbol: 0 for symbol in accelerations}
    remainder_w = sp.expand(scalar_row.subs(zero_accelerations))
    time_row = tuple(
        sp.expand(sp.diff(scalar_row, acceleration).subs(zero_accelerations))
        for acceleration in accelerations
    )
    affine_residual = sp.expand(
        scalar_row
        - remainder_w
        - sum(time_row[index] * accelerations[index] for index in range(11))
    )
    polynomial = sp.Poly(sp.expand(scalar_row), *accelerations)
    return {
        "inverse_symbols": inverse_symbols,
        "connection_symbols": connection_symbols,
        "scalar_gradient": tuple(scalar_gradient),
        "scalar_second_symbols": scalar_second_symbols,
        "einstein_bar_symbols": einstein_bar_symbols,
        "metric_accelerations": metric_acceleration_symbols,
        "scalar_acceleration": scalar_acceleration,
        "alpha": alpha,
        "c20": c20,
        "einstein_upper_acceleration": einstein_upper_acceleration,
        "scalar_row": scalar_row,
        "remainder_W": remainder_w,
        "time_row_A": time_row,
        "affine_residual": affine_residual,
        "acceleration_total_degree": polynomial.total_degree(),
        "acceleration_monomial_count": len(polynomial.terms()),
    }


def generic_scalar_row_affinity_control() -> tuple[bool, dict[str, Any]]:
    data = _universal_scalar_row_data()
    accelerations = (*data["metric_accelerations"], data["scalar_acceleration"])
    corrupted = sp.expand(
        data["scalar_row"]
        + data["metric_accelerations"][0] * data["scalar_acceleration"]
    )
    corrupted_polynomial = sp.Poly(corrupted, *accelerations)
    witness_specialized = data["affine_residual"].subs(
        {
            symbol: value
            for symbol, value in zip(
                data["inverse_symbols"], (-1, 0, 0, 0, 1, 0, 0, 1, 0, 1)
            )
        }
    )
    universal_inverse_symbol_count = len(
        set(data["inverse_symbols"]) & data["scalar_row"].free_symbols
    )
    passed = bool(
        data["einstein_upper_acceleration"][0, 0] == 0
        and data["affine_residual"] == 0
        and data["acceleration_total_degree"] == 1
        and corrupted_polynomial.total_degree() == 2
        and universal_inverse_symbol_count == 10
        and witness_specialized == 0
    )
    return passed, {
        "control": "universal scalar Euler-row acceleration affinity",
        "row": 10,
        "identity": "E_phi=A_phi,B*a_B+W_phi",
        "universal_inverse_metric_symbol_count": universal_inverse_symbol_count,
        "G_upper_00_acceleration_part": str(
            data["einstein_upper_acceleration"][0, 0]
        ),
        "acceleration_total_degree": data["acceleration_total_degree"],
        "acceleration_monomial_count": data["acceleration_monomial_count"],
        "affine_residual": str(data["affine_residual"]),
        "negative_controls": {
            "corrupted_quadratic_acceleration": {
                "corruption": "add ag00*aphi",
                "total_degree": corrupted_polynomial.total_degree(),
                "rejected": corrupted_polynomial.total_degree() != 1,
            },
            "witness_specialization": {
                "corruption": "substitute Minkowski inverse metric before the proof",
                "specialized_residual": str(witness_specialized),
                "rejected": universal_inverse_symbol_count == 10,
            },
        },
        "passed": passed,
        "scope": (
            "The scalar row is universal in ten inverse-metric components, forty "
            "connection components, scalar coordinate jets, and acceleration-free "
            "Einstein components. Other Euler rows are not lowered here."
        ),
    }


def _multi_indices() -> list[tuple[int, int]]:
    return [
        (left_order, total - left_order)
        for total in range(2, 5)
        for left_order in range(1, total)
    ]


@cache
def _universal_mixed_packet(
    atom_pairs: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    data = _universal_scalar_row_data()
    label_to_symbol = {
        f"p{index}[10]": symbol
        for index, symbol in enumerate(data["scalar_gradient"])
    }
    dag = ArithmeticDag()
    scalar_root = dag.from_sympy(data["scalar_row"])
    w_expression = data["remainder_W"]
    w_root = dag.from_sympy(w_expression)
    a_expressions = list(data["time_row_A"])
    a_roots = [dag.from_sympy(expression) for expression in a_expressions]
    checkpoint_packets: list[dict[str, Any]] = []
    derivative_component_count = 0
    for labels in atom_pairs:
        if len(labels) != 2 or any(label not in label_to_symbol for label in labels):
            raise QuarticScalarRowLoweringError(
                "scalar-row checkpoint pairs must be canonical scalar-gradient atoms"
            )
        left, right = (label_to_symbol[label] for label in labels)
        components: list[dict[str, Any]] = []
        for left_order, right_order in _multi_indices():
            targets = [w_expression, *a_expressions]
            roots = []
            for expression in targets:
                derivative = sp.diff(
                    expression, left, left_order, right, right_order
                )
                roots.append(dag.from_sympy(derivative))
            components.append(
                {
                    "multi_index": [left_order, right_order],
                    "total_order": left_order + right_order,
                    "target_order": ["W_phi", *[f"A_phi,{index}" for index in range(11)]],
                    "roots": roots,
                }
            )
            derivative_component_count += len(roots)
        body = {
            "atom_pair": list(labels),
            "mixed_multi_indices": len(components),
            "components": components,
            "exact_component_roots": len(components) * 12,
        }
        checkpoint_packets.append({**body, "content_sha256": _content_hash(body)})
    packet = dag.packet()
    roots = {
        "Euler_scalar_row": scalar_root,
        "W_scalar_row": w_root,
        "A_scalar_time_row": a_roots,
    }
    return {
        "arithmetic_dag": packet,
        "root_packet": {**roots, "content_sha256": _content_hash(roots)},
        "mixed_derivative_checkpoints": checkpoint_packets,
        "exact_mixed_component_roots": derivative_component_count,
    }


def _candidate_packet(
    coefficients: dict[str, Any], atom_pairs: list[list[str]]
) -> dict[str, Any]:
    universal = _universal_mixed_packet(
        tuple((str(pair[0]), str(pair[1])) for pair in atom_pairs)
    )
    bindings = {
        "alpha": str(coefficients["a10"]),
        "c20": str(coefficients["c20"]),
    }
    return {
        "arithmetic_dag": {
            "node_count": universal["arithmetic_dag"]["node_count"],
            "content_sha256": universal["arithmetic_dag"]["content_sha256"],
        },
        "root_packet": {
            "content_sha256": universal["root_packet"]["content_sha256"]
        },
        "mixed_derivative_checkpoints": [
            {
                "atom_pair": item["atom_pair"],
                "mixed_multi_indices": item["mixed_multi_indices"],
                "exact_component_roots": item["exact_component_roots"],
                "content_sha256": item["content_sha256"],
            }
            for item in universal["mixed_derivative_checkpoints"]
        ],
        "exact_mixed_component_roots": universal["exact_mixed_component_roots"],
        "coefficient_bindings": {
            **bindings,
            "content_sha256": _content_hash(bindings),
        },
    }


def _certify_candidate(
    semantic: dict[str, Any],
    nonlinear: dict[str, Any],
    config: dict[str, Any],
    source_code_sha256: str,
    conventions_sha256: str,
) -> dict[str, Any]:
    candidate_id = str(semantic.get("candidate_id"))
    if (
        nonlinear.get("candidate_id") != candidate_id
        or nonlinear.get("coefficients") != semantic.get("coefficients")
    ):
        raise QuarticScalarRowLoweringError("candidate identity mismatch")
    formula_hash = nonlinear["evolution_formula_contract_sha256"]
    if semantic["provenance"]["evolution_formula_contract_sha256"] != formula_hash:
        raise QuarticScalarRowLoweringError("formula provenance mismatch")
    packet = _candidate_packet(
        semantic["coefficients"], list(config["checkpoint_atom_pairs"])
    )
    return {
        "schema_version": "sigma-quartic-scalar-row-lowering-certificate-1.0",
        "status": "pass_universal_scalar_row_affinity_partial_mixed_tensor_checkpoint",
        "candidate_id": candidate_id,
        "coefficients": semantic["coefficients"],
        "provenance": {
            "evolution_formula_contract_sha256": formula_hash,
            "source_geometric_formula_contract_sha256": nonlinear[
                "source_geometric_formula_contract_sha256"
            ],
            "nonlinear_source_code_sha256": source_code_sha256,
            "tensor_conventions_sha256": conventions_sha256,
            "coordinate_atom_basis_sha256": semantic["provenance"][
                "coordinate_atom_basis_sha256"
            ],
        },
        "lowered_row": 10,
        "lowered_formula": (
            "-sum_(mu,nu)[(G2_X*g^munu-2*c20*p^mu*p^nu-"
            "2*alpha*Einstein^munu)*H_munu]"
        ),
        **packet,
        "acceleration_affine_residual_proved_zero": True,
        "rows_lowered": 1,
        "rows_remaining": 10,
        "solved_source_F_component_derivatives_emitted": 0,
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "next_blocker": (
            "lower the ten coupled metric Euler rows, prove their affine residuals, "
            "assemble and invert the complete A, then propagate these exact A/W "
            "mixed tensors through F=-A^-1W"
        ),
        "scope": (
            "This proves one complete universal Euler row and exact A/W mixed "
            "components for two scalar-gradient atom pairs. It does not claim a "
            "component derivative of the coupled solved source F."
        ),
    }


def run_quartic_scalar_row_lowering_campaign(
    semantic_dag_campaign: dict[str, Any],
    nonlinear_evolution_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticScalarRowLoweringError("unsupported campaign schema_version")
        if (
            semantic_dag_campaign.get("status")
            != "partial_all_12_exact_universal_source_operator_dag_checkpoints"
            or nonlinear_evolution_campaign.get("status")
            != "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations"
        ):
            raise QuarticScalarRowLoweringError("campaign prerequisite status mismatch")
        if not _content_hash_matches(semantic_dag_campaign) or not _content_hash_matches(
            nonlinear_evolution_campaign
        ):
            raise QuarticScalarRowLoweringError("campaign content hash mismatch")
        if semantic_dag_campaign.get("upstream_sha256", {}).get(
            "nonlinear_evolution"
        ) != nonlinear_evolution_campaign.get("content_sha256"):
            raise QuarticScalarRowLoweringError("nonlinear provenance mismatch")
        if (
            int(config["lowered_row"]) != 10
            or int(config["max_mixed_derivative_order"]) != 4
            or len(config["checkpoint_atom_pairs"]) != 2
        ):
            raise QuarticScalarRowLoweringError("unsupported row-lowering checkpoint")
        if bool(config.get("declare_solved_source_remainder_proved", False)):
            raise QuarticScalarRowLoweringError(
                "solved-source remainder cannot be declared from one Euler row"
            )
        control_passed, control = generic_scalar_row_affinity_control()
        if not control_passed:
            raise QuarticScalarRowLoweringError("scalar-row affinity control failed")
        source_path = Path(__file__).with_name("quartic_nonlinear_evolution_campaign.py")
        source_code_sha256 = _file_hash(source_path)
        conventions = {
            "dimension": 4,
            "symmetric_pairs": [list(pair) for pair in PAIRS],
            "off_diagonal_field_weight": "sqrt(2)",
            "riemann": "R^u_lij=d_i Gamma^u_jl-d_j Gamma^u_il+GammaGamma-GammaGamma",
            "einstein_upper": "g^ma g^nb R_ab-g^mn g^ab R_ab/2",
            "scalar_row_sign": "minus scalar_euler",
            "acceleration_atoms": "partial_0 partial_0 q_A",
        }
        conventions_sha256 = _content_hash(conventions)
        maps = (
            _candidate_records(semantic_dag_campaign),
            _candidate_records(nonlinear_evolution_campaign),
        )
        candidate_ids = set(maps[0])
        if len(candidate_ids) != int(config.get("expected_candidate_count", 12)) or set(
            maps[1]
        ) != candidate_ids:
            raise QuarticScalarRowLoweringError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id],
                maps[1][candidate_id],
                config,
                source_code_sha256,
                conventions_sha256,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        universal_packet = _universal_mixed_packet(
            tuple(
                (str(pair[0]), str(pair[1]))
                for pair in config["checkpoint_atom_pairs"]
            )
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_universal_scalar_row_affinity_partial_mixed_checkpoints",
            "errors": [],
            "upstream_sha256": {
                "semantic_source_dag": semantic_dag_campaign.get("content_sha256"),
                "nonlinear_evolution": nonlinear_evolution_campaign.get("content_sha256"),
            },
            "source_code_sha256": source_code_sha256,
            "tensor_conventions": {
                **conventions,
                "content_sha256": conventions_sha256,
            },
            "config_sha256": _content_hash(config),
            "generic_scalar_row_affinity_control": control,
            "universal_arithmetic_tensor_packet": universal_packet,
            "counts": {
                "selected": len(certificates),
                "Euler_rows_lowered_per_candidate": 1,
                "Euler_rows_remaining_per_candidate": 10,
                "atom_pairs_per_candidate": 2,
                "mixed_A_W_component_roots_per_candidate": 144,
                "solved_source_component_derivatives": 0,
                "remainder_bounds_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The scalar Euler row is explicitly lowered and universally affine in "
                "all eleven accelerations; exact mixed A/W tensors are checkpointed "
                "without promoting them to a coupled solved-source claim."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticScalarRowLoweringError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "Euler_rows_lowered_per_candidate": 0,
                "Euler_rows_remaining_per_candidate": 0,
                "atom_pairs_per_candidate": 0,
                "mixed_A_W_component_roots_per_candidate": 0,
                "solved_source_component_derivatives": 0,
                "remainder_bounds_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_scalar_row_lowering_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
