from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

from .composite_covariant_lift_campaign import compile_composite_aether_action
from .formal_backend import load_field_contract
from .production_covariant_provenance import (
    map_candidate_to_covariant_action,
    production_blocked_candidates,
)
from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-composite-q-degenerate-formal-campaign-1.0"
ADAPTER_SCHEMA = "sigma-composite-q-degenerate-formal-adapter-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _sha(body)}


def _literal(node: ast.AST) -> str | int | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, bool)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    raise ValueError("typed expression contains a nonliteral constructor argument")


def _parse_typed_expression(raw: str) -> sp.Expr:
    """Parse the small SymPy srepr emitted by the composite lift without eval."""

    root = ast.parse(raw, mode="eval")

    def build(node: ast.AST) -> sp.Expr:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise TypeError("typed expression contains an unsupported AST node")
        name = node.func.id
        if name == "Symbol":
            if len(node.args) != 1 or _literal(node.args[0]) not in {"Q_a_u", "X_a_u"}:
                raise ValueError("typed expression contains an undeclared symbol")
            if len(node.keywords) != 1 or node.keywords[0].arg != "real":
                raise ValueError("typed symbol is missing its real-domain declaration")
            if _literal(node.keywords[0].value) is not True:
                raise ValueError("typed symbol real-domain declaration is not true")
            return sp.Symbol(str(_literal(node.args[0])), real=True)
        if node.keywords:
            raise ValueError("typed expression constructor has unexpected keywords")
        if name == "Integer" and len(node.args) == 1:
            return sp.Integer(_literal(node.args[0]))
        if name == "Rational" and len(node.args) == 2:
            return sp.Rational(_literal(node.args[0]), _literal(node.args[1]))
        constructors = {"Add": sp.Add, "Mul": sp.Mul, "Pow": sp.Pow}
        if name not in constructors or (name == "Pow" and len(node.args) != 2):
            raise ValueError("typed expression contains an unsupported constructor")
        return constructors[name](*(build(argument) for argument in node.args))

    expression = build(root.body)
    if {str(symbol) for symbol in expression.free_symbols} - {"Q_a_u", "X_a_u"}:
        raise ValueError("typed expression escaped the declared invariant vocabulary")
    return expression


def _validate_lift(lift: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if lift.get("decision") != "mapped" or lift.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("formal adapter requires an exact fail-closed composite lift")
    action = lift["action_ir"]
    action_body = {key: value for key, value in action.items() if key != "content_sha256"}
    if action.get("content_sha256") != _sha(action_body):
        raise ValueError("composite action content hash mismatch")
    certificate = lift["static_shape_certificate"]
    certificate_body = {
        key: value for key, value in certificate.items() if key != "content_sha256"
    }
    if certificate.get("content_sha256") != _sha(certificate_body):
        raise ValueError("static shape certificate content hash mismatch")
    provenance = lift["covariant_action_provenance"]
    provenance_body = {
        key: value
        for key, value in provenance.items()
        if key != "provenance_binding_sha256"
    }
    if provenance.get("provenance_binding_sha256") != _sha(provenance_body):
        raise ValueError("covariant action provenance binding mismatch")
    if provenance.get("input_action_sha256") != action.get("content_sha256"):
        raise ValueError("provenance is not bound to the supplied composite action")
    if provenance.get("static_shape_certificate_sha256") != certificate.get(
        "content_sha256"
    ):
        raise ValueError("provenance is not bound to the supplied shape certificate")
    if action.get("static_shape_certificate_content_sha256") != certificate.get(
        "content_sha256"
    ):
        raise ValueError("action is not bound to the supplied shape certificate")
    if action.get("candidate_id") != lift.get("candidate_id") or action.get(
        "ordinal"
    ) != lift.get("ordinal"):
        raise ValueError("action candidate identity mismatch")
    if (
        action.get("universal_matter_coupling_preserved") is not True
        or action.get("observational_data_opened") is not False
    ):
        raise ValueError("action violates the sealed formal policy")
    terms = [
        term
        for term in action.get("terms", [])
        if term.get("role") == "derived_dimensionless_aether_scalar"
    ]
    if len(terms) != 1:
        raise ValueError("composite action does not contain exactly one typed scalar")
    return action, terms[0]


def evaluate_zero_local_acceleration_family(lift: dict[str, Any]) -> dict[str, Any]:
    """Classify F(X,Q) actions whose aligned local acceleration coefficient vanishes.

    X and Q are each second order in a spatial Aether perturbation about aligned
    Minkowski.  The exact quadratic action therefore depends only on F_X(0,0)
    and F_Q(0,0).  F_X(0,0)=0 makes the k=0 velocity Hessian singular, which is
    a decisive necessary-condition failure rather than a full covariant proof.
    """

    action, term = _validate_lift(lift)
    expression = _parse_typed_expression(str(term["typed_expression_srepr"]))
    symbols = {str(symbol): symbol for symbol in expression.free_symbols}
    x = symbols.get("X_a_u", sp.Symbol("X_a_u", real=True))
    q = symbols.get("Q_a_u", sp.Symbol("Q_a_u", real=True))
    origin = {x: 0, q: 0}
    value_at_origin = sp.simplify(expression.subs(origin))
    f_x = sp.simplify(sp.diff(expression, x).subs(origin))
    f_q = sp.simplify(sp.diff(expression, q).subs(origin))
    if any(value.has(sp.nan, sp.zoo, sp.oo, -sp.oo) for value in (value_at_origin, f_x, f_q)):
        return {
            "schema_version": ADAPTER_SCHEMA,
            "decision": "blocked",
            "candidate_id": lift["candidate_id"],
            "ordinal": lift["ordinal"],
            "blocker": "nonanalytic_composite_action_at_aligned_zero_invariant_background",
            "input_action_sha256": action["content_sha256"],
            "provenance_binding_sha256": lift["covariant_action_provenance"][
                "provenance_binding_sha256"
            ],
            "data_eligibility": dict(ELIGIBILITY),
        }
    if value_at_origin != 0:
        return {
            "schema_version": ADAPTER_SCHEMA,
            "decision": "blocked",
            "candidate_id": lift["candidate_id"],
            "ordinal": lift["ordinal"],
            "blocker": "nonzero_background_composite_density_outside_bounded_family",
            "input_action_sha256": action["content_sha256"],
            "provenance_binding_sha256": lift["covariant_action_provenance"][
                "provenance_binding_sha256"
            ],
            "data_eligibility": dict(ELIGIBILITY),
        }
    if f_x != 0:
        return {
            "schema_version": ADAPTER_SCHEMA,
            "decision": "blocked",
            "candidate_id": lift["candidate_id"],
            "ordinal": lift["ordinal"],
            "blocker": "nonzero_local_acceleration_coefficient_outside_bounded_family",
            "linearized_coefficients": {"F_X_at_origin": str(f_x), "F_Q_at_origin": str(f_q)},
            "input_action_sha256": action["content_sha256"],
            "provenance_binding_sha256": lift["covariant_action_provenance"][
                "provenance_binding_sha256"
            ],
            "data_eligibility": dict(ELIGIBILITY),
        }

    eta, x2, q2 = sp.symbols("eta X2 Q2", real=True)
    quadratic = sp.simplify(
        sp.diff(expression.subs({x: eta**2 * x2, q: eta**2 * q2}), eta, 2).subs(
            eta, 0
        )
        / 2
    )
    quadratic_expected = sp.simplify(f_x * x2 + f_q * q2)
    quadratic_residual = sp.simplify(quadratic - quadratic_expected)
    if quadratic_residual != 0:
        raise ValueError("exact composite weak-field chain-rule certificate failed")
    wave_number, length_scale = sp.symbols("k L_sigma", nonnegative=True)
    normalized_hessian = sp.factor(2 * (f_x + f_q * length_scale**2 * wave_number**2))
    homogeneous_hessian = sp.simplify(normalized_hessian.subs(wave_number, 0))
    if homogeneous_hessian != 0:
        raise ValueError("bounded-family homogeneous Hessian unexpectedly nonzero")
    has_quadratic_q = f_q != 0
    reason = (
        "aligned_velocity_hessian_rank_jumps_between_k_zero_and_nonzero"
        if has_quadratic_q
        else "composite_aether_sector_has_no_quadratic_vector_evolution"
    )
    body = {
        "schema_version": ADAPTER_SCHEMA,
        "decision": "reject",
        "candidate_id": lift["candidate_id"],
        "ordinal": lift["ordinal"],
        "reason": reason,
        "input_action_sha256": action["content_sha256"],
        "provenance_binding_sha256": lift["covariant_action_provenance"][
            "provenance_binding_sha256"
        ],
        "static_shape_certificate_sha256": lift["static_shape_certificate"][
            "content_sha256"
        ],
        "linearized_coefficients": {
            "F_at_origin": str(value_at_origin),
            "F_X_at_origin": str(f_x),
            "F_Q_at_origin": str(f_q),
        },
        "covariant_variation_preflight": {
            "status": "bounded_chain_rule_derived",
            "fixed_metric_first_variation": "delta F=F_X delta X_a_u+F_Q delta Q_a_u",
            "metric_variation_status": "unresolved_not_needed_for_decisive_negative_gate",
            "claim_scope": "exact chain rule for the hash-bound typed covariant scalar",
        },
        "weak_field_certificate": {
            "background": "Minkowski, u^mu=(1,0,0,0), X_a_u=Q_a_u=0",
            "invariant_scaling": {"X_a_u": "eta^2*X2", "Q_a_u": "eta^2*Q2"},
            "quadratic_expression": str(quadratic),
            "chain_rule_residual": str(quadratic_residual),
        },
        "adm_kinetic_preflight": {
            "status": "reject",
            "aligned_normalized_velocity_hessian": str(normalized_hessian),
            "homogeneous_k_zero_hessian": str(homogeneous_hessian),
            "spatial_vector_rank_at_k_zero": 0,
            "spatial_vector_rank_at_k_nonzero": 3 if has_quadratic_q else 0,
            "constant_rank": False,
        },
        "principal_preflight": {
            "status": "reject",
            "normalized_fourier_factor": (
                "omega^2*F_Q(0,0)*L_sigma^2*k^2" if has_quadratic_q else "0"
            ),
            "failure": (
                "no uniform time-block inverse across k=0"
                if has_quadratic_q
                else "zero linearized Aether principal symbol"
            ),
        },
        "formal_pass": False,
        "solar_bundle_outcome": {
            "decision": "blocked",
            "blocker": "upstream_formal_rejected",
            "bundle_generated": False,
        },
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "A decisive necessary kinetic/principal condition fails on an allowed background. "
            "This rejection does not claim a completed generic covariant variation."
        ),
    }
    return _content(body)


def run_composite_q_degenerate_formal_campaign(
    config: dict[str, Any],
    database_path: str | Path,
    generator_path: str | Path,
    grammar_path: str | Path,
    field_contract_path: str | Path,
    static_dictionary_path: str | Path,
) -> dict[str, Any]:
    paths = {
        "database": Path(database_path).resolve(),
        "generator": Path(generator_path).resolve(),
        "grammar": Path(grammar_path).resolve(),
        "field_contract": Path(field_contract_path).resolve(),
        "static_dictionary": Path(static_dictionary_path).resolve(),
    }
    for name, path in paths.items():
        if _file_sha(path) != config[f"{name}_file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
    generator = json.loads(paths["generator"].read_text(encoding="utf-8"))
    grammar = json.loads(paths["grammar"].read_text(encoding="utf-8"))
    field_contract = load_field_contract(paths["field_contract"])
    static_dictionary = json.loads(paths["static_dictionary"].read_text(encoding="utf-8"))
    if static_dictionary.get("content_sha256") != config[
        "static_dictionary_content_sha256"
    ]:
        raise ValueError("static dictionary content hash mismatch")
    candidates = production_blocked_candidates(paths["database"])
    if len(candidates) != int(config["expected_production_candidate_count"]):
        raise ValueError("production candidate count mismatch")
    outcomes: list[dict[str, Any]] = []
    for candidate in candidates:
        existing = map_candidate_to_covariant_action(
            candidate,
            generator,
            grammar,
            field_contract,
            source_sha256=config["source_summary_file_sha256"],
        )
        if existing["decision"] == "mapped":
            continue
        lift = compile_composite_aether_action(
            candidate,
            field_contract,
            field_contract_file_sha256=config["field_contract_file_sha256"],
            static_dictionary_file_sha256=config["static_dictionary_file_sha256"],
            static_dictionary_content_sha256=config[
                "static_dictionary_content_sha256"
            ],
            source_sha256=config["source_summary_file_sha256"],
        )
        outcomes.append(evaluate_zero_local_acceleration_family(lift))
    if len(outcomes) != int(config["expected_composite_candidate_count"]):
        raise ValueError("composite candidate count mismatch")
    decisions = Counter(outcome["decision"] for outcome in outcomes)
    reasons = Counter(
        outcome.get("reason", outcome.get("blocker", "unknown")) for outcome in outcomes
    )
    rejected = [outcome for outcome in outcomes if outcome["decision"] == "reject"]
    identities = [
        {
            "candidate_id": outcome["candidate_id"],
            "ordinal": outcome["ordinal"],
            "decision": outcome["decision"],
            "reason_or_blocker": outcome.get("reason", outcome.get("blocker")),
            "input_action_sha256": outcome["input_action_sha256"],
            "provenance_binding_sha256": outcome["provenance_binding_sha256"],
            "formal_evidence_sha256": outcome.get("content_sha256"),
        }
        for outcome in outcomes
    ]
    body = {
        "schema_version": SCHEMA_VERSION,
        "input_composite_candidate_count": len(outcomes),
        "decision_counts": dict(sorted(decisions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "bounded_family_reject_count": len(rejected),
        "remaining_formal_blocked_count": decisions.get("blocked", 0),
        "formal_pass_count": decisions.get("pass", 0),
        "solar_bundle_generated_count": 0,
        "candidate_evidence_root_sha256": _sha(identities),
        "sample_rejections": [
            rejected[0],
            next(item for item in rejected if item["linearized_coefficients"]["F_Q_at_origin"] == "0"),
            rejected[-1],
        ],
        "source_database_file_sha256": _file_sha(paths["database"]),
        "source_summary_file_sha256": config["source_summary_file_sha256"],
        "policy": {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "paid_llm_calls": False,
            "static_shape_treated_as_formal_proof": False,
        },
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "This bounded campaign rejects only the exact hash-bound F(X,Q) actions whose "
            "local acceleration coefficient vanishes at the aligned zero-invariant background. "
            "All other composite actions remain blocked for a later adapter."
        ),
    }
    return _content(body)
