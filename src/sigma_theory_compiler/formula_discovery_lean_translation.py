"""Fail-closed translation of supported Formula Discovery PASS results to Lean source."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy as sp

from .formula_discovery_job import (
    _parse_expression,
    _parse_problem,
    validate_formula_discovery_result,
)
from .math_lean_adapter import (
    AUDIT_PROTOCOL,
    LeanAdapterConfig,
    build_allowed_premise_manifest,
    validate_allowed_premise_manifest,
)
from .sigma_core import canonical_sha256

TRANSLATION_SCHEMA = "sigma-formula-discovery-lean-translation-1.0"
MAX_POLYNOMIAL_DEGREE = 8
MAX_SOURCE_BYTES = 64 * 1024
FORBIDDEN_PREMISES = ("Classical.choice", "False.elim")
FORBIDDEN_PREFIXES = ("KnownAnswer", "Unsafe")

_TRANSLATION_KEYS = {
    "candidate_content_sha256",
    "claims",
    "content_sha256",
    "counts",
    "job_id",
    "premise_manifest",
    "problem_sha256",
    "proof_certificate_content_sha256",
    "result_content_sha256",
    "schema_version",
    "scope",
    "source",
    "source_sha256",
    "target",
    "translation_kind",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class FormulaDiscoveryLeanTranslationError(ValueError):
    """Raised when a PASS result cannot be translated without weakening its semantics."""


@dataclass(frozen=True, slots=True)
class _Polynomial:
    coefficients: tuple[int, ...]

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1


def _canonical_source_sha(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _lean_suffix(job_id: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", job_id) if part]
    if not parts:
        raise FormulaDiscoveryLeanTranslationError("job id has no Lean-safe name material")
    suffix = "".join(part[0].upper() + part[1:] for part in parts)
    if not suffix[0].isalpha() or not suffix.isalnum() or len(suffix) > 96:
        raise FormulaDiscoveryLeanTranslationError("job id cannot form a bounded Lean name")
    return suffix


def _integer_polynomial(expression: sp.Expr, variable: sp.Symbol) -> _Polynomial:
    try:
        polynomial = sp.Poly(sp.expand(expression), variable, domain=sp.ZZ)
    except (sp.PolynomialError, sp.CoercionFailed) as error:
        raise FormulaDiscoveryLeanTranslationError(
            "expression is not an integer-coefficient polynomial"
        ) from error
    if polynomial.total_degree() > MAX_POLYNOMIAL_DEGREE:
        raise FormulaDiscoveryLeanTranslationError("polynomial degree exceeds translation cap")
    coefficients = [0] * (max(0, int(polynomial.degree())) + 1)
    for (power,), coefficient in polynomial.terms():
        coefficients[power] = int(coefficient)
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    if any(abs(value).bit_length() > 256 for value in coefficients):
        raise FormulaDiscoveryLeanTranslationError("polynomial coefficient exceeds bit cap")
    return _Polynomial(tuple(coefficients))


def _parse_candidate_expression(
    result: Mapping[str, Any], problem: Mapping[str, Any]
) -> tuple[sp.Symbol, sp.Expr, Mapping[str, Any]]:
    candidate = result.get("candidate")
    if not isinstance(candidate, Mapping):
        raise FormulaDiscoveryLeanTranslationError("PASS result has no candidate")
    representation = candidate.get("representation")
    if not isinstance(representation, Mapping):
        raise FormulaDiscoveryLeanTranslationError("candidate representation is missing")
    expected_keys = {
        "basis",
        "coefficients",
        "expression",
        "expression_srepr",
        "problem_sha256",
        "schema_version",
        "solver_receipt_sha256",
        "variable",
        "variable_domain",
    }
    if set(representation) != expected_keys:
        raise FormulaDiscoveryLeanTranslationError("candidate representation schema changed")
    parsed = _parse_problem(problem)
    expression = _parse_expression(
        representation["expression"],
        parsed.variable,
        max_bits=parsed.limits["max_integer_bits"],
    )
    if (
        representation.get("schema_version") != "sigma-formula-discovery-candidate-1.0"
        or representation.get("problem_sha256") != result.get("problem_sha256")
        or representation.get("variable") != problem.get("variable")
        or representation.get("variable_domain") != problem.get("variable_domain")
        or representation.get("basis") != problem.get("solver", {}).get("basis")
        or representation.get("coefficients") != result.get("synthesis", {}).get("coefficients")
        or representation.get("expression") != result.get("synthesis", {}).get("expression")
        or sp.srepr(expression) != representation.get("expression_srepr")
    ):
        raise FormulaDiscoveryLeanTranslationError("candidate representation binding changed")
    return parsed.variable, expression, representation


def _lean_int(value: int) -> str:
    return str(value)


def _lean_list(values: Sequence[int], *, natural: bool = False) -> str:
    if natural and any(value < 0 for value in values):
        raise FormulaDiscoveryLeanTranslationError("negative coefficient is unsafe over Nat")
    return "[" + ", ".join(str(value) if natural else _lean_int(value) for value in values) + "]"


def _nat_polynomial(coefficients: Sequence[int], variable: str) -> str:
    if any(value < 0 for value in coefficients):
        raise FormulaDiscoveryLeanTranslationError("negative coefficient is unsafe over Nat")
    terms = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[power]
        if coefficient == 0:
            continue
        atom = "1" if power == 0 else variable if power == 1 else f"{variable} ^ {power}"
        terms.append(atom if coefficient == 1 else f"{coefficient} * {atom}")
    return " + ".join(terms) if terms else "0"


def _audit_lines(target: str, dependencies: Sequence[str]) -> str:
    lines = [
        f'#eval IO.println "{AUDIT_PROTOCOL}_BEGIN"',
        f'#eval IO.println "target={target}"',
    ]
    lines.extend(f'#eval IO.println "dependency={dependency}"' for dependency in dependencies)
    lines.extend(
        [
            '#eval IO.println "result=checked"',
            f'#eval IO.println "{AUDIT_PROTOCOL}_END"',
        ]
    )
    return "\n".join(lines)


def _polynomial_source(
    suffix: str,
    target: str,
    representation: Mapping[str, Any],
    candidate: _Polynomial,
    variable: sp.Symbol,
) -> tuple[str, tuple[str, ...]]:
    basis_sources = representation["basis"]
    coefficient_data = representation["coefficients"]
    if not isinstance(basis_sources, list) or not isinstance(coefficient_data, list):
        raise FormulaDiscoveryLeanTranslationError("candidate basis or coefficients are malformed")
    if len(basis_sources) != len(coefficient_data) or not basis_sources:
        raise FormulaDiscoveryLeanTranslationError("candidate basis dimension changed")
    width = candidate.degree + 1
    basis_vectors = []
    multipliers = []
    for basis_source, coefficient in zip(basis_sources, coefficient_data, strict=True):
        if (
            not isinstance(coefficient, Mapping)
            or set(coefficient) != {"denominator", "numerator"}
            or coefficient.get("denominator") != 1
            or isinstance(coefficient.get("numerator"), bool)
            or not isinstance(coefficient.get("numerator"), int)
        ):
            raise FormulaDiscoveryLeanTranslationError(
                "rational or malformed basis coefficient is unsupported"
            )
        basis_expression = _parse_expression(basis_source, variable, max_bits=256)
        basis_polynomial = list(_integer_polynomial(basis_expression, variable).coefficients)
        if len(basis_polynomial) > width:
            raise FormulaDiscoveryLeanTranslationError("basis exceeds candidate degree")
        basis_vectors.append(basis_polynomial + [0] * (width - len(basis_polynomial)))
        multipliers.append(coefficient["numerator"])

    add_name = f"formulaDiscoveryPolyAdd{suffix}"
    scale_name = f"formulaDiscoveryPolyScale{suffix}"
    combine_name = f"formulaDiscoveryPolyCombine{suffix}"
    dependencies = (
        f"Invariant.{add_name}",
        f"Invariant.{scale_name}",
        f"Invariant.{combine_name}",
        "of_decide_eq_true",
    )
    basis_lean = "[" + ", ".join(_lean_list(row) for row in basis_vectors) + "]"
    source = f"""import Std.Tactic

namespace Invariant

def {add_name} : List Int → List Int → List Int
  | [], right => right
  | left, [] => left
  | a :: left, b :: right => (a + b) :: {add_name} left right

def {scale_name} (a : Int) : List Int → List Int
  | [] => []
  | b :: rest => (a * b) :: {scale_name} a rest

def {combine_name} : List Int → List (List Int) → List Int
  | [], _ => []
  | _, [] => []
  | a :: multipliers, basis :: bases =>
      {add_name} ({scale_name} a basis) ({combine_name} multipliers bases)

theorem formulaDiscovery{suffix} :
    {combine_name} {_lean_list(multipliers)} {basis_lean} =
      {_lean_list(candidate.coefficients)} := by
  native_decide

end Invariant

{_audit_lines(target, dependencies)}
"""
    return source, dependencies


def _recurrence_source(
    suffix: str,
    target: str,
    problem: Mapping[str, Any],
    candidate: _Polynomial,
    variable: sp.Symbol,
) -> tuple[str, tuple[str, ...]]:
    constraints = problem.get("constraints")
    proof = problem.get("proof")
    if (
        not isinstance(constraints, Mapping)
        or constraints.get("kind") != "first_order_recurrence"
        or proof != {"kind": "induction"}
    ):
        raise FormulaDiscoveryLeanTranslationError("Nat recurrence contract is missing")
    base = constraints.get("base")
    if (
        not isinstance(base, Mapping)
        or base.get("index") != 0
        or not isinstance(base.get("value"), Mapping)
        or base["value"].get("denominator") != 1
        or isinstance(base["value"].get("numerator"), bool)
        or not isinstance(base["value"].get("numerator"), int)
        or base["value"]["numerator"] < 0
    ):
        raise FormulaDiscoveryLeanTranslationError(
            "only nonnegative integral base-at-zero Nat recurrences are supported"
        )
    increment_expression = _parse_expression(
        constraints.get("successor_increment"), variable, max_bits=256
    )
    increment = _integer_polynomial(increment_expression, variable)
    if any(value < 0 for value in candidate.coefficients + increment.coefficients):
        raise FormulaDiscoveryLeanTranslationError(
            "negative candidate or increment coefficients are unsafe over Nat"
        )
    if candidate.coefficients[0] != base["value"]["numerator"]:
        raise FormulaDiscoveryLeanTranslationError("candidate and Nat base value disagree")
    proof_expression = _nat_polynomial(candidate.coefficients, "n")
    successor_expression = _nat_polynomial(candidate.coefficients, "(n + 1)")
    increment_source = _nat_polynomial(increment.coefficients, "n")
    sequence_name = f"formulaDiscoverySequence{suffix}"
    successor_name = f"formulaDiscoverySuccessor{suffix}"
    dependencies = (
        f"Invariant.{sequence_name}",
        f"Invariant.{successor_name}",
        "Nat.rec",
        "Nat.add_mul",
        "Nat.mul_add",
        "Nat.mul_assoc",
        "Nat.mul_one",
        "Nat.pow_succ",
        "Nat.pow_zero",
        "Lean.Parser.Tactic.omega",
    )
    source = f"""import Std.Tactic

namespace Invariant

def {sequence_name} : Nat → Nat
  | 0 => {base["value"]["numerator"]}
  | n + 1 => {sequence_name} n + ({increment_source})

private theorem {successor_name} (n : Nat) :
    {proof_expression} + ({increment_source}) = {successor_expression} := by
  simp only [
    Nat.pow_succ,
    Nat.pow_zero,
    Nat.one_mul,
    Nat.mul_one,
    Nat.mul_assoc,
    Nat.add_mul,
    Nat.mul_add,
  ]
  omega

theorem formulaDiscovery{suffix} (n : Nat) :
    {sequence_name} n = {proof_expression} := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [{sequence_name}, ih]
      exact {successor_name} n

end Invariant

{_audit_lines(target, dependencies)}
"""
    return source, dependencies


def translate_formula_discovery_pass(
    result: Mapping[str, Any], problem: Mapping[str, Any]
) -> dict[str, Any]:
    """Translate one validated PASS or reject it before emitting any Lean source."""

    try:
        validate_formula_discovery_result(result, problem)
    except Exception as error:
        raise FormulaDiscoveryLeanTranslationError(
            "formula-discovery result did not validate exactly"
        ) from error
    if result.get("decision") != "PASS":
        raise FormulaDiscoveryLeanTranslationError("only validated PASS results are translatable")
    if problem.get("variable_domain") != "integer":
        raise FormulaDiscoveryLeanTranslationError(
            "rational-domain candidates require an explicit nonzero-domain theorem"
        )
    variable, expression, representation = _parse_candidate_expression(result, problem)
    candidate = _integer_polynomial(expression, variable)
    suffix = _lean_suffix(str(result["job_id"]))
    target = f"Invariant.formulaDiscovery{suffix}"
    constraints = problem.get("constraints", {})
    proof_certificate = result.get("proof_certificate")
    if constraints.get("kind") == "evaluations":
        if problem.get("proof") != {"kind": "none"} or proof_certificate is not None:
            raise FormulaDiscoveryLeanTranslationError(
                "integer polynomial identity proof contract changed"
            )
        source, dependencies = _polynomial_source(
            suffix, target, representation, candidate, variable
        )
        translation_kind = "integer_polynomial_coefficient_identity"
    elif constraints.get("kind") == "first_order_recurrence":
        if (
            not isinstance(proof_certificate, Mapping)
            or proof_certificate.get("decision") != "proved_by_base_and_symbolic_successor_identity"
        ):
            raise FormulaDiscoveryLeanTranslationError(
                "checked induction certificate is required for Nat recurrence translation"
            )
        source, dependencies = _recurrence_source(suffix, target, problem, candidate, variable)
        translation_kind = "first_order_nat_recurrence_closed_form"
    else:
        raise FormulaDiscoveryLeanTranslationError("constraint kind is unsupported")
    if "sorry" in source.lower() or re.search(r"\baxiom\b", source.lower()):
        raise FormulaDiscoveryLeanTranslationError("unsafe Lean declaration was generated")
    source_bytes = source.encode("utf-8")
    if len(source_bytes) > MAX_SOURCE_BYTES:
        raise FormulaDiscoveryLeanTranslationError("generated Lean source exceeds byte cap")
    config = LeanAdapterConfig(
        target=target,
        allowed_premises=dependencies,
        forbidden_premises=FORBIDDEN_PREMISES,
        forbidden_prefixes=FORBIDDEN_PREFIXES,
    )
    manifest = build_allowed_premise_manifest(config)
    proof_sha = None if proof_certificate is None else str(proof_certificate.get("content_sha256"))
    body = {
        "schema_version": TRANSLATION_SCHEMA,
        "translation_kind": translation_kind,
        "job_id": result["job_id"],
        "problem_sha256": result["problem_sha256"],
        "result_content_sha256": result["content_sha256"],
        "candidate_content_sha256": result["candidate"]["content_sha256"],
        "proof_certificate_content_sha256": proof_sha,
        "target": target,
        "source": source,
        "source_sha256": _canonical_source_sha(source),
        "premise_manifest": manifest,
        "counts": {
            "candidate_polynomial_degree": candidate.degree,
            "allowed_premises": len(dependencies),
            "generated_source_bytes": len(source_bytes),
            "proof_certificates_bound": int(proof_certificate is not None),
        },
        "claims": {
            "source_generated_from_candidate_representation": True,
            "closed_premise_manifest_generated": True,
            "lean_kernel_executed": False,
            "general_formula_discovery_established": False,
            "novelty_established": False,
            "scientific_or_physics_truth_inferred": False,
        },
        "scope": (
            "translation of one exact validated Formula Discovery PASS into a supported Lean "
            "coefficient identity or first-order Nat recurrence theorem; translation alone is "
            "not a Lean kernel check"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_formula_discovery_lean_translation(
    value: Mapping[str, Any],
    result: Mapping[str, Any],
    problem: Mapping[str, Any],
) -> None:
    """Rebuild the translation and reject any resealed semantic or source change."""

    if set(value) != _TRANSLATION_KEYS or value.get("schema_version") != TRANSLATION_SCHEMA:
        raise FormulaDiscoveryLeanTranslationError("translation schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise FormulaDiscoveryLeanTranslationError("translation content seal changed")
    source = value.get("source")
    if (
        not isinstance(source, str)
        or value.get("source_sha256") != _canonical_source_sha(source)
        or not _HEX.fullmatch(str(value.get("source_sha256")))
        or "sorry" in source.lower()
        or re.search(r"\baxiom\b", source.lower())
    ):
        raise FormulaDiscoveryLeanTranslationError("translation source binding changed")
    manifest = value.get("premise_manifest")
    if not isinstance(manifest, Mapping):
        raise FormulaDiscoveryLeanTranslationError("translation premise manifest is missing")
    dependencies = tuple(manifest.get("allowed_premises", ()))
    config = LeanAdapterConfig(
        target=str(value.get("target")),
        allowed_premises=dependencies,
        forbidden_premises=FORBIDDEN_PREMISES,
        forbidden_prefixes=FORBIDDEN_PREFIXES,
    )
    try:
        validate_allowed_premise_manifest(manifest, config)
    except Exception as error:
        raise FormulaDiscoveryLeanTranslationError("translation premise closure changed") from error
    expected = translate_formula_discovery_pass(result, problem)
    if dict(value) != expected:
        raise FormulaDiscoveryLeanTranslationError("translation exact replay changed")


def write_lean_source(value: Mapping[str, Any], output_path: Path) -> Path:
    """Write only an already sealed translation to a caller-selected .lean path."""

    if output_path.suffix != ".lean" or not isinstance(value.get("source"), str):
        raise FormulaDiscoveryLeanTranslationError("Lean output path or source is invalid")
    if value.get("source_sha256") != _canonical_source_sha(value["source"]):
        raise FormulaDiscoveryLeanTranslationError("Lean source seal changed before write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(value["source"], encoding="utf-8", newline="\n")
    return output_path


__all__ = [
    "TRANSLATION_SCHEMA",
    "FormulaDiscoveryLeanTranslationError",
    "translate_formula_discovery_pass",
    "validate_formula_discovery_lean_translation",
    "write_lean_source",
]
