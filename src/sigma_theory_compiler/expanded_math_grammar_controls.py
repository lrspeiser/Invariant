"""Run positive and mutation controls for every expanded first-class mathematical formula kind."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import expanded_math_independent_evaluator as independent
from . import expanded_math_primary_evaluator as primary
from .independent_exact_evaluator import evaluate_recurrence
from .math_canonicalizer import canonical_sha256 as formula_sha256
from .math_counterexample import evaluate_formula
from .math_expression_ir import (
    Equation,
    FiniteProduct,
    FiniteSum,
    Formula,
    GeneratingFunction,
    ModularRelation,
    Recurrence,
    TensorIdentity,
    VariationalFunctional,
    add,
    call,
    literal,
    multiply,
    power,
    symbol,
)
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/expanded_typed_grammar.json"
OUTPUT_PATH = "runs/math/expanded-typed-grammar/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/expanded_math_grammar_controls.py"
IR_PATH = "src/sigma_theory_compiler/math_expression_ir.py"
PRIMARY_PATH = "src/sigma_theory_compiler/expanded_math_primary_evaluator.py"
INDEPENDENT_PATH = "src/sigma_theory_compiler/expanded_math_independent_evaluator.py"
INDEPENDENT_BASE_PATH = "src/sigma_theory_compiler/independent_exact_evaluator.py"
CANONICALIZER_PATH = "src/sigma_theory_compiler/math_canonicalizer.py"
COUNTEREXAMPLE_PATH = "src/sigma_theory_compiler/math_counterexample.py"
CONFIG_SCHEMA = "invariant-expanded-typed-grammar-config-1.0"
RESULT_SCHEMA = "invariant-expanded-typed-grammar-controls-1.0"
_KINDS = (
    "finite_product",
    "finite_sum",
    "generating_function",
    "modular_relation",
    "recurrence",
    "tensor_identity",
    "variational_functional",
)


class ExpandedGrammarControlError(ValueError):
    """The expanded grammar, evaluator independence, or a mutation control changed."""


def _normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if set(value) != {
        "admitted_formula_kinds",
        "claim_boundary",
        "evaluator_policy",
        "grammar_id",
        "resource_limits",
        "schema_version",
    }:
        raise ExpandedGrammarControlError("expanded grammar config keys changed")
    if set(value["evaluator_policy"]) != {
        "exact_primary_and_independent_agreement_required",
        "mutation_must_fail_both_evaluators",
        "positive_control_required_for_every_kind",
        "sympy_forbidden_in_independent_evaluator",
    } or set(value["claim_boundary"]) != {
        "control_pass_establishes_literature_novelty",
        "grammar_admission_establishes_formula_correctness",
        "llm_output_can_extend_executable_grammar_without_review",
    }:
        raise ExpandedGrammarControlError("expanded grammar policy keys changed")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["grammar_id"] != "invariant.math.expanded-exact-grammar-v1"
        or value["admitted_formula_kinds"] != list(_KINDS)
        or value["resource_limits"]
        != {
            "maximum_expression_nodes": 512,
            "maximum_finite_terms": 64,
            "maximum_polynomial_power_for_independent_variational_evaluator": 8,
            "maximum_tensor_components": 256,
            "maximum_tensor_rank": 4,
        }
        or any(value["evaluator_policy"].get(key) is not True for key in value["evaluator_policy"])
        or any(value["claim_boundary"].get(key) is not False for key in value["claim_boundary"])
    ):
        raise ExpandedGrammarControlError("expanded grammar policy changed")
    return value


def _formula_controls() -> dict[str, tuple[Formula, Formula, list[dict[str, Fraction]]]]:
    index = symbol("i")
    variable = symbol("x")
    field = symbol("q")
    first = symbol("q_dot")
    second = symbol("q_ddot")
    finite_sum = FiniteSum(index, 1, 5, index, literal(15))
    finite_sum_mutation = FiniteSum(index, 1, 5, index, literal(14))
    finite_product = FiniteProduct(index, 1, 4, index, literal(24))
    finite_product_mutation = FiniteProduct(index, 1, 4, index, literal(25))
    generating = GeneratingFunction(
        "geometric_prefix",
        variable,
        tuple(literal(1) for _ in range(5)),
        add(*(power(variable, exponent) for exponent in range(5))),
    )
    generating_mutation = GeneratingFunction(
        "geometric_prefix",
        variable,
        tuple(literal(1) for _ in range(5)),
        add(*(power(variable, exponent) for exponent in range(4))),
    )
    modular = ModularRelation(power(literal(2), literal(5)), literal(1), 31)
    modular_mutation = ModularRelation(power(literal(2), literal(5)), literal(2), 31)
    tensor = TensorIdentity(
        "symmetric_control",
        (2, 2),
        ("covariant", "covariant"),
        tuple(literal(value) for value in (1, 2, 2, 3)),
        tuple(literal(value) for value in (1, 2, 2, 3)),
        ((0, 1, 1),),
    )
    tensor_mutation = TensorIdentity(
        "symmetric_control",
        (2, 2),
        ("covariant", "covariant"),
        tuple(literal(value) for value in (1, 2, 2, 3)),
        tuple(literal(value) for value in (1, 2, 4, 3)),
        ((0, 1, 1),),
    )
    integrand = add(multiply(Fraction(1, 2), power(first, 2)), multiply(-2, power(field, 2)))
    variational = VariationalFunctional(
        "q",
        "t",
        "q_dot",
        "q_ddot",
        integrand,
        add(multiply(-4, field), -second),
    )
    variational_mutation = VariationalFunctional(
        "q",
        "t",
        "q_dot",
        "q_ddot",
        integrand,
        add(multiply(-4, field), second),
    )
    return {
        "finite_product": (finite_product, finite_product_mutation, [{}]),
        "finite_sum": (finite_sum, finite_sum_mutation, [{}]),
        "generating_function": (
            generating,
            generating_mutation,
            [{"x": Fraction(1, 2)}, {"x": Fraction(-1, 2)}, {"x": Fraction(2)}],
        ),
        "modular_relation": (modular, modular_mutation, [{}]),
        "tensor_identity": (tensor, tensor_mutation, [{}]),
        "variational_functional": (variational, variational_mutation, [{}]),
    }


def _recurrence_result() -> dict[str, Any]:
    index = symbol("n")
    positive = Recurrence(
        "a",
        index,
        2,
        Equation(call("a", index + 2), call("a", index + 1) + call("a", index)),
        ((0, literal(0)), (1, literal(1))),
    )
    mutation = Recurrence(
        "a",
        index,
        2,
        Equation(call("a", index + 2), call("a", index + 1) + 2 * call("a", index)),
        ((0, literal(0)), (1, literal(1))),
    )
    expected = (Fraction(0), Fraction(1), Fraction(1), Fraction(2), Fraction(3), Fraction(5), Fraction(8), Fraction(13))
    table = {index: value for index, value in enumerate(expected)}
    primary_positive = all(
        evaluate_formula(positive, {"n": value, "a": table}) for value in range(6)
    )
    primary_mutation_rejected = all(
        not evaluate_formula(mutation, {"n": value, "a": table}) for value in range(1, 6)
    )
    independent_positive = evaluate_recurrence((Fraction(1), Fraction(1)), expected[:2], range(8)) == expected
    independent_mutation_rejected = evaluate_recurrence(
        (Fraction(1), Fraction(2)), expected[:2], range(8)
    ) != expected
    return {
        "kind": "recurrence",
        "positive_formula_sha256": formula_sha256(positive),
        "mutation_formula_sha256": formula_sha256(mutation),
        "primary_positive_passed": primary_positive,
        "independent_positive_passed": independent_positive,
        "primary_mutation_rejected": primary_mutation_rejected,
        "independent_mutation_rejected": independent_mutation_rejected,
        "exact_evaluator_agreement": primary_positive
        and independent_positive
        and primary_mutation_rejected
        and independent_mutation_rejected,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root)
    results = []
    for kind, (formula, mutation, assignments) in sorted(_formula_controls().items()):
        primary_positive = all(primary.evaluate(formula, item) for item in assignments)
        independent_positive = all(independent.evaluate(formula, item) for item in assignments)
        primary_mutation_rejected = all(
            not primary.evaluate(mutation, item) for item in assignments
        )
        independent_mutation_rejected = all(
            not independent.evaluate(mutation, item) for item in assignments
        )
        results.append(
            {
                "kind": kind,
                "positive_formula_sha256": formula_sha256(formula),
                "mutation_formula_sha256": formula_sha256(mutation),
                "primary_positive_passed": primary_positive,
                "independent_positive_passed": independent_positive,
                "primary_mutation_rejected": primary_mutation_rejected,
                "independent_mutation_rejected": independent_mutation_rejected,
                "exact_evaluator_agreement": primary_positive
                and independent_positive
                and primary_mutation_rejected
                and independent_mutation_rejected,
            }
        )
    results.append(_recurrence_result())
    paths = {
        "canonicalizer": CANONICALIZER_PATH,
        "config": CONFIG_PATH,
        "control_runner": SOURCE_PATH,
        "counterexample_evaluator": COUNTEREXAMPLE_PATH,
        "independent_base": INDEPENDENT_BASE_PATH,
        "independent_evaluator": INDEPENDENT_PATH,
        "math_ir": IR_PATH,
        "primary_evaluator": PRIMARY_PATH,
    }
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "grammar_id": config["grammar_id"],
        "source_bindings": {
            key: {"path": path, "normalized_file_sha256": _normalized_sha256(root / path)}
            for key, path in sorted(paths.items())
        },
        "resource_limits": config["resource_limits"],
        "controls": sorted(results, key=lambda item: item["kind"]),
        "summary": {
            "admitted_formula_kinds": list(_KINDS),
            "controls_passed": sum(item["exact_evaluator_agreement"] for item in results),
            "controls_total": len(results),
            "status": "PASS_EXPANDED_TYPED_GRAMMAR_CONTROLS",
        },
        "claims": {
            "control_pass_establishes_literature_novelty": False,
            "grammar_admission_establishes_formula_correctness": False,
            "llm_output_can_extend_executable_grammar_without_review": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root)
    return body


def validate_receipt(value: Mapping[str, Any], root: Path | None = None) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("content_sha256") != canonical_sha256(body)
    ):
        raise ExpandedGrammarControlError("expanded grammar receipt identity or seal changed")
    controls = value.get("controls")
    summary = value.get("summary", {})
    expected_claims = {
        "control_pass_establishes_literature_novelty",
        "grammar_admission_establishes_formula_correctness",
        "llm_output_can_extend_executable_grammar_without_review",
    }
    expected_sources = {
        "canonicalizer": CANONICALIZER_PATH,
        "config": CONFIG_PATH,
        "control_runner": SOURCE_PATH,
        "counterexample_evaluator": COUNTEREXAMPLE_PATH,
        "independent_base": INDEPENDENT_BASE_PATH,
        "independent_evaluator": INDEPENDENT_PATH,
        "math_ir": IR_PATH,
        "primary_evaluator": PRIMARY_PATH,
    }
    if (
        value.get("grammar_id") != "invariant.math.expanded-exact-grammar-v1"
        or value.get("resource_limits")
        != {
            "maximum_expression_nodes": 512,
            "maximum_finite_terms": 64,
            "maximum_polynomial_power_for_independent_variational_evaluator": 8,
            "maximum_tensor_components": 256,
            "maximum_tensor_rank": 4,
        }
        or not isinstance(controls, list)
        or [item.get("kind") for item in controls] != list(_KINDS)
        or any(item.get("exact_evaluator_agreement") is not True for item in controls)
        or summary.get("admitted_formula_kinds") != list(_KINDS)
        or summary.get("controls_passed") != len(_KINDS)
        or summary.get("controls_total") != len(_KINDS)
        or summary.get("status") != "PASS_EXPANDED_TYPED_GRAMMAR_CONTROLS"
        or set(value.get("claims", {})) != expected_claims
        or any(value.get("claims", {}).get(key) is not False for key in expected_claims)
        or set(value.get("source_bindings", {})) != set(expected_sources)
    ):
        raise ExpandedGrammarControlError("expanded grammar control policy changed")
    for item in controls:
        if set(item) != {
            "exact_evaluator_agreement",
            "independent_mutation_rejected",
            "independent_positive_passed",
            "kind",
            "mutation_formula_sha256",
            "positive_formula_sha256",
            "primary_mutation_rejected",
            "primary_positive_passed",
        } or any(
            item.get(key) is not True
            for key in (
                "primary_positive_passed",
                "independent_positive_passed",
                "primary_mutation_rejected",
                "independent_mutation_rejected",
            )
        ):
            raise ExpandedGrammarControlError("expanded grammar control lost a required outcome")
    if root is not None:
        root = root.resolve()
        _load_config(root)
        for key, binding in value.get("source_bindings", {}).items():
            if not isinstance(binding, Mapping) or set(binding) != {
                "normalized_file_sha256",
                "path",
            }:
                raise ExpandedGrammarControlError("expanded grammar source binding changed")
            if binding["path"] != expected_sources[key]:
                raise ExpandedGrammarControlError("expanded grammar source path changed")
            path = (root / str(binding["path"])).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ExpandedGrammarControlError("expanded grammar source path escapes root") from error
            if _normalized_sha256(path) != binding["normalized_file_sha256"]:
                raise ExpandedGrammarControlError("expanded grammar source hash changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "run":
        receipt = build_receipt(root)
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_receipt(receipt, root)
    print(
        json.dumps(
            {
                "controls_passed": receipt["summary"]["controls_passed"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
