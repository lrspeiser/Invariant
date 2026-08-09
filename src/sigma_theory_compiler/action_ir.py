from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp
from sympy.core.relational import Relational

from .formal_backend import load_field_contract, validate_covariant_action_spec


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _coefficient_symbols(value: Any) -> set[str]:
    text = str(value).replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"invalid coefficient syntax: {error.msg}") from error
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise TypeError(f"unsupported coefficient syntax: {type(node).__name__}")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _canonical_parameter_domain(
    value: Any, universal_constants: set[str]
) -> tuple[dict[str, list[str]], list[str]]:
    errors: list[str] = []
    raw = value if isinstance(value, dict) else {}
    if value is not None and not isinstance(value, dict):
        errors.append("parameter_domain must be an object")
    sign_keys = ("positive", "nonnegative", "negative", "nonpositive", "nonzero")
    canonical: dict[str, list[str]] = {}
    for key in sign_keys:
        entries = raw.get(key, [])
        if not isinstance(entries, list) or any(not isinstance(item, str) for item in entries):
            errors.append(f"parameter_domain.{key} must be a list of constant names")
            entries = []
        names = sorted(set(entries))
        unknown = sorted(set(names) - universal_constants)
        if unknown:
            errors.append(
                f"parameter_domain.{key} uses undeclared constants: " + ", ".join(unknown)
            )
        canonical[key] = names

    incompatible = [
        ("positive", "nonpositive"),
        ("positive", "negative"),
        ("negative", "nonnegative"),
        ("negative", "positive"),
    ]
    for left, right in incompatible:
        overlap = sorted(set(canonical[left]) & set(canonical[right]))
        if overlap:
            errors.append(f"contradictory parameter signs in {left}/{right}: " + ", ".join(overlap))

    inequalities = raw.get("inequalities", [])
    if not isinstance(inequalities, list) or any(
        not isinstance(item, str) for item in inequalities
    ):
        errors.append("parameter_domain.inequalities must be a list of comparisons")
        inequalities = []
    symbols = {name: sp.Symbol(name, real=True) for name in universal_constants}
    normalized_inequalities: list[str] = []
    domain_properties: list[tuple[str, str, sp.Expr]] = []
    for name in canonical["positive"]:
        if name in symbols:
            domain_properties.append((f"{name}>0", "positive", symbols[name]))
    for name in canonical["negative"]:
        if name in symbols:
            domain_properties.append((f"{name}<0", "positive", -symbols[name]))
    for name in canonical["nonnegative"]:
        if name in symbols:
            domain_properties.append((f"{name}>=0", "nonnegative", symbols[name]))
    for name in canonical["nonpositive"]:
        if name in symbols:
            domain_properties.append((f"{name}<=0", "nonnegative", -symbols[name]))
    for name in canonical["nonzero"]:
        if name in symbols:
            domain_properties.append((f"{name}!=0", "nonzero", symbols[name]))
    relation_nodes = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Compare)
    arithmetic_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )
    for text in inequalities:
        source = text.replace("^", "**")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as error:
            errors.append(f"invalid parameter inequality syntax: {error.msg}")
            continue
        if not isinstance(tree.body, ast.Compare) or len(tree.body.ops) != 1:
            errors.append(f"parameter inequality must contain one comparison: {text}")
            continue
        if any(
            not isinstance(node, (*arithmetic_nodes, *relation_nodes)) for node in ast.walk(tree)
        ):
            errors.append(f"unsupported parameter inequality syntax: {text}")
            continue
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        unknown = sorted(names - universal_constants)
        if unknown:
            errors.append("parameter inequality uses undeclared constants: " + ", ".join(unknown))
            continue
        try:
            left = sp.sympify(ast.unparse(tree.body.left), locals=symbols, evaluate=True)
            right = sp.sympify(
                ast.unparse(tree.body.comparators[0]), locals=symbols, evaluate=True
            )
            relation_type = {
                ast.Gt: sp.Gt,
                ast.GtE: sp.Ge,
                ast.Lt: sp.Lt,
                ast.LtE: sp.Le,
                ast.Eq: sp.Eq,
                ast.NotEq: sp.Ne,
            }[type(tree.body.ops[0])]
            relation = relation_type(left, right, evaluate=False)
        except (TypeError, ValueError) as error:
            errors.append(f"invalid parameter inequality: {error}")
            continue
        if not isinstance(relation, Relational):
            errors.append(f"parameter inequality is not relational: {text}")
            continue
        normalized_inequalities.append(str(relation))
        if relation.rel_op == ">":
            kind, expression = "positive", relation.lhs - relation.rhs
        elif relation.rel_op == "<":
            kind, expression = "positive", relation.rhs - relation.lhs
        elif relation.rel_op == ">=":
            kind, expression = "nonnegative", relation.lhs - relation.rhs
        elif relation.rel_op == "<=":
            kind, expression = "nonnegative", relation.rhs - relation.lhs
        elif relation.rel_op == "!=":
            kind, expression = "nonzero", relation.lhs - relation.rhs
        else:
            kind, expression = "zero", relation.lhs - relation.rhs
        domain_properties.append((str(relation), kind, sp.factor(expression)))

    contradiction_messages: set[str] = set()
    for index, (left_label, left_kind, left_expression) in enumerate(domain_properties):
        for right_label, right_kind, right_expression in domain_properties[index + 1 :]:
            same = sp.cancel(left_expression - right_expression) == 0
            opposite = sp.cancel(left_expression + right_expression) == 0
            contradictory = (
                same
                and {left_kind, right_kind} in ({"positive", "zero"}, {"nonzero", "zero"})
            ) or (
                opposite
                and left_kind in {"positive", "nonnegative"}
                and right_kind in {"positive", "nonnegative"}
                and "positive" in {left_kind, right_kind}
            )
            if contradictory:
                contradiction_messages.add(
                    f"contradictory parameter-domain claims: {left_label} and {right_label}"
                )
    errors.extend(sorted(contradiction_messages))
    canonical["inequalities"] = sorted(set(normalized_inequalities))
    return canonical, errors


def _canonical_background_domain(
    value: Any, universal_constants: set[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate local field/background restrictions separately from constant parameters."""

    if value is None:
        return None, []
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["background_domain must be an object"]
    allowed_top = {"variables", "inequalities", "preservation"}
    unknown_top = sorted(set(value) - allowed_top)
    if unknown_top:
        errors.append("unknown background_domain fields: " + ", ".join(unknown_top))

    raw_variables = value.get("variables", [])
    if not isinstance(raw_variables, list):
        errors.append("background_domain.variables must be a list")
        raw_variables = []
    allowed_variable = {
        "id",
        "covariant_definition",
        "unitary_gauge_identification",
        "mass_dimension",
        "nonnegative",
        "locally_measurable",
    }
    variables: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(raw_variables):
        if not isinstance(item, dict):
            errors.append(f"background_domain.variables[{index}] must be an object")
            continue
        unknown = sorted(set(item) - allowed_variable)
        if unknown:
            errors.append(
                f"unknown background_domain.variables[{index}] fields: "
                + ", ".join(unknown)
            )
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.isidentifier():
            errors.append(
                f"background_domain.variables[{index}].id must be a safe identifier"
            )
            continue
        if identifier in ids or identifier in universal_constants:
            errors.append(f"duplicate or colliding background variable: {identifier}")
        ids.add(identifier)
        covariant_definition = item.get("covariant_definition")
        unitary_identification = item.get("unitary_gauge_identification")
        mass_dimension = item.get("mass_dimension")
        nonnegative = item.get("nonnegative")
        locally_measurable = item.get("locally_measurable")
        if not isinstance(covariant_definition, str) or not covariant_definition.strip():
            errors.append(
                f"background_domain.variables[{index}].covariant_definition must be text"
            )
        if not isinstance(unitary_identification, str) or not unitary_identification.strip():
            errors.append(
                f"background_domain.variables[{index}].unitary_gauge_identification must be text"
            )
        if not isinstance(mass_dimension, int):
            errors.append(
                f"background_domain.variables[{index}].mass_dimension must be int"
            )
        if not isinstance(nonnegative, bool):
            errors.append(
                f"background_domain.variables[{index}].nonnegative must be bool"
            )
        if locally_measurable is not True:
            errors.append(
                f"background_domain.variables[{index}].locally_measurable must be true"
            )
        variables.append(
            {
                "id": identifier,
                "covariant_definition": covariant_definition,
                "unitary_gauge_identification": unitary_identification,
                "mass_dimension": mass_dimension,
                "nonnegative": nonnegative,
                "locally_measurable": locally_measurable,
            }
        )

    raw_inequalities = value.get("inequalities", [])
    if not isinstance(raw_inequalities, list) or any(
        not isinstance(item, str) for item in raw_inequalities
    ):
        errors.append("background_domain.inequalities must be a list of comparisons")
        raw_inequalities = []
    symbol_names = universal_constants | ids
    symbols = {name: sp.Symbol(name, real=True) for name in symbol_names}
    relation_nodes = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Compare)
    arithmetic_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )
    normalized_inequalities: list[str] = []
    for text in raw_inequalities:
        source = text.replace("^", "**")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as error:
            errors.append(f"invalid background inequality syntax: {error.msg}")
            continue
        if not isinstance(tree.body, ast.Compare) or len(tree.body.ops) != 1:
            errors.append(f"background inequality must contain one comparison: {text}")
            continue
        if any(
            not isinstance(node, (*arithmetic_nodes, *relation_nodes))
            for node in ast.walk(tree)
        ):
            errors.append(f"unsupported background inequality syntax: {text}")
            continue
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        unknown = sorted(names - symbol_names)
        if unknown:
            errors.append(
                "background inequality uses undeclared names: " + ", ".join(unknown)
            )
            continue
        try:
            left = sp.sympify(ast.unparse(tree.body.left), locals=symbols, evaluate=True)
            right = sp.sympify(
                ast.unparse(tree.body.comparators[0]), locals=symbols, evaluate=True
            )
            relation_type = {
                ast.Gt: sp.Gt,
                ast.GtE: sp.Ge,
                ast.Lt: sp.Lt,
                ast.LtE: sp.Le,
                ast.Eq: sp.Eq,
                ast.NotEq: sp.Ne,
            }[type(tree.body.ops[0])]
            relation = relation_type(left, right, evaluate=False)
        except (TypeError, ValueError) as error:
            errors.append(f"invalid background inequality: {error}")
            continue
        normalized_inequalities.append(str(relation))

    preservation = value.get("preservation", {})
    if not isinstance(preservation, dict):
        errors.append("background_domain.preservation must be an object")
        preservation = {}
    allowed_preservation = {"status", "statement", "required_controls"}
    unknown_preservation = sorted(set(preservation) - allowed_preservation)
    if unknown_preservation:
        errors.append(
            "unknown background_domain.preservation fields: "
            + ", ".join(unknown_preservation)
        )
    preservation_status = preservation.get("status", "unresolved")
    if preservation_status not in {"proved", "unresolved", "rejected"}:
        errors.append(
            "background_domain.preservation.status must be proved, unresolved, or rejected"
        )
    statement = preservation.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        errors.append("background_domain.preservation.statement must be text")
    required_controls = preservation.get("required_controls", [])
    if not isinstance(required_controls, list) or any(
        not isinstance(item, str) or not item for item in required_controls
    ):
        errors.append(
            "background_domain.preservation.required_controls must be a list of names"
        )
        required_controls = []
    canonical = {
        "variables": sorted(variables, key=lambda item: item["id"]),
        "inequalities": sorted(set(normalized_inequalities)),
        "preservation": {
            "status": preservation_status,
            "statement": statement,
            "required_controls": sorted(set(required_controls)),
        },
    }
    return canonical, errors


def load_action_grammar(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sigma-covariant-action-grammar-1.0":
        raise ValueError("Unsupported or missing covariant action-grammar version")
    return payload


def compile_action_spec(
    spec: dict[str, Any],
    grammar: dict[str, Any],
    field_contract: dict[str, Any],
) -> dict[str, Any]:
    """Compile a declarative action into a deterministic, fail-closed intermediate form."""

    errors: list[str] = []
    if spec.get("schema_version") != "sigma-action-spec-1.0":
        errors.append("unsupported or missing action-spec schema_version")
    term_library = {item["id"]: item for item in grammar["term_library"]}
    requested = list(spec.get("terms", []))
    if len(requested) != len(set(requested)):
        errors.append("duplicate term ids are forbidden; combine coefficients before compilation")
    unknown = sorted(set(requested) - set(term_library))
    if unknown:
        errors.append("unknown action terms: " + ", ".join(unknown))
    bounds = grammar["bounds"]
    if len(requested) > int(bounds["maximum_terms"]):
        errors.append("action exceeds the finite term-count bound")
    if "EH_R" not in requested:
        errors.append("EH_R is required by this grammar")
    control_only_terms = sorted(
        term_id
        for term_id in set(requested) & set(term_library)
        if term_library[term_id].get("control_only")
    )
    if control_only_terms and spec.get("role") != "known_answer_control":
        errors.append(
            "control-only terms cannot enter candidate generation: " + ", ".join(control_only_terms)
        )

    declared_fields = set(spec.get("fields", []))
    allowed_fields = {item["id"] for item in field_contract["fields"]}
    undeclared_fields = sorted(declared_fields - allowed_fields)
    if undeclared_fields:
        errors.append("unknown fields: " + ", ".join(undeclared_fields))
    extra_dynamical_fields = declared_fields & {"phi", "u_mu", "A_mu"}
    if len(extra_dynamical_fields) > int(bounds["maximum_extra_dynamical_fields"]):
        errors.append(
            "action exceeds the extra-dynamical-field bound: "
            + ", ".join(sorted(extra_dynamical_fields))
        )
    required_fields: set[str] = set()
    invariants: set[str] = set()
    compiled_terms: list[dict[str, Any]] = []
    universal_constants = sorted(set(spec.get("universal_constants", [])))
    if len(universal_constants) > int(bounds["maximum_universal_constants"]):
        errors.append("action exceeds the universal-constant bound")
    parameter_domain, domain_errors = _canonical_parameter_domain(
        spec.get("parameter_domain"), set(universal_constants)
    )
    errors.extend(domain_errors)
    background_domain, background_errors = _canonical_background_domain(
        spec.get("background_domain"), set(universal_constants)
    )
    errors.extend(background_errors)
    used_coefficient_symbols: set[str] = set()
    required_term_constants: set[str] = set()
    for term_id in sorted(set(requested) & set(term_library)):
        term = term_library[term_id]
        required_fields.update(term["fields"])
        if term.get("invariant"):
            invariants.add(term["invariant"])
        required_term_constants.update(term.get("required_constants", []))
        if int(term["maximum_derivatives_per_field"]) > int(
            bounds["maximum_derivatives_per_field_in_term"]
        ):
            errors.append(f"term exceeds derivative bound: {term_id}")
        coefficient = spec.get("coefficients", {}).get(term_id, term["coefficient"])
        try:
            used_coefficient_symbols.update(_coefficient_symbols(coefficient))
        except (TypeError, ValueError) as error:
            errors.append(f"invalid coefficient for {term_id}: {error}")
        compiled_terms.append(
            {
                "id": term_id,
                "coefficient": str(coefficient),
                "density": term["density"],
                "fields": term["fields"],
                "invariant": term.get("invariant"),
                "maximum_derivatives_per_field": term["maximum_derivatives_per_field"],
                "static_legacy_atom": term.get("static_legacy_atom"),
                "required_constants": sorted(term.get("required_constants", [])),
            }
        )
    missing_fields = sorted(required_fields - declared_fields)
    if missing_fields:
        errors.append("terms require undeclared fields: " + ", ".join(missing_fields))

    proca_control = spec.get("role") == "known_answer_control" and "PROCA_F2" in requested
    if (
        "u_mu" in required_fields
        and not proca_control
        and "UNIT_VECTOR_CONSTRAINT" not in requested
    ):
        errors.append("unit timelike u_mu requires UNIT_VECTOR_CONSTRAINT")
    if "lambda_u" in declared_fields and "UNIT_VECTOR_CONSTRAINT" not in requested:
        errors.append("lambda_u is declared but its constraint term is absent")

    undeclared_coefficient_symbols = sorted(used_coefficient_symbols - set(universal_constants))
    if undeclared_coefficient_symbols:
        errors.append(
            "coefficient symbols must be declared universal constants: "
            + ", ".join(undeclared_coefficient_symbols)
        )
    missing_term_constants = sorted(required_term_constants - set(universal_constants))
    if missing_term_constants:
        errors.append(
            "terms require undeclared universal constants: "
            + ", ".join(missing_term_constants)
        )
    policy = validate_covariant_action_spec(
        {
            "matter_metric": spec.get("matter_metric"),
            "invariants": sorted(invariants),
            "action": " + ".join(item["density"] for item in compiled_terms),
            "static_dictionary_status": spec.get("static_dictionary_status"),
        },
        field_contract,
    )
    errors.extend(policy["errors"])

    canonical = {
        "schema_version": "sigma-action-ir-1.0",
        "source_role": spec.get("role", "candidate"),
        "fields": sorted(declared_fields),
        "extra_dynamical_fields": sorted(extra_dynamical_fields),
        "matter_metric": spec.get("matter_metric"),
        "universal_constants": universal_constants,
        "coefficient_symbols": sorted(used_coefficient_symbols),
        "parameter_domain": parameter_domain,
        "static_dictionary_status": spec.get("static_dictionary_status"),
        "terms": compiled_terms,
    }
    if background_domain is not None:
        canonical["background_domain"] = background_domain
    generator_origin = spec.get("generator_origin")
    if generator_origin is not None:
        if not isinstance(generator_origin, dict):
            errors.append("generator_origin must be an object")
        else:
            required_origin = {
                "family_id": str,
                "ordinal": int,
                "correction_expression": str,
                "pareto_front": int,
                "source_priority_sha256": str,
            }
            for key, expected_type in required_origin.items():
                if not isinstance(generator_origin.get(key), expected_type):
                    errors.append(
                        f"generator_origin.{key} must be {expected_type.__name__}"
                    )
            unknown_origin = sorted(set(generator_origin) - set(required_origin))
            if unknown_origin:
                errors.append(
                    "unknown generator_origin fields: " + ", ".join(unknown_origin)
                )
            canonical["generator_origin"] = {
                key: generator_origin.get(key) for key in sorted(required_origin)
            }
    completion = spec.get("covariant_completion")
    if completion is not None:
        if not isinstance(completion, dict):
            errors.append("covariant_completion must be an object")
        else:
            required_completion = {
                "completion_id": str,
                "static_null_terms": list,
                "purpose": str,
            }
            for key, expected_type in required_completion.items():
                if not isinstance(completion.get(key), expected_type):
                    errors.append(
                        f"covariant_completion.{key} must be {expected_type.__name__}"
                    )
            static_null_terms = completion.get("static_null_terms", [])
            if isinstance(static_null_terms, list) and any(
                not isinstance(item, str) for item in static_null_terms
            ):
                errors.append(
                    "covariant_completion.static_null_terms must contain term ids"
                )
            unknown_completion = sorted(set(completion) - set(required_completion))
            if unknown_completion:
                errors.append(
                    "unknown covariant_completion fields: "
                    + ", ".join(unknown_completion)
                )
            canonical["covariant_completion"] = {
                "completion_id": completion.get("completion_id"),
                "purpose": completion.get("purpose"),
                "static_null_terms": sorted(set(static_null_terms))
                if isinstance(static_null_terms, list)
                else [],
            }
    canonical_text = _canonical_json(canonical)
    return {
        "valid": not errors,
        "errors": errors,
        "content_sha256": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
        "canonical": canonical,
        "policy_validation": policy,
    }


def compile_action_file(
    spec_path: str | Path,
    grammar_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    grammar = load_action_grammar(grammar_path)
    contract = load_field_contract(contract_path)
    return compile_action_spec(spec, grammar, contract)
