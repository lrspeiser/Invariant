from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
from typing import Any

SCHEMA_VERSION = "sigma-physics-concept-language-1.0"

VERIFICATION_STAGES = (
    "syntax",
    "type",
    "dimension",
    "covariance",
    "variation",
    "noether",
    "adm",
    "dirac",
    "hamiltonian",
    "principal",
    "observation",
)

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_FIELD_KINDS = {
    "metric",
    "real_scalar",
    "complex_scalar",
    "covector",
    "unit_timelike_covector",
    "p_form",
    "spinor",
}


def _primitive(
    inputs: tuple[tuple[str, ...], ...], output: str, capabilities: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "input_kinds": [list(item) for item in inputs],
        "output_kind": output,
        "capabilities": list(capabilities),
    }


_FORMAL = VERIFICATION_STAGES[:-1]
_TYPED = VERIFICATION_STAGES[:3]
_KINEMATIC = VERIFICATION_STAGES[:4]
BUILTIN_PRIMITIVES: dict[str, dict[str, Any]] = {
    "operator.einstein_hilbert": _primitive((("metric",),), "operator", _FORMAL),
    "operator.real_scalar_kinetic": _primitive(
        (("metric",), ("real_scalar",)), "operator", _FORMAL
    ),
    "operator.complex_scalar_kinetic": _primitive(
        (("metric",), ("complex_scalar",)), "operator", _TYPED
    ),
    "operator.quartic_horndeski_linear_x": _primitive(
        (("metric",), ("real_scalar",)), "operator", _FORMAL
    ),
    "operator.horndeski_l2_l3_l4_pack": _primitive(
        (("metric",), ("real_scalar",)), "operator", _KINEMATIC
    ),
    "operator.horndeski_g2_function": _primitive(
        (("metric",), ("real_scalar",)), "operator", VERIFICATION_STAGES[:6]
    ),
    "operator.horndeski_g3_function": _primitive(
        (("metric",), ("real_scalar",)), "operator", VERIFICATION_STAGES[:6]
    ),
    "operator.horndeski_g4_phi_function": _primitive(
        (("metric",), ("real_scalar",)), "operator", VERIFICATION_STAGES[:6]
    ),
    "operator.dhost_quadratic_reduced_pack": _primitive(
        (("metric",), ("real_scalar",)), "operator", _TYPED
    ),
    "operator.proca": _primitive((("metric",), ("covector",)), "operator", _FORMAL),
    "operator.einstein_aether": _primitive(
        (("metric",), ("unit_timelike_covector",)), "operator", VERIFICATION_STAGES[:8]
    ),
    "action.sum": _primitive((("operator",),), "action", VERIFICATION_STAGES[:10]),
    "state.coherent": _primitive(
        (("complex_scalar", "covector", "p_form"),), "state", _TYPED
    ),
    "observable.first_order_coherence": _primitive(
        (("state",),), "observable", _TYPED
    ),
    "observable.local_stress_energy": _primitive(
        (("action",),), "observable", VERIFICATION_STAGES[:6]
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_identifier(value: Any, label: str, errors: list[str]) -> str:
    text = str(value)
    if not _IDENTIFIER.fullmatch(text):
        errors.append(f"{label} must match {_IDENTIFIER.pattern}: {text!r}")
    return text


def _custom_primitives(program: dict[str, Any], errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(program.get("primitive_declarations", [])):
        if not isinstance(raw, dict):
            errors.append(f"primitive_declarations[{index}] must be an object")
            continue
        primitive_id = _validate_identifier(raw.get("id", ""), "primitive id", errors)
        input_kinds = raw.get("input_kinds", [])
        if not isinstance(input_kinds, list) or any(
            not isinstance(item, list) or not item for item in input_kinds
        ):
            errors.append(f"primitive {primitive_id} input_kinds must be nonempty kind lists")
            input_kinds = []
        semantics = raw.get("semantic_contract")
        if not isinstance(semantics, str) or not semantics.strip():
            errors.append(f"primitive {primitive_id} requires a semantic_contract")
        capabilities = raw.get("capabilities", ["syntax", "type"])
        if not isinstance(capabilities, list) or any(
            item not in VERIFICATION_STAGES for item in capabilities
        ):
            errors.append(f"primitive {primitive_id} has invalid capabilities")
            capabilities = ["syntax", "type"]
        result[primitive_id] = {
            "input_kinds": input_kinds,
            "output_kind": str(raw.get("output_kind", "concept")),
            "capabilities": capabilities,
            "semantic_contract": semantics,
            "custom": True,
        }
    return result


def _topological_order(
    node_ids: set[str], dependencies: dict[str, list[str]], errors: list[str]
) -> list[str]:
    indegree = {node_id: 0 for node_id in node_ids}
    children = {node_id: [] for node_id in node_ids}
    for node_id, inputs in dependencies.items():
        for source in inputs:
            if source in node_ids:
                indegree[node_id] += 1
                children[source].append(node_id)
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(node_ids):
        errors.append("concept graph contains a dependency cycle")
    return order


def _mutation_cardinality(program: dict[str, Any], errors: list[str]) -> tuple[int, list[dict[str, Any]]]:
    cardinality = 1
    spaces: list[dict[str, Any]] = []
    for index, raw in enumerate(program.get("mutation_spaces", [])):
        if not isinstance(raw, dict):
            errors.append(f"mutation_spaces[{index}] must be an object")
            continue
        space_id = _validate_identifier(raw.get("id", ""), "mutation-space id", errors)
        choices = raw.get("choices", [])
        if not isinstance(choices, list) or not choices:
            errors.append(f"mutation space {space_id} must contain choices")
            continue
        cardinality *= len(choices)
        spaces.append({"id": space_id, "choice_count": len(choices)})
    return cardinality, spaces


def compile_physics_program(program: dict[str, Any]) -> dict[str, Any]:
    """Compile a typed concept graph and fail closed on missing proof capabilities.

    This front end deliberately separates expressibility from scientific admission. A custom
    primitive can be represented immediately, but it remains unresolved until its required
    variation, constraint, stability, and observational adapters exist.
    """

    errors: list[str] = []
    if program.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    primitive_registry = {**BUILTIN_PRIMITIVES, **_custom_primitives(program, errors)}
    fields: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(program.get("fields", [])):
        if not isinstance(raw, dict):
            errors.append(f"fields[{index}] must be an object")
            continue
        field_id = _validate_identifier(raw.get("id", ""), "field id", errors)
        kind = str(raw.get("kind", ""))
        if kind not in _FIELD_KINDS:
            errors.append(f"field {field_id} has unsupported kind {kind!r}")
        if field_id in fields:
            errors.append(f"duplicate field id: {field_id}")
        fields[field_id] = {
            "kind": kind,
            "matter": bool(raw.get("matter", False)),
            "dimension": raw.get("dimension"),
        }

    metric_ids = sorted(field_id for field_id, field in fields.items() if field["kind"] == "metric")
    if len(metric_ids) != 1:
        errors.append("exactly one metric field is required")
    coupling = program.get("matter_coupling", {})
    universal_metric = coupling.get("universal_metric") if isinstance(coupling, dict) else None
    if universal_metric not in metric_ids:
        errors.append("matter_coupling.universal_metric must name the unique metric field")
    exceptional = coupling.get("exceptions", []) if isinstance(coupling, dict) else []
    if exceptional:
        errors.append("universal matter coupling cannot declare exceptions")

    raw_nodes = program.get("concepts", [])
    nodes: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            errors.append(f"concepts[{index}] must be an object")
            continue
        node_id = _validate_identifier(raw.get("id", ""), "concept id", errors)
        if node_id in fields or node_id in nodes:
            errors.append(f"duplicate field/concept id: {node_id}")
        primitive_id = str(raw.get("primitive", ""))
        if primitive_id not in primitive_registry:
            errors.append(
                f"concept {node_id} uses undeclared primitive {primitive_id!r}; "
                "add a typed primitive_declaration"
            )
        inputs = raw.get("inputs", [])
        if not isinstance(inputs, list) or any(not isinstance(item, str) for item in inputs):
            errors.append(f"concept {node_id} inputs must be identifier strings")
            inputs = []
        nodes[node_id] = {"primitive": primitive_id, "inputs": inputs}

    known_ids = set(fields) | set(nodes)
    for node_id, node in nodes.items():
        unknown = sorted(set(node["inputs"]) - known_ids)
        if unknown:
            errors.append(f"concept {node_id} has unknown inputs: {', '.join(unknown)}")

    order = _topological_order(set(nodes), {key: value["inputs"] for key, value in nodes.items()}, errors)
    inferred_kinds = {field_id: field["kind"] for field_id, field in fields.items()}
    primitive_counts: Counter[str] = Counter()
    for node_id in order:
        node = nodes[node_id]
        primitive_id = node["primitive"]
        primitive = primitive_registry.get(primitive_id)
        if primitive is None:
            continue
        primitive_counts[primitive_id] += 1
        inputs = node["inputs"]
        constraints = primitive["input_kinds"]
        variadic_action = primitive_id == "action.sum"
        if variadic_action:
            if not inputs or any(inferred_kinds.get(item) != "operator" for item in inputs):
                errors.append(f"concept {node_id} action.sum requires one or more operators")
        elif len(inputs) != len(constraints):
            errors.append(
                f"concept {node_id} expects {len(constraints)} inputs, received {len(inputs)}"
            )
        else:
            for position, (source, allowed) in enumerate(zip(inputs, constraints, strict=True)):
                actual = inferred_kinds.get(source)
                if actual not in allowed:
                    errors.append(
                        f"concept {node_id} input {position} expects {allowed}, received {actual!r}"
                    )
        inferred_kinds[node_id] = primitive["output_kind"]

    required = program.get("required_verification_stages", list(_FORMAL))
    if not isinstance(required, list) or any(item not in VERIFICATION_STAGES for item in required):
        errors.append("required_verification_stages contains an invalid stage")
        required = list(_FORMAL)
    missing_adapters: list[dict[str, str]] = []
    for primitive_id in sorted(primitive_counts):
        available = set(primitive_registry[primitive_id]["capabilities"])
        for stage in required:
            if stage not in available:
                missing_adapters.append({"primitive": primitive_id, "stage": stage})

    cardinality, mutation_spaces = _mutation_cardinality(program, errors)
    canonical_program = json.loads(_canonical_json(program))
    content_sha256 = hashlib.sha256(_canonical_json(canonical_program).encode()).hexdigest()
    status = "reject" if errors else "unresolved_missing_adapters" if missing_adapters else "ready"
    return {
        "schema_version": "sigma-physics-concept-ir-1.0",
        "source_schema_version": SCHEMA_VERSION,
        "status": status,
        "content_sha256": content_sha256,
        "universal_metric": universal_metric,
        "field_count": len(fields),
        "concept_count": len(nodes),
        "topological_order": order,
        "inferred_kinds": dict(sorted(inferred_kinds.items())),
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "required_verification_stages": required,
        "missing_adapters": missing_adapters,
        "errors": errors,
        "mutation_space": {
            "axes": mutation_spaces,
            "declared_cardinality": cardinality,
            "log10_cardinality": 0.0 if cardinality == 1 else math.log10(cardinality),
            "enumerated": False,
        },
        "admission_rule": (
            "expressibility is not evidence of physical validity; every required capability must "
            "be supplied by an action- and concept-bound verifier before promotion"
        ),
    }
