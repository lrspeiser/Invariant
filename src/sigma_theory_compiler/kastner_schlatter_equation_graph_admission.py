"""Typed equation-graph admission for the Schlatter--Kastner proposal intake."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .equation_universe import SCHEMA_VERSION as EQUATION_UNIVERSE_SCHEMA
from .equation_universe import canonicalize_record

CONFIG_SCHEMA = "sigma-kastner-schlatter-equation-graph-admission-config-1.0"
RESULT_SCHEMA = "sigma-kastner-schlatter-equation-graph-admission-1.0"
FIRST_BLOCKER = "no_candidate_bound_fundamental_action_or_complete_variational_field_system"
SOURCE_CONTENT_SHA256 = "560caa71caacd5172ff170d6619f77c99b878c019431c5a1a82982db50117c37"
SOURCE_FILE_SHA256 = "4c142f202cc30a39ad62039ae01355b91e9264260ec0ec4fd02f45a3a16f82e2"
SOURCE_PDF_SHA256 = "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _bound_path(root: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"}:
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    value = _load(_bound_path(root, binding, label))
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "campaign_implementation",
        "source_intake_artifact",
        "source_intake_config",
        "source_intake_implementation",
        "admission_policy",
        "budget",
        "data_seals",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("equation-graph admission config is invalid")
    if config["admission_policy"] != {
        "equation_only": True,
        "exact_duplicates_admitted": False,
        "semantic_equivalence_is_not_theory_equivalence": True,
        "absent_action_edges_required": True,
        "observational_execution": False,
        "cuda_execution": False,
    }:
        raise ValueError("equation-graph admission policy changed")
    if config["budget"] != {
        "maximum_formula_nodes": 25,
        "maximum_graph_nodes": 64,
        "maximum_graph_edges": 160,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("equation-graph admission budget changed")
    if config["data_seals"] != {
        "observational_data_opened": False,
        "dark_matter_or_halo_data_opened": False,
        "redshift_or_cosmology_data_opened": False,
        "solar_system_data_opened": False,
    }:
        raise ValueError("equation-graph data seals changed")
    if config["external_paid_llm_calls"] is not False:
        raise ValueError("equation-graph admission opened paid LLM calls")


def _v(
    symbol: str,
    meaning: str,
    dimension: dict[str, int] | None,
    *,
    canonical_name: str | None = None,
    field_kind: str = "scalar",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "canonical_name": canonical_name or symbol,
        "meaning": meaning,
        "dimension": dimension,
        "field_kind": field_kind,
        "tensor_rank": 0,
    }


DIMENSIONLESS: dict[str, int] = {}
LENGTH = {"L": 1}
MASS = {"M": 1}
TIME = {"T": 1}
SPEED = {"L": 1, "T": -1}
ACCELERATION = {"L": 1, "T": -2}
PRESSURE = {"M": 1, "L": -1, "T": -2}
ACTION = {"M": 1, "L": 2, "T": -1}
NEWTON_G = {"M": -1, "L": 3, "T": -2}
FOUR_RATE = {"L": -4}
INV_LENGTH2 = {"L": -2}


ASSUMPTIONS = {
    "constant_poisson_four_rate": "constant average transaction rate per four-volume",
    "poisson_count_domain": "mu >= 0 and n is a nonnegative integer",
    "local_inertial_frames": "local inertial frames around every spacetime point",
    "known_tensor_transformations": "known tensor transformation rules",
    "homogeneous_local_energy": "energy is homogeneous near the local origin",
    "repulsive_pressure_sign": "transaction pressure uses the paper's repulsive sign",
    "static_sds_chart": "static Schwarzschild-de Sitter chart and positive radii",
    "positive_effective_entropy": "effective de Sitter entropy is positive",
    "weak_potential_first_order": "first-order weak-potential entropy construction",
    "outer_radius_branch": "r is above r0=sqrt(M*G/a0)",
    "deep_mond_approximation": "the equation-68 square-root-dominance approximation holds",
    "circular_motion": "effective radial acceleration equals v**2/r",
}

DOMAINS = {
    "transaction_stochastic_process",
    "transaction_cosmological_identification",
    "local_trace_reversed_gravity",
    "schwarzschild_de_sitter",
    "effective_entropy",
    "galaxy_acceleration_approximation",
}


def _record(
    equation_id: str,
    name: str,
    domain: str,
    representation: str,
    expression: str,
    variables: list[dict[str, Any]],
    assumptions: list[str],
    locator: str,
    *,
    latex: str | None = None,
    tags: list[str] | None = None,
    source_id: str = "SRC-KS-2209.04025V1",
) -> dict[str, Any]:
    return {
        "equation_id": equation_id,
        "name": name,
        "domain": domain,
        "representation": representation,
        "expression": expression,
        "latex": latex,
        "variables": variables,
        "assumptions": assumptions,
        "validity": ["equation-only proposal intake; no theory validation inference"],
        "tags": ["kastner_schlatter", "equation_only", *(tags or [])],
        "source_id": source_id,
        "source_locator": locator,
        "independently_encoded": True,
    }


def _formula_records() -> list[dict[str, Any]]:
    c = _v("c", "speed of light", SPEED, field_kind="constant")
    g_newton = _v("G", "Newton gravitational constant", NEWTON_G, field_kind="constant")
    mass = _v("M", "central mass", MASS)
    a0 = _v("a0", "paper acceleration scale", ACCELERATION)
    radius = _v("r", "radial coordinate", LENGTH)
    entropy = _v("S", "entropy-like quantity", DIMENSIONLESS)
    deficit = _v("D", "absolute effective entropy deficit", DIMENSIONLESS)
    abar = _v("abar", "effective acceleration", ACCELERATION)
    pull = _v("g", "gravitative pull", ACCELERATION)
    return [
        _record(
            "EQ-KS-POISSON-PMF-IMPLEMENTATION",
            "Standard Poisson PMF implementation",
            "transaction_stochastic_process",
            "scalar_sympy",
            "p_n = exp(-mu)*mu**n/fact_n",
            [
                _v("p_n", "count probability", DIMENSIONLESS),
                _v("mu", "Poisson mean", DIMENSIONLESS),
                _v("n", "count", DIMENSIONLESS),
                _v("fact_n", "n factorial", DIMENSIONLESS),
            ],
            ["poisson_count_domain"],
            "p.9 Poisson assertion; standard PMF is implementation-only",
            tags=["implementation_not_printed_equation"],
        ),
        _record(
            "EQ-KS-33-FOUR-RATE",
            "Average transaction-density derivative",
            "transaction_stochastic_process",
            "scalar_sympy",
            "d_lambda_dx0 = q_gamma",
            [
                _v("d_lambda_dx0", "average density derivative", FOUR_RATE),
                _v("q_gamma", "transaction rate per four-volume", FOUR_RATE),
            ],
            ["constant_poisson_four_rate"],
            "p.9 equation (33) following text",
        ),
        _record(
            "EQ-KS-34-PRESSURE",
            "Transaction pressure",
            "transaction_cosmological_identification",
            "scalar_sympy",
            "Pbar = -c*h*q_gamma",
            [
                _v("Pbar", "average transaction pressure", PRESSURE),
                c,
                _v("h", "Planck constant as printed", ACTION, field_kind="constant"),
                _v("q_gamma", "transaction rate per four-volume", FOUR_RATE),
            ],
            ["constant_poisson_four_rate", "repulsive_pressure_sign"],
            "p.9 equation (34)",
        ),
        _record(
            "EQ-KS-35-LAMBDA-PRESSURE",
            "Cosmological term from transaction pressure",
            "transaction_cosmological_identification",
            "scalar_sympy",
            "Lambda = -4*pi*G*Pbar/c**4",
            [
                _v("Lambda", "cosmological term", INV_LENGTH2),
                g_newton,
                _v("Pbar", "average transaction pressure", PRESSURE),
                c,
            ],
            ["repulsive_pressure_sign"],
            "p.9 equation (35), first equality",
        ),
        _record(
            "EQ-KS-35-LAMBDA-RATE",
            "Cosmological term from transaction rate",
            "transaction_cosmological_identification",
            "scalar_sympy",
            "Lambda = 4*pi*G*h*q_gamma/c**3",
            [
                _v("Lambda", "cosmological term", INV_LENGTH2),
                g_newton,
                _v("h", "Planck constant as printed", ACTION, field_kind="constant"),
                _v("q_gamma", "transaction rate per four-volume", FOUR_RATE),
                c,
            ],
            ["constant_poisson_four_rate"],
            "p.9 equation (35), middle equality",
        ),
        _record(
            "EQ-KS-35-LAMBDA-PLANCK",
            "Printed Planck-length cosmological term",
            "transaction_cosmological_identification",
            "scalar_sympy",
            "Lambda = 4*pi**2*lP**2*q_gamma",
            [
                _v("Lambda", "cosmological term", INV_LENGTH2),
                _v("lP", "Planck length", LENGTH, field_kind="constant"),
                _v("q_gamma", "transaction rate per four-volume", FOUR_RATE),
            ],
            ["constant_poisson_four_rate"],
            "p.9 equation (35), printed final equality; normalization blocked in intake",
            tags=["normalization_clarification_required"],
        ),
        _record(
            "EQ-KS-38-LOCAL-00",
            "Local trace-reversed zero component",
            "local_trace_reversed_gravity",
            "tensor_dsl",
            "R_00 + Lambda delta_00 = (8 pi G/c^4) (T_00 - (1/2) T delta_00)",
            [],
            ["local_inertial_frames", "homogeneous_local_energy"],
            "p.10 equation (38)",
        ),
        _record(
            "EQ-KS-39-TRACE-REVERSED",
            "Full trace-reversed Einstein-form equation",
            "local_trace_reversed_gravity",
            "tensor_dsl",
            "R_mn + Lambda g_mn = (8 pi G/c^4) (T_mn - (1/2) T g_mn)",
            [],
            ["local_inertial_frames", "known_tensor_transformations"],
            "p.10 equation (39)",
            tags=["successive_generalization_not_action_equivalence"],
        ),
        _record(
            "EQ-KS-42-SDS-METRIC",
            "Schwarzschild-de Sitter line element",
            "schwarzschild_de_sitter",
            "tensor_dsl",
            "ds2 = f(r)c2dt2 - inv(f(r))dr2 - r2dOmega2",
            [],
            ["static_sds_chart"],
            "p.11 equation (42), line element",
        ),
        _record(
            "EQ-KS-42-METRIC-FACTOR",
            "Schwarzschild-de Sitter metric factor",
            "schwarzschild_de_sitter",
            "scalar_sympy",
            "f = 1-r**2/R0**2-Rs/r",
            [
                _v("f", "static metric factor", DIMENSIONLESS),
                radius,
                _v("R0", "de Sitter horizon radius", LENGTH),
                _v("Rs", "Schwarzschild radius", LENGTH),
            ],
            ["static_sds_chart"],
            "p.11 equation (42), metric factor",
        ),
        _record(
            "EQ-KS-42-R0",
            "de Sitter radius",
            "schwarzschild_de_sitter",
            "scalar_sympy",
            "R0 = sqrt(3/Lambda)",
            [
                _v("R0", "de Sitter horizon radius", LENGTH),
                _v("Lambda", "cosmological term", INV_LENGTH2),
            ],
            ["static_sds_chart"],
            "p.11 equation (42), definition after line element",
        ),
        _record(
            "EQ-KS-42-RS",
            "Schwarzschild radius",
            "schwarzschild_de_sitter",
            "scalar_sympy",
            "Rs = 2*G*M/c**2",
            [_v("Rs", "Schwarzschild radius", LENGTH), g_newton, mass, c],
            ["static_sds_chart"],
            "p.11 text immediately before equation (42)",
        ),
        _record(
            "EQ-KS-44-AINF-H0",
            "de Sitter horizon acceleration from Hubble constant",
            "schwarzschild_de_sitter",
            "scalar_sympy",
            "a_inf = c*H0",
            [
                _v("a_inf", "horizon acceleration", ACCELERATION),
                c,
                _v("H0", "Hubble constant", {"T": -1}),
            ],
            ["static_sds_chart"],
            "p.11 equation (44), first equality",
        ),
        _record(
            "EQ-KS-44-AINF-R0",
            "de Sitter horizon acceleration from radius",
            "schwarzschild_de_sitter",
            "scalar_sympy",
            "a_inf = c**2/R0",
            [
                _v("a_inf", "horizon acceleration", ACCELERATION),
                c,
                _v("R0", "de Sitter horizon radius", LENGTH),
            ],
            ["static_sds_chart"],
            "p.11 equation (44), second equality",
        ),
        _record(
            "EQ-KS-44-AINF-LAMBDA",
            "de Sitter horizon acceleration from Lambda",
            "schwarzschild_de_sitter",
            "scalar_sympy",
            "a_inf = c**2*sqrt(Lambda/3)",
            [
                _v("a_inf", "horizon acceleration", ACCELERATION),
                c,
                _v("Lambda", "cosmological term", INV_LENGTH2),
            ],
            ["static_sds_chart"],
            "p.11 equation (44), third equality",
        ),
        _record(
            "EQ-KS-A0-DEFINITION",
            "Paper acceleration-scale definition",
            "schwarzschild_de_sitter",
            "scalar_sympy",
            "a0 = a_inf/2",
            [a0, _v("a_inf", "horizon acceleration", ACCELERATION)],
            ["static_sds_chart"],
            "p.11 text following equation (44)",
        ),
        _record(
            "EQ-KS-45-DE-SITTER-ACCELERATION",
            "Hypothetical de Sitter radial acceleration",
            "schwarzschild_de_sitter",
            "scalar_sympy",
            "a = r*a0/R0",
            [
                _v("a", "hypothetical acceleration", ACCELERATION),
                radius,
                a0,
                _v("R0", "de Sitter horizon radius", LENGTH),
            ],
            ["static_sds_chart"],
            "p.11 equation (45); paper calls this hypothetical in empty space",
        ),
        _record(
            "EQ-KS-55-EFFECTIVE-ENTROPY",
            "Effective de Sitter entropy definition",
            "effective_entropy",
            "latex_only",
            "Sbar_dS(r)=S_dS(r)+DeltaS_dS(r)",
            [],
            ["weak_potential_first_order", "positive_effective_entropy"],
            "p.13 equation (55); full printed right side retained in intake contract",
        ),
        _record(
            "EQ-KS-59-ENTROPY-QUADRATIC",
            "Effective acceleration entropy quadratic",
            "effective_entropy",
            "scalar_sympy",
            "abar**2*S/a0 = abar*D+g*S",
            [abar, entropy, a0, deficit, pull],
            ["positive_effective_entropy", "outer_radius_branch"],
            "p.13 equation (59), D=Abs(DeltaS_dS)",
            latex=r"(\bar a^2/a_0)S-\bar aD-gS=0",
        ),
        _record(
            "EQ-KS-60-POSITIVE-ROOT",
            "Positive branch of the entropy quadratic",
            "effective_entropy",
            "scalar_sympy",
            "abar = a0*(D/S+sqrt((D/S)**2+4*g/a0))/2",
            [abar, a0, deficit, entropy, pull],
            ["positive_effective_entropy", "outer_radius_branch"],
            "p.13 equation (60), D=Abs(DeltaS_dS)",
        ),
        _record(
            "EQ-KS-62-SDS-MOND",
            "Schwarzschild-de Sitter effective acceleration",
            "galaxy_acceleration_approximation",
            "scalar_sympy",
            "abar = a0*Phi_abs*(1+sqrt(1+c**4/(M*G*a0)))",
            [
                abar,
                a0,
                _v("Phi_abs", "absolute dimensionless potential", DIMENSIONLESS),
                c,
                mass,
                g_newton,
            ],
            ["weak_potential_first_order", "outer_radius_branch"],
            "p.14 equation (62)",
        ),
        _record(
            "EQ-KS-65-LIMIT-RADIUS",
            "Effective-entropy limit radius",
            "galaxy_acceleration_approximation",
            "scalar_sympy",
            "r0 = sqrt(M*G/a0)",
            [_v("r0", "limit radius", LENGTH), mass, g_newton, a0],
            ["positive_effective_entropy"],
            "p.14 equation (65)",
        ),
        _record(
            "EQ-KS-68-DEEP-ACCELERATION",
            "Paper MOND-like outer acceleration approximation",
            "galaxy_acceleration_approximation",
            "scalar_sympy",
            "abar = sqrt(M*G*a0)/r",
            [abar, mass, g_newton, a0, radius],
            ["outer_radius_branch", "deep_mond_approximation"],
            "p.14 equation (68)",
        ),
        _record(
            "EQ-KS-69-VELOCITY",
            "Paper asymptotic circular velocity relation",
            "galaxy_acceleration_approximation",
            "scalar_sympy",
            "v**2 = sqrt(M*G*a0)",
            [
                _v("v", "circular speed", SPEED),
                mass,
                g_newton,
                a0,
            ],
            ["deep_mond_approximation", "circular_motion"],
            "p.15 equation (69)",
        ),
        _record(
            "EQ-KS-59-REARRANGED-CONTROL",
            "Entropy quadratic rearrangement control",
            "effective_entropy",
            "scalar_sympy",
            "abar*D+g*S = abar**2*S/a0",
            [abar, entropy, a0, deficit, pull],
            ["positive_effective_entropy", "outer_radius_branch"],
            "internal exact rearrangement of p.13 equation (59)",
            tags=["equivalence_control", "not_additional_paper_equation"],
            source_id="SRC-SIGMA-INTERNAL-EQUIVALENCE-CONTROL",
        ),
    ]


DEPENDENCIES = [
    ("EQ-KS-34-PRESSURE", "EQ-KS-33-FOUR-RATE"),
    ("EQ-KS-35-LAMBDA-PRESSURE", "EQ-KS-34-PRESSURE"),
    ("EQ-KS-35-LAMBDA-RATE", "EQ-KS-34-PRESSURE"),
    ("EQ-KS-39-TRACE-REVERSED", "EQ-KS-38-LOCAL-00"),
    ("EQ-KS-42-SDS-METRIC", "EQ-KS-39-TRACE-REVERSED"),
    ("EQ-KS-42-SDS-METRIC", "EQ-KS-42-METRIC-FACTOR"),
    ("EQ-KS-44-AINF-R0", "EQ-KS-42-R0"),
    ("EQ-KS-44-AINF-LAMBDA", "EQ-KS-42-R0"),
    ("EQ-KS-A0-DEFINITION", "EQ-KS-44-AINF-R0"),
    ("EQ-KS-45-DE-SITTER-ACCELERATION", "EQ-KS-A0-DEFINITION"),
    ("EQ-KS-59-ENTROPY-QUADRATIC", "EQ-KS-55-EFFECTIVE-ENTROPY"),
    ("EQ-KS-60-POSITIVE-ROOT", "EQ-KS-59-ENTROPY-QUADRATIC"),
    ("EQ-KS-62-SDS-MOND", "EQ-KS-60-POSITIVE-ROOT"),
    ("EQ-KS-65-LIMIT-RADIUS", "EQ-KS-55-EFFECTIVE-ENTROPY"),
    ("EQ-KS-68-DEEP-ACCELERATION", "EQ-KS-62-SDS-MOND"),
    ("EQ-KS-68-DEEP-ACCELERATION", "EQ-KS-65-LIMIT-RADIUS"),
    ("EQ-KS-69-VELOCITY", "EQ-KS-68-DEEP-ACCELERATION"),
    ("EQ-KS-59-REARRANGED-CONTROL", "EQ-KS-59-ENTROPY-QUADRATIC"),
]

ABSENT_CAPABILITIES = [
    "fundamental_action",
    "closed_field_content",
    "variational_principle",
    "euler_lagrange_map",
    "boundary_terms",
    "formal_gr_equivalence",
    "dark_sector_elimination_proof",
    "observational_validation",
]


def _compile_formula_nodes() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _formula_records()
    if len(records) != 25 or len({record["equation_id"] for record in records}) != 25:
        raise ValueError("formula registry count or IDs changed")
    nodes = []
    exact_groups: dict[str, list[str]] = defaultdict(list)
    semantic_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        canonical = canonicalize_record(record)
        if canonical["dimension_status"] == "fail":
            raise ValueError(f"dimension audit failed for {record['equation_id']}")
        exact_payload = {
            key: value
            for key, value in record.items()
            if key not in {"equation_id", "name", "source_locator", "tags"}
        }
        exact_hash = _sha(exact_payload)
        exact_groups[exact_hash].append(record["equation_id"])
        semantic_groups[canonical["semantic_hash"]].append(record["equation_id"])
        nodes.append(
            {
                "node_id": record["equation_id"],
                "node_type": "formula",
                "equation_universe_schema": EQUATION_UNIVERSE_SCHEMA,
                "record": record,
                "record_sha256": _sha(record),
                "exact_formula_sha256": exact_hash,
                "normalized_expression": canonical["normalized_expression"],
                "semantic_hash": canonical["semantic_hash"],
                "structural_hash": canonical["structural_hash"],
                "dimension_status": canonical["dimension_status"],
                "dimension_detail": canonical["dimension_detail"],
            }
        )
    duplicate_groups = sorted(
        sorted(items) for items in exact_groups.values() if len(items) > 1
    )
    if duplicate_groups:
        raise ValueError("exact duplicate formula node admission blocked")
    semantic_only_groups = []
    for items in semantic_groups.values():
        if len(items) < 2:
            continue
        hashes = {next(node for node in nodes if node["node_id"] == item)["exact_formula_sha256"] for item in items}
        if len(hashes) > 1:
            semantic_only_groups.append(sorted(items))
    audit = {
        "exact_duplicate_groups": sorted(duplicate_groups),
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_nodes_suppressed": 0,
        "semantic_equivalence_not_exact_duplicate_groups": sorted(semantic_only_groups),
        "semantic_equivalence_not_exact_duplicate_group_count": len(semantic_only_groups),
        "theory_equivalence_edges": 0,
    }
    return nodes, audit


def _edge(edge_type: str, source: str, target: str, detail: str) -> dict[str, Any]:
    body = {"edge_type": edge_type, "source": source, "target": target, "detail": detail}
    return {"edge_id": f"EDGE-{_sha(body)[:24]}", **body}


def _build_graph(formulas: list[dict[str, Any]], duplicate_audit: dict[str, Any]):
    nodes = list(formulas)
    nodes.extend(
        {
            "node_id": f"ASSUME-{key.upper()}",
            "node_type": "assumption",
            "statement": value,
        }
        for key, value in sorted(ASSUMPTIONS.items())
    )
    nodes.extend(
        {"node_id": f"DOMAIN-{domain.upper()}", "node_type": "domain", "name": domain}
        for domain in sorted(DOMAINS)
    )
    nodes.extend(
        [
            {
                "node_id": "SOURCE-KS-2209.04025V1",
                "node_type": "primary_source",
                "arxiv_id": "2209.04025",
                "version": "v1",
                "pdf_sha256": SOURCE_PDF_SHA256,
            },
            {
                "node_id": "SOURCE-SIGMA-EQUIVALENCE-CONTROL",
                "node_type": "internal_source",
                "scope": "algebraic rearrangement control only",
            },
            {
                "node_id": "ACTION-CONTRACT-ABSENT",
                "node_type": "action_contract",
                "fundamental_action": None,
                "status": "absent",
            },
        ]
    )
    nodes.extend(
        {
            "node_id": f"ABSENT-{capability.upper()}",
            "node_type": "absent_capability",
            "capability": capability,
            "status": "not_registered_or_proven",
        }
        for capability in ABSENT_CAPABILITIES
    )
    edges: list[dict[str, Any]] = []
    for formula in formulas:
        record = formula["record"]
        source_node = (
            "SOURCE-SIGMA-EQUIVALENCE-CONTROL"
            if record["source_id"] == "SRC-SIGMA-INTERNAL-EQUIVALENCE-CONTROL"
            else "SOURCE-KS-2209.04025V1"
        )
        edges.append(_edge("sourced_from", formula["node_id"], source_node, record["source_locator"]))
        edges.append(
            _edge(
                "valid_in_domain",
                formula["node_id"],
                f"DOMAIN-{record['domain'].upper()}",
                "declared equation domain only",
            )
        )
        edges.append(
            _edge(
                "not_derived_from_action",
                formula["node_id"],
                "ACTION-CONTRACT-ABSENT",
                "intake provides equations but no fundamental variational action",
            )
        )
        for assumption in record["assumptions"]:
            edges.append(
                _edge(
                    "assumes",
                    formula["node_id"],
                    f"ASSUME-{assumption.upper()}",
                    "explicit proposal or implementation premise",
                )
            )
    for target, source in DEPENDENCIES:
        edges.append(_edge("depends_on", target, source, "declared formula lineage"))
    for group in duplicate_audit["semantic_equivalence_not_exact_duplicate_groups"]:
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                edges.append(
                    _edge(
                        "semantic_algebraic_equivalence",
                        left,
                        right,
                        "equal equation-universe semantic hash; not exact duplicate or theory equivalence",
                    )
                )
    for capability in ABSENT_CAPABILITIES:
        edges.append(
            _edge(
                "lacks",
                "ACTION-CONTRACT-ABSENT",
                f"ABSENT-{capability.upper()}",
                "explicit fail-closed absence edge",
            )
        )
    if len({node["node_id"] for node in nodes}) != len(nodes):
        raise ValueError("graph node IDs are not unique")
    if len({edge["edge_id"] for edge in edges}) != len(edges):
        raise ValueError("graph edge IDs are not unique")
    return nodes, edges


def _validate_source(source: dict[str, Any]) -> None:
    if source.get("content_sha256") != SOURCE_CONTENT_SHA256:
        raise ValueError("source intake content lineage changed")
    if source.get("primary_source", {}).get("pdf_sha256") != SOURCE_PDF_SHA256:
        raise ValueError("source primary PDF lineage changed")
    if source.get("decision") != "blocked" or source.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("source intake fail-closed decision changed")
    seals = source.get("claim_seals", {})
    if not seals or any(value is not False for value in seals.values()):
        raise ValueError("source intake claim seals opened")
    expected_contracts = {
        "transaction_poisson_rate_and_pressure",
        "standard_poisson_cuda_reference",
        "transaction_cosmological_term",
        "einstein_trace_reversed_recovery_scope",
        "schwarzschild_de_sitter_background",
        "sds_effective_entropy_quadratic",
        "sds_mond_galaxy_relation",
        "action_and_validation_absence_contract",
    }
    if {item.get("contract_id") for item in source.get("formula_contracts", [])} != expected_contracts:
        raise ValueError("source formula-contract lineage changed")


def _validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("equation-graph artifact schema changed")
    body = {key: item for key, item in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != _sha(body):
        raise ValueError("equation-graph artifact content hash mismatch")
    if result.get("decision") != "blocked" or result.get("first_blocker") != FIRST_BLOCKER:
        raise ValueError("equation-graph decision is not fail-closed")
    counts = result.get("graph_counts", {})
    graph = result.get("knowledge_graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if any(
        edge.get("edge_type") in {"theory_equivalent_to", "proves_gr_equivalence"}
        for edge in edges
    ):
        raise ValueError("forbidden theory-equivalence edge was admitted")
    expected_formula_nodes, expected_audit = _compile_formula_nodes()
    expected_nodes, expected_edges = _build_graph(expected_formula_nodes, expected_audit)
    if graph != {"nodes": expected_nodes, "edges": expected_edges}:
        raise ValueError("equation-graph records differ from the deterministic source-bound graph")
    if result.get("graph_sha256") != _sha(graph):
        raise ValueError("equation-graph registry hash changed")
    if len({node.get("node_id") for node in nodes}) != len(nodes):
        raise ValueError("equation-graph node IDs are not unique")
    if len({edge.get("edge_id") for edge in edges}) != len(edges):
        raise ValueError("equation-graph edge IDs are not unique")
    node_ids = {node["node_id"] for node in nodes}
    if any(edge.get("source") not in node_ids or edge.get("target") not in node_ids for edge in edges):
        raise ValueError("equation-graph edge endpoint is missing")
    if counts.get("nodes") != len(nodes) or counts.get("edges") != len(edges):
        raise ValueError("equation-graph aggregate counts changed")
    node_counts = Counter(node["node_type"] for node in nodes)
    edge_counts = Counter(edge["edge_type"] for edge in edges)
    expected_counts = {
        "nodes": len(nodes),
        "edges": len(edges),
        "formula_nodes": node_counts["formula"],
        "assumption_nodes": node_counts["assumption"],
        "domain_nodes": node_counts["domain"],
        "source_nodes": node_counts["primary_source"] + node_counts["internal_source"],
        "action_contract_nodes": node_counts["action_contract"],
        "absent_capability_nodes": node_counts["absent_capability"],
        "dependency_edges": edge_counts["depends_on"],
        "assumption_edges": edge_counts["assumes"],
        "semantic_algebraic_equivalence_edges": edge_counts[
            "semantic_algebraic_equivalence"
        ],
        "exact_duplicate_edges": edge_counts["exact_duplicate"],
        "theory_equivalence_edges": 0,
        "absent_action_edges": edge_counts["not_derived_from_action"] + edge_counts["lacks"],
    }
    if counts != expected_counts or counts.get("formula_nodes") != 25:
        raise ValueError("equation/theory equivalence partition changed")
    audit = result.get("duplicate_equivalence_audit", {})
    if audit.get("exact_duplicate_group_count") != 0:
        raise ValueError("exact duplicate was admitted")
    if audit.get("semantic_equivalence_not_exact_duplicate_group_count") != 1:
        raise ValueError("semantic equivalence control changed")
    expected_group = ["EQ-KS-59-ENTROPY-QUADRATIC", "EQ-KS-59-REARRANGED-CONTROL"]
    if audit.get("semantic_equivalence_not_exact_duplicate_groups") != [expected_group]:
        raise ValueError("semantic equivalence group changed")
    if any(edge["edge_type"] in {"theory_equivalent_to", "proves_gr_equivalence"} for edge in edges):
        raise ValueError("forbidden theory-equivalence edge was admitted")
    if result.get("admission_contract") != {
        "kind": "equation_universe_compatible_typed_knowledge_graph",
        "equation_universe_schema": EQUATION_UNIVERSE_SCHEMA,
        "equation_only": True,
        "fundamental_action": None,
        "variational_edges_present": False,
        "theory_equivalence_edges_present": False,
        "observational_edges_present": False,
    }:
        raise ValueError("equation-graph admission contract changed")
    expected_secondary = [
        "equation_35_h_vs_hbar_factor_normalization_clarification",
        "no_formal_transaction_process_to_lorentzian_continuum_derivation",
        "no_registered_global_or_initial_boundary_value_completion",
        "no_registered_observational_likelihood",
    ]
    if result.get("secondary_blockers") != expected_secondary:
        raise ValueError("equation-graph blocker ledger changed")
    claims = result.get("claim_seals", {})
    if not claims or any(value is not False for value in claims.values()):
        raise ValueError("equation-graph claim seal opened")
    expected_data_seals = {
        "observational_data_opened": False,
        "dark_matter_or_halo_data_opened": False,
        "redshift_or_cosmology_data_opened": False,
        "solar_system_data_opened": False,
    }
    if result.get("data_seals") != expected_data_seals:
        raise ValueError("equation-graph data seal changed")
    if result.get("external_paid_llm_calls") is not False:
        raise ValueError("equation-graph paid-call seal changed")
    root = Path(__file__).resolve().parents[2]
    lineage = result.get("source_lineage", {})
    if lineage != {
        "source_intake_content_sha256": SOURCE_CONTENT_SHA256,
        "source_intake_file_sha256": SOURCE_FILE_SHA256,
        "source_intake_config_sha256": _file_sha(
            root / "configs/kastner_schlatter_transactional_gravity_intake.json"
        ),
        "source_intake_implementation_sha256": _file_sha(
            root / "src/sigma_theory_compiler/kastner_schlatter_transactional_gravity_intake.py"
        ),
        "primary_pdf_sha256": SOURCE_PDF_SHA256,
        "bridge_config_sha256": _file_sha(
            root / "configs/kastner_schlatter_equation_graph_admission.json"
        ),
        "bridge_implementation_sha256": _file_sha(Path(__file__).resolve()),
    }:
        raise ValueError("equation-graph source lineage changed")


def build_admission(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Compile the reviewed intake into a typed, hash-bound equation knowledge graph."""

    _validate_config(config)
    source = _bound_json(root, config["source_intake_artifact"], "source intake artifact")
    _bound_path(root, config["source_intake_config"], "source intake config")
    _bound_path(root, config["source_intake_implementation"], "source intake implementation")
    _validate_source(source)
    formula_nodes, duplicate_audit = _compile_formula_nodes()
    nodes, edges = _build_graph(formula_nodes, duplicate_audit)
    if len(nodes) > config["budget"]["maximum_graph_nodes"]:
        raise ValueError("graph node budget exceeded")
    if len(edges) > config["budget"]["maximum_graph_edges"]:
        raise ValueError("graph edge budget exceeded")
    node_counts = Counter(node["node_type"] for node in nodes)
    edge_counts = Counter(edge["edge_type"] for edge in edges)
    config_path = root / "configs/kastner_schlatter_equation_graph_admission.json"
    implementation_path = Path(__file__).resolve()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_lineage": {
            "source_intake_content_sha256": SOURCE_CONTENT_SHA256,
            "source_intake_file_sha256": config["source_intake_artifact"]["file_sha256"],
            "source_intake_config_sha256": config["source_intake_config"]["file_sha256"],
            "source_intake_implementation_sha256": config["source_intake_implementation"][
                "file_sha256"
            ],
            "primary_pdf_sha256": SOURCE_PDF_SHA256,
            "bridge_config_sha256": _file_sha(config_path),
            "bridge_implementation_sha256": _file_sha(implementation_path),
        },
        "admission_contract": {
            "kind": "equation_universe_compatible_typed_knowledge_graph",
            "equation_universe_schema": EQUATION_UNIVERSE_SCHEMA,
            "equation_only": True,
            "fundamental_action": None,
            "variational_edges_present": False,
            "theory_equivalence_edges_present": False,
            "observational_edges_present": False,
        },
        "knowledge_graph": {"nodes": nodes, "edges": edges},
        "graph_counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "formula_nodes": node_counts["formula"],
            "assumption_nodes": node_counts["assumption"],
            "domain_nodes": node_counts["domain"],
            "source_nodes": node_counts["primary_source"] + node_counts["internal_source"],
            "action_contract_nodes": node_counts["action_contract"],
            "absent_capability_nodes": node_counts["absent_capability"],
            "dependency_edges": edge_counts["depends_on"],
            "assumption_edges": edge_counts["assumes"],
            "semantic_algebraic_equivalence_edges": edge_counts[
                "semantic_algebraic_equivalence"
            ],
            "exact_duplicate_edges": edge_counts["exact_duplicate"],
            "theory_equivalence_edges": 0,
            "absent_action_edges": edge_counts["not_derived_from_action"] + edge_counts["lacks"],
        },
        "duplicate_equivalence_audit": duplicate_audit,
        "decision": "blocked",
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "equation_35_h_vs_hbar_factor_normalization_clarification",
            "no_formal_transaction_process_to_lorentzian_continuum_derivation",
            "no_registered_global_or_initial_boundary_value_completion",
            "no_registered_observational_likelihood",
        ],
        "claim_seals": {
            "fundamental_action_registered": False,
            "variational_derivation_registered": False,
            "formal_gr_equivalence_proven": False,
            "dark_matter_elimination_proven": False,
            "dark_energy_elimination_proven": False,
            "observational_pass": False,
            "theory_validity_claimed": False,
            "automatic_downstream_enqueue_performed": False,
        },
        "data_seals": config["data_seals"],
        "external_paid_llm_calls": False,
    }
    result["graph_sha256"] = _sha(result["knowledge_graph"])
    result["content_sha256"] = _sha(result)
    _validate_result(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _load(args.config)
    root = Path(__file__).resolve().parents[2]
    result = build_admission(config, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
