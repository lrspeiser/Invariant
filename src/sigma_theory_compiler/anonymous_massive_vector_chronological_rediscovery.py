"""Chronological rediscovery benchmark for an anonymous massive rank-one field."""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import io
import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Self

CONFIG_SCHEMA = "sigma-anonymous-massive-vector-chronological-rediscovery-config-1.0"
RESULT_SCHEMA = "sigma-anonymous-massive-vector-chronological-rediscovery-1.0"
BENCHMARK_ID = "anonymous-massive-vector-chronological-rediscovery-001"
CONFIG_PATH = "configs/backgrounds/anonymous_massive_vector_chronological_rediscovery.json"
SOURCE_PATH = "src/sigma_theory_compiler/anonymous_massive_vector_chronological_rediscovery.py"
TEST_PATH = "tests/test_anonymous_massive_vector_chronological_rediscovery.py"
OUTPUT_PATH = (
    "runs/physics-language/anonymous-massive-vector-chronological-rediscovery/campaign.json"
)
PRIMITIVES = ("q0", "q1", "q2", "qm")
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "argparse",
    "ast",
    "builtins",
    "collections",
    "fractions",
    "hashlib",
    "io",
    "itertools",
    "json",
    "math",
    "pathlib",
    "re",
    "types",
    "typing",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("anonymous rediscovery path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "schema_version",
        "benchmark_id",
        "output_path",
        "anonymous_field_contract",
        "discovery_constraints",
        "equivalence_contract",
        "chronology_contract",
        "negative_controls",
        "policies",
    }:
        raise ValueError("anonymous rediscovery config keys changed")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["benchmark_id"] != BENCHMARK_ID
        or value["output_path"] != OUTPUT_PATH
        or value["anonymous_field_contract"]
        != {
            "field_rank": 1,
            "field_components": 4,
            "spacetime_dimension": 4,
            "background_signature": "minus_plus_plus_plus",
            "coefficient_domain": [-2, -1, 0, 1, 2],
            "primitive_order": list(PRIMITIVES),
            "primitives": {
                "q0": "metric contraction of first-jet slots (mu,nu) with (mu,nu)",
                "q1": "metric contraction of first-jet slots (mu,nu) with (nu,mu)",
                "q2": "product of the two contracted first-jet traces",
                "qm": "metric contraction of the two undifferentiated rank-one fields",
            },
            "quadratic_locality": True,
            "maximum_action_derivative_count": 2,
            "metric_coupling": (
                "one universal metric contracts every spacetime index; no preferred tensor"
            ),
        }
        or value["discovery_constraints"]
        != {
            "massive": True,
            "physical_degrees_of_freedom": 3,
            "algebraic_divergence_constraint": True,
            "positive_spatial_kinetic_energy": True,
            "positive_mass_squared": True,
        }
        or value["equivalence_contract"]
        != {
            "positive_overall_normalization": True,
            "flat_quadratic_integration_by_parts": True,
            "rank_one_field_sign_relabel": True,
            "no_other_equivalences": True,
        }
        or value["chronology_contract"]
        != {
            "seal_generation_inputs_before_enumeration": True,
            "audit_pre_unseal_dependency_closure": True,
            "enforce_pre_unseal_file_reads": True,
            "require_denied_post_unseal_verifier_probe": True,
            "seal_blinded_pareto_ranking_before_unseal": True,
            "post_unseal_reference_verification_only": True,
        }
        or value["negative_controls"]
        != [
            "kinetic_sign_reversal",
            "zero_algebraic_scale",
            "propagating_longitudinal_mode",
            "four_derivative_intrusion",
        ]
        or value["policies"]
        != {
            "candidate_generation": "deterministic_complete_bounded_cartesian_enumeration",
            "scoring": "derive_constraints_without_reference_action",
            "leakage": "fail_closed",
            "observational_claims": "forbidden",
            "novelty_claims": "forbidden",
        }
    ):
        raise ValueError("anonymous rediscovery config boundary changed")


class _PreUnsealReadGuard:
    def __init__(self, root: Path, allowed_paths: Sequence[str]) -> None:
        self.root = root.resolve()
        self.allowed_paths = tuple(allowed_paths)
        self._allowed = set(allowed_paths)
        self.accesses: list[dict[str, Any]] = []
        self._original_builtin_open = builtins.open
        self._original_io_open = io.open
        self._original_path_open = Path.open

    def _relative(self, file: Any) -> str:
        if isinstance(file, int):
            return f"<file_descriptor:{file}>"
        path = Path(file).resolve()
        if path == self.root:
            return "."
        if self.root not in path.parents:
            return path.as_posix()
        return path.relative_to(self.root).as_posix()

    def _check(self, file: Any, mode: str, surface: str) -> None:
        relative = self._relative(file)
        read_only = not any(token in mode for token in ("w", "a", "x", "+"))
        allowed = relative in self._allowed and read_only
        self.accesses.append(
            {
                "sequence": len(self.accesses),
                "path": relative,
                "mode": mode,
                "surface": surface,
                "decision": "allowed" if allowed else "denied",
            }
        )
        if not allowed:
            raise PermissionError(f"pre-unseal file access denied: {relative}")

    def _builtin_open(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        self._check(file, mode, "builtins.open")
        return self._original_builtin_open(file, mode, *args, **kwargs)

    def _io_open(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        self._check(file, mode, "io.open")
        return self._original_io_open(file, mode, *args, **kwargs)

    def _path_open(
        self,
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> Any:
        self._check(path, mode, "pathlib.Path.open")
        return self._original_io_open(path, mode, buffering, encoding, errors, newline)

    def __enter__(self) -> Self:
        def guarded_path_open(
            path: Path,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> Any:
            return self._path_open(path, mode, buffering, encoding, errors, newline)

        builtins.open = self._builtin_open
        io.open = self._io_open
        Path.open = guarded_path_open  # type: ignore[method-assign]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        Path.open = self._original_path_open  # type: ignore[method-assign]
        io.open = self._original_io_open
        builtins.open = self._original_builtin_open

    def certificate(self) -> dict[str, Any]:
        allowed = [row for row in self.accesses if row["decision"] == "allowed"]
        denied = [row for row in self.accesses if row["decision"] == "denied"]
        return {
            "enforcement_surfaces": ["builtins.open", "io.open", "pathlib.Path.open"],
            "enforcement_scope": (
                "owned_single_threaded_python_file_read_surfaces_with_static_import_allowlist_"
                "not_an_operating_system_sandbox"
            ),
            "allowed_paths": list(self.allowed_paths),
            "attempted_accesses": self.accesses,
            "attempted_access_count": len(self.accesses),
            "allowed_access_count": len(allowed),
            "denied_access_count": len(denied),
            "denied_paths": sorted({row["path"] for row in denied}),
            "denied_content_bytes_exposed": 0,
            "single_threaded_phase": True,
        }


def _forbidden_concepts() -> dict[str, str]:
    fragments = {
        "named_theory": ("pr", "oca"),
        "named_operator_square": ("f", "2"),
        "named_vector_square": ("a", "2"),
        "answer_role": ("known", "answer"),
        "answer_adapter": ("action", "health"),
        "variation_adapter": ("covariant", "variation"),
        "reference_equivalence": ("expected", "equivalence"),
        "reference_coefficients": ("target", "coefficients"),
        "reference_digest": ("target", "hash"),
        "reference_identifier": ("target", "action", "id"),
    }
    return {label: "[_ -]?".join(parts) for label, parts in fragments.items()}


def _import_roots(source_text: str) -> list[str]:
    tree = ast.parse(source_text)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return sorted(roots)


def _leakage_audit(root: Path) -> dict[str, Any]:
    paths = [CONFIG_PATH, SOURCE_PATH]
    matches: list[dict[str, str]] = []
    total_bytes = 0
    for relative in paths:
        raw = _inside(root, relative).read_bytes()
        total_bytes += len(raw)
        text = raw.decode("utf-8").lower()
        for label, pattern in _forbidden_concepts().items():
            if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text):
                matches.append({"concept_id": label, "path": relative})
    source_text = _inside(root, SOURCE_PATH).read_text(encoding="utf-8")
    imports = _import_roots(source_text)
    unexpected_imports = sorted(set(imports) - ALLOWED_IMPORT_ROOTS)
    result = {
        "pre_unseal_dependency_paths": paths,
        "pre_unseal_dependency_root_sha256": _sha(
            [{"path": path, "file_sha256": _file_sha(_inside(root, path))} for path in paths]
        ),
        "bytes_scanned": total_bytes,
        "forbidden_concept_count": len(matches),
        "forbidden_concept_matches": matches,
        "import_roots": imports,
        "unexpected_import_roots": unexpected_imports,
        "passed": not matches and not unexpected_imports,
    }
    if not result["passed"]:
        raise ValueError("anonymous rediscovery leakage audit failed")
    return result


def _canonical_signature(raw: Sequence[int]) -> tuple[int, int, int]:
    q0, q1, q2, qm = raw
    reduced = (q0, q1 + q2, qm)
    divisor = math.gcd(math.gcd(abs(reduced[0]), abs(reduced[1])), abs(reduced[2]))
    if divisor == 0:
        return reduced
    return tuple(value // divisor for value in reduced)


def _rank(diagonal: Sequence[int]) -> int:
    return sum(value != 0 for value in diagonal)


def _canonical_constraint_certificate(signature: tuple[int, int, int]) -> dict[str, Any]:
    q0, qcross, qm = signature
    velocity_hessian = [q0 + qcross, -q0, -q0, -q0]
    hessian_rank = _rank(velocity_hessian)
    primary_exists = q0 + qcross == 0 and q0 != 0
    secondary_exists = primary_exists
    bracket_laplacian = q0 + qcross if primary_exists else None
    bracket_algebraic = -qm if primary_exists else None
    bracket_determinant = qm * qm if primary_exists else None
    if primary_exists and bracket_determinant:
        first_class = 0
        second_class = 2
        constraint_class = "second_class_pair"
    elif primary_exists and qm == 0:
        first_class = 2
        second_class = 0
        constraint_class = "first_class_pair"
    elif hessian_rank == 4:
        first_class = 0
        second_class = 0
        constraint_class = "no_primary_velocity_constraint"
    else:
        first_class = 0
        second_class = 0
        constraint_class = "degenerate_unclassified"
    phase_space_dimension = 8
    dof_numerator = phase_space_dimension - 2 * first_class - second_class
    dof = (
        dof_numerator // 2
        if constraint_class != "degenerate_unclassified"
        and dof_numerator >= 0
        and dof_numerator % 2 == 0
        else None
    )
    return {
        "canonical_momenta": {
            "pi0_velocity_coefficient": q0 + qcross,
            "pi0_spatial_divergence_coefficient": -qcross,
            "pi_spatial_velocity_coefficient": -q0,
        },
        "primary_constraint": {
            "exists": primary_exists,
            "pi0_coefficient": 1 if primary_exists else None,
            "spatial_divergence_coefficient": qcross if primary_exists else None,
        },
        "secondary_constraint": {
            "exists": secondary_exists,
            "spatial_momentum_divergence_coefficient": -1 if secondary_exists else None,
            "laplacian_time_component_coefficient": -q0 if secondary_exists else None,
            "algebraic_time_component_coefficient": qm if secondary_exists else None,
        },
        "primary_secondary_poisson_bracket": {
            "laplacian_delta_coefficient": bracket_laplacian,
            "algebraic_delta_coefficient": bracket_algebraic,
            "two_constraint_matrix_determinant": bracket_determinant,
            "nonzero": bracket_determinant not in (None, 0),
        },
        "velocity_hessian_diagonal": velocity_hessian,
        "velocity_hessian_rank": hessian_rank,
        "constraint_class": constraint_class,
        "first_class_constraint_count": first_class,
        "second_class_constraint_count": second_class,
        "phase_space_dimension": phase_space_dimension,
        "physical_degrees_of_freedom": dof,
    }


def _derive(signature: tuple[int, int, int], *, action_derivative_count: int = 2) -> dict[str, Any]:
    q0, qcross, qm = signature
    divergence_wave = -(q0 + qcross)
    hamiltonian = _canonical_constraint_certificate(signature)
    hessian = hamiltonian["velocity_hessian_diagonal"]
    hessian_rank = hamiltonian["velocity_hessian_rank"]
    algebraic_divergence = divergence_wave == 0 and qm != 0
    mass_ratio = Fraction(qm, q0) if q0 else None
    positive_mass = mass_ratio is not None and mass_ratio > 0
    positive_kinetic = -q0 > 0
    dof = hamiltonian["physical_degrees_of_freedom"]
    violations = {
        "derivative_bound": action_derivative_count > 2,
        "dynamical_spatial_sector": q0 == 0,
        "algebraic_divergence": not algebraic_divergence,
        "three_degrees_of_freedom": dof != 3,
        "positive_spatial_kinetic": not positive_kinetic,
        "positive_mass_squared": not positive_mass,
    }
    return {
        "canonical_coefficients": {"q0": q0, "qcross": qcross, "qm": qm},
        "euler_operator_coefficients": {
            "box_rank_one_field": -q0,
            "gradient_of_divergence": -qcross,
            "algebraic_rank_one_field": qm,
        },
        "divergence_operator_coefficients": {
            "box_of_divergence": divergence_wave,
            "algebraic_divergence": qm,
        },
        "velocity_hessian_diagonal": hessian,
        "velocity_hessian_rank": hessian_rank,
        "canonical_hamiltonian_derivation": hamiltonian,
        "constraint_class": hamiltonian["constraint_class"],
        "first_class_constraint_count": hamiltonian["first_class_constraint_count"],
        "second_class_constraint_count": hamiltonian["second_class_constraint_count"],
        "phase_space_dimension": hamiltonian["phase_space_dimension"],
        "physical_degrees_of_freedom": dof,
        "mass_squared": (
            None
            if mass_ratio is None
            else {"numerator": mass_ratio.numerator, "denominator": mass_ratio.denominator}
        ),
        "maximum_action_derivative_count": action_derivative_count,
        "violations": violations,
        "eligible": not any(violations.values()),
    }


def _candidate_id(signature: tuple[int, int, int]) -> str:
    return "AVC-" + _sha({"canonical_signature": list(signature)})[:16]


def _enumerate(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    domain = config["anonymous_field_contract"]["coefficient_domain"]
    raw_candidates = list(itertools.product(domain, repeat=len(PRIMITIVES)))
    multiplicities: dict[tuple[int, int, int], int] = {}
    for raw in raw_candidates:
        signature = _canonical_signature(raw)
        multiplicities[signature] = multiplicities.get(signature, 0) + 1
    candidates = []
    for signature in sorted(multiplicities):
        derived = _derive(signature)
        support = sum(value != 0 for value in signature)
        mass = derived["mass_squared"]
        mass_complexity = 0 if mass is None else abs(mass["numerator"]) + mass["denominator"]
        candidates.append(
            {
                "candidate_id": _candidate_id(signature),
                "signature": signature,
                "raw_orbit_multiplicity": multiplicities[signature],
                "derived": derived,
                "objectives": [support, sum(abs(value) for value in signature), mass_complexity],
            }
        )
    return candidates, {
        "raw_cartesian_candidates": len(raw_candidates),
        "canonical_equivalence_classes": len(candidates),
        "raw_orbit_multiplicity_sum": sum(multiplicities.values()),
        "integration_by_parts_reduction": "q1_plus_q2_to_qcross",
        "normalization": "greatest_common_divisor_positive_scale_only",
        "field_relabel_orbit_size": 1,
    }


def _dominates(left: Sequence[int], right: Sequence[int]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True)) and any(
        a < b for a, b in zip(left, right, strict=True)
    )


def _blinded_rank(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if candidate["derived"]["eligible"]]
    front = [
        candidate
        for candidate in eligible
        if not any(
            _dominates(other["objectives"], candidate["objectives"])
            for other in eligible
            if other["candidate_id"] != candidate["candidate_id"]
        )
    ]
    front.sort(key=lambda row: (row["objectives"], row["candidate_id"]))
    if len(front) != 1:
        raise ValueError("anonymous rediscovery Pareto front is not unique")
    blinded_rows = [
        {"candidate_id": row["candidate_id"], "objectives": row["objectives"]} for row in front
    ]
    return {
        "eligible_candidate_count": len(eligible),
        "pareto_front_size": len(front),
        "pareto_front": blinded_rows,
        "pareto_root_sha256": _sha(blinded_rows),
        "selected_candidate_id": front[0]["candidate_id"],
        "coefficient_visibility_before_unseal": False,
    }


def _negative_controls(selected: Mapping[str, Any]) -> list[dict[str, Any]]:
    signature = tuple(selected["signature"])
    controls = [
        ("kinetic_sign_reversal", tuple(-value for value in signature), 2),
        ("zero_algebraic_scale", (signature[0], signature[1], 0), 2),
        ("propagating_longitudinal_mode", (signature[0], 0, signature[2]), 2),
        ("four_derivative_intrusion", signature, 4),
    ]
    records = []
    for control_id, control_signature, derivative_count in controls:
        derived = _derive(control_signature, action_derivative_count=derivative_count)
        records.append(
            {
                "control_id": control_id,
                "eligible": derived["eligible"],
                "violated_constraints": sorted(
                    key for key, value in derived["violations"].items() if value
                ),
                "physical_degrees_of_freedom": derived["physical_degrees_of_freedom"],
                "maximum_action_derivative_count": derivative_count,
                "primary_constraint_exists": derived["canonical_hamiltonian_derivation"][
                    "primary_constraint"
                ]["exists"],
                "primary_secondary_poisson_bracket_determinant": derived[
                    "canonical_hamiltonian_derivation"
                ]["primary_secondary_poisson_bracket"]["two_constraint_matrix_determinant"],
                "constraint_class": derived["constraint_class"],
            }
        )
    if any(record["eligible"] for record in records):
        raise ValueError("anonymous rediscovery negative control admitted")
    return records


def _run_pre_unseal(root: Path) -> dict[str, Any]:
    allowed_paths = (CONFIG_PATH, SOURCE_PATH)
    guard = _PreUnsealReadGuard(root, allowed_paths)
    with guard:
        config = json.loads(_inside(root, CONFIG_PATH).read_text(encoding="utf-8"))
        _validate_config(config)
        input_bindings = [
            {"path": path, "file_sha256": _file_sha(_inside(root, path))} for path in allowed_paths
        ]
        leakage = _leakage_audit(root)
        try:
            _inside(root, TEST_PATH).read_bytes()
        except PermissionError:
            pass
        else:
            raise ValueError("post-unseal verifier denial probe unexpectedly succeeded")
        candidates, enumeration = _enumerate(config)
        ranking = _blinded_rank(candidates)
        selected = next(
            row for row in candidates if row["candidate_id"] == ranking["selected_candidate_id"]
        )
        controls = _negative_controls(selected)
    io_certificate = guard.certificate()
    if (
        io_certificate["denied_paths"] != [TEST_PATH]
        or io_certificate["denied_access_count"] != 1
        or io_certificate["denied_content_bytes_exposed"] != 0
    ):
        raise ValueError("pre-unseal file-read enforcement boundary changed")
    blinded_seal = _sha(
        {
            "input_root_sha256": _sha(input_bindings),
            "phase_io_contract_root_sha256": _sha(io_certificate),
            "leakage_root_sha256": _sha(leakage),
            "enumeration_root_sha256": _sha(enumeration),
            "pareto_root_sha256": ranking["pareto_root_sha256"],
        }
    )
    return {
        "config": config,
        "input_bindings": input_bindings,
        "io_certificate": io_certificate,
        "leakage": leakage,
        "candidates": candidates,
        "enumeration": enumeration,
        "ranking": ranking,
        "selected": selected,
        "controls": controls,
        "blinded_seal": blinded_seal,
    }


def _expected_body(root: Path) -> dict[str, Any]:
    pre_unseal = _run_pre_unseal(root)
    input_bindings = pre_unseal["input_bindings"]
    io_certificate = pre_unseal["io_certificate"]
    leakage = pre_unseal["leakage"]
    candidates = pre_unseal["candidates"]
    enumeration = pre_unseal["enumeration"]
    ranking = pre_unseal["ranking"]
    selected = pre_unseal["selected"]
    controls = pre_unseal["controls"]
    blinded_seal = pre_unseal["blinded_seal"]
    derived = selected["derived"]
    eligible = sorted(
        (row for row in candidates if row["derived"]["eligible"]),
        key=lambda row: (
            Fraction(
                row["derived"]["mass_squared"]["numerator"],
                row["derived"]["mass_squared"]["denominator"],
            ),
            row["candidate_id"],
        ),
    )
    eligible_representatives = [
        {
            "candidate_id": row["candidate_id"],
            "canonical_representative": row["derived"]["canonical_coefficients"],
            "mass_squared": row["derived"]["mass_squared"],
            "simplicity_objectives": row["objectives"],
            "selected_by_pareto": row["candidate_id"] == selected["candidate_id"],
        }
        for row in eligible
    ]
    unsealed = {
        "selected_candidate_id": selected["candidate_id"],
        "canonical_representative": derived["canonical_coefficients"],
        "raw_orbit_multiplicity": selected["raw_orbit_multiplicity"],
        "independent_euler_derivation": derived["euler_operator_coefficients"],
        "independent_divergence_derivation": derived["divergence_operator_coefficients"],
        "velocity_hessian_diagonal": derived["velocity_hessian_diagonal"],
        "velocity_hessian_rank": derived["velocity_hessian_rank"],
        "canonical_hamiltonian_derivation": derived["canonical_hamiltonian_derivation"],
        "constraint_class": derived["constraint_class"],
        "first_class_constraint_count": derived["first_class_constraint_count"],
        "second_class_constraint_count": derived["second_class_constraint_count"],
        "phase_space_dimension": derived["phase_space_dimension"],
        "physical_degrees_of_freedom": derived["physical_degrees_of_freedom"],
        "mass_squared": derived["mass_squared"],
        "discovered_structure": {
            "first_jet_coefficients_sum_to_zero": (
                derived["canonical_coefficients"]["q0"]
                + derived["canonical_coefficients"]["qcross"]
                == 0
            ),
            "algebraic_scale_nonzero": derived["canonical_coefficients"]["qm"] != 0,
            "positive_spatial_kinetic": not derived["violations"]["positive_spatial_kinetic"],
            "positive_mass_squared": not derived["violations"]["positive_mass_squared"],
        },
        "eligible_positive_mass_representatives": eligible_representatives,
        "eligible_mass_squared_values": [row["mass_squared"] for row in eligible_representatives],
        "selection_statement": (
            "unit_mass_representative_selected_by_declared_simplicity_objectives_from_three_"
            "eligible_positive_mass_ratios_not_unique_theory_or_free_mass_family"
        ),
        "reference_comparison_permitted_only_after_blinded_seal": True,
    }
    chronology = [
        {"sequence": 0, "phase": "generation_inputs_sealed", "root_sha256": _sha(input_bindings)},
        {
            "sequence": 1,
            "phase": "pre_unseal_file_reads_enforced",
            "root_sha256": _sha(io_certificate),
        },
        {"sequence": 2, "phase": "dependency_leakage_audited", "root_sha256": _sha(leakage)},
        {"sequence": 3, "phase": "bounded_grammar_enumerated", "root_sha256": _sha(enumeration)},
        {
            "sequence": 4,
            "phase": "euler_divergence_hamiltonian_and_constraint_scores_derived",
            "root_sha256": _sha(
                [
                    {"candidate_id": row["candidate_id"], "derived": row["derived"]}
                    for row in candidates
                ]
            ),
        },
        {"sequence": 5, "phase": "negative_controls_rejected", "root_sha256": _sha(controls)},
        {"sequence": 6, "phase": "blinded_pareto_ranking_sealed", "root_sha256": blinded_seal},
        {"sequence": 7, "phase": "selected_structure_unsealed", "root_sha256": _sha(unsealed)},
    ]
    return {
        "schema_version": RESULT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "decision": "pass_blinded_unit_mass_pareto_representative_in_declared_bounded_grammar",
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(_inside(root, CONFIG_PATH))},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
        },
        "pre_unseal_input_bindings": input_bindings,
        "pre_unseal_phase_io_contract": io_certificate,
        "pre_unseal_leakage_audit": leakage,
        "enumeration": enumeration,
        "derivation_contract": {
            "euler_equation_derived_from_each_canonical_quadratic_action": True,
            "divergence_equation_derived_from_euler_coefficients": True,
            "velocity_hessian_derived_independently": True,
            "canonical_momenta_derived": True,
            "primary_and_secondary_constraints_derived": True,
            "primary_secondary_poisson_bracket_determinant_derived": True,
            "constraint_class_and_phase_space_count_derived": True,
            "universal_metric_contractions_only": True,
            "integration_by_parts_scope": "flat_quadratic_only",
        },
        "negative_controls": controls,
        "blinded_pareto_ranking": ranking,
        "blinded_pre_unseal_root_sha256": blinded_seal,
        "unsealed_result": unsealed,
        "chronology": chronology,
        "claims": {
            "complete_for_declared_finite_coefficient_box": True,
            "unique_blinded_pareto_winner": True,
            "pre_unseal_file_reads_enforced": True,
            "three_positive_mass_representatives_exposed": True,
            "flat_integration_by_parts_only": True,
            "post_unseal_reference_equivalence_check_defined": True,
            "unique_massive_equivalence_class_proved": False,
            "free_mass_family_proved": False,
            "unbounded_coefficient_space_exhausted": False,
            "interacting_vector_theories_classified": False,
            "curvature_coupling_classes_classified": False,
            "novel_theory_discovered": False,
            "observational_support": False,
        },
        "first_remaining_blocker": (
            "extend_beyond_quadratic_two_derivative_single_rank_one_field_grammar_without_"
            "introducing_answer_bearing_generation_dependencies"
        ),
        "scope": (
            "chronological file-read-enforced and leak-audited rediscovery in one finite anonymous "
            "quadratic Lorentz-invariant rank-one grammar with flat-only integration-by-parts; "
            "selects one unit-mass simplicity representative from three eligible positive mass "
            "ratios rather than proving a unique theory, free mass family, novelty, interacting or "
            "curved nonminimal classification, or data support"
        ),
    }


def build_benchmark(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    if config_path != _inside(root, CONFIG_PATH):
        raise ValueError("anonymous rediscovery config path changed")
    body = _expected_body(root)
    return {**body, "content_sha256": _sha(body)}


def validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if set(value) != {
        "schema_version",
        "benchmark_id",
        "decision",
        "source_bindings",
        "pre_unseal_input_bindings",
        "pre_unseal_phase_io_contract",
        "pre_unseal_leakage_audit",
        "enumeration",
        "derivation_contract",
        "negative_controls",
        "blinded_pareto_ranking",
        "blinded_pre_unseal_root_sha256",
        "unsealed_result",
        "chronology",
        "claims",
        "first_remaining_blocker",
        "scope",
        "content_sha256",
    }:
        raise ValueError("anonymous rediscovery result keys changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("anonymous rediscovery content hash changed")
    expected = _expected_body(validation_root)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("anonymous rediscovery result boundary changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    arguments = parser.parse_args()
    result = build_benchmark(Path(arguments.config))
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_result(result, root=Path(arguments.config).resolve().parents[2])


if __name__ == "__main__":
    main()
