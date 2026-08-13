from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-readiness-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-readiness-config-1.0"
)
CHECKPOINT_SCHEMA = (
    "sigma-quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-checkpoint-1.0"
)
STATUS = "block_coordinate_free_D4_recurrence_emitter_missing_symbolic_P_and_Taylor_packets"
CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_readiness.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_readiness.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_readiness.py"

STATE_DIMENSION = 55
COKERNEL_COORDINATES = 558
ODD_SPHERE_MODES = 210
REQUIRED_ROWS = 117_180
SPECTRUM_TEXT = ["0", "1", "-1", "1/2", "-1/2", "1/3", "-1/3"]
SPECTRUM = [Fraction(value) for value in SPECTRUM_TEXT]
EIGENSPACE_RANKS = {
    "0": 33,
    "1": 3,
    "-1": 3,
    "1/2": 4,
    "-1/2": 4,
    "1/3": 4,
    "-1/3": 4,
}
EXPECTED_CAPS = {
    "maximum_projector_recipe_terms": 49,
    "maximum_required_input_manifest_records": 512,
    "maximum_output_row_descriptors": REQUIRED_ROWS,
    "maximum_emitted_sparse_entries": 2_000_000,
    "maximum_checkpoint_bytes": 16_777_216,
}
EXPECTED_UPSTREAMS = {
    "coefficient_map_registration_audit": {
        "content_sha256": ("1c3146fa32b309c700d6cce6647fc13d33e5ae6091b61b6d3af16264d03e4838"),
        "status": "block_coordinate_free_D4_sparse_coefficient_map_not_symbolically_emitted",
    },
    "full_sphere_readiness_gate": {
        "content_sha256": ("07ba08c8057823b03733d39bf8c2d1c04ce3d506d4dd4103c18d195943a1724b"),
        "status": ("block_full_sphere_degree_seven_exact_solve_coefficient_map_not_registered"),
    },
    "canonical_D4_obstruction_certificate": {
        "content_sha256": ("bef3246a17942c74e8f3cdb09bc14a36c6bdc44d030a9a70ce833c30ec04bc65"),
        "status": "pass_exact_canonical_d4_obstruction_cokernel_classification",
    },
    "rational_chart_counterexample_gate": {
        "content_sha256": ("48b8ecfe63336071721baeb90a41f379ac1b4235629b380abdf0124e7008152c"),
        "status": (
            "pass_exact_rational_chart_counterexample_disproves_current_full_sphere_"
            "D4_compatibility"
        ),
    },
    "reference_Sylvester_space": {
        "content_sha256": ("43e40ccbae728364121e63dc44945b2ca7fd6d733aab662bf04f6a1842193212"),
        "status": "pass_exact_d4_obstruction_invariant_under_all_lower_homogeneous_freedom",
    },
}

FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "complete_D2F_tensor_registered",
    "complete_coordinate_free_coefficient_map_emitted",
    "complete_coordinate_free_rhs_emitted",
    "covariant_action_origin_proved",
    "full_direction_sphere_D4_compatibility_proved",
    "full_high_atom_identity_proved",
    "full_tube_Sylvester_identity_proved",
    "global_H7_closed",
    "lifespan_proved",
    "local_differential_operator_origin_proved",
    "nonlinear_PDE_closure_proved",
    "phase_two_exact_solve_admitted",
    "theory_candidate_rejected",
    "unregistered_symbolic_inputs_treated_as_zero",
    "variable_coefficient_constraint_calculus_proved",
}

NEGATIVE_CONTROLS = {
    "emit_rows_from_point_samples": {"rejected": True},
    "evaluate_projectors_without_P_pencil_matrices": {"rejected": True},
    "infer_missing_Taylor_packet_as_zero": {"rejected": True},
    "omit_any_Lagrange_spectrum_factor": {"rejected": True},
    "omit_lower_recurrence_orders": {"rejected": True},
    "promote_projector_recipes_to_matrix_projectors": {"rejected": True},
    "promote_row_descriptors_to_emitted_coefficients": {"rejected": True},
    "run_phase_two_with_incomplete_input_manifest": {"rejected": True},
}


class SymbolicRecurrenceEmitterReadinessError(ValueError):
    """Raised when the symbolic emitter readiness record is inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SymbolicRecurrenceEmitterReadinessError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise SymbolicRecurrenceEmitterReadinessError("bound path escaped project root")
    return path


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def _poly_multiply(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    output = [Fraction(0) for _ in range(len(left) + len(right) - 1)]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            output[left_index + right_index] += left_value * right_value
    return output


def lagrange_projector_recipes(spectrum: list[Fraction]) -> list[dict[str, Any]]:
    """Return exact coefficients of Π_lambda(P)=prod_(mu!=lambda)(P-mu I)/(lambda-mu)."""

    if len(set(spectrum)) != len(spectrum):
        raise SymbolicRecurrenceEmitterReadinessError("projector spectrum is not distinct")
    recipes: list[dict[str, Any]] = []
    for eigenvalue in spectrum:
        coefficients = [Fraction(1)]
        denominator = Fraction(1)
        factors: list[str] = []
        for other in spectrum:
            if other == eigenvalue:
                continue
            coefficients = _poly_multiply(coefficients, [-other, Fraction(1)])
            denominator *= eigenvalue - other
            factors.append(_fraction_text(other))
        coefficients = [coefficient / denominator for coefficient in coefficients]
        evaluations = [
            sum(coefficient * point**power for power, coefficient in enumerate(coefficients))
            for point in spectrum
        ]
        expected = [Fraction(int(point == eigenvalue)) for point in spectrum]
        if evaluations != expected or len(coefficients) != len(spectrum):
            raise SymbolicRecurrenceEmitterReadinessError("exact Lagrange projector recipe failed")
        recipes.append(
            {
                "eigenvalue": _fraction_text(eigenvalue),
                "degree": len(spectrum) - 1,
                "excluded_spectrum_factors": factors,
                "denominator": _fraction_text(denominator),
                "coefficients_low_to_high": [
                    _fraction_text(coefficient) for coefficient in coefficients
                ],
                "Kronecker_evaluations_in_declared_spectrum_order": [
                    _fraction_text(value) for value in evaluations
                ],
            }
        )
    return recipes


def _matmul(left: list[list[Fraction]], right: list[list[Fraction]]) -> list[list[Fraction]]:
    if not left or not right or len(left[0]) != len(right):
        raise SymbolicRecurrenceEmitterReadinessError("small matrix shape mismatch")
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def _transpose(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    return [list(column) for column in zip(*matrix, strict=True)]


def _matrix_add(left: list[list[Fraction]], right: list[list[Fraction]]) -> list[list[Fraction]]:
    return [
        [left[row][column] + right[row][column] for column in range(len(left[0]))]
        for row in range(len(left))
    ]


def _small_model_controls() -> dict[str, Any]:
    two_spectrum = [Fraction(-1), Fraction(1)]
    recipes = lagrange_projector_recipes(two_spectrum)
    pencil = [[Fraction(-1), Fraction(0)], [Fraction(0), Fraction(1)]]
    rhs = [[Fraction(0), Fraction(1)], [Fraction(-1), Fraction(0)]]
    correction = [
        [Fraction(0), Fraction(-1, 2)],
        [Fraction(-1, 2), Fraction(0)],
    ]
    residual = _matrix_add(
        _matrix_add(_matmul(correction, pencil), rhs),
        [[-value for value in row] for row in _matmul(_transpose(pencil), correction)],
    )
    pass_zero = all(value == 0 for row in residual for value in row)

    repeated_pencil = [
        [Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1)],
    ]
    obstructed_rhs = [
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(-1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0)],
    ]
    zero_compression = [row[:2] for row in obstructed_rhs[:2]]
    obstruction_nonzero = any(value != 0 for row in zero_compression for value in row)
    if not pass_zero or not obstruction_nonzero or repeated_pencil[2][2] != 1:
        raise SymbolicRecurrenceEmitterReadinessError("small exact controls failed")
    return {
        "exact_projector_recipe_control": {
            "spectrum": ["-1", "1"],
            "recipes": recipes,
            "Kronecker_identities_verified": True,
        },
        "exact_solvable_Sylvester_control": {
            "pencil": [[_fraction_text(value) for value in row] for row in pencil],
            "rhs": [[_fraction_text(value) for value in row] for row in rhs],
            "symmetric_correction": [
                [_fraction_text(value) for value in row] for row in correction
            ],
            "residual": [[_fraction_text(value) for value in row] for row in residual],
            "decision": "PASS",
        },
        "exact_equal_eigenspace_obstruction_control": {
            "pencil_spectrum": ["0", "0", "1"],
            "zero_eigenspace_dimension": 2,
            "compressed_rhs": [
                [_fraction_text(value) for value in row] for row in zero_compression
            ],
            "witness_entry": {"row": 0, "column": 1, "value": "1"},
            "decision": "OBSTRUCTED_CLASS",
        },
    }


def _gradient_lift_coefficients() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Exact registered state order z=(q,w2,w3), y=(v,w1).
    for spatial, row_offset in ((1, 44), (2, 11), (3, 22)):
        records.append(
            {
                "pencil_component": f"L_{spatial}",
                "shape": [55, 11],
                "nonzero_entries": [
                    {"row": row_offset + field, "column": field, "value": "1"}
                    for field in range(11)
                ],
            }
        )
    return records


def _required_input_manifest() -> list[dict[str, Any]]:
    return [
        {
            "input_id": "physical_spatial_pencil_coefficients",
            "schema_version": "sigma-exact-sparse-linear-P55-pencil-1.0",
            "required_packets": 3,
            "registered_packets": 0,
            "required_shape_each": [55, 55],
            "required_fields": [
                "spatial_axis",
                "sparse_rational_or_radical_entries",
                "matrix_sha256",
                "exact_linearity_residual_zero",
                "minimal_polynomial_on_unit_sphere_certificate",
            ],
            "status": "missing",
            "evidence_boundary": (
                "upstreams bind hashes and point-evaluated pencils but serialize no three "
                "coordinate-free matrix coefficients"
            ),
        },
        {
            "input_id": "physical_gradient_lift_coefficients",
            "schema_version": "sigma-exact-sparse-gradient-lift-pencil-1.0",
            "required_packets": 3,
            "registered_packets": 3,
            "required_shape_each": [55, 11],
            "required_fields": [
                "spatial_axis",
                "sparse_exact_entries",
                "gradient_definition_constraint_residual_zero",
            ],
            "status": "registered_in_this_emitter",
            "registered_nonzero_entries": 33,
        },
        {
            "input_id": "polarized_P55_Taylor_packets",
            "schema_version": "sigma-coordinate-free-P55-Taylor-polarization-packet-1.0",
            "required_packets": 75,
            "registered_packets": 0,
            "count_identity": "15 polarization evaluations * 5 Taylor orders",
            "required_shape_each": [55, 55],
            "status": "missing",
        },
        {
            "input_id": "polarized_K55_Taylor_packets",
            "schema_version": "sigma-coordinate-free-K55-Taylor-polarization-packet-1.0",
            "required_packets": 75,
            "registered_packets": 0,
            "count_identity": "15 polarization evaluations * 5 Taylor orders",
            "required_shape_each": [55, 55],
            "status": "missing",
        },
        {
            "input_id": "polarized_TC2_Taylor_packets",
            "schema_version": "sigma-coordinate-free-TC2-Taylor-polarization-packet-1.0",
            "required_packets": 75,
            "registered_packets": 0,
            "count_identity": "15 polarization evaluations * 5 Taylor orders",
            "required_shape_each": [55, 55],
            "status": "missing",
        },
        {
            "input_id": "lower_Sylvester_correction_recurrence",
            "schema_version": "sigma-coordinate-free-deltaK-orders-zero-through-three-1.0",
            "required_packets": 60,
            "registered_packets": 0,
            "count_identity": "15 polarization evaluations * 4 lower recurrence orders",
            "required_shape_each": [55, 55],
            "status": "missing",
        },
        {
            "input_id": "candidate_normalization_table",
            "schema_version": "sigma-D4-common-shape-candidate-normalization-1.0",
            "required_packets": 12,
            "registered_packets": 0,
            "required_fields": [
                "candidate_id",
                "a10",
                "c20",
                "eta",
                "common_shape_factorization_residual_zero",
            ],
            "status": "missing_as_coordinate_free_emitter_input",
        },
        {
            "input_id": "sphere_mode_normal_form_reducer",
            "schema_version": "sigma-exact-odd-sphere-harmonic-normal-form-1.0",
            "required_packets": 1,
            "registered_packets": 0,
            "required_fields": [
                "degree_ceiling_19",
                "mode_ordering_210",
                "sphere_ideal_reduction",
                "exact_coefficient_extractor",
                "replay_certificate",
            ],
            "status": "missing",
        },
    ]


def _validate_config(config: dict[str, Any]) -> None:
    target = config.get("target", {})
    if (
        set(config)
        != {
            "schema_version",
            "global_claim_policy",
            "emission_policy",
            "target",
            "resource_caps",
            "upstream_bindings",
            "content_sha256",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("emission_policy")
        != "emit_exact_symbolic_coefficients_only_never_fill_missing_with_zero"
        or not _hash_matches(config)
        or target.get("state_dimension") != STATE_DIMENSION
        or target.get("spectrum") != SPECTRUM_TEXT
        or target.get("eigenspace_ranks") != EIGENSPACE_RANKS
        or target.get("equal_eigenspace_skew_coordinates") != COKERNEL_COORDINATES
        or target.get("odd_sphere_modes_through_degree_19") != ODD_SPHERE_MODES
        or target.get("required_output_rows") != REQUIRED_ROWS
        or target.get("recurrence_orders") != [1, 2, 3, 4]
        or target.get("polarization_evaluations") != 15
        or target.get("candidate_specializations") != 12
        or config.get("resource_caps") != EXPECTED_CAPS
        or set(config.get("upstream_bindings", {})) != set(EXPECTED_UPSTREAMS)
    ):
        raise SymbolicRecurrenceEmitterReadinessError("invalid emitter readiness config")


def _validate_upstream(name: str, value: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_UPSTREAMS[name]
    if (
        not _hash_matches(value)
        or value.get("content_sha256") != expected["content_sha256"]
        or value.get("status") != expected["status"]
        or value.get("errors") != []
    ):
        raise SymbolicRecurrenceEmitterReadinessError(f"upstream seal mismatch: {name}")
    if name == "coefficient_map_registration_audit":
        counts = value.get("counts", {})
        if (
            counts.get("required_coefficient_rows") != REQUIRED_ROWS
            or counts.get("registered_coefficient_rows") != 0
            or counts.get("global_numerator_polynomials_materialized_upstream") != 0
            or value.get("phase_two", {}).get("attempted") is not False
        ):
            raise SymbolicRecurrenceEmitterReadinessError("registration boundary mismatch")
    elif name == "full_sphere_readiness_gate":
        counts = value.get("counts", {})
        if (
            counts.get("unknown_columns") != 33_880
            or counts.get("symbolic_row_descriptors") != REQUIRED_ROWS
            or counts.get("coefficient_entries_materialized") != 0
        ):
            raise SymbolicRecurrenceEmitterReadinessError("full-sphere boundary mismatch")
    elif name == "canonical_D4_obstruction_certificate":
        counts = value.get("counts", {})
        if (
            counts.get("candidate_obstructions_certified") != 12
            or counts.get("compression_generic_rank") != 2
        ):
            raise SymbolicRecurrenceEmitterReadinessError("D4 obstruction boundary mismatch")
    elif name == "rational_chart_counterexample_gate":
        reduction = value.get("exact_gate", {}).get("symbolic_chart_reduction", {})
        if reduction.get("global_two_variable_numerator_polynomials_materialized") != 0:
            raise SymbolicRecurrenceEmitterReadinessError("chart emission boundary mismatch")
    else:
        reference = value.get("reference_sylvester_space", {})
        if (
            reference.get("state_dimension") != STATE_DIMENSION
            or reference.get("spectrum") != SPECTRUM_TEXT
            or reference.get("eigenspace_ranks") != EIGENSPACE_RANKS
            or reference.get("equal_eigenspace_cokernel_dimension") != COKERNEL_COORDINATES
        ):
            raise SymbolicRecurrenceEmitterReadinessError("reference space mismatch")
    return {
        "path_status": "verified",
        "schema_version": value.get("schema_version"),
        "status": value["status"],
        "content_sha256": value["content_sha256"],
    }


def _checkpoint(
    projector_recipes: list[dict[str, Any]],
    lift: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "projector_recipe_sha256": hashlib.sha256(_canonical_bytes(projector_recipes)).hexdigest(),
        "gradient_lift_sha256": hashlib.sha256(_canonical_bytes(lift)).hexdigest(),
        "required_input_manifest_sha256": hashlib.sha256(_canonical_bytes(manifest)).hexdigest(),
        "row_emission_cursor": {
            "next_cokernel_coordinate": 0,
            "next_odd_sphere_mode": 0,
            "next_flat_row_offset": 0,
        },
        "emitted_rows": [],
        "emitted_rhs_rows": [],
        "emitted_sparse_entries": [],
        "counts": {
            "projector_recipes_registered": len(projector_recipes),
            "gradient_lift_matrix_packets_registered": len(lift),
            "gradient_lift_nonzero_entries_registered": sum(
                len(record["nonzero_entries"]) for record in lift
            ),
            "required_input_manifest_records": len(manifest),
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "missing_output_rows": REQUIRED_ROWS,
            "emitted_rhs_rows": 0,
            "missing_rhs_rows": REQUIRED_ROWS,
            "emitted_sparse_entries": 0,
        },
        "complete": False,
        "first_missing_input": "physical_spatial_pencil_coefficients",
        "first_missing_output_row": {
            "flat_offset": 0,
            "equal_eigenspace_cokernel_coordinate": 0,
            "odd_sphere_mode": 0,
        },
    }
    checkpoint["content_sha256"] = _content_hash(checkpoint)
    return checkpoint


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path.resolve())
    _validate_config(config)
    receipts: dict[str, Any] = {}
    for name, binding in config["upstream_bindings"].items():
        if set(binding) != {"path", "content_sha256"}:
            raise SymbolicRecurrenceEmitterReadinessError(f"invalid upstream binding: {name}")
        if binding["content_sha256"] != EXPECTED_UPSTREAMS[name]["content_sha256"]:
            raise SymbolicRecurrenceEmitterReadinessError(f"config upstream hash mismatch: {name}")
        document = _load_json(_resolve_under(root, binding["path"]))
        receipts[name] = _validate_upstream(name, document) | {"path": binding["path"]}

    projector_recipes = lagrange_projector_recipes(SPECTRUM)
    lift = _gradient_lift_coefficients()
    manifest = _required_input_manifest()
    controls = _small_model_controls()
    total_projector_terms = sum(
        len(recipe["coefficients_low_to_high"]) for recipe in projector_recipes
    )
    if (
        total_projector_terms > config["resource_caps"]["maximum_projector_recipe_terms"]
        or len(manifest) > config["resource_caps"]["maximum_required_input_manifest_records"]
    ):
        raise SymbolicRecurrenceEmitterReadinessError("readiness construction cap exceeded")
    checkpoint = _checkpoint(projector_recipes, lift, manifest)
    checkpoint_bytes = len(_canonical_bytes(checkpoint))
    if checkpoint_bytes > config["resource_caps"]["maximum_checkpoint_bytes"]:
        raise SymbolicRecurrenceEmitterReadinessError("checkpoint byte cap exceeded")
    registered_input_packets = sum(record["registered_packets"] for record in manifest)
    required_input_packets = sum(record["required_packets"] for record in manifest)
    missing_input_packets = required_input_packets - registered_input_packets
    source_path = _resolve_under(root, SOURCE_PATH)
    test_path = _resolve_under(root, TEST_PATH)
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "config_sha256": config["content_sha256"],
        "implementation_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(source_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(test_path)},
        },
        "upstream_receipts": receipts,
        "exact_Lagrange_projector_recipes": {
            "operator_variable": "P(n)",
            "formula": "Pi_lambda(P)=product_(mu!=lambda)(P-mu*I)/(lambda-mu)",
            "degree": 6,
            "recipes": projector_recipes,
            "recipes_verified_by_exact_Kronecker_evaluation": True,
            "matrix_projectors_evaluated": False,
            "matrix_evaluation_blocker": "three_exact_sparse_P55_spatial_pencil_matrices_missing",
        },
        "exact_gradient_lift_pencil": {
            "formula": "L(n)=n1*L1+n2*L2+n3*L3",
            "state_order": "z=(q,w2,w3), y=(v,w1)",
            "field_columns": 11,
            "spatial_gradient_rows": 33,
            "coefficient_matrices": lift,
            "registered": True,
        },
        "required_symbolic_input_manifest": manifest,
        "bounded_sparse_emission_checkpoint": checkpoint,
        "small_exact_controls": controls,
        "resource_admission": {
            "caps": config["resource_caps"],
            "projector_recipe_terms": total_projector_terms,
            "required_input_manifest_records": len(manifest),
            "checkpoint_bytes": checkpoint_bytes,
            "all_readiness_objects_within_caps": True,
        },
        "phase_two": {
            "decision": "BLOCK",
            "admitted": False,
            "attempted": False,
            "PASS": False,
            "OBSTRUCTED_CLASS": False,
            "BLOCK": True,
            "admission_rule": (
                "all required symbolic input packets registered, all 117180 coefficient rows "
                "and RHS rows exactly emitted and replayed, and zero rows missing"
            ),
        },
        "counts": {
            "upstream_seals_verified": len(receipts),
            "Lagrange_projector_recipes_registered": len(projector_recipes),
            "Lagrange_projector_scalar_terms": total_projector_terms,
            "matrix_projectors_evaluated": 0,
            "gradient_lift_matrix_packets_registered": len(lift),
            "gradient_lift_nonzero_entries_registered": 33,
            "required_symbolic_input_packets": required_input_packets,
            "registered_symbolic_input_packets": registered_input_packets,
            "missing_symbolic_input_packets": missing_input_packets,
            "required_output_rows": REQUIRED_ROWS,
            "emitted_output_rows": 0,
            "missing_output_rows": REQUIRED_ROWS,
            "emitted_rhs_rows": 0,
            "missing_rhs_rows": REQUIRED_ROWS,
            "emitted_sparse_entries": 0,
            "small_model_PASS_controls": 1,
            "small_model_OBSTRUCTED_CLASS_controls": 1,
            "phase_two_solve_attempts": 0,
            "negative_controls": len(NEGATIVE_CONTROLS),
        },
        "claims": {
            "all_five_upstream_content_seals_verified": True,
            "all_seven_exact_Lagrange_projector_recipes_registered": True,
            "exact_gradient_lift_pencil_registered": True,
            "exact_required_symbolic_input_schema_closed": True,
            "small_exact_PASS_and_OBSTRUCTED_CLASS_controls_verified": True,
        }
        | {claim: False for claim in sorted(FALSE_CLAIMS)},
        "negative_controls": NEGATIVE_CONTROLS,
        "scope": (
            "Exact construction of all seven degree-six Lagrange projector polynomial "
            "recipes in the operator variable P(n), the three exact sparse gradient-lift "
            "coefficient matrices, a closed required-input manifest, and a bounded emission "
            "checkpoint. The sealed artifacts provide hashes and point evaluations but not "
            "the three coordinate-free P55 pencil matrices or the 15-by-orders-zero-through-"
            "four P55/K55/TC2 Taylor packets, so matrix projectors and all 117,180 coefficient "
            "and RHS rows remain un-emitted rather than being assigned zero. No full-sphere "
            "D4, D2F, high-atom, TC2, H7, tube, PDE, lifespan, local, covariant, candidate-pass, "
            "or candidate-rejection claim follows."
        ),
        "first_blocker": (
            "register_the_three_exact_sparse_coordinate_free_P55_spatial_pencil_matrices_"
            "with_their_unit_sphere_minimal_polynomial_certificate"
        ),
        "errors": [],
    }
    artifact["content_sha256"] = _content_hash(artifact)
    return artifact


def validate_campaign(document: dict[str, Any], project_root: Path | None = None) -> None:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    expected = build_campaign(root, root / CONFIG_PATH)
    checkpoint = document.get("bounded_sparse_emission_checkpoint", {})
    if (
        document != expected
        or not _hash_matches(document)
        or not isinstance(checkpoint, dict)
        or not _hash_matches(checkpoint)
    ):
        raise SymbolicRecurrenceEmitterReadinessError("campaign replay mismatch")


def write_campaign(document: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root, args.config)
    write_campaign(document, args.output)
    validate_campaign(document, args.project_root)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
